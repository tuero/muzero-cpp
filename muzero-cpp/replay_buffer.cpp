#include "muzero-cpp/replay_buffer.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>

#include "absl/strings/str_cat.h"

namespace muzero_cpp {
using namespace types;
namespace buffer {

PrioritizedReplayBuffer::PrioritizedReplayBuffer(const muzero_config::MuZeroConfig &config)
    : alpha_(config.per_alpha),
      beta_(config.per_beta),
      epsilon_(config.per_epsilon),
      beta_increment_(config.per_beta_increment),
      discount_(config.discount),
      batch_size_(config.batch_size),
      min_sample_size_(config.min_sample_size),
      num_stacked_observations_(config.stacked_observations),
      action_channels_(config.action_channels),
      td_steps_(config.td_steps),
      num_unroll_steps_(config.num_unroll_steps),
      obs_shape_(config.observation_shape),
      action_rep_func_(config.action_representation_initial),
      tree_(config.replay_buffer_size, absl::StrCat(config.path, "/buffer/")),
      path_(absl::StrCat(config.path, "/buffer/")) {
    std::filesystem::create_directories(path_);
}

// Check if enough items are stored to start sampling.
bool PrioritizedReplayBuffer::can_sample() const {
    return tree_.get_size() >= min_sample_size_;
}

// Get the number of stored items.
int PrioritizedReplayBuffer::size() const {
    return tree_.get_size();
}

// Helper to convert errors to priorities (See Appendix G Training).
void PrioritizedReplayBuffer::error_to_priorities(std::vector<double> &errors) const {
    for (auto &e : errors) {
        e = std::pow(std::abs(e) + epsilon_, alpha_);
    }
}

// Insert a game history into the replay buffer.
void PrioritizedReplayBuffer::save_game_history(const GameHistory &game_history) {
    absl::MutexLock lock(&m_);
    // Initial priorities (|root value - n-step return|, Appendix G Training)
    std::vector<double> errors;
    errors.reserve(game_history.root_values.size());

    // Find errors and convert to priorities
    for (int i = 0; i < (int)game_history.root_values.size(); ++i) {
        errors.push_back(game_history.root_values[i] -
                         game_history.compute_target_value(i, td_steps_, discount_));
    }
    error_to_priorities(errors);
    assert(game_history.child_visits.size() > 0);
    tree_.add(errors, game_history);
}

// Update the priorities of the sample from observed errors.
void PrioritizedReplayBuffer::update_history_priorities(const std::vector<int> &indices,
                                                        std::vector<double> &errors) {
    absl::MutexLock lock(&m_);
    assert(indices.size() == errors.size());
    error_to_priorities(errors);
    for (int i = 0; i < (int)indices.size(); ++i) {
        tree_.update(indices[i], errors[i]);
    }
}

// Sample a single game uniform randomly, used for reanalyze
std::tuple<int, GameHistory> PrioritizedReplayBuffer::sample_game(std::mt19937 &rng) {
    absl::MutexLock lock(&m_);
    std::uniform_int_distribution<> uniform_dist(1, tree_.get_num_histories());
    int history_id = uniform_dist(rng);
    GameHistory history = tree_.get_history(history_id);
    return {history_id, history};
}

// Get a batched sample from the replay buffer
Batch PrioritizedReplayBuffer::sample(std::mt19937 &rng) {
    absl::MutexLock lock(&m_);
    // Update beta
    beta_ = std::min(1.0, beta_ + beta_increment_);
    double priority_segment = tree_.total_priority() / batch_size_;

    Batch samples;
    samples.num_samples = batch_size_;

    // Sample N items for batch
    for (int i = 0; i < batch_size_; ++i) {
        // Get sample via priority weighting
        std::uniform_real_distribution<double> uniform_dist(priority_segment * i, priority_segment * (i + 1));
        double value = uniform_dist(rng);
        std::tuple<int, double, int, GameHistory> sample = tree_.get_leaf(value);
        // Unpack
        int index = std::get<0>(sample);
        double priority = std::get<1>(sample);
        int step = std::get<2>(sample);
        GameHistory game_history = std::get<3>(sample);

        // Make target values/rewards/policy/actions and insert into batch
        game_history.make_target(step, td_steps_, num_unroll_steps_, discount_, samples, rng);
        samples.gradient_scale.push_back(
            1.0 / std::min(num_unroll_steps_, (int)game_history.action_history.size() - step));

        // Make a stacked observation + action sequence and insert into batch
        const Observation current_stacked_obs = game_history.get_stacked_observations(
            step, num_stacked_observations_, obs_shape_, action_channels_, action_rep_func_);
        samples.priorities.push_back(priority);
        samples.indices.push_back(index);
        samples.stacked_observations.insert(samples.stacked_observations.end(), current_stacked_obs.begin(),
                                            current_stacked_obs.end());
    }

    // Get importance sampling weights
    for (int i = 0; i < batch_size_; ++i) {
        samples.priorities[i] =
            std::pow(tree_.get_size() * samples.priorities[i] / tree_.total_priority(), -beta_);
    }
    double max_value = *std::max_element(samples.priorities.begin(), samples.priorities.end());
    for (int i = 0; i < batch_size_; ++i) {
        samples.priorities[i] /= max_value;
    }

    // Returned values are flattened raw data vectors, need to reshape when converting to tensors
    return samples;
}

// Update the priorities of the sample from observed errors.
void PrioritizedReplayBuffer::update_game_history(int history_id, const GameHistory &game_history) {
    absl::MutexLock lock(&m_);
    tree_.update_hist(history_id, game_history);
}

// Save the replay buffer
void PrioritizedReplayBuffer::save() {
    absl::MutexLock lock(&m_);
    const std::string path = absl::StrCat(path_, "priority_buffer.nop");
    nop::Serializer<nop::StreamWriter<std::ofstream>> serializer{path};
    serializer.Write(this->alpha_);
    serializer.Write(this->beta_);
    tree_.save();
}

// Load the replay buffer
void PrioritizedReplayBuffer::load() {
    absl::MutexLock lock(&m_);
    // Check if we should quick exit because we are missiing files.
    const std::string path = absl::StrCat(path_, "priority_buffer.nop");
    if (!std::filesystem::exists(path)) {
        std::cerr << "Error: " << path << " does not exist. Resuming with empty buffer." << std::endl;
        return;
    }
    std::string tree_path = tree_.get_path();
    if (!std::filesystem::exists(tree_path)) {
        std::cerr << "Error: " << tree_path << " does not exist. Resuming with empty buffer." << std::endl;
        return;
    }
    nop::Deserializer<nop::StreamReader<std::ifstream>> deserializer{path};
    deserializer.Read(&(this->alpha_));
    deserializer.Read(&(this->beta_));
    tree_.load();
}

}    // namespace buffer
}    // namespace muzero_cpp