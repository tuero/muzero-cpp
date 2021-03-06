#include "muzero-cpp/replay_buffer.h"

#include <filesystem>
#include <memory>
#include <random>

#include "muzero-cpp/config.h"
#include "muzero-cpp/types.h"
#include "tests/test_macros.h"

namespace muzero_cpp {
using namespace muzero_config;
using namespace types;
namespace buffer {
namespace {

constexpr int NUM_ACTIONS = 8;

Observation action_to_observation(Action action) {
    Observation obs(1 * 10 * 10, (double)action / NUM_ACTIONS);
    return obs;
}

// Test creating a game history trajectory
GameHistory make_history(int idx) {
    GameHistory history;
    history.action_history = std::vector<Action>(10, idx % NUM_ACTIONS);
    history.reward_history = std::vector<double>(10, idx);
    history.to_play_history = std::vector<Player>(10, idx % 2);
    history.root_values = std::vector<double>(10, idx % 4);
    for (int i = 0; i < 10; ++i) {
        history.child_visits.push_back(std::vector<double>(NUM_ACTIONS, i));
        history.observation_history.push_back(Observation(3 * 10 * 10, 1));
    }
    return history;
}

// Test adding/sampling from the buffer
void replay_buffer_test() {
    MuZeroConfig config;
    config.per_alpha = 0.4;
    config.per_beta = 0.6;
    config.per_epsilon = 0.01;
    config.per_beta_increment = 0.001;
    config.discount = 0.99;
    config.batch_size = 16;
    config.min_sample_size = 128;
    config.stacked_observations = 4;
    config.action_channels = 1;
    config.td_steps = 5;
    config.num_unroll_steps = 5;
    config.observation_shape = ObservationShape{3, 10, 10};
    config.action_representation_initial = action_to_observation;
    config.replay_buffer_size = 10000;
    config.path = ".";
    PrioritizedReplayBuffer replay_buffer(config, config.replay_buffer_size, "buffer");

    // Insert items
    int counter = 0;
    for (int i = 0; i < config.min_sample_size; ++i) {
        REQUIRE_TRUE(replay_buffer.can_sample() == (counter >= config.min_sample_size));
        auto game_history = make_history(i);
        counter += game_history.action_history.size();
        replay_buffer.save_game_history(game_history);
    }

    // Sample
    std::mt19937 rng(0);
    auto samples = replay_buffer.sample(rng, config.batch_size);
    REQUIRE_TRUE(samples.num_samples == config.batch_size);
    REQUIRE_TRUE((int)samples.actions.size() == config.batch_size * (config.num_unroll_steps + 1));
    REQUIRE_TRUE((int)samples.target_rewards.size() == config.batch_size * (config.num_unroll_steps + 1));
    REQUIRE_TRUE((int)samples.target_values.size() == config.batch_size * (config.num_unroll_steps + 1));
    REQUIRE_TRUE((int)samples.target_policies.size() ==
                 config.batch_size * NUM_ACTIONS * (config.num_unroll_steps + 1));
    REQUIRE_TRUE((int)samples.gradient_scale.size() == config.batch_size);

    // Update priorities
    std::vector<double> errors;
    std::vector<int> indices;
    for (int i = 0; i < samples.num_samples; ++i) {
        indices.push_back(samples.indices[i]);
        errors.push_back(samples.priorities[i] * 0.5);
    }

    // Update priorities
    replay_buffer.update_history_priorities(indices, errors);
}

}    // namespace
}    // namespace buffer
}    // namespace muzero_cpp

int main() {
    muzero_cpp::buffer::replay_buffer_test();
}
