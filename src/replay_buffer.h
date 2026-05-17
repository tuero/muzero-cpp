#ifndef MUZERO_REPLAY_BUFFER_H_
#define MUZERO_REPLAY_BUFFER_H_

#include <muzero/config.h>
#include <muzero/types.h>

#include "game_history.h"

#include <nop/serializer.h>
#include <nop/utility/stream_reader.h>
#include <nop/utility/stream_writer.h>

#include <absl/strings/str_cat.h>
#include <absl/synchronization/mutex.h>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace muzero::buffer {

// Sum tree binary tree structure for the prioritized replay
template <typename T>
class SumTree {
public:
    SumTree(int capacity, std::string path, std::string name)
        : capacity_(capacity),
          num_entries_(0),
          position_(0),
          map_counter_(0),
          path_(std::move(path)),
          tree_(static_cast<std::size_t>(2 * capacity - 1), 0),
          data_(static_cast<std::size_t>(capacity), {0, 0}),
          name_(std::move(name))
    {
        assert(capacity > 0);
    }
    SumTree() = delete;

    /**
     * Update the priority of the given index.
     * @param index The index to update
     * @param priority The updated priority value
     */
    void update(int index, double priority)
    {
        double change = priority - tree_[static_cast<std::size_t>(index)];
        tree_[static_cast<std::size_t>(index)] = priority;
        while (index != 0) {
            index = (index - 1) / 2;
            tree_[static_cast<std::size_t>(index)] += change;
        }
    }

    void update_hist(int history_id, const T &hist)
    {
        // Only update if we have not lost reference to this game history ID
        if (hist_map_.find(history_id) != hist_map_.end()) {
            hist_map_[history_id] = hist;
        }
    }

    /**
     * Get the total individual game histories stored
     * @note This is not the total samples of the buffer
     * @note This is used in uniform random sampling for reanalyze
     */
    auto get_num_histories() const -> int
    {
        return map_counter_;
    }

    /**
     * Get a stored history representing the given ID
     * @note This is used in uniform random sampling for reanalyze
     * @param history_id The ID of the history
     * @return The game history
     */
    auto get_history(int history_id) -> T &
    {
        return hist_map_[history_id];
    }

    /**
     * Store the trajectory of priorities and history.
     * @param priorities List of priorities for the game played
     * @param history The history trajectory of the game played
     */
    void add(const std::vector<double> &priorities, const T &history)
    {
        map_counter_ += 1;
        hist_map_[map_counter_] = history;
        for (int step = 0; step < (int)priorities.size(); ++step) {
            int index = position_ + capacity_ - 1;
            // If we are overwritting previous data, decrement history counter
            // and potentially remove stored history
            int old_hist_id = data_[static_cast<std::size_t>(position_)].second;
            if (old_hist_id > 0) {
                --hist_count_[old_hist_id];
                if (hist_count_[old_hist_id] == 0) {
                    num_entries_ -= static_cast<int>(hist_map_[old_hist_id].root_values.size());
                    hist_map_.erase(old_hist_id);
                }
            }
            // Add
            data_[static_cast<std::size_t>(position_)] = {step, map_counter_};
            ++hist_count_[map_counter_];
            update(index, priorities[static_cast<std::size_t>(step)]);
            position_ = (position_ + 1) % capacity_;
            ++num_entries_;
        }
    }

    /**
     * Get the data stored represented by the given value.
     * @param value The value to search for
     * @return A tuple containing the leaf index, the value at the leaf index,
     *         the sample game step, and the sample game history
     */
    auto get_leaf(double value)
    {
        int leaf_index = retrieve(0, value);
        int buffer_index = leaf_index - capacity_ + 1;
        return std::make_tuple(
            leaf_index,
            tree_[static_cast<std::size_t>(leaf_index)],
            data_[static_cast<std::size_t>(buffer_index)].first,
            std::ref(hist_map_[data_[static_cast<std::size_t>(buffer_index)].second])
        );
    }

    /**
     * Get the total items stored in the tree.
     * @return the number of items stored
     */
    auto get_size() const -> int
    {
        return num_entries_;
    }

    /**
     * Get the total priority stored.
     * @return The total priority
     */
    auto total_priority() const -> double
    {
        return tree_[0];
    }

    /**
     * Get the stored path that the sum tree resides.
     * @return full path of sum tree
     */
    [[nodiscard]] auto get_path() -> std::string
    {
        return absl::StrCat(path_, name_, ".nop");
    }

    /**
     * Save the SumTree for resume training
     */
    void save()
    {
        const std::string path = absl::StrCat(path_, name_, ".nop");
        nop::Serializer<nop::StreamWriter<std::ofstream>> serializer{path};
        serializer.Write(this->num_entries_);
        serializer.Write(this->position_);
        serializer.Write(this->map_counter_);
        serializer.Write(this->tree_);
        serializer.Write(this->hist_map_);
        serializer.Write(this->hist_count_);
        serializer.Write(this->data_);
    }

    /**
     * Load the SumTree for resume training
     */
    void load()
    {
        // Check if we should quick exit because we are missiing files.
        const std::string path = absl::StrCat(path_, name_, ".nop");
        if (!std::filesystem::exists(path)) {
            std::cerr << "Error: " << path << " does not exist. Resuming with empty buffer." << std::endl;
            return;
        }
        nop::Deserializer<nop::StreamReader<std::ifstream>> deserializer{path};
        deserializer.Read(&(this->num_entries_));
        deserializer.Read(&(this->position_));
        deserializer.Read(&(this->map_counter_));
        deserializer.Read(&(this->tree_));
        deserializer.Read(&(this->hist_map_));
        deserializer.Read(&(this->hist_count_));
        deserializer.Read(&(this->data_));
        if ((int)data_.size() != capacity_) {
            std::cerr << "Fatal error: Attempting to load SumTree of size " << data_.size()
                      << " with configured max size " << capacity_ << "." << std::endl
                      << std::endl;
            std::exit(1);
        }
    }

private:
    // Find index corresponding for the value to search.
    [[nodiscard]] auto retrieve(int index, double value) -> int
    {
        int left = 2 * index + 1;
        int right = left + 1;
        if (left >= static_cast<int>(tree_.size())) {
            return index;
        }
        return (value <= tree_[static_cast<std::size_t>(left)])
                   ? retrieve(left, value)
                   : retrieve(right, value - tree_[static_cast<std::size_t>(left)]);
    }

    int capacity_;                               // The maximum capacity of the sum tree
    int num_entries_;                            // The current number of entries in the tree
    int position_;                               // The current position of storage for the tree
    int map_counter_;                            // Index into map
    std::string path_;                           // Base path for storing tree
    std::vector<double> tree_;                   // The array representing the tree sum values
    std::unordered_map<int, T> hist_map_;        // Map containing stored histories
    std::unordered_map<int, int> hist_count_;    // Map containing reference counts to histories
    std::vector<std::pair<int, int>> data_;      // The data items the tree holds
    std::string name_;                           // Name for storing/loading
};

struct GameSample {
    int id;
    GameHistory game_history;
};

// Prioritized replay
class PrioritizedReplayBuffer {
public:
    PrioritizedReplayBuffer(const Config &config, int max_size, const std::string &name);
    PrioritizedReplayBuffer() = delete;

    /**
     * Check if enough items are stored to start sampling.
     * @return True if enough items are stored to start sampling
     */
    [[nodiscard]] auto can_sample() const -> bool;

    /**
     * Get the number of stored items
     * @return Number of stored items
     */
    [[nodiscard]] auto size() const -> int;

    /**
     * Sample a single game uniform randomly, used for reanalyze
     * Can't just return a ref as it game history could be deleted, need to copy
     * @param rng Source of randomness
     * @return Sampled game history reference and id
     */
    auto sample_game(std::mt19937 &rng) -> GameSample;

    /**
     * Get a batched sample from the replay buffer
     * @param rng Source of randomness
     * @param batch_size Number of samples to get
     * @return Flat vectors representing the priorities, indicies, actions, observations, target rewards,
     * target values, target policies, and gradient scale. The caller needs to convert into tensors of the
     * correct size.
     */
    auto sample(std::mt19937 &rng, int batch_size) -> Batch;

    /**
     * Insert a game history into the replay buffer.
     * @param game_history The game history
     */
    void save_game_history(const GameHistory &game_history);

    /**
     * Update the priorities of the sample from observed errors.
     * @param indices The indicies from the tree chosen
     * @param errors The observed errors, used to update the priorities
     */
    void update_history_priorities(const std::vector<int> &indices, std::vector<double> &errors);

    /**
     * Update the stored game history with the given
     * This is used in reanalyze
     * @param history_id The correponding history ID in the sumtree
     * @param game_history The updated history
     */
    void update_game_history(int history_id, const GameHistory &game_history);

    /**
     * Save the replay buffer
     */
    void save();

    /**
     * Load the replay buffer
     */
    void load();

private:
    /**
     * Helper to convert errors to priorities (See Appendix G Training).
     * @param errors The errors to inplace convert
     */
    void error_to_priorities(std::vector<double> &errors) const;

    double alpha_;                    // Priority exponent
    double beta_;                     // Correction for sampling bias
    double epsilon_;                  // Epsilon added to error to avoid 0's
    double beta_increment_;           // How much to increment beta (caps at 1.0)
    double discount_;                 // Discounting factor for future rewards/values
    int min_sample_size_;             // Minimum samples needed to be stored before we can sample
    int num_stacked_observations_;    // Number of stacked observations used
    int action_channels_;             // Numbers of channels in the action representation
    int td_steps_;                    // Number of future td steps to take into account for future value
    int num_unroll_steps_;            // Number of steps to unroll for each sample
    ObservationShape obs_shape_;      // Observation shape
    ActionRepresentationFunction
        action_rep_func_;          // Function to convert raw action ints to the channel representation
    SumTree<GameHistory> tree_;    // Underlying sum tree datastructure
    std::string path_;             // Base path for storing saved buffer
    std::string name_;             // Name of buffer for store/loading
    absl::Mutex m_;                // Lock for multithreading access
};

}    // namespace muzero::buffer

#endif    // MUZERO_REPLAY_BUFFER_H_
