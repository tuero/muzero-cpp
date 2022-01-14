#ifndef MUZERO_CPP_REPLAY_BUFFER_H_
#define MUZERO_CPP_REPLAY_BUFFER_H_

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "muzero-cpp/config.h"
#include "muzero-cpp/types.h"
#include "nop/serializer.h"
#include "nop/utility/stream_reader.h"
#include "nop/utility/stream_writer.h"

namespace muzero_cpp {
namespace buffer {

// Sum tree binary tree structure for the prioritized replay
template <typename T>
class SumTree {
public:
    SumTree(int capacity, const std::string &path)
        : capacity_(capacity),
          num_entries_(0),
          position_(0),
          map_counter_(0),
          path_(path),
          tree_(2 * capacity - 1, 0),
          data_(capacity, {0, 0}) {}
    SumTree() = delete;

    /**
     * Update the priority of the given index.
     * @param index The index to update
     * @param priority The updated priority value
     */
    void update(int index, double priority) {
        double change = priority - tree_[index];
        tree_[index] = priority;
        while (index != 0) {
            index = (index - 1) / 2;
            tree_[index] += change;
        }
    }

    void update_hist(int history_id, const T &hist) {
        // Only update if we have not lost reference to this game history ID
        if (hist_map_.find(history_id) != hist_map_.end()) { hist_map_[history_id] = hist; }
    }

    /**
     * Get the total individual game histories stored
     * @note This is not the total samples of the buffer
     * @note This is used in uniform random sampling for reanalyze
     */
    int get_num_histories() const {
        return map_counter_;
    }

    /**
     * Get a stored history representing the given ID
     * @note This is used in uniform random sampling for reanalyze
     * @param history_id The ID of the history
     * @return The game history
     */
    T get_history(int history_id) {
        return hist_map_[history_id];
    }

    /**
     * Store the trajectory of priorities and history.
     * @param priorities List of priorities for the game played
     * @param history The history trajectory of the game played
     */
    void add(const std::vector<double> &priorities, const T &history) {
        map_counter_ += 1;
        hist_map_[map_counter_] = history;
        for (int step = 0; step < (int)priorities.size(); ++step) {
            int index = position_ + capacity_ - 1;
            // If we are overwritting previous data, decrement history counter
            // and potentially remove stored history
            int old_hist_id = data_[position_].second;
            if (old_hist_id > 0) {
                --hist_count_[old_hist_id];
                if (hist_count_[old_hist_id] == 0) { hist_count_.erase(old_hist_id); }
            }
            // Add
            data_[position_] = {step, map_counter_};
            ++hist_count_[map_counter_];
            update(index, priorities[step]);
            position_ = (position_ + 1) % capacity_;
            num_entries_ = std::min(num_entries_ + 1, capacity_);
        }
    }

    /**
     * Get the data stored represented by the given value.
     * @param value The value to search for
     * @return A tuple containing the leaf index, the value at the leaf index,
     *         the sample game step, and the sample game history
     */
    auto get_leaf(double value) {
        int leaf_index = retrieve(0, value);
        int buffer_index = leaf_index - capacity_ + 1;
        return std::make_tuple(leaf_index, tree_[leaf_index], data_[buffer_index].first,
                               std::ref(hist_map_[data_[buffer_index].second]));
    }

    /**
     * Get the total items stored in the tree.
     * @return the number of items stored
     */
    int get_size() const {
        return num_entries_;
    }

    /**
     * Get the total priority stored.
     * @return The total priority
     */
    double total_priority() const {
        return tree_[0];
    }

    /**
     * Get the stored path that the sum tree resides.
     * @return full path of sum tree
     */
    std::string get_path() {
        return absl::StrCat(path_, "sum_tree.nop");
    }

    /**
     * Save the SumTree for resume training
     */
    void save() {
        const std::string path = absl::StrCat(path_, "sum_tree.nop");
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
    void load() {
        // Check if we should quick exit because we are missiing files.
        const std::string path = absl::StrCat(path_, "sum_tree.nop");
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
    int retrieve(int index, double value) {
        int left = 2 * index + 1;
        int right = left + 1;
        if (left >= (int)tree_.size()) { return index; }
        return (value <= tree_[left]) ? retrieve(left, value) : retrieve(right, value - tree_[left]);
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
};

// Prioritized replay
class PrioritizedReplayBuffer {
public:
    PrioritizedReplayBuffer(const muzero_config::MuZeroConfig &config);
    PrioritizedReplayBuffer() = delete;

    /**
     * Check if enough items are stored to start sampling.
     * @return True if enough items are stored to start sampling
     */
    bool can_sample() const;

    /**
     * Get the number of stored items
     * @return Number of stored items
     */
    int size() const;

    /**
     * Sample a single game uniform randomly, used for reanalyze
     * @param rng Source of randomness
     * @return Sampled game history along with the corresponding index
     */
    std::tuple<int, types::GameHistory> sample_game(std::mt19937 &rng);

    /**
     * Get a batched sample from the replay buffer
     * @param rng Source of randomness
     * @return Flat vectors representing the priorities, indicies, actions, observations, target rewards,
     * target values, target policies, and gradient scale. The caller needs to convert into tensors of the
     * correct size.
     */
    types::Batch sample(std::mt19937 &rng);

    /**
     * Insert a game history into the replay buffer.
     * @param game_history The game history
     */
    void save_game_history(const types::GameHistory &game_history);

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
    void update_game_history(int history_id, const types::GameHistory &game_history);

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

    double alpha_;                         // Priority exponent
    double beta_;                          // Correction for sampling bias
    double epsilon_;                       // Epsilon added to error to avoid 0's
    double beta_increment_;                // How much to increment beta (caps at 1.0)
    double discount_;                      // Discounting factor for future rewards/values
    int batch_size_;                       // Samples per batch
    int min_sample_size_;                  // Minimum samples needed to be stored before we can sample
    int num_stacked_observations_;         // Number of stacked observations used
    int action_channels_;                  // Numbers of channels in the action representation
    int td_steps_;                         // Number of future td steps to take into account for future value
    int num_unroll_steps_;                 // Number of steps to unroll for each sample
    types::ObservationShape obs_shape_;    // Observation shape
    types::ActionRepresentationFunction
        action_rep_func_;                 // Function to convert raw action ints to the channel representation
    SumTree<types::GameHistory> tree_;    // Underlying sum tree datastructure
    std::string path_;                    // Base path for storing saved buffer
    absl::Mutex m_;                       // Lock for multithreading access
};

}    // namespace buffer
}    // namespace muzero_cpp

#endif    // MUZERO_CPP_REPLAY_BUFFER_H_