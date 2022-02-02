#ifndef MUZERO_CPP_SHARED_STATS_H_
#define MUZERO_CPP_SHARED_STATS_H_

#include <chrono>
#include <fstream>
#include <queue>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "nop/serializer.h"
#include "nop/utility/stream_reader.h"
#include "nop/utility/stream_writer.h"

namespace muzero_cpp {

const std::string shared_stats_ext = ".nop";

class SharedStats {
public:
    struct Stats {
        int training_step;                   // Number of training steps by the learner thread
        int num_reanalyze_games_;            // Number of games reanalyzed
        int num_reanalyze_steps_;            // Number of steps reanalyzed
        int num_played_games;                // Number of played games by the actors
        int num_played_steps;                // Number of played game steps by the actors
        double evaluator_episode_length;     // Length of games played by evaluator
        double evaluator_total_reward;       // Running average reward sum
        double evaluator_mean_value;         // Running average mean root values
        double evaluator_muzero_reward;      // Running average sum of rewards for muzero player
        double evaluator_opponent_reward;    // Running average sum of rewards for opponent
        double total_loss;                   // Recent total weighted loss by the learner
        double value_loss;                   // Recent value loss by the learner
        double policy_loss;                  // Recent policy loss by the learner
        double reward_loss;                  // Recent reward loss by the learner
        int logging_step;                    // Step which our logger is on
        double selfplay_speed;               // Number of selfplayed steps per second
        double train_speed;                  // Number of training steps per selfplayed step
    };

    SharedStats()
        : training_step_(0),
          num_reanalyze_games_(0),
          num_reanalyze_steps_(0),
          num_played_games_(0),
          num_played_steps_(0),
          evaluator_episode_length_(0),
          evaluator_total_reward_(0),
          evaluator_mean_value_(0),
          evaluator_muzero_reward_(0),
          evaluator_opponent_reward_(0),
          total_loss_(0),
          value_loss_(0),
          policy_loss_(0),
          reward_loss_(0),
          logging_step_(0) {}

    // Increment the training steps
    void add_training_step(int steps) {
        absl::MutexLock lock(&m_);
        training_step_ += steps;
    }

    // Increment number of games reanalyzed
    void add_reanalyze_game(int steps) {
        absl::MutexLock lock(&m_);
        num_reanalyze_games_ += 1;
        num_reanalyze_steps_ += steps;
    }

    // Set the stats from an actor
    void add_actor_stats(int steps, int games) {
        absl::MutexLock lock(&m_);
        num_played_steps_ += steps;
        num_played_games_ += games;
    }

    // Set the stats from the evaluator
    // Since there could be multiple evaluators, the metrics are a running average of the last 10
    void set_evaluator_stats(int episode_length, double total_reward, double mean_value, double muzero_reward,
                             double opponent_reward) {
        absl::MutexLock lock(&m_);
        evaluator_episode_length_.push_back(episode_length);
        evaluator_total_reward_.push_back(total_reward);
        evaluator_mean_value_.push_back(mean_value);
        evaluator_muzero_reward_.push_back(muzero_reward);
        evaluator_opponent_reward_.push_back(opponent_reward);
        if (evaluator_episode_length_.size() > 10) {
            evaluator_episode_length_.erase(evaluator_episode_length_.begin());
            evaluator_total_reward_.erase(evaluator_total_reward_.begin());
            evaluator_mean_value_.erase(evaluator_mean_value_.begin());
            evaluator_muzero_reward_.erase(evaluator_muzero_reward_.begin());
            evaluator_opponent_reward_.erase(evaluator_opponent_reward_.begin());
        }
    }

    // Set the current loss values
    void set_loss(double total_loss, double value_loss, double policy_loss, double reward_loss) {
        absl::MutexLock lock(&m_);
        training_step_ += 1;
        total_loss_ = total_loss;
        value_loss_ = value_loss;
        policy_loss_ = policy_loss;
        reward_loss_ = reward_loss;
    }

    // Increment the metric logger
    void set_metric_logger() {
        logging_step_ += 1;
        time_queue_.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::system_clock::now().time_since_epoch())
                                  .count());
        played_steps_.push_back(num_played_steps_ + num_reanalyze_steps_);
        train_steps_.push_back(training_step_);
        if (time_queue_.size() > 10) {
            time_queue_.erase(time_queue_.begin());
            played_steps_.erase(played_steps_.begin());
            train_steps_.erase(train_steps_.begin());
        }
    }

    // Get all stats, used for logging
    Stats get_all_stats() {
        absl::MutexLock lock(&m_);
        double delta_time = time_queue_.empty() ? 1 : (time_queue_.back() - time_queue_.front()) / 1000;
        double delta_selfplay = played_steps_.empty() ? 0 : (played_steps_.back() - played_steps_.front());
        double delta_train = train_steps_.empty() ? 0 : (train_steps_.back() - train_steps_.front());
        // Running average of last N games for evaluator (in case multiple evaluators running)
        double eval_ep_len =
            (double)std::reduce(evaluator_episode_length_.begin(), evaluator_episode_length_.end()) /
            std::max((int)evaluator_episode_length_.size(), 1);
        double eval_total_reward =
            std::reduce(evaluator_total_reward_.begin(), evaluator_total_reward_.end()) /
            std::max((int)evaluator_total_reward_.size(), 1);
        double eval_mean_value = std::reduce(evaluator_mean_value_.begin(), evaluator_mean_value_.end()) /
                                 std::max((int)evaluator_mean_value_.size(), 1);
        double eval_muzero_reward =
            std::reduce(evaluator_muzero_reward_.begin(), evaluator_muzero_reward_.end()) /
            std::max((int)evaluator_muzero_reward_.size(), 1);
        double eval_opp_reward =
            std::reduce(evaluator_opponent_reward_.begin(), evaluator_opponent_reward_.end()) /
            std::max((int)evaluator_opponent_reward_.size(), 1);

        return {training_step_,
                num_reanalyze_games_,
                num_reanalyze_steps_,
                num_played_games_,
                num_played_steps_,
                eval_ep_len,
                eval_total_reward,
                eval_mean_value,
                eval_muzero_reward,
                eval_opp_reward,
                total_loss_,
                value_loss_,
                policy_loss_,
                reward_loss_,
                logging_step_,
                delta_selfplay / delta_time,
                delta_train / delta_selfplay};
    }

    // Get the running average evaluator muzero reward
    double get_evaluator_muzero_reward() {
        absl::MutexLock lock(&m_);
        return std::reduce(evaluator_muzero_reward_.begin(), evaluator_muzero_reward_.end()) /
               std::max((int)evaluator_muzero_reward_.size(), 1);
    }

    // Get total number of training steps
    int get_training_step() {
        absl::MutexLock lock(&m_);
        return training_step_;
    }

    // Get total number of played games
    int get_num_played_games() {
        absl::MutexLock lock(&m_);
        return num_played_games_;
    }

    // Get total number of played steps
    int get_num_played_steps() {
        absl::MutexLock lock(&m_);
        return num_played_steps_;
    }

    // Get the current step of logging
    int get_logging_step() {
        absl::MutexLock lock(&m_);
        return logging_step_;
    }

    // Set the path to load/save from
    void set_path(const std::string &base_path) {
        absl::MutexLock lock(&m_);
        path_ = base_path;
    }

    // Save the shared stats
    void save(int checkpoint_step) {
        absl::MutexLock lock(&m_);
        const std::string path = absl::StrCat(path_, "shared_stats-", checkpoint_step, shared_stats_ext);
        nop::Serializer<nop::StreamWriter<std::ofstream>> serializer{path};
        serializer.Write(*this);
    }

    // Load the shared stats
    void load(int checkpoint_step) {
        absl::MutexLock lock(&m_);
        const std::string path = absl::StrCat(path_, "shared_stats-", checkpoint_step, shared_stats_ext);
        nop::Deserializer<nop::StreamReader<std::ifstream>> deserializer{path};
        deserializer.Read(this);
    }

private:
    int training_step_;                                // Number of training steps by the learner thread
    int num_reanalyze_games_;                          // Number of games reanalyzed
    int num_reanalyze_steps_;                          // Number of steps reanalyzed
    int num_played_games_;                             // Number of played games by the actors
    int num_played_steps_;                             // Number of played game steps by the actors
    std::vector<int> evaluator_episode_length_;        // Length of games played by evaluator
    std::vector<double> evaluator_total_reward_;       // Running average reward sum
    std::vector<double> evaluator_mean_value_;         // Running average mean root values
    std::vector<double> evaluator_muzero_reward_;      // Running average sum of rewards for muzero player
    std::vector<double> evaluator_opponent_reward_;    // Running average sum of rewards for opponent
    double total_loss_;                                // Recent total weighted loss by the learner
    double value_loss_;                                // Recent value loss by the learner
    double policy_loss_;                               // Recent policy loss by the learner
    double reward_loss_;                               // Recent reward loss by the learner
    int logging_step_;                                 // Step which our logger is on
    std::vector<uint64_t> time_queue_;                 // Queue of when we last updated logging metrics
    std::vector<int> played_steps_;                    // Queue of previous number of selfplayed steps
    std::vector<int> train_steps_;                     // Queue of previous training steps
    std::string path_;                                 // Base path for loading/saving
    absl::Mutex m_;                                    // Lock for multithreading access

    // Required data to be stored/loaded
    NOP_STRUCTURE(SharedStats, training_step_, num_reanalyze_games_, num_reanalyze_steps_, num_played_games_,
                  num_played_steps_, evaluator_episode_length_, evaluator_total_reward_,
                  evaluator_mean_value_, evaluator_muzero_reward_, evaluator_opponent_reward_, total_loss_,
                  value_loss_, policy_loss_, reward_loss_, logging_step_, time_queue_, played_steps_,
                  train_steps_);
};

}    // namespace muzero_cpp

#endif    // MUZERO_CPP_SHARED_STATS_H_