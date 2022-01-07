#include "muzero-cpp/shared_stats.h"

#include <chrono>
#include <thread>

#include "tests/test_macros.h"

namespace muzero_cpp {
namespace {

// Test adding/sampling from the buffer
void save_load_test() {
    const int training_step = 5;
    const int num_played_games = 100;
    const int num_played_steps = 10;
    const int evaluator_episode_length = 100;
    const double evaluator_total_reward = 1;
    const double evaluator_mean_value = 2;
    const double evaluator_muzero_reward = -3;
    const double evaluator_opponent_reward = -4.32;
    const double total_loss = 12.5;
    const double value_loss = 13;
    const double policy_loss = -1;
    const double reward_loss = 0;
    const int logging_step = 12;
    {
        // Create stats
        SharedStats stats;
        stats.set_path(".");
        for (int i = 0; i < training_step; ++i) {
            stats.set_loss(total_loss, value_loss, policy_loss, reward_loss);
            stats.set_metric_logger();
        }
        stats.set_evaluator_stats(evaluator_episode_length, evaluator_total_reward, evaluator_mean_value,
                                  evaluator_muzero_reward, evaluator_opponent_reward);
        stats.add_actor_stats(num_played_steps, num_played_games);
        for (int i = 0; i < (logging_step - training_step); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            stats.set_metric_logger();
        }
        // Write to file
        stats.save(0);

        SharedStats::Stats s = stats.get_all_stats();
        REQUIRE_EQUAL(s.training_step, training_step);
        REQUIRE_EQUAL(s.num_played_games, num_played_games);
        REQUIRE_EQUAL(s.num_played_steps, num_played_steps);
        REQUIRE_EQUAL(s.evaluator_episode_length, evaluator_episode_length);
        REQUIRE_EQUAL(s.evaluator_total_reward, evaluator_total_reward);
        REQUIRE_EQUAL(s.evaluator_mean_value, evaluator_mean_value);
        REQUIRE_EQUAL(s.evaluator_muzero_reward, evaluator_muzero_reward);
        REQUIRE_EQUAL(s.evaluator_opponent_reward, evaluator_opponent_reward);
        REQUIRE_EQUAL(s.total_loss, total_loss);
        REQUIRE_EQUAL(s.value_loss, value_loss);
        REQUIRE_EQUAL(s.policy_loss, policy_loss);
        REQUIRE_EQUAL(s.reward_loss, reward_loss);
        REQUIRE_EQUAL(s.logging_step, logging_step);
        REQUIRE_TRUE(s.selfplay_speed > 1);
        REQUIRE_TRUE(s.train_speed > 0);
    }

    {
        // Create and load new stats
        SharedStats stats;
        stats.set_path(".");
        stats.load(0);

        SharedStats::Stats s = stats.get_all_stats();
        REQUIRE_EQUAL(s.training_step, training_step);
        REQUIRE_EQUAL(s.num_played_games, num_played_games);
        REQUIRE_EQUAL(s.num_played_steps, num_played_steps);
        REQUIRE_EQUAL(s.evaluator_episode_length, evaluator_episode_length);
        REQUIRE_EQUAL(s.evaluator_total_reward, evaluator_total_reward);
        REQUIRE_EQUAL(s.evaluator_mean_value, evaluator_mean_value);
        REQUIRE_EQUAL(s.evaluator_muzero_reward, evaluator_muzero_reward);
        REQUIRE_EQUAL(s.evaluator_opponent_reward, evaluator_opponent_reward);
        REQUIRE_EQUAL(s.total_loss, total_loss);
        REQUIRE_EQUAL(s.value_loss, value_loss);
        REQUIRE_EQUAL(s.policy_loss, policy_loss);
        REQUIRE_EQUAL(s.reward_loss, reward_loss);
        REQUIRE_EQUAL(s.logging_step, logging_step);
        REQUIRE_TRUE(s.selfplay_speed > 1);
        REQUIRE_TRUE(s.train_speed > 0);
    }
}

}    // namespace
}    // namespace muzero_cpp

int main() {
    muzero_cpp::save_load_test();
}
