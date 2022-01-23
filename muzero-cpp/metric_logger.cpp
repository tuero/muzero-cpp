#include "muzero-cpp/metric_logger.h"

#include <algorithm>
#include <filesystem>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "tensorboard_logger.h"

namespace muzero_cpp {
using namespace muzero_config;

void metric_logger(const MuZeroConfig& config, std::shared_ptr<SharedStats> shared_stats,
                   util::StopToken* stop) {
    // Create directory if not exists
    std::filesystem::create_directories(absl::StrCat(config.path, "/metrics/"));

    // Tensorboard writer from external library
    // If resuming, writer will append to the existing tfevents file
    TensorBoardLogger writer(absl::StrCat(config.path, "/metrics/", "tfevents.pb").c_str(), config.resume);

    shared_stats->set_metric_logger();

    // Continue to log metrics until we are told to stop
    for (int step = shared_stats->get_logging_step(); !stop->stop_requested(); ++step) {
        // Sleep on first step so we can let data fill
        if (step == 1) { absl::SleepFor(absl::Milliseconds(2500)); }
        // Get stats
        shared_stats->set_metric_logger();
        SharedStats::Stats stats = shared_stats->get_all_stats();

        // Evaluator
        writer.add_scalar("1.Evaluator/1.Episode_length", step, stats.evaluator_episode_length);
        writer.add_scalar("1.Evaluator/2.Total_reward", step, stats.evaluator_total_reward);
        writer.add_scalar("1.Evaluator/3.Mean_value", step, stats.evaluator_mean_value);
        writer.add_scalar("1.Evaluator/4.Muzero_reward", step, stats.evaluator_muzero_reward);
        writer.add_scalar("1.Evaluator/5.Opponent_reward", step, stats.evaluator_opponent_reward);

        // Workers
        writer.add_scalar("2.Workers/1.Self_played_games", step, (double)stats.num_played_games);
        writer.add_scalar("2.Workers/2.Self_played_steps", step, (double)stats.num_played_steps);
        writer.add_scalar("2.Workers/3.Training_steps", step, (double)stats.training_step);
        writer.add_scalar("2.Workers/4.Reanalyze_games", step, (double)stats.num_reanalyze_games_);
        writer.add_scalar("2.Workers/5.Reanalyze_steps", step, (double)stats.num_reanalyze_steps_);
        writer.add_scalar("2.Workers/6.Training_per_selfplay_step_ratio", step,
                          (double)stats.training_step / std::max(1, stats.num_played_steps));

        // Loss
        writer.add_scalar("3.Loss/1.Total_weighted_loss", step, stats.total_loss);
        writer.add_scalar("3.Loss/2.Value_loss", step, stats.value_loss);
        writer.add_scalar("3.Loss/3.Policy_loss", step, stats.policy_loss);
        writer.add_scalar("3.Loss/4.Reward_loss", step, stats.reward_loss);

        // Log to console
        std::cout << "\33[2K\r";
        std::cout << absl::StrFormat("Train step: %d/%d, Selfplay games: %d, Reanalyze "
                                     "trajs: %d, Train speed: %.2f s/s, Selfplay speed %.2f steps/s",
                                     stats.training_step, config.max_training_steps, stats.num_played_games,
                                     stats.num_reanalyze_games_, stats.train_speed, stats.selfplay_speed);
        std::fflush(stdout);

        // Sleep
        absl::SleepFor(absl::Milliseconds(1000));
    }
}

}    // namespace muzero_cpp