#include "muzero-cpp/learner.h"

#include <tuple>
#include <vector>

#include "absl/types/optional.h"

namespace muzero_cpp {
using namespace types;
using namespace buffer;

// Learner thread logic.
// Continuously updates the muzero network model
void learn(const muzero_config::MuZeroConfig& config, DeviceManager* device_manager,
           std::shared_ptr<PrioritizedReplayBuffer> replay_buffer,
           std::shared_ptr<PrioritizedReplayBuffer> reanalyze_buffer,
           ThreadedQueue<GameHistory>* trajectory_queue, std::shared_ptr<SharedStats> shared_stats,
           util::StopToken* stop) {
    const int device_id = 0;
    std::mt19937 rng(config.seed);
    double max_reward = NINF_D;

    // Could be resuming, so ask what the current step is
    int step = shared_stats->get_training_step();

    // Continue to learn until asked to stop
    while (!stop->stop_requested() && step <= config.max_training_steps) {
        // Get trajectories from self-play actors and store into buffer
        int queue_size = trajectory_queue->Size();
        int num_trajectories = 0;
        while (!stop->stop_requested() && num_trajectories < queue_size) {
            absl::optional<GameHistory> trajectory = trajectory_queue->Pop();
            if (trajectory) {
                ++num_trajectories;
                replay_buffer->save_game_history(trajectory.value());
                reanalyze_buffer->save_game_history(trajectory.value());
            }
        }

        // Check if we have enough samples
        if (!replay_buffer->can_sample()) {
            absl::SleepFor(absl::Milliseconds(1000));
            continue;
        }

        // Learning step (scope so model given back immediately)
        {
            // Ask manager for model
            DeviceManager::DeviceLoan model = device_manager->Get(config.batch_size, device_id);

            // Sample and send to model
            // Here we find ratio of reanalyze to self play samples
            int num_reanalyze_samples = static_cast<int>(config.train_reanalyze_ratio * config.batch_size);
            int num_fresh_samples = config.batch_size - num_reanalyze_samples;
            Batch batch = replay_buffer->sample(rng, num_fresh_samples);
            if (num_reanalyze_samples > 0) {
                batch += reanalyze_buffer->sample(rng, num_reanalyze_samples);
            }

            // The learner needs a non-const reference due to torch requiring non-cost pointers for tensor
            // initialization
            VPRNetModel::LossInfo loss = model->Learn(batch);
            // Update stats and handoff model back
            shared_stats->set_loss(loss.total_loss, loss.value, loss.policy, loss.reward);

            // Update PER using the loss info for each buffer type
            std::vector<int> indices_replay =
                std::vector<int>(loss.indices.begin(), loss.indices.begin() + num_fresh_samples);
            std::vector<double> errors_replay =
                std::vector<double>(loss.errors.begin(), loss.errors.begin() + num_fresh_samples);
            replay_buffer->update_history_priorities(indices_replay, errors_replay);
            if (num_reanalyze_samples > 0) {
                std::vector<int> indices_reanalyze =
                    std::vector<int>(loss.indices.begin() + num_fresh_samples, loss.indices.end());
                std::vector<double> errors_reanalyze =
                    std::vector<double>(loss.errors.begin() + num_fresh_samples, loss.errors.end());
                reanalyze_buffer->update_history_priorities(indices_reanalyze, errors_reanalyze);
            }
        }

        // Checkpoint models and buffer
        if (step % config.checkpoint_interval == 0) {
            std::string checkpoint_path =
                device_manager->Get(0, device_id)->SaveCheckpoint(kMostRecentCheckpointStep);
            for (int i = 0; i < device_manager->Count(); ++i) {
                if (i != device_id) { device_manager->Get(0, i)->LoadCheckpoint(checkpoint_path); }
            }
            replay_buffer->save();
            reanalyze_buffer->save();
            shared_stats->save(kMostRecentCheckpointStep);
            std::cout << "\33[2K\rcheckpoint saved: " << checkpoint_path << std::endl;
        }
        if (step % config.model_sync_interval == 0) {
            std::string checkpoint_path =
                device_manager->Get(0, device_id)->SaveCheckpoint(kMostRecentCheckpointStep);
            for (int i = 0; i < device_manager->Count(); ++i) {
                if (i != device_id) { device_manager->Get(0, i)->LoadCheckpoint(checkpoint_path); }
            }
            shared_stats->save(kMostRecentCheckpointStep);
            std::cout << "\33[2K\rmodel synced" << std::endl;
        }

        // Checkpoint best performance model
        double current_reward = shared_stats->get_evaluator_muzero_reward();
        if (current_reward > max_reward) {
            max_reward = current_reward;
            std::string checkpoint_path =
                device_manager->Get(0, 1)->SaveCheckpoint(kBestPerformanceCheckpointStep);
            std::cout << "\33[2K\rcheckpoint saved: " << checkpoint_path << std::endl;
        }

        // Get updated step counter
        step = shared_stats->get_training_step();
    }
    // Learning finished, send kill signal
    stop->stop();
}

};    // namespace muzero_cpp