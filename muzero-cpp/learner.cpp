#include "muzero-cpp/learner.h"

#include <tuple>
#include <vector>

#include "absl/types/optional.h"

namespace muzero_cpp {
using namespace types;
using namespace buffer;

// Reanalyze thread logic.
// Continuously updates samples root value estimates using the most recent model
void reanalyze(const muzero_config::MuZeroConfig& config, std::shared_ptr<Evaluator> vpr_eval,
               std::shared_ptr<PrioritizedReplayBuffer> replay_buffer,
               std::shared_ptr<SharedStats> shared_stats, util::StopToken* stop) {
    std::mt19937 rng(config.seed);
    // Continue to reanalyze until we are done
    for (int step = 1; !stop->stop_requested(); ++step) {
        // Check if we have enough samples and sleep if not
        if (!replay_buffer->can_sample()) {
            absl::SleepFor(absl::Milliseconds(1000));
            continue;
        }

        // Sample
        std::tuple<int, GameHistory> sample = replay_buffer->sample_game(rng);
        int history_id = std::get<0>(sample);
        GameHistory game_history = std::get<1>(sample);

        // Not efficient, ideally we would batch and send to model once but this works
        // Reanalyze root values by sending to most recent model
        std::vector<double> reanalysed_predicted_root_values;
        for (int i = 0; i < (int)game_history.root_values.size(); ++i) {
            Observation stacked_observation = game_history.get_stacked_observations(
                i, config.stacked_observations, config.observation_shape, config.action_channels,
                config.action_representation);
            VPRNetModel::InferenceOutputs inference_output = vpr_eval->InitialInference(stacked_observation);
            reanalysed_predicted_root_values.push_back(inference_output.value);
        }

        // Set root values to updated values
        game_history.reanalysed_predicted_root_values = reanalysed_predicted_root_values;

        // Store history back in buffer
        replay_buffer->update_game_history(history_id, game_history);

        // Update stats to reflect number of 
        shared_stats->add_reanalyze_game(1);
    }
}

// Learner thread logic.
// Continuously updates the muzero network model
void learn(const muzero_config::MuZeroConfig& config, DeviceManager* device_manager,
           std::shared_ptr<PrioritizedReplayBuffer> replay_buffer,
           ThreadedQueue<GameHistory>* trajectory_queue, std::shared_ptr<SharedStats> shared_stats,
           util::StopToken* stop) {
    const int device_id = 0;
    std::mt19937 rng(config.seed);

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
            }
        }

        // Check if we have enough samples
        if (!replay_buffer->can_sample()) {
            absl::SleepFor(absl::Milliseconds(1000));
            continue;
        }

        // Learning step (scope so model given back immediately)
        {
            // Set first device off-limits from inference if requested
            device_manager->SetLearning(config.explicit_learning);
            // Ask manager for model
            DeviceManager::DeviceLoan model = device_manager->Get(config.batch_size, device_id);
            // Sample and send to model
            // The learner needs a non-const reference due to torch requiring non-cost pointers for tensor initialization
            std::vector<BatchItem> batch = replay_buffer->sample(rng);
            VPRNetModel::LossInfo loss = model->Learn(batch);
            // Update stats and handoff model back 
            shared_stats->set_loss(loss.total_loss, loss.value, loss.policy, loss.reward);
            device_manager->SetLearning(false);

            // Update PER using the loss info
            replay_buffer->update_history_priorities(loss.indices, loss.errors);
        }

        // Checkpoint models and buffer
        if (step % config.checkpoint_interval == 0) {
            std::string checkpoint_path =
                device_manager->Get(0, device_id)->SaveCheckpoint(VPRNetModel::kMostRecentCheckpointStep);
            for (int i = 0; i < device_manager->Count(); ++i) {
                if (i != device_id) { device_manager->Get(0, i)->LoadCheckpoint(checkpoint_path); }
            }
            replay_buffer->save();
            shared_stats->save(VPRNetModel::kMostRecentCheckpointStep);
            std::cout << "\33[2K\rcheckpoint saved: " << checkpoint_path << std::endl;
        }

        // Check if we should wait due to training being too fast
        if (config.train_selfplay_ratio > 0) {
            while (!stop->stop_requested() &&
                   (double)step / std::max(1, shared_stats->get_num_played_steps()) >
                       config.train_selfplay_ratio * 1.25) {
                absl::SleepFor(absl::Milliseconds(500));
            }
        }

        // Get updated step counter
        step = shared_stats->get_training_step();
    }
    // Learning finished, send kill signal
    stop->stop();
}

};    // namespace muzero_cpp