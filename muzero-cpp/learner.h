#ifndef MUZERO_CPP_LEARNER_H_
#define MUZERO_CPP_LEARNER_H_

#include <memory>
#include <random>

#include "muzero-cpp/abstract_game.h"
#include "muzero-cpp/device_manager.h"
#include "muzero-cpp/queue.h"
#include "muzero-cpp/replay_buffer.h"
#include "muzero-cpp/shared_stats.h"
#include "muzero-cpp/types.h"
#include "muzero-cpp/util.h"
#include "muzero-cpp/vprnet_evaluator.h"

namespace muzero_cpp {

/**
 * Learner thread logic.
 * Continuously updates the muzero network model
 * @param config Muzero config
 * @param device_manager Access to muzero network for learning
 * @param replay_buffer Shared pointer to the replay buffer
 * @param reanalyze_buffer Shared pointer to the reanalyze buffer
 * @param trajectory_queue Queue of history trajectories from self-play actors
 * @param shared_stats Statistics to update for logger
 * @param stop Stop token, used to terminate the reanalyze actor
 */
void learn(const muzero_config::MuZeroConfig& config, DeviceManager* device_manager,
           std::shared_ptr<buffer::PrioritizedReplayBuffer> replay_buffer,
           std::shared_ptr<buffer::PrioritizedReplayBuffer> reanalyze_buffer,
           ThreadedQueue<types::GameHistory>* trajectory_queue, std::shared_ptr<SharedStats> shared_stats,
           util::StopToken* stop);

}    // namespace muzero_cpp

#endif    // MUZERO_CPP_LEARNER_H_