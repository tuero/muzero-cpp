#ifndef MUZERO_LEARNER_H_
#define MUZERO_LEARNER_H_

#include <muzero/abstract_game.h>
#include <muzero/config.h>

#include "device_manager.h"
#include "game_history.h"
#include "queue.h"
#include "replay_buffer.h"
#include "shared_stats.h"
#include "stop_token.h"

#include <memory>

namespace muzero {

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
void learn(
    const Config &config,
    DeviceManager *device_manager,
    std::shared_ptr<buffer::PrioritizedReplayBuffer> replay_buffer,
    std::shared_ptr<buffer::PrioritizedReplayBuffer> reanalyze_buffer,
    ThreadedQueue<GameHistory> *trajectory_queue,
    std::shared_ptr<SharedStats> shared_stats,
    std::shared_ptr<StopToken> stop
);

}    // namespace muzero

#endif    // MUZERO_LEARNER_H_
