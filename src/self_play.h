#ifndef MUZERO_SELF_PLAY_H_
#define MUZERO_SELF_PLAY_H_

#include <muzero/abstract_game.h>
#include <muzero/config.h>

#include "game_history.h"
#include "queue.h"
#include "replay_buffer.h"
#include "shared_stats.h"
#include "stop_token.h"
#include "vprnet_evaluator.h"

#include <memory>

namespace muzero {

/**
 * Actor's main self play logic.
 * Continues to self play games and feed results to the trajectory queue. The learning thread is responsible
 * for inserting these trajectories into the replay buffer.
 * @param config Muzero config
 * @param game Copy of game (separate for each actor thread)
 * @param actor_num Id of the actor
 * @param trajectory_queue Queue to insert game histories in
 * @param vpr_eval Evaluator for inference during MCTS
 * @param shared_stats Common self play and training stats
 * @param stop Stop token, used to terminate the actor
 */
void self_play_actor(
    const Config &config,
    std::unique_ptr<AbstractGame> game,
    int actor_num,
    ThreadedQueue<GameHistory> *trajectory_queue,
    std::shared_ptr<Evaluator> vpr_eval,
    std::shared_ptr<SharedStats> shared_stats,
    std::shared_ptr<StopToken> stop
);

/**
 * Reanalyze actor's main logic
 * Continues to sample old trajectories and update the policy/value using a fresher model
 * @param config Muzero config
 * @param reanalyze_buffer Reanalyze buffer to pull samples from
 * @param actor_num Id of the actor
 * @param vpr_eval Evaluator for inference during MCTS
 * @param shared_stats Common self play and training stats
 * @param stop Stop token, used to terminate the actor
 */
void reanalyze_actor(
    const Config &config,
    std::shared_ptr<buffer::PrioritizedReplayBuffer> reanalyze_buffer,
    int actor_num,
    std::shared_ptr<Evaluator> vpr_eval,
    std::shared_ptr<SharedStats> shared_stats,
    std::shared_ptr<StopToken> stop
);

/**
 * Evaluators's main self play logic.
 * Continues to self play games and sets evaluator stats for logging
 * @param config Muzero config
 * @param game Copy of game (separate for each actor thread)
 * @param actor_num Id of the actor
 * @param vpr_eval Evaluator for inference during MCTS
 * @param shared_stats Common self play and training stats
 * @param stop Stop token, used to terminate the actor
 */
void self_play_evaluator(
    const Config &config,
    std::unique_ptr<AbstractGame> game,
    int actor_num,
    std::shared_ptr<Evaluator> vpr_eval,
    std::shared_ptr<SharedStats> shared_stats,
    std::shared_ptr<StopToken> stop
);

/**
 * Play against muzero for testing
 * @param config Muzero config
 * @param game Copy of game (separate for each actor thread)
 * @param vpr_eval Evaluator for inference during MCTS
 * @param shared_stats Common self play and training stats
 * @param stop Stop token, used to terminate the actor
 */
void self_play_test(
    const Config &config,
    std::unique_ptr<AbstractGame> game,
    std::shared_ptr<Evaluator> vpr_eval,
    std::shared_ptr<SharedStats> shared_stats,
    std::shared_ptr<StopToken> stop
);

}    // namespace muzero

#endif    // MUZERO_SELF_PLAY_H_
