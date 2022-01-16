#ifndef MUZERO_CPP_SELF_PLAY_H_
#define MUZERO_CPP_SELF_PLAY_H_

#include <memory>
#include <random>

#include "muzero-cpp/abstract_game.h"
#include "muzero-cpp/mcts.h"
#include "muzero-cpp/queue.h"
#include "muzero-cpp/shared_stats.h"
#include "muzero-cpp/types.h"
#include "muzero-cpp/util.h"
#include "muzero-cpp/vprnet_evaluator.h"

namespace muzero_cpp {

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
void self_play_actor(const muzero_config::MuZeroConfig& config, std::unique_ptr<AbstractGame> game,
                     int actor_num, ThreadedQueue<types::GameHistory>* trajectory_queue,
                     std::shared_ptr<Evaluator> vpr_eval, std::shared_ptr<SharedStats> shared_stats,
                     util::StopToken* stop);

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
void self_play_evaluator(const muzero_config::MuZeroConfig& config, std::unique_ptr<AbstractGame> game,
                         int actor_num, std::shared_ptr<Evaluator> vpr_eval,
                         std::shared_ptr<SharedStats> shared_stats, util::StopToken* stop);

/**
 * Play against muzero for testing
 * @param config Muzero config
 * @param game Copy of game (separate for each actor thread)
 * @param vpr_eval Evaluator for inference during MCTS
 * @param shared_stats Common self play and training stats
 * @param stop Stop token, used to terminate the actor
 */
void self_play_test(const muzero_config::MuZeroConfig& config, std::unique_ptr<AbstractGame> game,
                    std::shared_ptr<Evaluator> vpr_eval, std::shared_ptr<SharedStats> shared_stats,
                    util::StopToken* stop);

}    // namespace muzero_cpp

#endif    // MUZERO_CPP_SELF_PLAY_H_