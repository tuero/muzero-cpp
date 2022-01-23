#include "muzero-cpp/self_play.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "muzero-cpp/mcts.h"
#include "muzero-cpp/types.h"

namespace muzero_cpp {
using namespace buffer;
using namespace types;
using namespace algorithm;

namespace {
// Selects the action using the search statistics and temperature
Action select_action(const MCTSReturn& mcts_stats, double temperature, std::mt19937& rng) {
    assert(mcts_stats.children_relative_visit.size() == mcts_stats.child_actions.size());
    if (temperature <= 0) {
        // Greedily select action
        auto max_element_itr = std::max_element(mcts_stats.children_relative_visit.begin(),
                                                mcts_stats.children_relative_visit.end());
        auto action_idx = std::distance(mcts_stats.children_relative_visit.begin(), max_element_itr);
        return mcts_stats.child_actions[action_idx];    // Chosen action
    } else {
        // Sample action with probabilities given by visit count scaled by temperature
        std::vector<double> relative_visits = mcts_stats.children_relative_visit;
        for (auto& child_visit : relative_visits) {
            child_visit = std::pow(child_visit, 1.0 / temperature);
        }
        std::discrete_distribution<> d(relative_visits.begin(), relative_visits.end());
        auto action_idx = d(rng);
        return mcts_stats.child_actions[action_idx];    // Chosen action
    }
}

// Randomly selects the opponents action if MuZero is not being used for selection,
// or asks for an expert action
Action select_opponent_action(AbstractGame& game, OpponentTypes opponent_type, std::mt19937& rng) {
    if (opponent_type == OpponentTypes::Expert) {
        return game.expert_action();
    } else if (opponent_type == OpponentTypes::Random) {
        const std::vector<Action>& legal_actions = game.legal_actions();
        std::uniform_int_distribution<> d(0, legal_actions.size() - 1);
        return legal_actions[d(rng)];
    } else if (opponent_type == OpponentTypes::Human) {
        return game.human_to_action();
    }
    std::cerr << "Unknown opponent type." << std::endl;
    std::exit(1);
}

// Construct the necessary stats to add from an evaluator
void add_evaluator_stats(const std::shared_ptr<SharedStats>& shared_stats, const GameHistory& game_history,
                         Player muzero_player, bool render) {
    double reward_sum = std::reduce(game_history.reward_history.begin(), game_history.reward_history.end());
    double mean_value = std::reduce(game_history.root_values.begin(), game_history.root_values.end()) /
                        game_history.root_values.size();
    double muzero_reward = 0;
    double opponent_reward = 0;
    for (int i = 0; i < (int)game_history.reward_history.size(); ++i) {
        if (game_history.to_play_history[i - 1] == muzero_player) {
            muzero_reward += game_history.reward_history[i];
        } else {
            opponent_reward += game_history.reward_history[i];
        }
    }
    shared_stats->set_evaluator_stats(game_history.action_history.size() - 1, reward_sum, mean_value,
                                      muzero_reward, opponent_reward);
    if (render) {
        std::cout << absl::StrFormat("muzero_reward: %.2f, opponent_reward: %.2f", muzero_reward,
                                     opponent_reward)
                  << std::endl;
    }
}

}    // namespace

// Actor's main self play logic.
void play_game(const muzero_config::MuZeroConfig& config, algorithm::MCTS& mcts, AbstractGame& game,
               const std::shared_ptr<SharedStats>& shared_stats, std::mt19937& rng,
               ThreadedQueue<GameHistory>* trajectory_queue, bool is_evaluator, bool render,
               util::StopToken* stop) {
    Observation observation = game.reset();
    GameHistory game_history;
    // Set the initial game history values
    game_history.action_history.push_back(0);
    game_history.observation_history.push_back(observation);
    game_history.reward_history.push_back(0);
    game_history.to_play_history.push_back(game.to_play());
    game_history.legal_actions.push_back(game.legal_actions());

    int hist_idx_start = 0;
    int hist_idx_end = 0;

    // Render
    if (render) { game.render(); }

    // Find temperature using the temperature schedule
    // If we are an evaluator, we don't use a temperature
    double temperature =
        is_evaluator ? 0 : config.visit_softmax_temperature(shared_stats->get_training_step());

    // We always play against ourself if we are an actor or 1 player games
    OpponentTypes opponent_type =
        (!is_evaluator || config.num_players == 1) ? OpponentTypes::Self : config.opponent_type;

    // Loop game until game is done
    // Paper also suggests to store partial histories in longer games (Atari),
    // but this isn't supported yet.
    bool done = false;
    while (!done) {
        // Exit early if stop requested
        if (stop->stop_requested()) { return; }

        // Get stacked observation
        Observation stacked_observation = game_history.get_stacked_observations(
            -1, config.stacked_observations, config.observation_shape, config.action_channels,
            config.action_representation_initial);
        Action action;
        MCTSReturn mcts_stats;

        // Choose action according to current player
        if (opponent_type == OpponentTypes::Self || game.to_play() == config.muzero_player) {
            // Find action using mcts
            mcts_stats = mcts.run(stacked_observation, game.legal_actions(), game.to_play(), !is_evaluator);
            action = select_action(mcts_stats, temperature, rng);
            // Only store root statistics if we performed an mcts search
            game_history.store_search_statistics(mcts_stats.root_value, mcts_stats.emperical_policy);

            // Display search stats on render, which is usually done when testing the model
            if (render) {
                std::cout << "Tree depth: " << mcts_stats.max_tree_depth << std::endl;
                std::cout << "Root value for player " << game.to_play() << ": "
                          << absl::StrFormat("%.3f", mcts_stats.root_value) << std::endl;
            }
        } else {
            action = select_opponent_action(game, opponent_type, rng);
        }

        // Step environment
        StepReturn step_return = game.step(action);
        done = step_return.done ||
               (config.max_moves > 0 && (int)game_history.root_values.size() < config.max_moves);
        observation = step_return.observation;

        // Render
        if (render) {
            std::cout << "Played action: " << game.action_to_string(action) << std::endl;
            game.render();
        }

        // Store step history
        game_history.action_history.push_back(action);
        game_history.observation_history.push_back(observation);
        game_history.reward_history.push_back(step_return.reward);
        game_history.to_play_history.push_back(game.to_play());
        game_history.legal_actions.push_back(game.legal_actions());
        ++hist_idx_end;

        // Send history slice if game over or exceed maximum history length
        bool send_hist =
            (config.max_history_len > 0 && (hist_idx_end - hist_idx_start > config.max_history_len));
        if (!is_evaluator && (done || send_hist)) {
            if (!trajectory_queue->Push(game_history.get_slice(hist_idx_start, hist_idx_end),
                                        absl::Seconds(10))) {
                std::cerr << "Error: Unable to push trajectory to queue" << std::endl;
            }
            hist_idx_start = hist_idx_end;
        }
    }

    // Add self play game stats for logging
    if (is_evaluator) {
        add_evaluator_stats(shared_stats, game_history, config.muzero_player, render);
    } else {
        shared_stats->add_actor_stats(game_history.action_history.size() - 1, 1);
    }
}

// Perform reanalyze
void reanalyze(const muzero_config::MuZeroConfig& config, algorithm::MCTS& mcts,
               std::shared_ptr<PrioritizedReplayBuffer> reanalyze_buffer,
               std::shared_ptr<SharedStats> shared_stats, std::mt19937& rng, util::StopToken* stop) {
    for (int step = 1; !stop->stop_requested(); ++step) {
        // Check if we have enough samples and sleep if not
        if (!reanalyze_buffer->can_sample()) {
            absl::SleepFor(absl::Milliseconds(1000));
            continue;
        }

        // Sample
        std::tuple<int, GameHistory> sample = reanalyze_buffer->sample_game(rng);
        int history_id = std::get<0>(sample);
        GameHistory game_history = std::get<1>(sample);

        // Iterate over history
        for (int i = 0; i < (int)game_history.root_values.size(); ++i) {
            if (stop->stop_requested()) { return; }
            // Get stacked observation and re-run MCTS
            Observation stacked_observation = game_history.get_stacked_observations(
                i, config.stacked_observations, config.observation_shape, config.action_channels,
                config.action_representation_initial);
            MCTSReturn mcts_stats = mcts.run(stacked_observation, game_history.legal_actions[i],
                                             game_history.to_play_history[i], true);

            // Update values
            game_history.root_values[i] = mcts_stats.root_value;
            game_history.child_visits[i] = mcts_stats.emperical_policy;
        }

        // Update game history to buffer
        // If buffer removed this history because we are overriding due to being at max size, this will do
        // nothing (but this should be a low probability event, and all that happens is 1 wasted cycle of
        // reanalyze)
        reanalyze_buffer->update_game_history(history_id, game_history);

        // Update reanalyze stats
        shared_stats->add_reanalyze_game((int)game_history.root_values.size());
    }
}

// Actor's main self play logic.
void self_play_actor(const muzero_config::MuZeroConfig& config, std::unique_ptr<AbstractGame> game,
                     int actor_num, ThreadedQueue<GameHistory>* trajectory_queue,
                     std::shared_ptr<Evaluator> vpr_eval, std::shared_ptr<SharedStats> shared_stats,
                     util::StopToken* stop) {
    MCTS mcts(config, config.seed + actor_num, vpr_eval);
    std::mt19937 rng(config.seed + actor_num);
    // Continue to play games until we are told to stop
    for (int game_num = 1; !stop->stop_requested(); ++game_num) {
        play_game(config, mcts, *game, shared_stats, rng, trajectory_queue, false, false, stop);
    }
}

// Evaluators's main self play logic.
void self_play_evaluator(const muzero_config::MuZeroConfig& config, std::unique_ptr<AbstractGame> game,
                         int actor_num, std::shared_ptr<Evaluator> vpr_eval,
                         std::shared_ptr<SharedStats> shared_stats, util::StopToken* stop) {
    MCTS mcts(config, config.seed + actor_num, vpr_eval);
    std::mt19937 rng(config.seed + actor_num);
    // Continue to play games until we are told to stop
    for (int game_num = 1; !stop->stop_requested(); ++game_num) {
        play_game(config, mcts, *game, shared_stats, rng, nullptr, true, false, stop);
    }
}

// Reanalyze actor logic
void reanalyze_actor(const muzero_config::MuZeroConfig& config,
                     std::shared_ptr<PrioritizedReplayBuffer> reanalyze_buffer, int actor_num,
                     std::shared_ptr<Evaluator> vpr_eval, std::shared_ptr<SharedStats> shared_stats,
                     util::StopToken* stop) {
    MCTS mcts(config, config.seed + actor_num, vpr_eval);
    std::mt19937 rng(config.seed + actor_num);
    for (int game_num = 1; !stop->stop_requested(); ++game_num) {
        reanalyze(config, mcts, reanalyze_buffer, shared_stats, rng, stop);
    }
}

// Play against muzero for testing
void self_play_test(const muzero_config::MuZeroConfig& config, std::unique_ptr<AbstractGame> game,
                    std::shared_ptr<Evaluator> vpr_eval, std::shared_ptr<SharedStats> shared_stats,
                    util::StopToken* stop) {
    MCTS mcts(config, config.seed, vpr_eval);
    std::mt19937 rng(config.seed);
    play_game(config, mcts, *game, shared_stats, rng, nullptr, true, true, stop);
}

}    // namespace muzero_cpp