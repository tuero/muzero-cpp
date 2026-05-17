#ifndef MUZERO_GAME_HISTORY_H_
#define MUZERO_GAME_HISTORY_H_

#include <muzero/types.h>

#include "batch.h"

#include <nop/base/optional.h>
#include <nop/base/serializer.h>
#include <nop/structure.h>

#include <random>
#include <vector>

namespace muzero {

// All necessary stored items for a game history
struct GameHistory {
    std::vector<Observation> observation_history;
    std::vector<Action> action_history;
    std::vector<double> reward_history;
    std::vector<Player> to_play_history;
    std::vector<std::vector<Action>> legal_actions;
    std::vector<double> root_values;
    std::vector<std::vector<double>> child_visits;
    nop::Optional<std::vector<double>> reanalysed_predicted_root_values;

    /**
     * Store the MCTS search statistics into the game history
     * @param root_value The root value of the MCTS tree
     * @param relative_visits The relative visits of the available children (emperical policy)
     */
    void store_search_statistics(double root_value, const std::vector<double> &relative_visits);

    /**
     * Get the current observation + num_stacked_observations previous observations/actions all stacked
     * @param step The starting step to retrieve
     * @param num_stacked_observations Number of historical previous steps to stack (can be 0)
     * @param action_channels The number of channels each action is represented as
     * @param action_rep_func Function which converts the (int) action into a feature
     * @return Single observation which stacks the historical + current observations
     */
    [[nodiscard]] auto get_stacked_observations(
        int step,
        int num_stacked_observations,
        const ObservationShape &obs_shape,
        int action_channels,
        ActionRepresentationFunction action_rep_func
    ) const -> Observation;

    /**
     * Compute the target value for the given step
     * @param step The starting step to retrieve
     * @param td_steps Number of future td steps to take into account for future value
     * @param discount Discount factor for reward
     * @returns Target value for the given step
     */
    [[nodiscard]] auto compute_target_value(int step, int td_steps, double discount) const -> double;

    /**
     * Create the target values used for learning
     * Passed vectors are given as refs so we can insert the values directly
     * @note If step + num_unroll_steps > history, we assume the state is absorbing and randomly select
     * actions and give a uniform target policy.
     * @param step The starting step to retrieve
     * @param td_steps Number of future td steps to take into account for future value
     * @param num_unroll_steps Number of steps to unroll forward for each sample
     * @param sample Reference to batched sample to append to
     * @param rng Source of randomness, used for absorbing states
     */
    void make_target(
        int step,
        int td_steps,
        int num_unroll_steps,
        double discount,
        Batch &sample,
        std::mt19937 &rng
    ) const;

    /**
     * Get a slice of the full history.
     * @param start_idx The starting index
     * @param end_idx The ending index
     * @returns A partial game history
     */
    [[nodiscard]] auto get_slice(int start_idx, int end_idx) const -> GameHistory;

    // Requirements for loading/saving struct
    NOP_STRUCTURE(
        GameHistory,
        observation_history,
        action_history,
        reward_history,
        to_play_history,
        legal_actions,
        root_values,
        child_visits,
        reanalysed_predicted_root_values
    );
};

}    // namespace muzero

#endif    // MUZERO_GAME_HISTORY_H_
