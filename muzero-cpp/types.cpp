#include "muzero-cpp/types.h"

#include <algorithm>
#include <cassert>

namespace muzero_cpp {
namespace types {

// Store the MCTS search statistics into the game history
void GameHistory::store_search_statistics(double root_value, const std::vector<double> &relative_visits) {
    root_values.push_back(root_value);
    child_visits.push_back(relative_visits);
}

// Get the current observation + num_stacked_observations previous observations/actions all stacked
Observation GameHistory::get_stacked_observations(int step, int num_stacked_observations,
                                                  const ObservationShape &obs_shape, int action_channels,
                                                  ActionRepresentationFunction action_rep_func) const {
    // python style modulo to convert negative indices to the relative positive by wrapping around
    // C++ has negative modulo results stay negative, and we can't use that for indexing
    int size = observation_history.size();
    step = ((step % size) + size) % size;
    // Always start with the current observaion (i.e. no history)
    Observation stacked_observations = observation_history[step];

    // Append as channels previous observations/action planes
    int flat_size = obs_shape.h * obs_shape.w;
    for (int past_index = step - 1; past_index >= step - num_stacked_observations; --past_index) {
        if (past_index >= 0) {    // Seen this history
            const Observation &obs_slice = observation_history[past_index];
            stacked_observations.insert(stacked_observations.end(), obs_slice.begin(), obs_slice.end());
            Observation embeded_actions = action_rep_func(action_history[past_index + 1]);
            assert((int)embeded_actions.size() == flat_size * action_channels);
            stacked_observations.insert(stacked_observations.end(), embeded_actions.begin(),
                                        embeded_actions.end());
        } else {    // Run off trajectory, need C+action_channels planes of 0's for state + action
            stacked_observations.insert(stacked_observations.end(),
                                        (obs_shape.c + action_channels) * flat_size, 0);
        }
    }
    return stacked_observations;
}

// Compute the target value for the given step
double GameHistory::compute_target_value(int step, int td_steps, double discount) const {
    int bootstrap_index = step + td_steps;
    double value = 0;

    // Discounted root value td_steps into future + discounted sum of rewards
    if (bootstrap_index < (int)root_values.size()) {
        // Use reanalyze values if available
        const std::vector<double> &values =
            (reanalysed_predicted_root_values) ? reanalysed_predicted_root_values.get() : root_values;
        value = values[bootstrap_index] * std::pow(discount, td_steps);
        // Value in view of player
        if (to_play_history[step] == to_play_history[bootstrap_index]) { value *= -1; }
    }

    // Sum of rewards with discounting
    int start_idx = step + 1;
    int end_idx = std::min(bootstrap_index + 1, (int)reward_history.size());
    for (int i = 0; i < (end_idx - start_idx); ++i) {
        // Reward in view of player
        bool same_player = (to_play_history[step] == to_play_history[step + i]);
        double reward = same_player ? reward_history[start_idx + i] : -reward_history[start_idx + i];
        value += reward * std::pow(discount, i);
    }

    return value;
}

// Create the target values used for learning
void GameHistory::make_target(int step, int td_steps, int num_unroll_steps, double discount, Batch &sample,
                              std::mt19937 &rng) const {
    assert((int)child_visits.size() > 0);
    int NUM_ACTIONS = child_visits[0].size();
    std::uniform_int_distribution<int> uniform_dist(0, NUM_ACTIONS - 1);

    // Unroll
    for (int i = step; i < step + num_unroll_steps + 1; ++i) {
        // Compute target value
        double value = compute_target_value(step, td_steps, discount);
        // Absorbing states are past the end of game history -> random policy + 0 value/rewards
        // Special case needed for one step off end as actions are shifted off by 1
        // (observation index is next to action which led to this observation)
        bool NON_ABSORBING = (i < (int)root_values.size());
        bool NON_ABSORBING_END = (i <= (int)root_values.size());
        sample.target_values.push_back(NON_ABSORBING ? value : 0);
        sample.target_rewards.push_back(NON_ABSORBING_END ? reward_history[i] : 0);
        std::vector<double> policy =
            NON_ABSORBING ? child_visits[i] : std::vector<double>(NUM_ACTIONS, 1.0 / NUM_ACTIONS);
        sample.target_policies.insert(sample.target_policies.end(), policy.begin(), policy.end());
        sample.actions.push_back(NON_ABSORBING_END ? action_history[i] : uniform_dist(rng));
    }
}

}    // namespace types
}    // namespace muzero_cpp