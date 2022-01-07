#ifndef MUZERO_CPP_TYPES_H_
#define MUZERO_CPP_TYPES_H_

#include <torch/torch.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <vector>

#include "nop/base/optional.h"
#include "nop/serializer.h"

namespace muzero_cpp {
namespace types {

using Player = int;
using Action = int;
constexpr Player InvalidPlayer = -1;
constexpr Action InvalidAction = -1;

using Observation = std::vector<float>;
struct ObservationShape {
    int c;    // Number of channels
    int h;    // Height of observation
    int w;    // Width of observation
    bool operator==(const ObservationShape &rhs) const {
        return c == rhs.c && h == rhs.h && w == rhs.w;
    }
    bool operator!=(const ObservationShape &rhs) const {
        return c != rhs.c || h != rhs.h || w != rhs.w;
    }
};

// Function which takes as input an action and returns an observation representing that action
// This is used to convert actions into feature observations and stack with previous observations
// For simple games this can just be a single channel plane of values 1/action_id, but some
// games like chess/go might want more informative representations (See AlphaZero papers)
using ActionRepresentationFunction = std::function<Observation(Action)>;

// Function which takes the current learning step and returns the softmax temperature to apply to the action
// selection during self play. This can be a constant temperature, or one which follows a complex schedule.
// Values should fall in range of [0, 1].
using SoftmaxTemperatureFunction = std::function<double(int)>;

// These are the opponents MuZero will play against during train evaluation or testing.
// MuZero always plays against itself during acting (samples used for training), or in 1 player games (acting,
// evaluating, and testing; and this selection will be ignored in that case).
enum class OpponentTypes {
    Self,      // Play against itself
    Random,    // Opponent randomly choose a legal action
    Expert,    // User game-defined expert (see abstract_game.h / Examples)
    Human      // Human types in input action (see abstract_game.h / Examples)
};

// AbstractGame
// These are the values returned after taking a step in the environment
struct StepReturn {
    Observation observation;
    double reward;
    bool done;
};

// MCTS
// These are all the necessary information required from the MCTS
struct MCTSReturn {
    double root_value;
    int max_tree_depth;
    std::vector<double> emperical_policy;
    std::vector<double> children_relative_visit;
    std::vector<Action> child_actions;
};
constexpr double NINF_D = std::numeric_limits<double>::lowest();
constexpr double INF_D = std::numeric_limits<double>::max();

// Replay buffer
struct BatchItem {
    double priority;
    int index;
    std::vector<Action> actions;
    Observation stacked_observation;
    std::vector<double> target_rewards;
    std::vector<double> target_values;
    std::vector<std::vector<double>> target_policies;
    std::vector<double> gradient_scale;
};

// All necessary stored items for a game history
struct GameHistory {
    std::vector<Observation> observation_history;
    std::vector<Action> action_history;
    std::vector<double> reward_history;
    std::vector<Player> to_play_history;
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
    Observation get_stacked_observations(int step, int num_stacked_observations,
                                         const ObservationShape &obs_shape, int action_channels,
                                         ActionRepresentationFunction action_rep_func) const;

    /**
     * Compute the target value for the given step
     * @param step The starting step to retrieve
     * @param td_steps Number of future td steps to take into account for future value
     * @param discount Discount factor for reward
     * @returns Target value for the given step
     */
    double compute_target_value(int step, int td_steps, double discount) const;

    /**
     * Create the target values used for learning
     * Passed vectors are given as refs so we can insert the values directly
     * @note If step + num_unroll_steps > history, we assume the state is absorbing and randomly select
     * actions and give a uniform target policy.
     * @param step The starting step to retrieve
     * @param td_steps Number of future td steps to take into account for future value
     * @param num_unroll_steps Number of steps to unroll forward for each sample
     * @param action_batch Actions taken along the trajectory
     * @param target_rewards Rewards observed along the trajectory
     * @param target_values Computed target values (sum of discounted rewards) along the trajectory
     * @param target_policies Empirical policy as found from the MCTS
     * @param rng Source of randomness, used for absorbing states
     */
    void make_target(int step, int td_steps, int num_unroll_steps, double discount,
                     std::vector<Action> &action_batch, std::vector<double> &target_rewards,
                     std::vector<double> &target_values, std::vector<std::vector<double>> &target_policies,
                     std::mt19937 &rng) const;

    // Requirements for loading/saving struct
    NOP_STRUCTURE(GameHistory, observation_history, action_history, reward_history, to_play_history,
                  root_values, child_visits, reanalysed_predicted_root_values);
};

}    // namespace types
}    // namespace muzero_cpp

#endif    // MUZERO_CPP_TYPES_H_