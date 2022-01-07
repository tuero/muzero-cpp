#ifndef MUZERO_CPP_MCTS_H_
#define MUZERO_CPP_MCTS_H_

#include <cassert>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "muzero-cpp/config.h"
#include "muzero-cpp/types.h"
#include "muzero-cpp/util.h"
#include "muzero-cpp/vprnet_evaluator.h"

namespace muzero_cpp {
namespace algorithm {

// Node used in MCTS search tree
struct MCTSNode {
    int visit_count = 0;                             // Number of times this node has been visited
    types::Player to_play = types::InvalidPlayer;    // Current player to play at this node
    types::Action action =
        types::InvalidAction;            // Action taken to get to this node from the previous state
    double prior = 0;                    // Prior probability for choosing action which led to this node
    double value_sum = 0;                // Sum of values from passing through this node
    double reward = 0;                   // Reward received by transitioning to this node
    std::vector<MCTSNode> children;      // Child nodes
    types::Observation encoded_state;    // Hidden state which the nodes represents

    MCTSNode(types::Action action, double prior) : action(action), prior(prior) {}
    MCTSNode() = delete;    // Default constructor is probably fine, but doesn't make sense to use ...

    /**
     * Check whether the node has been previously expanded.
     * @return True if the node has been expanded, false otherwise
     */
    bool is_expanded() const;

    /**
     * Get the mean value of the node.
     * @return The mean value of the node
     */
    double value() const;

    /**
     * Expand the current node using the value/policy and reward prediction
     * from the recurrent dynamics model.
     * @param actions All possible game actions
     * @param to_play The current player to play
     * @param reward The predicted reward
     * @param policy_probs The policy probabilities over tne entire action space
     * @param encoded_state The hidden state which the current node represents
     */
    void expand(const std::vector<types::Action> &actions, types::Player to_play, double reward,
                const std::vector<double> &policy_probs, const types::Observation &encoded_state);

    /**
     * Add a weighted sum of dirichlet exploration noise over the priors
     * @param dirichlet_alpha The alpha parameter for the Dirichlet distribution
     * @param dirichlet_epsilon The fractional component of the weighted sum for the Dirichlet noise
     * @param rng A random number generator
     */
    void add_exploration_noise(double dirichlet_alpha, double dirichlet_epsilon, std::mt19937 &rng);

    /**
     * Get the search statistics in form of a policy.
     * @param action_space Action space to gather statistics over
     * @return vector of visit count over total visits for all actions
     */
    // std::vector<double> get_search_statistics(const std::vector<types::Action> &action_space) const;
    void get_search_statistics(const std::vector<types::Action> &action_space,
                                         std::vector<types::Action> &child_actions,
                                         std::vector<double> &children_relative_visit,
                                         std::vector<double> &root_policy) const;
};

// Class to hold the min-max values seen during the tree search
// Used to normalize values (Appendix B Search)
class MinMaxStats {
public:
    MinMaxStats(double minimum, double maximum)
        : init_minimum_(minimum), init_maximum_(maximum), minimum_(minimum), maximum_(maximum){};
    MinMaxStats()
        : init_minimum_(types::INF_D),
          init_maximum_(types::NINF_D),
          minimum_(types::INF_D),
          maximum_(types::NINF_D){};

    /**
     * Reset the stored min/max bounds to the init values
     */
    void reset();

    /**
     * Update the seen min/max bounds.
     * @param value The new value seen.
     */
    void update(double value);

    /**
     * Normalize the given value using the known/seen min/max bounds.
     * @param value The value to normalize
     * @return The normalized value
     */
    double normalize(double value) const;

private:
    double init_minimum_;   // Minimum starting value
    double init_maximum_;   // Maximum starting value
    double minimum_;        // Minimum observed bound on value
    double maximum_;        // Maximum observed bound on value
};

// Class which handles the MCTS for self-play
class MCTS {
public:
    /**
     * Constructor for MCTS object
     * @param config A muzero configuration struct
     * @param seed Initial seed to source all randomness used in the search
     * @param vpr_eval Muzero network evaluator for inference
     */
    MCTS(const muzero_config::MuZeroConfig &config, int seed, std::shared_ptr<Evaluator> vpr_eval);
    MCTS() = delete;

    /**
     * Run MCTS for a set number of iterations
     * @param stacked_observation The stacked observation for the current game state
     * @param legal_actions The list of legal actions for the current game state
     * @param to_play The player to move for the current game state
     * @param add_exploration_noise flag to add Dirichlet exploration noise to the prior distribution
     * @return Struct containing root value, max depth searched, child visit counts, and root child actions
     */
    types::MCTSReturn run(const types::Observation &stacked_observation,
                          const std::vector<types::Action> &legal_actions, types::Player to_play,
                          bool add_exploration_noise);

    /**
     * Selects a child node which maximizes the modifed pUCT formulation
     * @param node Pointer to node which to select for
     * @return Pointer to child node selected
     */
    MCTSNode *select_child(MCTSNode *node);

    /**
     * Calculate the UCB score for the node using the modified pUCT rule (See Appendix B Search)
     * @param Parent Pointer to parent of node to consider
     * @param child node to find the value of
     * @return The UCB score for the given child node
     */
    double ucb_score(MCTSNode *parent, MCTSNode *child);

    /**
     * Backpropogate the value up along the nodes for the given search path
     * @param search_path The path along which to backpropogate
     * @param value The value to popogate
     * @param to_play The current player for the leaf node along the path
     */
    void backpropogate(std::vector<MCTSNode *> &search_path, double value, types::Player to_play);

private:
    int num_players_;             // Number of players the game requires
    int num_simulations_;         // Number of MCTS simulations per move
    double dirichlet_alpha_;      // Dirichlet distribution alpha parameter
    double dirichlet_epsilon_;    // The fractional component of the weighted sum for the Dirichlet noise
    double pb_c_base_;            // pUCT c_base constant
    double pb_c_init_;            // pUCT c_base constant
    double discount_;             // Discount factor for reward
    MinMaxStats minmax_stats_;    // Normalizes values seen into [0, 1] range
    std::mt19937 rng_;            // Source of randomness for Dirichlet noise
    std::vector<types::Action> action_space_;    // List of all possible actions
    std::shared_ptr<Evaluator> vpr_eval_;        // Muzero network evaluator for inference
};

}    // namespace algorithm
}    // namespace muzero_cpp

#endif    // MUZERO_CPP_MCTS_H_