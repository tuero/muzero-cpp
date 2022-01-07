#include "muzero-cpp/mcts.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <unordered_map>

#include "muzero-cpp/types.h"
#include "muzero-cpp/util.h"

namespace muzero_cpp {
using namespace types;
namespace algorithm {

// ---------------------------------- MCTS ----------------------------------
// Constructor for MCTS object
MCTS::MCTS(const muzero_config::MuZeroConfig &config, int seed, std::shared_ptr<Evaluator> vpr_eval)
    : num_players_(config.num_players),
      num_simulations_(config.num_simulations),
      dirichlet_alpha_(config.dirichlet_alpha),
      dirichlet_epsilon_(config.dirichlet_epsilon),
      pb_c_base_(config.pb_c_base),
      pb_c_init_(config.pb_c_init),
      discount_(config.discount),
      minmax_stats_(config.value_lowerbound, config.value_upperbound),
      rng_(seed),
      action_space_(config.action_space),
      vpr_eval_(vpr_eval) {
    // Only deal with 1 or 2 player games
    assert(num_players_ <= 2);
}

// Run MCTS for a set number of iterations
MCTSReturn MCTS::run(const types::Observation &stacked_observation,
                     const std::vector<types::Action> &legal_actions, types::Player to_play,
                     bool add_exploration_noise) {
    assert((int)legal_actions.size() > 0);
    // Create root and send observation into the encoded state space
    MCTSNode root(types::InvalidAction, 0);
    VPRNetModel::InferenceOutputs initial_inference_output = vpr_eval_->InitialInference(stacked_observation);
    root.expand(legal_actions, to_play, initial_inference_output.reward, initial_inference_output.policy,
                initial_inference_output.encoded_state);
    
    // Add dirichlet exploration noise to the policy
    if (add_exploration_noise) { root.add_exploration_noise(dirichlet_alpha_, dirichlet_epsilon_, rng_); }

    // Reset values to empty search tree
    minmax_stats_.reset();
    int max_tree_depth = 0;

    // Run N simulations
    for (int i = 0; i < num_simulations_; ++i) {
        types::Player virtual_to_play = to_play;
        MCTSNode *node = &root;
        std::vector<MCTSNode *> search_path;
        search_path.push_back(node);
        int current_tree_depth = 0;

        // Resursively select child according to tree search policy
        while (node->is_expanded()) {
            ++current_tree_depth;
            node = select_child(node);
            search_path.push_back(node);
            // Players take turns (unless only a single player)
            virtual_to_play = (virtual_to_play + 1) % num_players_;
        }

        // Found node to expand; use the dynamics function to find next hidden state
        // starting at the parent's saved encoded state
        assert((int)search_path.size() >= 2);
        MCTSNode *parent = search_path[search_path.size() - 2];
        VPRNetModel::InferenceOutputs recurrent_inference_output =
            vpr_eval_->RecurrentInference(parent->encoded_state, node->action);
        node->expand(action_space_, virtual_to_play, recurrent_inference_output.reward,
                     recurrent_inference_output.policy, recurrent_inference_output.encoded_state);
        // backprop inference value along our search path
        backpropogate(search_path, recurrent_inference_output.value, virtual_to_play);
        max_tree_depth = std::max(max_tree_depth, current_tree_depth);
    }

    // Extract necessary search statistics
    std::vector<Action> child_actions;
    std::vector<double> children_relative_visit;
    std::vector<double> emperical_policy;
    root.get_search_statistics(action_space_, child_actions, children_relative_visit, emperical_policy);
    return {root.value(), max_tree_depth, emperical_policy, children_relative_visit, child_actions};
}

// Backpropogate the value up along the nodes for the given search path
void MCTS::backpropogate(std::vector<MCTSNode *> &search_path, double value, types::Player to_play) {
    for (auto it = search_path.rbegin(); it != search_path.rend(); ++it) {
        MCTSNode *node = *it;
        bool same_player = node->to_play == to_play;
        // Update node values
        node->value_sum += same_player ? value : -value;
        node->visit_count += 1;
        // Update minmax stats
        // reward stored at current node is reward received by transitioning from parent to current. In 2
        // player games, the value of parent is reward + discounted value in view of parent
        double new_q = node->reward + (discount_ * (num_players_ == 2 ? -node->value() : node->value()));
        minmax_stats_.update(new_q);
        // Value propogated upwards will handle having value in correct orentation above, so we only need to
        // correct for reward here
        double reward = (same_player && num_players_ == 2) ? -node->reward : node->reward;
        value = reward + (discount_ * value);
    }
}

// Selects a child node which maximizes the modifed pUCT formulation
MCTSNode *MCTS::select_child(MCTSNode *node) {
    // Shuffle to reduce selection bias, and select child with max UCB score
    std::shuffle(node->children.begin(), node->children.end(), rng_);
    MCTSNode *selected_child = nullptr;
    double child_value = 0;
    for (int i = 0; i < (int)node->children.size(); ++i) {
        double value = ucb_score(node, &node->children[i]);
        if (!selected_child || value > child_value) {
            selected_child = &(node->children[i]);
            child_value = value;
        }
    }
    assert(selected_child);
    return selected_child;
}

// Calculate the UCB score for the node using the modified pUCT rule (See Appendix B Search)
double MCTS::ucb_score(MCTSNode *parent, MCTSNode *child) {
    // Prior score
    double pb_c = std::log((parent->visit_count + pb_c_base_ + 1) / pb_c_base_) + pb_c_init_;
    pb_c *= std::sqrt(parent->visit_count) / (child->visit_count + 1);
    double prior_score = pb_c * child->prior;
    // Value score
    double value_score = 0;
    if (child->visit_count > 0) {
        // Value is in view of child node, assume we alternate every move (thus value for parent is -value of
        // child in 2 player games). Value of current node is reward + value of child node.
        double child_value = (num_players_ == 1) ? child->value() : -child->value();
        value_score = minmax_stats_.normalize(child->reward + (discount_ * child_value));
    }
    return prior_score + value_score;
}
// ---------------------------------- MCTS ----------------------------------

// -------------------------------- MCTSNode --------------------------------
// Check whether the node has been previously expanded.
bool MCTSNode::is_expanded() const {
    return children.size() > 0;
}

// Get the mean value of the node.
double MCTSNode::value() const {
    return (visit_count == 0) ? 0 : value_sum / visit_count;
}

// Expand the current node using the value/policy and reward prediction
//  from the recurrent dynamics model.
void MCTSNode::expand(const std::vector<types::Action> &actions, types::Player to_play, double reward,
                      const std::vector<double> &policy_probs, const types::Observation &encoded_state) {
    this->to_play = to_play;
    this->reward = reward;
    this->encoded_state = encoded_state;
    children.reserve(actions.size());
    for (int i = 0; i < (int)actions.size(); ++i) {
        assert(actions[i] < (int)policy_probs.size());
        children.push_back(MCTSNode(actions[i], policy_probs[actions[i]]));
    }
}

// Add a weighted sum of dirichlet exploration noise over the priors
void MCTSNode::add_exploration_noise(double dirichlet_alpha, double dirichlet_epsilon, std::mt19937 &rng) {
    std::vector<double> noise = util::sample_dirichlet(dirichlet_alpha, children.size(), rng);
    assert(children.size() == noise.size());
    for (int i = 0; i < (int)children.size(); ++i) {
        children[i].prior = children[i].prior * (1 - dirichlet_epsilon) + (noise[i] * dirichlet_epsilon);
    }
}

// Get the search statistics in form of a policy.
void MCTSNode::get_search_statistics(const std::vector<types::Action> &action_space,
                                     std::vector<Action> &child_actions,
                                     std::vector<double> &children_relative_visit,
                                     std::vector<double> &emperical_policy) const {
    int sum_visits = 0;
    // std::unordered_map<types::Action, int> child_action_map;
    emperical_policy = std::vector<double>(action_space.size(), 0);
    for (int i = 0; i < (int)children.size(); ++i) {
        child_actions.push_back(children[i].action);
        children_relative_visit.push_back(children[i].visit_count);
        emperical_policy[children[i].action] = children[i].visit_count;
        sum_visits += children[i].visit_count;
    }
    // Relative children
    for (int i = 0; i < (int)children_relative_visit.size(); ++i) {
        children_relative_visit[i] /= sum_visits;
    }
    // Set relative emperical policy if action taken, 0 otherwise
    for (int i = 0; i < (int)emperical_policy.size(); ++i) {
        emperical_policy[i] /= sum_visits;
    }
}
// -------------------------------- MCTSNode --------------------------------

// ------------------------------- MinMaxStats ------------------------------
// Reset the stored min/max bounds
void MinMaxStats::reset() {
    maximum_ = init_maximum_;
    minimum_ = init_minimum_;
}

// Update the seen min/max bounds.
void MinMaxStats::update(double value) {
    maximum_ = std::max(maximum_, value);
    minimum_ = std::min(minimum_, value);
}

// Normalize the given value using the known/seen min/max bounds.
double MinMaxStats::normalize(double value) const {
    // Only normalize if we have seen max/min values
    return (maximum_ > minimum_) ? (value - minimum_) / (maximum_ - minimum_) : value;
}
// ------------------------------- MinMaxStats ------------------------------

}    // namespace algorithm
}    // namespace muzero_cpp
