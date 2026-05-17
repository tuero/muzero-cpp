#include "mcts.h"

#include <muzero/types.h>

#include "util.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <ranges>
#include <vector>

namespace muzero::algorithm {

using namespace model;

// ---------------------------------- MCTS ----------------------------------
// Constructor for MCTS object
MCTS::MCTS(const Config &config, int seed, std::shared_ptr<Evaluator> vpr_eval)
    : num_players_(config.num_players),
      num_simulations_(config.num_simulations),
      dirichlet_alpha_(config.dirichlet_alpha),
      dirichlet_epsilon_(config.dirichlet_epsilon),
      pb_c_base_(config.pb_c_base),
      pb_c_init_(config.pb_c_init),
      discount_(config.discount),
      minmax_stats_(config.value_lowerbound, config.value_upperbound),
      rng_(static_cast<std::mt19937::result_type>(seed)),
      action_space_(config.action_space),
      vpr_eval_(std::move(vpr_eval))
{
    // Only deal with 1 or 2 player games
    assert(num_players_ <= 2);
}

// Run MCTS for a set number of iterations
MCTSReturn MCTS::run(
    const Observation &stacked_observation,
    const std::vector<Action> &legal_actions,
    Player to_play,
    bool add_exploration_noise
)
{
    assert(legal_actions.size() > 0);
    // Create root and send observation into the encoded state space
    MCTSNode root(InvalidAction, 0);
    VPRNetModel::InferenceOutputs initial_inference_output = vpr_eval_->InitialInference(stacked_observation);
    root.expand(
        legal_actions,
        to_play,
        initial_inference_output.reward,
        initial_inference_output.policy,
        initial_inference_output.encoded_state
    );

    // Add dirichlet exploration noise to the policy
    if (add_exploration_noise) {
        root.add_exploration_noise(dirichlet_alpha_, dirichlet_epsilon_, rng_);
    }

    // Reset values to empty search tree
    minmax_stats_.reset();
    int max_tree_depth = 0;

    // Run N simulations
    for (int i = 0; i < num_simulations_; ++i) {
        Player virtual_to_play = to_play;
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
        node->expand(
            action_space_,
            virtual_to_play,
            recurrent_inference_output.reward,
            recurrent_inference_output.policy,
            recurrent_inference_output.encoded_state
        );
        // backprop inference value along our search path
        backpropogate(search_path, recurrent_inference_output.value, virtual_to_play);
        max_tree_depth = std::max(max_tree_depth, current_tree_depth);
    }

    // Extract necessary search statistics
    std::vector<Action> child_actions;
    std::vector<double> children_relative_visit;
    std::vector<double> emperical_policy;
    root.get_search_statistics(action_space_, child_actions, children_relative_visit, emperical_policy);
    return {
        .root_value = root.value(),
        .max_tree_depth = max_tree_depth,
        .emperical_policy = emperical_policy,
        .children_relative_visit = children_relative_visit,
        .child_actions = child_actions
    };
}

// Backpropogate the value up along the nodes for the given search path
void MCTS::backpropogate(std::vector<MCTSNode *> &search_path, double value, Player to_play)
{
    for (auto node : search_path | std::views::reverse) {
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
auto MCTS::select_child(MCTSNode *node) -> MCTSNode *
{
    // Shuffle to reduce selection bias, and select child with max UCB score
    std::ranges::shuffle(node->children, rng_);
    MCTSNode *selected_child = nullptr;
    double child_value = 0;
    for (std::size_t i = 0; i < node->children.size(); ++i) {
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
auto MCTS::ucb_score(MCTSNode *parent, MCTSNode *child) -> double
{
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
auto MCTSNode::is_expanded() const -> bool
{
    return children.size() > 0;
}

// Get the mean value of the node.
auto MCTSNode::value() const -> double
{
    return (visit_count == 0) ? 0 : value_sum / visit_count;
}

// Expand the current node using the value/policy and reward prediction
//  from the recurrent dynamics model.
void MCTSNode::expand(
    const std::vector<Action> &actions,
    Player tp,
    double r,
    const std::vector<double> &policy_probs,
    const Observation &es
)
{
    this->to_play = tp;
    this->reward = r;
    this->encoded_state = es;
    children.reserve(actions.size());
    for (const auto &a : actions) {
        assert(static_cast<std::size_t>(a) < policy_probs.size());
        children.emplace_back(a, policy_probs[static_cast<std::size_t>(a)]);
    }
}

// Add a weighted sum of dirichlet exploration noise over the priors
void MCTSNode::add_exploration_noise(double dirichlet_alpha, double dirichlet_epsilon, std::mt19937 &rng)
{
    std::vector<double> noise = sample_dirichlet(dirichlet_alpha, static_cast<int>(children.size()), rng);
    assert(children.size() == noise.size());
    for (std::size_t i = 0; i < children.size(); ++i) {
        children[i].prior = children[i].prior * (1 - dirichlet_epsilon) + (noise[i] * dirichlet_epsilon);
    }
}

// Get the search statistics in form of a policy.
void MCTSNode::get_search_statistics(
    const std::vector<Action> &action_space,
    std::vector<Action> &child_actions,
    std::vector<double> &children_relative_visit,
    std::vector<double> &emperical_policy
) const
{
    int sum_visits = 0;
    // std::unordered_map<types::Action, int> child_action_map;
    emperical_policy = std::vector<double>(action_space.size(), 0);
    for (const auto &child : children) {
        child_actions.push_back(child.action);
        children_relative_visit.push_back(child.visit_count);
        emperical_policy[static_cast<std::size_t>(child.action)] = child.visit_count;
        sum_visits += child.visit_count;
    }
    // Relative children
    for (auto &child_relative_visit : children_relative_visit) {
        child_relative_visit /= sum_visits;
    }
    // Set relative emperical policy if action taken, 0 otherwise
    for (auto &ep : emperical_policy) {
        ep /= sum_visits;
    }
}
// -------------------------------- MCTSNode --------------------------------

// ------------------------------- MinMaxStats ------------------------------
// Reset the stored min/max bounds
void MinMaxStats::reset()
{
    maximum_ = init_maximum_;
    minimum_ = init_minimum_;
}

// Update the seen min/max bounds.
void MinMaxStats::update(double value)
{
    maximum_ = std::max(maximum_, value);
    minimum_ = std::min(minimum_, value);
}

// Normalize the given value using the known/seen min/max bounds.
auto MinMaxStats::normalize(double value) const -> double
{
    // Only normalize if we have seen max/min values
    return (maximum_ > minimum_) ? (value - minimum_) / (maximum_ - minimum_) : value;
}
// ------------------------------- MinMaxStats ------------------------------

}    // namespace muzero::algorithm
