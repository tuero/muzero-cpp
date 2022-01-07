#include "muzero-cpp/mcts.h"

#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "muzero-cpp/config.h"
#include "muzero-cpp/types.h"
#include "tests/test_macros.h"

namespace muzero_cpp {
using namespace types;
namespace algorithm {

class TestEvaluator : public Evaluator {
public:
    VPRNetModel::InferenceOutputs InitialInference(const Observation& stacked_observations) override {
        (void)stacked_observations;
        return {1, 1, {0.01, 0.2, 0.3, 0.1, 0.5, 0.25}, {0}};
    }
    VPRNetModel::InferenceOutputs RecurrentInference(const Observation& hidden_state,
                                                     Action action) override {
        (void)hidden_state;
        return {1, (double)action, {0.01, 0.2, 0.3, 0.1, 0.5, 0.25}, {0}};
    }
};

// Test backpropogating values
void mcts_backpropogate_test(const muzero_config::MuZeroConfig& config) {
    // 2 player game
    {
        MCTS mcts_(config, config.seed, std::make_shared<TestEvaluator>());
        MCTSNode root(-1, 0);
        std::vector<Action> actions{0, 1, 4, 5};
        Player player = 0;
        std::vector<double> policy_probs{0.01, 0.2, 0.3, 0.1, 0.5, 0.4};
        Observation encoded_state;

        // Expand out left subtree
        MCTSNode* node = &root;
        std::vector<MCTSNode*> search_path{node};
        for (int i = 0; i < 4; ++i) {
            node->expand(actions, player, 0.1, policy_probs, encoded_state);
            player = (player + 1) % 2;
            node = &node->children[0];
            node->reward = 0.1;
            search_path.push_back(node);
        }
        node->expand(actions, player, 0.1, policy_probs, encoded_state);

        // value * discount + r, with r in view of opposite player,
        // and value set in view of player argument
        std::vector<double> expected_values = {1, -0.4, 0.3, -0.05, 0.125};

        // Backprop and check values
        mcts_.backpropogate(search_path, 1, player);
        for (int i = (int)search_path.size() - 1; i >= 0; --i) {
            node = search_path[i];
            REQUIRE_NEAR(node->value(), expected_values[expected_values.size() - i - 1], 0.001);
        }
    }
    // 1 player game
    {
        muzero_config::MuZeroConfig config_single = config;
        config_single.num_players = 1;
        MCTS mcts_(config_single, config.seed, std::make_shared<TestEvaluator>());
        MCTSNode root(-1, 0);
        std::vector<Action> actions{0, 1, 4, 5};
        Player player = 0;
        std::vector<double> policy_probs{0.01, 0.2, 0.3, 0.1, 0.5, 0.4};
        Observation encoded_state;

        // Expand out left subtree
        MCTSNode* node = &root;
        std::vector<MCTSNode*> search_path{node};
        for (int i = 0; i < 4; ++i) {
            node->expand(actions, player, 0.1, policy_probs, encoded_state);
            player = (player + 1) % 1;
            node = &node->children[0];
            node->reward = 0.1;
            search_path.push_back(node);
        }
        node->expand(actions, player, 0.1, policy_probs, encoded_state);

        // value * discount + r, with r in view of opposite player,
        // and value set in view of player argument
        std::vector<double> expected_values = {1, 0.6, 0.4, 0.3, 0.25};

        // Backprop and check values
        mcts_.backpropogate(search_path, 1, player);
        for (int i = (int)search_path.size() - 1; i >= 0; --i) {
            node = search_path[i];
            REQUIRE_NEAR(node->value(), expected_values[expected_values.size() - i - 1], 0.001);
        }
    }
}

// Test selecting a child
void mcts_select_child_test(const muzero_config::MuZeroConfig& config) {
    MCTS mcts_(config, config.seed, std::make_shared<TestEvaluator>());
    MCTSNode root(-1, 0);
    std::vector<Action> actions{0, 1, 4, 5};
    Player player = 0;
    double reward = 10;
    std::vector<double> policy_probs{0.01, 0.2, 0.3, 0.1, 0.5, 0.4};
    Observation encoded_state;

    // Expand and set children values
    root.expand(actions, player, reward, policy_probs, encoded_state);
    for (int i = 0; i < (int)root.children.size(); ++i) {
        root.children[i].reward = 1;
        root.children[i].visit_count = 1;
        root.children[i].value_sum = 0.5;
    }

    // Root has no visits, so prior has no influence
    MCTSNode* child = mcts_.select_child(&root);
    REQUIRE_EQUAL(child->action, 0);

    // Set root count, selected child should be one with higheset prior
    root.visit_count += 1;
    child = mcts_.select_child(&root);
    REQUIRE_EQUAL(child->action, 4);

    // Set low prior child with high value and visit count
    for (int i = 0; i < (int)root.children.size(); ++i) {
        if (root.children[i].action == 1) {
            root.children[i].value_sum -= 100;
            root.children[i].visit_count += 100;
        }
    }
    child = mcts_.select_child(&root);
    REQUIRE_EQUAL(child->action, 1);
}

// Test the UCB value selection
void mcts_ucb_test(const muzero_config::MuZeroConfig& config) {
    MCTS mcts_(config, config.seed, std::make_shared<TestEvaluator>());
    MCTSNode root(-1, 0);
    std::vector<Action> actions{0, 1, 4, 5};
    Player player = 0;
    double reward = 10;
    std::vector<double> policy_probs{0.01, 0.2, 0.3, 0.1, 0.5, 0.4};
    Observation encoded_state;

    root.visit_count = 2;
    root.expand(actions, player, reward, policy_probs, encoded_state);
    root.children[0].reward = 2;
    root.children[0].value_sum = -8;
    root.children[0].visit_count = 10;
    REQUIRE_NEAR(mcts_.ucb_score(&root, &root.children[0]), 2.401782, 0.001);
}

// Test minmax normalization stats
void minmax_stats_stest() {
    // default bounds
    {
        MinMaxStats minmax_stats;
        REQUIRE_EQUAL(minmax_stats.normalize(10), 10);
        minmax_stats.update(2);
        REQUIRE_EQUAL(minmax_stats.normalize(10), 10);
        minmax_stats.update(12);
        REQUIRE_EQUAL(minmax_stats.normalize(10), 0.8);
        REQUIRE_EQUAL(minmax_stats.normalize(2), 0.0);
        REQUIRE_EQUAL(minmax_stats.normalize(12), 1.0);
        minmax_stats.reset();
        REQUIRE_EQUAL(minmax_stats.normalize(10), 10);
        minmax_stats.update(2);
        REQUIRE_EQUAL(minmax_stats.normalize(10), 10);
    }

    // Bounds given
    {
        MinMaxStats minmax_stats(0, 1);
        REQUIRE_EQUAL(minmax_stats.normalize(0.5), 0.5);
        REQUIRE_EQUAL(minmax_stats.normalize(0), 0.0);
        REQUIRE_EQUAL(minmax_stats.normalize(1), 1.0);
        minmax_stats.update(0.5);
        REQUIRE_EQUAL(minmax_stats.normalize(0.5), 0.5);
        REQUIRE_EQUAL(minmax_stats.normalize(0), 0.0);
        REQUIRE_EQUAL(minmax_stats.normalize(1), 1.0);
        minmax_stats.reset();
        REQUIRE_EQUAL(minmax_stats.normalize(0.5), 0.5);
        REQUIRE_EQUAL(minmax_stats.normalize(0), 0.0);
        REQUIRE_EQUAL(minmax_stats.normalize(1), 1.0);
    }
}

// Tst expanding a leaf node
void mctsnode_test() {
    MCTSNode root(-1, 0);
    REQUIRE_FALSE(root.is_expanded());
    REQUIRE_EQUAL(root.value(), 0);

    std::vector<Action> actions{0, 1, 4, 5};
    Player player = 0;
    double reward = 10;
    std::vector<double> policy_probs{0.01, 0.2, 0.3, 0.1, 0.5, 0.25};
    Observation encoded_state;
    std::mt19937 rng(0);

    // Expand and check root
    root.expand(actions, player, reward, policy_probs, encoded_state);
    root.value_sum += 10;
    root.visit_count += 2;
    REQUIRE_EQUAL(root.children.size(), actions.size());
    REQUIRE_EQUAL(root.reward, reward);
    REQUIRE_EQUAL(root.value(), 5);
    REQUIRE_TRUE(root.is_expanded());

    // Check children
    for (int i = 0; i < (int)root.children.size(); ++i) {
        REQUIRE_EQUAL(root.children[i].prior, policy_probs[actions[i]]);
        REQUIRE_EQUAL(root.children[i].action, actions[i]);
        REQUIRE_FALSE(root.children[i].is_expanded());
    }

    // Check if adding noise changes policy probs
    root.add_exploration_noise(0.3, 0.25, rng);
    for (int i = 0; i < (int)root.children.size(); ++i) {
        REQUIRE_NEQUAL(root.children[i].prior, policy_probs[actions[i]]);
    }
}

}    // namespace algorithm
}    // namespace muzero_cpp

int main() {
    muzero_cpp::muzero_config::MuZeroConfig config;
    config.seed = 0;
    config.num_players = 2;
    config.num_simulations = 10;
    config.dirichlet_alpha = 0.3;
    config.dirichlet_epsilon = 0.25;
    config.pb_c_base = 1;
    config.pb_c_init = 0;
    config.discount = 0.5;
    config.value_lowerbound = 0;
    config.value_upperbound = 1;
    config.action_space = {0, 1, 2, 3, 4, 5};

    muzero_cpp::algorithm::mcts_backpropogate_test(config);
    muzero_cpp::algorithm::mcts_select_child_test(config);
    muzero_cpp::algorithm::mcts_ucb_test(config);
    muzero_cpp::algorithm::minmax_stats_stest();
    muzero_cpp::algorithm::mctsnode_test();
}
