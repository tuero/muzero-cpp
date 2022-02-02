#include <algorithm>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "connect4.h"
#include "muzero-cpp/abstract_game.h"
#include "muzero-cpp/config.h"
#include "muzero-cpp/default_flags.h"
#include "muzero-cpp/muzero.h"
#include "muzero-cpp/types.h"

using namespace muzero_cpp;
using namespace muzero_cpp::types;
using namespace muzero_cpp::muzero_config;

class Connect4Env : public AbstractGame {
public:
    Connect4Env(int seed) : env_(seed) {}
    Connect4Env() = delete;
    ~Connect4Env() = default;

    /**
     * Reset the environment for a new game.
     */
    Observation reset() override {
        return env_.reset();
    }

    /**
     * Apply the given action to the environment
     * @param action The action to send to the environment
     * @return A struct containing the observation, reward, and a flag indicating if the game is done
     */
    StepReturn step(Action action) override {
        StepReturn step_return = env_.step(action);
        step_return.reward *= 10;
        return step_return;
    }

    /**
     * Returns the current player to play
     * @return The player number to play
     */
    Player to_play() const override {
        return env_.to_play();
    }

    /**
     * Return the legal actions for the current environment state.
     * @returns Vector of legal action ids
     */
    std::vector<Action> legal_actions() const override {
        return env_.legal_actions();
    }

    /**
     * Returns an action given by an expert player/bot.
     * @returns An expert action which is legal
     */
    Action expert_action() override {
        return env_.expert_action();
    }

    /**
     * Returns a legal action given by human input.
     * @returns An action which is legal
     */
    Action human_to_action() override {
        std::vector<Action> legal_actions = env_.legal_actions();
        Action action;
        while (true) {
            std::cout << "Enter a column to play: ";
            std::cin >> action;
            if (std::find(legal_actions.begin(), legal_actions.end(), action) != legal_actions.end()) {
                break;
            }
        }
        return action;
    }

    /**
     * Render the environment for testing games.
     */
    void render() override {
        std::cout << env_.board_to_str() << std::endl;
    }

    /**
     * Convert action to human readable string.
     * @param action The action to convert
     * @returns The string format of the action
     */
    std::string action_to_string(types::Action action) const override {
        return std::to_string(action);
    }

private:
    Connect4 env_;    // Environment
};

// Encode action as feature plane of values 1/action
Observation encode_action(Action action) {
    static const int num_actions = Connect4::action_space().size();
    ObservationShape obs_shape = Connect4::obs_shape();
    Observation obs(obs_shape.w * obs_shape.h, (double)action / num_actions);
    return obs;
}

// Simple softmax schedule
double get_softmax(int step) {
    if (step < 25000) {
        return 1.0;
    } else if (step < 50000) {
        return 0.5;
    }
    return 0.125;
}

// Additional flag to choose whether to test or not
ABSL_FLAG(bool, test, false, "Test using human input.");

int main(int argc, char** argv) {
    // parse flags
    parse_flags(argc, argv);
    MuZeroConfig config = get_initial_config();

    // Set specific values for the game
    config.observation_shape = Connect4::obs_shape();
    config.action_space = Connect4::action_space();
    config.network_config.normalize_hidden_states = true;
    config.action_channels = 1;
    config.num_players = 2;
    config.value_upperbound = 10;
    config.value_lowerbound = -10;
    config.min_reward = -10;
    config.max_reward = 10;
    config.min_value = -10;
    config.max_value = 10;
    config.opponent_type = OpponentTypes::Expert;
    config.action_representation_initial = encode_action;
    config.action_representation_recurrent = encode_action;
    config.visit_softmax_temperature = get_softmax;

    // Perform learning or testing
    if (absl::GetFlag(FLAGS_test)) {
        config.opponent_type = OpponentTypes::Human;
        return play_test_model(config, game_factory<Connect4Env>);
    } else {
        return muzero(config, game_factory<Connect4Env>);
    }

    return 0;
}