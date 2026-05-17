#include <muzero/muzero.h>

#include "tictactoe.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace muzero;

using TTT = TicTacToe<3>;

namespace {

class TicTacToeEnv : public AbstractGame {
public:
    TicTacToeEnv(int seed)
        : env_(seed)
    {}
    TicTacToeEnv() = delete;
    ~TicTacToeEnv() = default;

    /**
     * Reset the environment for a new game.
     */
    auto reset() -> Observation override
    {
        step_ = 0;
        return env_.reset();
    }

    /**
     * Apply the given action to the environment
     * @param action The action to send to the environment
     * @return A struct containing the observation, reward, and a flag indicating if the game is done
     */
    [[nodiscard]] auto step(Action action) -> StepReturn override
    {
        ++step_;
        StepReturn step_return = env_.step(action);
        step_return.reward *= 10;
        return step_return;
    }

    /**
     * Returns the current player to play
     * @return The player number to play
     */
    [[nodiscard]] auto to_play() const -> Player override
    {
        return env_.to_play();
    }

    /**
     * Return the legal actions for the current environment state.
     * @returns Vector of legal action ids
     */
    [[nodiscard]] auto legal_actions() const -> std::vector<Action> override
    {
        return env_.legal_actions();
    }

    /**
     * Returns an action given by an expert player/bot.
     * @returns An expert action which is legal
     */
    auto expert_action() -> Action override
    {
        return env_.expert_action();
    }

    /**
     * Returns a legal action given by human input.
     * @returns An action which is legal
     */
    auto human_to_action() -> Action override
    {
        std::vector<Action> legal_actions = env_.legal_actions();
        Action action = -1;
        while (true) {
            std::cout << "Enter an action to play: ";
            std::cin >> action;
            if (std::ranges::find(legal_actions, action) != legal_actions.end()) {
                break;
            }
        }
        return action;
    }

    /**
     * Render the environment for testing games.
     */
    void render() override
    {
        std::cout << env_.board_to_str() << std::endl;
    }

    /**
     * Convert action to human readable string.
     * @param action The action to convert
     * @returns The string format of the action
     */
    [[nodiscard]] auto action_to_string(Action action) const -> std::string override
    {
        return std::to_string(action);
    }

private:
    TTT env_;         // environment
    int step_ = 0;    // current step of the environment
};

// Encode action as feature plane of values 1/action
auto encode_action(Action action) -> Observation
{
    static const auto num_actions = static_cast<float>(TTT::action_space().size());
    const auto obs_shape = TTT::obs_shape();
    const auto N = static_cast<std::size_t>(obs_shape.w * obs_shape.h);
    const auto a = static_cast<float>(action) / num_actions;
    Observation obs(N, a);
    return obs;
}

// Simple softmax schedule
auto get_softmax(int step) -> double
{
    if (step < 50000) {
        return 1.0;
    } else if (step < 75000) {
        return 0.5;
    }
    return 0.25;
}

}    // namespace

// Additional flag to choose whether to test or not
ABSL_FLAG(bool, test, false, "Test using human input.");

int main(int argc, char **argv)
{
    // parse flags
    parse_flags(argc, argv);
    Config config = initial_config();

    // Set specific values for the game
    config.observation_shape = TTT::obs_shape();
    config.action_space = TTT::action_space();
    config.action_channels = 1;
    config.num_players = 2;
    config.value_upperbound = 10;
    config.value_lowerbound = -10;
    config.min_reward = -10;
    config.max_reward = 10;
    config.min_value = -10;
    config.max_value = 10;
    config.opponent_type = OpponentType::Expert;
    config.action_representation_initial = encode_action;
    config.action_representation_recurrent = encode_action;
    config.visit_softmax_temperature = get_softmax;

    // Perform learning or testing
    if (absl::GetFlag(FLAGS_test)) {
        config.opponent_type = OpponentType::Human;
        return play_test_model(config, game_factory<TicTacToeEnv>);
    } else {
        return ::muzero::muzero(config, game_factory<TicTacToeEnv>);
    }

    return 0;
}
