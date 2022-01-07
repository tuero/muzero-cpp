#ifndef MUZERO_CPP_ABSTRACT_GAME_H_
#define MUZERO_CPP_ABSTRACT_GAME_H_

#include <memory>
#include <vector>

#include "muzero-cpp/types.h"

namespace muzero_cpp {

// Abstract game class which wraps and exposes the game to MuZero
class AbstractGame {
public:
    AbstractGame(int seed) {
        (void)seed;
    };
    AbstractGame() = delete;
    virtual ~AbstractGame() = default;

    /**
     * Reset the environment for a new game.
     */
    virtual types::Observation reset() = 0;

    /**
     * Apply the given action to the environment
     * @param action The action to send to the environment
     * @return A struct containing the observation, reward, and a flag indicating if the game is done
     */
    virtual types::StepReturn step(types::Action action) = 0;

    /**
     * Returns the current player to play
     * @return The player number to play
     */
    virtual types::Player to_play() const = 0;

    /**
     * Return the legal actions for the current environment state.
     * This can return the entire action space if all actions are legal for the current state.
     * @returns Vector of legal action ids
     */
    virtual std::vector<types::Action> legal_actions() const = 0;

    /**
     * Returns an action given by an expert player/bot.
     * @note If you're game is 1 player or you don't want to support this, return a random action
     *       so that it compiles, and don't use OpponentTypes::Expert in the configuration.
     * @returns An expert action which is legal
     */
    virtual types::Action expert_action() = 0;

    /**
     * Returns a legal action given by human input.
     * @returns An action which is legal
     */
    virtual types::Action human_to_action() const = 0;

    /**
     * Render the environment for testing games.
     */
    virtual void render() {};

    /**
     * Convert action to human readable string.
     * @note In most cases, this is a trivial int to str conversion
     * @param action The action to convert
     * @returns The string format of the action
     */
    virtual std::string action_to_string(types::Action action) const = 0;
};

// Factory to create instances of the abstract game
template <typename T>
std::unique_ptr<AbstractGame> game_factory(int seed) {
    return std::make_unique<T>(seed);
}

}    // namespace muzero_cpp

#endif    // MUZERO_CPP_ABSTRACT_GAME_H_