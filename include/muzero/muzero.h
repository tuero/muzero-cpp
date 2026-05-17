#ifndef MUZERO_MUZERO_H_
#define MUZERO_MUZERO_H_

#include <muzero/abstract_game.h>
#include <muzero/config.h>
#include <muzero/default_flags.h>
#include <muzero/types.h>

#include <memory>

namespace muzero {

/**
 * Use the model to play a test game
 * @param config A muzero configuration struct
 * @param game_factory Factory to create instances of the game (see abstract_game.h and examples for usage)
 */
bool play_test_model(const Config &config, std::function<std::unique_ptr<AbstractGame>(int)> game_factory);

/**
 * Train a model using the muzero algorithm
 * @param config A muzero configuration struct
 * @param game_factory Factory to create instances of the game (see abstract_game.h and examples for usage)
 */
bool muzero(const Config &config, std::function<std::unique_ptr<AbstractGame>(int)> game_factory);

}    // namespace muzero

#endif    // MUZERO_MUZERO_H_
