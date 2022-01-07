#ifndef MUZERO_CPP_MUZERO_H_
#define MUZERO_CPP_MUZERO_H_

#include <memory>

#include "muzero-cpp/abstract_game.h"
#include "muzero-cpp/config.h"

namespace muzero_cpp {

/**
 * Use the model to play a test game
 * @param config A muzero configuration struct
 * @param game_factory Factory to create instances of the game (see abstract_game.h and examples for usage)
 */
bool play_test_model(const muzero_config::MuZeroConfig& config,
                     std::function<std::unique_ptr<AbstractGame>(int)> game_factory);

/**
 * Train a model using the muzero algorithm
 * @param config A muzero configuration struct
 * @param game_factory Factory to create instances of the game (see abstract_game.h and examples for usage)
 */
bool muzero(const muzero_config::MuZeroConfig& config,
            std::function<std::unique_ptr<AbstractGame>(int)> game_factory);

}    // namespace muzero_cpp

#endif    // MUZERO_CPP_MUZERO_H_