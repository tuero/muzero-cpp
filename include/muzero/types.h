#ifndef MUZERO_TYPES_H_
#define MUZERO_TYPES_H_

#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace muzero {

using Player = int;
using Action = int;
constexpr Player InvalidPlayer = -1;
constexpr Action InvalidAction = -1;

using Observation = std::vector<float>;
struct ObservationShape {
    int c;    // Number of channels
    int h;    // Height of observation
    int w;    // Width of observation
    auto operator==(const ObservationShape &rhs) const -> bool
    {
        return c == rhs.c && h == rhs.h && w == rhs.w;
    }
    auto operator!=(const ObservationShape &rhs) const -> bool
    {
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
enum class OpponentType {
    Self,      // Play against itself
    Random,    // Opponent randomly choose a legal action
    Expert,    // User game-defined expert (see abstract_game.h / Examples)
    Human      // Human types in input action (see abstract_game.h / Examples)
};
inline auto to_string(const OpponentType opponent_type) -> std::string
{
    switch (opponent_type) {
    case OpponentType::Self:
        return "self";
    case OpponentType::Random:
        return "random";
    case OpponentType::Expert:
        return "Expert";
    case OpponentType::Human:
        return "human";
    }
    // @NOTE: If upgrading to C++23, replace this with std::unreachable
#if defined(_MSC_VER) && !defined(__clang__)    // MSVC
    __assume(false);
#else    // GCC, Clang
    __builtin_unreachable();
#endif
}

// Checkpoint steps
enum CheckpointStep {
    kMostRecentCheckpointStep = -1,
    kBestPerformanceCheckpointStep = -2
};

// AbstractGame
// These are the values returned after taking a step in the environment
struct StepReturn {
    Observation observation;
    double reward;
    bool done;
};

constexpr double NINF_D = std::numeric_limits<double>::lowest();
constexpr double INF_D = std::numeric_limits<double>::max();

}    // namespace muzero

#endif    // MUZERO_TYPES_H_
