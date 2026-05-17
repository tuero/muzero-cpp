
#ifndef MUZERO_EXAMPLE_TICTACTOE_H_
#define MUZERO_EXAMPLE_TICTACTOE_H_

#include <muzero/muzero.h>

#include <absl/strings/str_format.h>

#include <array>
#include <cassert>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// Simple connect4 environment implementation
// Some of the logic taken from: https://github.com/werner-duvaud/muzero-general/blob/master/games/tictactoe.py
template <std::size_t SIZE>
class TicTacToe {
public:
    static_assert(SIZE >= 3, "TicTacToe SIZE must be at least 3");
    static_assert(SIZE % 2 == 1, "TicTacToe SIZE must be odd");

    TicTacToe(int seed = 0)
        : rng_(static_cast<std::mt19937::result_type>(seed))
    {}

    // Get the observations shape
    static auto obs_shape() -> muzero::ObservationShape
    {
        return {.c = 3, .h = SIZE, .w = SIZE};
    }

    // Get the entire action space
    static auto action_space() -> std::vector<muzero::Action>
    {
        std::vector<muzero::Action> actions;
        for (std::size_t i = 0; i < SIZE * SIZE; ++i) {
            actions.push_back(static_cast<muzero::Action>(i));
        }
        return actions;
    }

    // Get the current player
    [[nodiscard]] auto to_play() const -> muzero::Player
    {
        return (player_ == 1) ? 0 : 1;
    }

    // Reset environment and send starting observation
    auto reset() -> muzero::Observation
    {
        player_ = 1;
        empty_squares_ = 0;
        for (auto &p : board_) {
            p = 0;
        }
        return get_observation();
    }

    // Step the environment, send next observation, reward, and done flag
    auto step(muzero::Action action) -> muzero::StepReturn
    {
        const auto action_idx = static_cast<std::size_t>(action);
        assert(action_idx < SIZE * SIZE && board_[action_idx] == 0);
        board_[action_idx] = player_;
        ++empty_squares_;
        bool is_win = have_winner();
        bool done = (is_win || empty_squares_ == board_.size());
        double reward = is_win ? 1 : 0;
        player_ *= -1;
        return {.observation = get_observation(), .reward = reward, .done = done};
    }

    // Current legal actions, subset of action space
    [[nodiscard]] auto legal_actions() const -> std::vector<muzero::Action>
    {
        std::vector<muzero::Action> actions;
        for (std::size_t i = 0; i < SIZE * SIZE; ++i) {
            if (board_[i] == 0) {
                actions.push_back(static_cast<int>(i));
            }
        }
        return actions;
    }

    // Find an expert action
    auto expert_action() -> muzero::Action
    {
        // Check if single move can win and play
        for (std::size_t i = 0; i < SIZE * SIZE; ++i) {
            if (board_[i] != 0) {
                continue;
            }
            auto temp_board = board_;
            temp_board[i] = player_;
            if (have_winner(player_, temp_board)) {
                return static_cast<muzero::Action>(i);
            }
        }

        // Check if single move can win and block
        for (std::size_t i = 0; i < SIZE * SIZE; ++i) {
            if (board_[i] != 0) {
                continue;
            }
            auto temp_board = board_;
            temp_board[i] = -1 * player_;
            if (have_winner(-1 * player_, temp_board)) {
                return static_cast<muzero::Action>(i);
            }
        }

        // Try to play corners
        std::array<std::size_t, 4> corners{{0, SIZE - 1, SIZE * SIZE - SIZE, SIZE * SIZE - 1}};
        for (const auto &c : corners) {
            if (board_[c] == 0) {
                return static_cast<muzero::Action>(c);
            }
        }

        // Try to play center
        const std::size_t center = (SIZE / 2) * SIZE + (SIZE / 2);
        if (board_[center] == 0) {
            return static_cast<muzero::Action>(center);
        }

        // Otherwise random move
        auto actions = legal_actions();
        std::uniform_int_distribution<std::size_t> dist(0, actions.size() - 1);
        return actions[dist(rng_)];
    }

    // Pretty string of board
    [[nodiscard]] auto board_to_str() const -> std::string
    {
        std::string out;
        for (std::size_t r = 0; r < SIZE; ++r) {
            out += "|";
            for (std::size_t c = 0; c < SIZE; ++c) {
                out += " " + print_map_.at(static_cast<std::size_t>(board_[r * SIZE + c] + 1)) + " ";
            }
            out += "|\n";
        }
        return out;
    }

private:
    using BoardT = std::array<int, SIZE * SIZE>;
    using SubArrayT = std::array<int, SIZE>;
    inline static const std::array<std::string, 3> print_map_{"2", "-", "1"};

    // Get the current observation (3 x SIZE x SIZE)
    [[nodiscard]] auto get_observation() const -> muzero::Observation
    {
        muzero::Observation observation(3 * SIZE * SIZE);
        std::size_t offset = 0;

        // player 1 slice
        for (std::size_t i = 0; i < SIZE * SIZE; ++i) {
            observation[offset + i] = board_[i] == 1 ? 1 : 0;
        }

        // player 2 slice
        offset += SIZE * SIZE;
        for (std::size_t i = 0; i < SIZE * SIZE; ++i) {
            observation[offset + i] = board_[i] == -1 ? 1 : 0;
        }

        // to play slice
        offset += SIZE * SIZE;
        for (std::size_t i = 0; i < SIZE * SIZE; ++i) {
            observation[offset + i] = static_cast<float>(player_);
        }

        return observation;
    }

    // Check if given board has a winner in view of given player
    auto have_winner(int player, const BoardT &board) -> bool
    {
        const auto win_count = static_cast<int>(SIZE) * player;
        // Horizontal/vertical checks
        for (std::size_t i = 0; i < SIZE; ++i) {
            if (sum_array(get_row(i, board)) == win_count) {
                return true;
            }
            if (sum_array(get_col(i, board)) == win_count) {
                return true;
            }
        }
        if (sum_array(get_diag(board)) == win_count) {
            return true;
        }
        if (sum_array(get_antidiag(board)) == win_count) {
            return true;
        }
        return false;
    }

    // Check if game has a winner
    auto have_winner() -> bool
    {
        const auto n = static_cast<int>(SIZE);
        // Horizontal/vertical checks
        for (std::size_t i = 0; i < SIZE; ++i) {
            if (std::abs(sum_array(get_row(i, board_))) == n) {
                return true;
            }
            if (std::abs(sum_array(get_col(i, board_))) == n) {
                return true;
            }
        }
        if (std::abs(sum_array(get_diag(board_))) == n) {
            return true;
        }
        if (std::abs(sum_array(get_antidiag(board_))) == n) {
            return true;
        }
        return false;
    }

    // Sum a sub_array
    auto sum_array(const SubArrayT &sub_array) -> int
    {
        return std::reduce(std::begin(sub_array), std::end(sub_array));
    }

    // Get row from board
    auto get_row(std::size_t row, const BoardT &board) -> SubArrayT
    {
        SubArrayT array;
        for (std::size_t i = 0; i < SIZE; ++i) {
            array[i] = board[row * SIZE + i];
        }
        return array;
    }

    // Get column from board
    auto get_col(std::size_t col, const BoardT &board) -> SubArrayT
    {
        SubArrayT array;
        for (std::size_t i = 0; i < SIZE; ++i) {
            array[i] = board[i * SIZE + col];
        }
        return array;
    }

    // Get board diagonal
    auto get_diag(const BoardT &board) -> SubArrayT
    {
        SubArrayT array;
        for (std::size_t i = 0; i < SIZE; ++i) {
            array[i] = board[i * SIZE + i];
        }
        return array;
    }

    // Get board anti-diagonal
    auto get_antidiag(const BoardT &board) -> SubArrayT
    {
        SubArrayT array;
        for (std::size_t i = 0; i < SIZE; ++i) {
            array[i] = board[i * SIZE + (SIZE - 1 - i)];
        }
        return array;
    }

    muzero::Player player_ = 1;
    std::size_t empty_squares_ = 0;
    BoardT board_{};
    std::mt19937 rng_;
};

#endif    // MUZERO_EXAMPLE_TICTACTOE_H_
