#ifndef MUZERO_CPP_EXAMPLE_TICTACTOE_H_
#define MUZERO_CPP_EXAMPLE_TICTACTOE_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/strings/str_format.h"
#include "muzero-cpp/types.h"

// Simple connect4 environment implementation
// Some of the logic taken from: https://github.com/werner-duvaud/muzero-general/blob/master/games/tictactoe.py
class TicTacToe {
public:
    TicTacToe(int seed = 0) : player_(1), board_{}, rng_(seed) {
        for (int r = 0; r < SIZE; ++r) {
            for (int c = 0; c < SIZE; ++c) {
                board_[r][c] = 0;
            }
        }
    }

    // Get the observations shape
    static muzero_cpp::types::ObservationShape obs_shape() {
        return {3, SIZE, SIZE};
    }

    // Get the entire action space
    static std::vector<muzero_cpp::types::Action> action_space() {
        std::vector<muzero_cpp::types::Action> actions;
        for (int i = 0; i < 9; ++i) {
            actions.push_back(i);
        }
        return actions;
    }

    // Get the current player
    muzero_cpp::types::Player to_play() const {
        return (player_ == 1) ? 0 : 1;
    }

    // Reset environment and send starting observation
    muzero_cpp::types::Observation reset() {
        player_ = 1;
        for (int r = 0; r < SIZE; ++r) {
            for (int c = 0; c < SIZE; ++c) {
                board_[r][c] = 0;
            }
        }
        return get_observation();
    }

    // Step the environment, send next observation, reward, and done flag
    muzero_cpp::types::StepReturn step(muzero_cpp::types::Action action) {
        assert(std::find(legal_actions().begin(), legal_actions().end(), action) != legal_actions().end());
        int row = action / 3;
        int col = action % 3;
        assert(board_[row][col] == 0);
        board_[row][col] = player_;
        bool is_win = have_winner();
        bool done = (is_win || legal_actions().size() == 0);
        double reward = is_win ? 1 : 0;
        player_ *= -1;
        return {get_observation(), reward, done};
    }

    // Current legal actions, subset of action space
    std::vector<muzero_cpp::types::Action> legal_actions() const {
        std::vector<muzero_cpp::types::Action> actions;
        for (int i = 0; i < 9; ++i) {
            if (board_[i / 3][i % 3] == 0) { actions.push_back(i); }
        }
        return actions;
    }

    // Find an expert action
    muzero_cpp::types::Action expert_action() {
        auto actions = legal_actions();

        // Check if single move can win and play
        for (int i = 0; i < SIZE * SIZE; ++i) {
            if (board_[i / SIZE][i % SIZE] != 0) { continue; }
            BoardT temp_board = board_;
            temp_board[i / SIZE][i % SIZE] = player_;
            if (have_winner(player_, temp_board)) {
                assert(board_[i / SIZE][i % SIZE] == 0);
                return i;
            }
        }

        // Check if single move can win and block
        for (int i = 0; i < SIZE * SIZE; ++i) {
            if (board_[i / SIZE][i % SIZE] != 0) { continue; }
            BoardT temp_board = board_;
            temp_board[i / SIZE][i % SIZE] = -1 * player_;
            if (have_winner(-1 * player_, temp_board)) {
                assert(board_[i / SIZE][i % SIZE] == 0);
                return i;
            }
        }

        // Try to play corners
        std::array<int, 4> corners{{0, 2, 6, 8}};
        for (const auto &c : corners) {
            if (board_[c / SIZE][c % SIZE] == 0) {
                assert(board_[c / SIZE][c % SIZE] == 0);
                return c;
            }
        }

        // Try to play center
        if (board_[1][1] == 0) { return 4; }

        // Otherwise random move
        std::uniform_int_distribution<int> dist(0, actions.size() - 1);
        muzero_cpp::types::Action action = actions[dist(rng_)];
        assert(board_[action / SIZE][action % SIZE] == 0);
        return action;
    }

    // Pretty string of board
    std::string board_to_str() const {
        std::string out;
        for (int r = 0; r < SIZE; ++r) {
            out += "|";
            for (int c = 0; c < SIZE; ++c) {
                out += " " + print_map_.at(board_[r][c]) + " ";
            }
            out += "|\n";
        }
        return out;
    }

private:
    constexpr static int SIZE = 3;
    using BoardT = std::array<std::array<int, SIZE>, SIZE>;
    using SubArrayT = std::array<int, SIZE>;

    // Get the current observation (3 x 3 x 3)
    muzero_cpp::types::Observation get_observation() {
        muzero_cpp::types::Observation board_player1;
        board_player1.reserve(SIZE * SIZE);
        muzero_cpp::types::Observation board_player2;
        board_player2.reserve(SIZE * SIZE);
        muzero_cpp::types::Observation board_to_play(SIZE * SIZE, player_);

        for (int r = 0; r < SIZE; ++r) {
            for (int c = 0; c < SIZE; ++c) {
                board_player1.push_back(board_[r][c] == 1 ? 1 : 0);
                board_player2.push_back(board_[r][c] == -1 ? 1 : 0);
            }
        }

        // Set observation
        muzero_cpp::types::Observation observation;
        observation.reserve(3 * SIZE * SIZE);
        observation.insert(observation.end(), board_player1.begin(), board_player1.end());
        observation.insert(observation.end(), board_player2.begin(), board_player2.end());
        observation.insert(observation.end(), board_to_play.begin(), board_to_play.end());
        return observation;
    }

    // Check if given board has a winner in view of given player
    bool have_winner(int player, const BoardT &board) {
        // Horizontal/vertical checks
        for (int i = 0; i < SIZE; ++i) {
            if (sum_array(get_row(i, board)) == 3 * player) { return true; }
            if (sum_array(get_col(i, board)) == 3 * player) { return true; }
        }
        if (sum_array(get_diag(board)) == 3 * player) { return true; }
        if (sum_array(get_antidiag(board)) == 3 * player) { return true; }
        return false;
    }

    // Check if game has a winner
    bool have_winner() {
        // Horizontal/vertical checks
        for (int i = 0; i < SIZE; ++i) {
            if (std::abs(sum_array(get_row(i, board_))) == 3) { return true; }
            if (std::abs(sum_array(get_col(i, board_))) == 3) { return true; }
        }
        if (std::abs(sum_array(get_diag(board_))) == 3) { return true; }
        if (std::abs(sum_array(get_antidiag(board_))) == 3) { return true; }
        return false;
    }

    // Sum a sub_array
    int sum_array(const SubArrayT &sub_array) {
        return std::reduce(std::begin(sub_array), std::end(sub_array));
    }

    // Get row from board
    SubArrayT get_row(int row, const BoardT &board) {
        return board[row];
    }

    // Get column from board
    SubArrayT get_col(int col, const BoardT &board) {
        SubArrayT array;
        for (int i = 0; i < SIZE; ++i) {
            array[i] = board[i][col];
        }
        return array;
    }

    // Get board diagonal
    SubArrayT get_diag(const BoardT &board) {
        SubArrayT array;
        for (int i = 0; i < SIZE; ++i) {
            array[i] = board[i][i];
        }
        return array;
    }

    // Get board anti-diagonal
    SubArrayT get_antidiag(const BoardT &board) {
        SubArrayT array;
        for (int i = 0; i < SIZE; ++i) {
            array[i] = board[i][SIZE - i - 1];
        }
        return array;
    }

    muzero_cpp::types::Player player_;
    BoardT board_;
    std::mt19937 rng_;
    const std::unordered_map<int, std::string> print_map_{{1, "1"}, {-1, "2"}, {0, "-"}};
};

#endif    // MUZERO_CPP_EXAMPLE_TICTACTOE_H_