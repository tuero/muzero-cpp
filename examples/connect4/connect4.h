#ifndef MUZERO_CPP_EXAMPLE_CONNECT4_H_
#define MUZERO_CPP_EXAMPLE_CONNECT4_H_

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
class Connect4 {
public:
    Connect4() : player_(1), board_{} {
        for (int r = 0; r < NUM_ROWS; ++r) {
            for (int c = 0; c < NUM_COLS; ++c) {
                board_[r][c] = 0;
            }
        }
    }

    // Get the observations shape
    static muzero_cpp::types::ObservationShape obs_shape() {
        return {3, NUM_ROWS, NUM_COLS};
    }

    // Get the entire action space
    static std::vector<muzero_cpp::types::Action> action_space() {
        std::vector<muzero_cpp::types::Action> actions;
        for (int c = 0; c < NUM_COLS; ++c) {
            actions.push_back(c);
        }
        return actions;
    }

    // Get the current player
    muzero_cpp::types::Player to_play() const {
        return (player_ == 1) ? 0 : 1;
    }

    // Reset environment and send starting observation
    muzero_cpp::types::Observation reset() {
        board_ = std::array<std::array<int, 7>, 6>{};
        player_ = 1;
        for (int r = 0; r < NUM_ROWS; ++r) {
            for (int c = 0; c < NUM_COLS; ++c) {
                board_[r][c] = 0;
            }
        }
        return get_observation();
    }

    // Step the environment, send next observation, reward, and done flag
    muzero_cpp::types::StepReturn step(muzero_cpp::types::Action action) {
        assert(std::find(legal_actions().begin(), legal_actions().end(), action) != legal_actions().end());
        for (int r = 0; r < NUM_ROWS; ++r) {
            if (board_[r][action] == 0) {
                board_[r][action] = player_;
                break;
            }
        }
        bool is_win = have_winner();
        bool done = (is_win || legal_actions().size() == 0);
        double reward = is_win ? 1 : 0;
        player_ *= -1;
        return {get_observation(), reward, done};
    }

    // Current legal actions, subset of action space
    std::vector<muzero_cpp::types::Action> legal_actions() const {
        std::vector<muzero_cpp::types::Action> actions;
        for (int c = 0; c < NUM_COLS; ++c) {
            if (board_[NUM_ROWS - 1][c] == 0) { actions.push_back(c); }
        }
        return actions;
    }

    // Find an expert action
    muzero_cpp::types::Action expert_action() {
        auto actions = legal_actions();
        std::uniform_int_distribution<int> dist(0, actions.size() - 1);
        auto action = actions[dist(rng_)];

        for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 4; ++l) {
                SubBoardT sub_board = create_sub(k, l);

                // Horizontal and vertical checks
                for (int i = 0; i < SUBBOARD_N; ++i) {
                    SubArrayT sub_row = get_row(sub_board, i);
                    if (std::abs(sum_array(sub_row)) == 3) {
                        int ind = find_where(sub_row, 0);
                        if (count_nonzero(ind + l) == i + k) {
                            action = ind + l;
                            if (player_ * sum_array(sub_row) > 0) { return action; }
                        }
                    }
                    SubArrayT sub_col = get_col(sub_board, i);
                    if (std::abs(sum_array(sub_col)) == 3) {
                        action = i + l;
                        if (player_ * sum_array(sub_col) > 0) { return action; }
                    }
                }
                // Diagonal checks
                SubArrayT diag = get_diag(sub_board);
                if (std::abs(sum_array(diag)) == 3) {
                    int ind = find_where(diag, 0);
                    if (count_nonzero(ind + l) == ind + k) {
                        action = ind + l;
                        if (player_ * sum_array(diag) > 0) { return action; }
                    }
                }
                SubArrayT antidiag = get_antidiag(sub_board);
                if (std::abs(sum_array(antidiag)) == 3) {
                    int ind = find_where(antidiag, 0);
                    if (count_nonzero(3 - ind + l) == ind + k) {
                        action = 3 - ind + l;
                        if (player_ * sum_array(antidiag) > 0) { return action; }
                    }
                }
            }
        }
        return action;
    }

    // Pretty string of board
    std::string board_to_str() const {
        std::string out;
        for (int r = 0; r < NUM_ROWS; ++r) {
            out += "|";
            for (int c = 0; c < NUM_COLS; ++c) {
                out += " " + print_map_.at(board_[NUM_ROWS - r - 1][c]) + " ";
            }
            out += "|\n";
        }
        return out;
    }

private:
    constexpr static int NUM_ROWS = 6;
    constexpr static int NUM_COLS = 7;
    constexpr static int SUBBOARD_N = 4;
    using SubBoardT = std::array<std::array<int, SUBBOARD_N>, SUBBOARD_N>;
    using SubArrayT = std::array<int, SUBBOARD_N>;

    // Get the current observation (3 x 6 x 7)
    muzero_cpp::types::Observation get_observation() {
        muzero_cpp::types::Observation board_player1;
        board_player1.reserve(NUM_ROWS * NUM_COLS);
        muzero_cpp::types::Observation board_player2;
        board_player2.reserve(NUM_ROWS * NUM_COLS);
        muzero_cpp::types::Observation board_to_play(NUM_ROWS * NUM_COLS, player_);

        for (int r = 0; r < NUM_ROWS; ++r) {
            for (int c = 0; c < NUM_COLS; ++c) {
                board_player1.push_back(board_[r][c] == 1 ? 1 : 0);
                board_player2.push_back(board_[r][c] == -1 ? 1 : 0);
            }
        }

        muzero_cpp::types::Observation observation;
        observation.reserve(3 * NUM_ROWS * NUM_COLS);
        observation.insert(observation.end(), board_player1.begin(), board_player1.end());
        observation.insert(observation.end(), board_player2.begin(), board_player2.end());
        observation.insert(observation.end(), board_to_play.begin(), board_to_play.end());
        return observation;
    }

    // Check if there is a winner
    bool have_winner() {
        // Horizontal check
        for (int c = 0; c < 4; ++c) {
            for (int r = 0; r < NUM_ROWS; ++r) {
                if (board_[r][c] == player_ && board_[r][c + 1] == player_ && board_[r][c + 2] == player_ &&
                    board_[r][c + 3] == player_) {
                    return true;
                }
            }
        }

        // Vertical check
        for (int c = 0; c < NUM_COLS; ++c) {
            for (int r = 0; r < 3; ++r) {
                if (board_[r][c] == player_ && board_[r + 1][c] == player_ && board_[r + 2][c] == player_ &&
                    board_[r + 3][c] == player_) {
                    return true;
                }
            }
        }

        // Positive diagonal check
        for (int c = 0; c < 4; ++c) {
            for (int r = 0; r < 3; ++r) {
                if (board_[r][c] == player_ && board_[r + 1][c + 1] == player_ &&
                    board_[r + 2][c + 2] == player_ && board_[r + 3][c + 3] == player_) {
                    return true;
                }
            }
        }

        // Negative diagonal check
        for (int c = 0; c < 4; ++c) {
            for (int r = 3; r < NUM_ROWS; ++r) {
                if (board_[r][c] == player_ && board_[r - 1][c + 1] == player_ &&
                    board_[r - 2][c + 2] == player_ && board_[r - 3][c + 3] == player_) {
                    return true;
                }
            }
        }
        return false;
    }

    // Create subboard, used for conditional checking
    SubBoardT create_sub(int start_r, int start_c) {
        SubBoardT sub_board;
        for (int r = 0; r < SUBBOARD_N; ++r) {
            for (int c = 0; c < SUBBOARD_N; ++c) {
                sub_board[r][c] = board_[start_r + r][start_c + c];
            }
        }
        return sub_board;
    }

    // Numpy like version of find where
    int find_where(const SubArrayT &sub_array, int val) {
        for (int i = 0; i < SUBBOARD_N; ++i) {
            if (sub_array[i] == val) { return i; }
        }
        assert(0);
        return -1;
    }

    // Count nonzero items in a column
    int count_nonzero(int col) {
        int count = 0;
        for (int i = 0; i < NUM_ROWS; ++i) {
            if (board_[i][col] != 0) { ++count; }
        }
        return count;
    }

    // Sum a sub_array
    int sum_array(const SubArrayT &sub_array) {
        return std::reduce(std::begin(sub_array), std::end(sub_array));
    }

    // Get board row
    SubArrayT get_row(const SubBoardT &sub_board, int row) {
        return sub_board[row];
    }

    // Get board column
    SubArrayT get_col(const SubBoardT &sub_board, int col) {
        SubArrayT array;
        for (int i = 0; i < SUBBOARD_N; ++i) {
            array[i] = sub_board[i][col];
        }
        return array;
    }

    // Get board diagonal
    SubArrayT get_diag(const SubBoardT &sub_board) {
        SubArrayT array;
        for (int i = 0; i < SUBBOARD_N; ++i) {
            array[i] = sub_board[i][i];
        }
        return array;
    }

    // Get board anti-diagonal
    SubArrayT get_antidiag(const SubBoardT &sub_board) {
        SubArrayT array;
        for (int i = 0; i < SUBBOARD_N; ++i) {
            array[i] = sub_board[i][SUBBOARD_N - i - 1];
        }
        return array;
    }

    muzero_cpp::types::Player player_;
    std::array<std::array<int, NUM_COLS>, NUM_ROWS> board_;
    std::mt19937 rng_;
    const std::unordered_map<int, std::string> print_map_{{1, "1"}, {-1, "2"}, {0, "-"}};
};

#endif    // MUZERO_CPP_EXAMPLE_CONNECT4_H_