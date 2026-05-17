#ifndef MUZERO_EXAMPLE_CONNECT4_H_
#define MUZERO_EXAMPLE_CONNECT4_H_

#include <muzero/muzero.h>

#include <array>
#include <cassert>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// Simple connect4 environment implementation
// Some of the logic taken from: https://github.com/werner-duvaud/muzero-general/blob/master/games/connect4.py
class Connect4 {
public:
    Connect4(int seed)
        : board_{}
    {
        rng_.seed(static_cast<std::mt19937::result_type>(seed));
        for (std::size_t r = 0; r < NUM_ROWS; ++r) {
            for (std::size_t c = 0; c < NUM_COLS; ++c) {
                board_[r][c] = 0;
            }
        }
    }

    // Get the observations shape
    static auto obs_shape() -> muzero::ObservationShape
    {
        return {.c = 3, .h = static_cast<int>(NUM_ROWS), .w = static_cast<int>(NUM_COLS)};
    }

    // Get the entire action space
    static auto action_space() -> std::vector<muzero::Action>
    {
        std::vector<muzero::Action> actions;
        for (std::size_t c = 0; c < NUM_COLS; ++c) {
            actions.push_back(static_cast<muzero::Action>(c));
        }
        return actions;
    }

    // Get the current player
    auto to_play() const -> muzero::Player
    {
        return (player_ == 1) ? 0 : 1;
    }

    // Reset environment and send starting observation
    auto reset() -> muzero::Observation
    {
        board_ = std::array<std::array<int, 7>, 6>{};
        player_ = 1;
        for (std::size_t r = 0; r < NUM_ROWS; ++r) {
            for (std::size_t c = 0; c < NUM_COLS; ++c) {
                board_[r][c] = 0;
            }
        }
        return get_observation();
    }

    // Step the environment, send next observation, reward, and done flag
    auto step(muzero::Action action) -> muzero::StepReturn
    {
        for (std::size_t r = 0; r < NUM_ROWS; ++r) {
            if (board_[r][static_cast<std::size_t>(action)] == 0) {
                board_[r][static_cast<std::size_t>(action)] = player_;
                break;
            }
        }
        bool is_win = have_winner();
        bool done = (is_win || legal_actions().size() == 0);
        double reward = is_win ? 1 : 0;
        player_ *= -1;
        return {.observation = get_observation(), .reward = reward, .done = done};
    }

    // Current legal actions, subset of action space
    auto legal_actions() const -> std::vector<muzero::Action>
    {
        std::vector<muzero::Action> actions;
        for (std::size_t c = 0; c < NUM_COLS; ++c) {
            if (board_[NUM_ROWS - 1][c] == 0) {
                actions.push_back(static_cast<muzero::Action>(c));
            }
        }
        return actions;
    }

    // Find an expert action
    auto expert_action() -> muzero::Action
    {
        auto actions = legal_actions();
        std::uniform_int_distribution<std::size_t> dist(0, actions.size() - 1);
        auto action = actions[dist(rng_)];

        for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 4; ++l) {
                SubBoardT sub_board = create_sub(k, l);

                // Horizontal and vertical checks
                for (std::size_t i = 0; i < SUBBOARD_N; ++i) {
                    SubArrayT sub_row = get_row(sub_board, i);
                    if (std::abs(sum_array(sub_row)) == 3) {
                        int ind = find_where(sub_row, 0);
                        if (count_nonzero(ind + l) == static_cast<int>(i) + k) {
                            action = ind + l;
                            if (player_ * sum_array(sub_row) > 0) {
                                return action;
                            }
                        }
                    }
                    SubArrayT sub_col = get_col(sub_board, i);
                    if (std::abs(sum_array(sub_col)) == 3) {
                        action = static_cast<int>(i) + l;
                        if (player_ * sum_array(sub_col) > 0) {
                            return action;
                        }
                    }
                }
                // Diagonal checks
                SubArrayT diag = get_diag(sub_board);
                if (std::abs(sum_array(diag)) == 3) {
                    int ind = find_where(diag, 0);
                    if (count_nonzero(ind + l) == ind + k) {
                        action = ind + l;
                        if (player_ * sum_array(diag) > 0) {
                            return action;
                        }
                    }
                }
                SubArrayT antidiag = get_antidiag(sub_board);
                if (std::abs(sum_array(antidiag)) == 3) {
                    int ind = find_where(antidiag, 0);
                    if (count_nonzero(3 - ind + l) == ind + k) {
                        action = 3 - ind + l;
                        if (player_ * sum_array(antidiag) > 0) {
                            return action;
                        }
                    }
                }
            }
        }
        return action;
    }

    // Pretty string of board
    std::string board_to_str() const
    {
        std::string out;
        for (std::size_t r = 0; r < NUM_ROWS; ++r) {
            out += "|";
            for (std::size_t c = 0; c < NUM_COLS; ++c) {
                out += " " + print_map_.at(board_[NUM_ROWS - r - 1][c]) + " ";
            }
            out += "|\n";
        }
        return out;
    }

private:
    constexpr static std::size_t NUM_ROWS = 6;
    constexpr static std::size_t NUM_COLS = 7;
    constexpr static std::size_t SUBBOARD_N = 4;
    using SubBoardT = std::array<std::array<int, SUBBOARD_N>, SUBBOARD_N>;
    using SubArrayT = std::array<int, SUBBOARD_N>;

    // Get the current observation (3 x 6 x 7)
    [[nodiscard]] auto get_observation() const -> muzero::Observation
    {
        muzero::Observation observation(3 * NUM_ROWS * NUM_COLS);
        std::size_t offset = 0;

        // player 1 slice
        for (std::size_t r = 0; r < NUM_ROWS; ++r) {
            for (std::size_t c = 0; c < NUM_COLS; ++c) {
                observation[offset + (r * NUM_COLS + c)] = board_[r][c] == 1 ? 1 : 0;
            }
        }

        // player 2 slice
        offset += NUM_ROWS * NUM_COLS;
        for (std::size_t r = 0; r < NUM_ROWS; ++r) {
            for (std::size_t c = 0; c < NUM_COLS; ++c) {
                observation[offset + (r * NUM_COLS + c)] = board_[r][c] == -1 ? 1 : 0;
            }
        }

        // to play slice
        offset += NUM_ROWS * NUM_COLS;
        for (std::size_t i = 0; i < NUM_ROWS * NUM_COLS; ++i) {
            observation[offset + i] = static_cast<float>(player_);
        }

        return observation;
    }

    // Check if there is a winner
    auto have_winner() -> bool
    {
        // Horizontal check
        for (std::size_t c = 0; c < 4; ++c) {
            for (std::size_t r = 0; r < NUM_ROWS; ++r) {
                if (board_[r][c] == player_ && board_[r][c + 1] == player_ && board_[r][c + 2] == player_
                    && board_[r][c + 3] == player_)
                {
                    return true;
                }
            }
        }

        // Vertical check
        for (std::size_t c = 0; c < NUM_COLS; ++c) {
            for (std::size_t r = 0; r < 3; ++r) {
                if (board_[r][c] == player_ && board_[r + 1][c] == player_ && board_[r + 2][c] == player_
                    && board_[r + 3][c] == player_)
                {
                    return true;
                }
            }
        }

        // Positive diagonal check
        for (std::size_t c = 0; c < 4; ++c) {
            for (std::size_t r = 0; r < 3; ++r) {
                if (board_[r][c] == player_ && board_[r + 1][c + 1] == player_ && board_[r + 2][c + 2] == player_
                    && board_[r + 3][c + 3] == player_)
                {
                    return true;
                }
            }
        }

        // Negative diagonal check
        for (std::size_t c = 0; c < 4; ++c) {
            for (std::size_t r = 3; r < NUM_ROWS; ++r) {
                if (board_[r][c] == player_ && board_[r - 1][c + 1] == player_ && board_[r - 2][c + 2] == player_
                    && board_[r - 3][c + 3] == player_)
                {
                    return true;
                }
            }
        }
        return false;
    }

    // Create subboard, used for conditional checking
    auto create_sub(int start_r, int start_c) -> SubBoardT
    {
        SubBoardT sub_board;
        for (std::size_t r = 0; r < SUBBOARD_N; ++r) {
            for (std::size_t c = 0; c < SUBBOARD_N; ++c) {
                sub_board[r][c] = board_[static_cast<std::size_t>(start_r) + r][static_cast<std::size_t>(start_c) + c];
            }
        }
        return sub_board;
    }

    // Numpy like version of find where
    auto find_where(const SubArrayT &sub_array, int val) -> int
    {
        for (std::size_t i = 0; i < SUBBOARD_N; ++i) {
            if (sub_array[i] == val) {
                return static_cast<int>(i);
            }
        }
        assert(0);
        return -1;
    }

    // Count nonzero items in a column
    auto count_nonzero(int col) -> int
    {
        int count = 0;
        for (std::size_t i = 0; i < NUM_ROWS; ++i) {
            if (board_[i][static_cast<std::size_t>(col)] != 0) {
                ++count;
            }
        }
        return count;
    }

    // Sum a sub_array
    auto sum_array(const SubArrayT &sub_array) -> int
    {
        return std::reduce(std::begin(sub_array), std::end(sub_array));
    }

    // Get board row
    auto get_row(const SubBoardT &sub_board, std::size_t row) -> SubArrayT
    {
        return sub_board[row];
    }

    // Get board column
    auto get_col(const SubBoardT &sub_board, std::size_t col) -> SubArrayT
    {
        SubArrayT array;
        for (std::size_t i = 0; i < SUBBOARD_N; ++i) {
            array[i] = sub_board[i][col];
        }
        return array;
    }

    // Get board diagonal
    auto get_diag(const SubBoardT &sub_board) -> SubArrayT
    {
        SubArrayT array;
        for (std::size_t i = 0; i < SUBBOARD_N; ++i) {
            array[i] = sub_board[i][i];
        }
        return array;
    }

    // Get board anti-diagonal
    auto get_antidiag(const SubBoardT &sub_board) -> SubArrayT
    {
        SubArrayT array;
        for (std::size_t i = 0; i < SUBBOARD_N; ++i) {
            array[i] = sub_board[i][SUBBOARD_N - i - 1];
        }
        return array;
    }

    muzero::Player player_ = 1;
    std::array<std::array<int, NUM_COLS>, NUM_ROWS> board_;
    std::mt19937 rng_;
    const std::unordered_map<int, std::string> print_map_{{1, "1"}, {-1, "2"}, {0, "-"}};
};

#endif    // MUZERO_EXAMPLE_CONNECT4_H_
