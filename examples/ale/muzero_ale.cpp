#include <muzero/muzero.h>

#include <ale/ale_interface.hpp>
#include <opencv2/opencv.hpp>
#include <SDL2/SDL.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace muzero;

namespace {

class ALEEnv : public AbstractGame {
public:
    ALEEnv(int seed)
    {
        was_real_done_ = true;
        lives_ = 0;
        rng_.seed(static_cast<std::mt19937::result_type>(seed));
        env_.setInt("random_seed", seed);
        // Assumes the static variables are set before factory constructs
        env_.loadROM(game_file_path);
        env_.reset_game();
        if (should_render) {
            init_render();
        }
        // If max frame, set both buffers to blank
        if (max_frame) {
            obs_buffer1_.insert(obs_buffer1_.end(), HEIGHT * WIDTH, 0);
            obs_buffer2_.insert(obs_buffer2_.end(), HEIGHT * WIDTH, 0);
        }
    }
    ALEEnv() = delete;
    ~ALEEnv()
    {
        if (should_render) {
            cleanup_render();
        }
    }

    /**
     * Reset the environment for a new game.
     */
    auto reset() -> Observation override
    {
        // If episodic, we perform a dummy reset if lives still remain,
        // so that we explore the entire state space
        if (episodic && !was_real_done_) {
            env_.act((ale::Action)0);
        } else if (episodic_pong && !was_real_done_) {
            env_.act((ale::Action)0);
        } else {
            env_.reset_game();
        }
        // Apply random noops
        std::uniform_int_distribution<> dist(1, noop_max);
        int num_noops = dist(this->rng_);
        for (int i = 0; i < num_noops; ++i) {
            env_.act((ale::Action)0);
            if (env_.game_over()) {
                env_.reset_game();
            }
        }
        // Check if we need to fire
        if (fire_reset) {
            env_.act(static_cast<ale::Action>(1));
        }
        lives_ = env_.lives();
        return get_obs();
    }

    /**
     * Apply the given action to the environment
     * @param action The action to send to the environment
     * @return A struct containing the observation, reward, and a flag indicating if the game is done
     */
    auto step(Action action) -> StepReturn override
    {
        assert(action >= 0);
        assert(static_cast<std::size_t>(action) < minimal_actions.size());
        StepReturn step_return;
        float reward = 0;
        int lives = env_.lives();
        // Step N times collecting all rewards, and return Nth frame
        for (int i = 0; i < frame_skip; ++i) {
            const auto ale_action = static_cast<ale::Action>(minimal_actions[static_cast<std::size_t>(action)]);
            reward += static_cast<float>(env_.act(ale_action));
            step_return.done = env_.game_over();
            was_real_done_ = step_return.done;
            lives = env_.lives();
            // Store obs in buffer if using max frame
            if (max_frame && i == frame_skip - 2) {
                obs_buffer2_ = get_obs();
            }
            if (max_frame && i == frame_skip - 1) {
                obs_buffer1_ = get_obs();
            }

            // Check if we should set done because episodic life
            if (episodic && lives < lives_ && lives > 0) {
                step_return.done = true;
            }
            if (episodic_pong && reward < 0) {
                step_return.done = true;
            }
            // Break frame_skip early if done
            if (step_return.done) {
                break;
            }
        }
        // Max over the 2 frames in the buffer if using max frame
        if (max_frame) {
            step_return.observation.clear();
            std::transform(
                obs_buffer2_.begin(),
                obs_buffer2_.end(),
                obs_buffer1_.begin(),
                std::back_inserter(step_return.observation),
                [](float a, float b) { return std::max(a, b); }
            );
        } else {
            step_return.observation = get_obs();
        }
        step_return.reward = reward;
        return step_return;
    }

    /**
     * Returns the current player to play
     * @return The player number to play
     */
    auto to_play() const -> Player override
    {
        return 0;
    }

    /**
     * Return the legal actions for the current environment state.
     * @returns Vector of legal action ids
     */
    auto legal_actions() const -> std::vector<Action> override
    {
        return ALEEnv::action_space();
    }

    /**
     * Get the entire action space
     * @returns Vector of legal action ids
     */
    static auto action_space() -> std::vector<Action>
    {
        std::vector<Action> actions;
        assert(minimal_actions.size() > 0);
        for (std::size_t a = 0; a < minimal_actions.size(); ++a) {
            actions.push_back(static_cast<Action>(a));
        }
        return actions;
    }

    auto get_minimal_actions() -> std::vector<Action>
    {
        std::vector<Action> actions;
        for (auto const &v : env_.getMinimalActionSet()) {
            actions.push_back(static_cast<Action>(v));
        }
        return actions;
    }

    // Get the observations shape
    static auto obs_shape() -> ObservationShape
    {
        return {.c = 1, .h = HEIGHT, .w = WIDTH};
    }

    /**
     * Returns an action given by an expert player/bot.
     * @returns An expert action which is legal
     */
    auto expert_action() -> Action override
    {
        auto actions = ALEEnv::action_space();
        std::uniform_int_distribution<std::size_t> dist(0, actions.size() - 1);
        return actions[dist(this->rng_)];
    }

    /**
     * Returns a legal action given by human input.
     * @returns An action which is legal
     */
    auto human_to_action() -> Action override
    {
        auto actions = ALEEnv::action_space();
        std::uniform_int_distribution<std::size_t> dist(0, actions.size() - 1);
        return actions[dist(this->rng_)];
    }

    /**
     * Render the environment for testing games.
     * This assumes an SDL context has already been created
     */
    void render() override
    {
        if (win == nullptr || screen_surface == nullptr) {
            return;
        }
        // Get screen values
        auto screen = env_.getScreen();
        std::vector<unsigned char> rgb_output_buffer;
        env_.getScreenRGB(rgb_output_buffer);
        // Resize
        cv::Mat image(
            static_cast<int>(screen.height()),
            static_cast<int>(screen.width()),
            CV_8UC3,
            rgb_output_buffer.data()
        );
        cv::Mat resized;
        cv::resize(
            image,
            resized,
            cv::Size(static_cast<int>(screen.width()) * RENDER_SCALE, static_cast<int>(screen.height()) * RENDER_SCALE),
            cv::INTER_LINEAR
        );

        // Convert to SDL_Surface
        image_surface = SDL_CreateRGBSurfaceWithFormatFrom(
            static_cast<void *>(resized.data),
            resized.cols,
            resized.rows,
            static_cast<int>(resized.elemSize() * 8),
            static_cast<int>(resized.step),
            SDL_PIXELFORMAT_RGB24
        );
        if (image_surface == nullptr) {
            std::cerr << "Error creating SDL surface: " << SDL_GetError() << std::endl;
            return;
        }
        // Draw
        SDL_BlitSurface(image_surface, nullptr, screen_surface, nullptr);
        SDL_UpdateWindowSurface(win);
        SDL_FreeSurface(image_surface);
    }

    /**
     * Convert action to human readable string.
     * @param action The action to convert
     * @returns The string format of the action
     */
    auto action_to_string(Action action) const -> std::string override
    {
        return std::to_string(action);
    }

    inline static std::string game_file_path = "";        // Path of game ROM
    inline static int noop_max = 30;                      // Number of frames to NOOP at start
    inline static int frame_skip = 4;                     // Number of consecutive frames to skip
    inline static bool fire_reset = false;                // Flag to send fire action on reset (needed for some games)
    inline static bool episodic = false;                  // Flag to tread end-of-life as end of episode
    inline static bool episodic_pong = false;             // Same as above, but specific to pong
    inline static bool should_render = false;             // Flag if we should render (used for testing)
    inline static bool max_frame = false;                 // Flag if we should max pool 2 consecutive frames
    inline static std::vector<Action> minimal_actions;    //  (determined by ALE)

private:
    // Get screen from environment and convert to observation
    auto get_obs() -> Observation
    {
        return get_obs_bw();
    }

    auto get_obs_bw() -> Observation
    {
        // Get screen data from ale env
        auto screen = env_.getScreen();
        std::vector<unsigned char> grayscale_output_buffer;
        env_.getScreenGrayscale(grayscale_output_buffer);

        // Resize
        cv::Mat image(
            static_cast<int>(screen.height()),
            static_cast<int>(screen.width()),
            CV_8UC1,
            grayscale_output_buffer.data()
        );
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(WIDTH, HEIGHT), cv::INTER_LINEAR);

        // Convert to normalized obs
        Observation obs;
        const auto rows = static_cast<std::size_t>(resized.rows);
        const auto cols = static_cast<std::size_t>(resized.cols);
        obs.reserve(rows * cols);
        for (int r = 0; r < resized.rows; ++r) {
            for (int c = 0; c < resized.cols; ++c) {
                obs.push_back(static_cast<float>(resized.at<uchar>(r, c)) / 255);
            }
        }
        return obs;
    }

    auto get_obs_rgb() -> Observation
    {
        // Get screen data from ale env
        auto screen = env_.getScreen();
        std::vector<unsigned char> rgb_output_buffer;
        env_.getScreenRGB(rgb_output_buffer);

        // Resize
        cv::Mat image(
            static_cast<int>(screen.height()),
            static_cast<int>(screen.width()),
            CV_8UC3,
            rgb_output_buffer.data()
        );
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(WIDTH, HEIGHT), cv::INTER_LINEAR);

        // Convert to normalized obs
        const auto rows = static_cast<std::size_t>(resized.rows);
        const auto cols = static_cast<std::size_t>(resized.cols);
        Observation obs(3 * rows * cols, 0.0F);
        for (int r = 0; r < resized.rows; ++r) {
            for (int c = 0; c < resized.cols; ++c) {
                auto rgb = resized.at<cv::Vec3b>(r, c);
                for (std::size_t i = 0; i < 3; ++i) {
                    const auto idx =
                        (i * rows * cols) + (static_cast<std::size_t>(r) * cols + static_cast<std::size_t>(c));
                    obs[idx] = static_cast<float>(rgb[static_cast<int>(i)]) / 255;
                }
            }
        }
        return obs;
    }

    // Initialize SDL assets for rendering (during testing)
    void init_render()
    {
        auto screen = env_.getScreen();
        win = SDL_CreateWindow(
            "MuZero ALE",
            SDL_WINDOWPOS_UNDEFINED,
            SDL_WINDOWPOS_UNDEFINED,
            static_cast<int>(screen.width()) * RENDER_SCALE,
            static_cast<int>(screen.height()) * RENDER_SCALE,
            SDL_WINDOW_SHOWN
        );
        if (win == nullptr) {
            std::cerr << "Error creating SDL window: " << SDL_GetError() << std::endl;
            return;
        }
        screen_surface = SDL_GetWindowSurface(win);
        if (screen_surface == nullptr) {
            std::cerr << "Error getting SDL window surface: " << SDL_GetError() << std::endl;
        }
    }

    // Cleanup initialized SDL assets
    void cleanup_render()
    {
        image_surface = nullptr;
        SDL_DestroyWindow(win);
        win = nullptr;
    }

    mutable ale::ALEInterface env_;           // Environmnet
    std::mt19937 rng_;                        // Source of RNG
    int lives_;                               // Number of lives outstanding
    bool was_real_done_;                      // Flag if environment actually ended
    Observation obs_buffer1_;                 // Buffer for the even step before (if using max frame)
    Observation obs_buffer2_;                 // Buffer for the odd step before (if using max frame)
    SDL_Window *win = nullptr;                // Window to draw (if rendering)
    SDL_Surface *screen_surface = nullptr;    // Surface for the window to draw (if rendering)
    SDL_Surface *image_surface = nullptr;     // Surface for the state obs image to exist (if rendering)
    static const int WIDTH = 96;              // State observation width
    static const int HEIGHT = 96;             // State observation height
    static const int RENDER_SCALE = 4;        // Scaling of state observation to render window
};

// Encode action as feature plane of values 1/action
auto encode_action_initial(Action action) -> Observation
{
    static const auto num_actions = static_cast<float>(ALEEnv::action_space().size());
    const auto obs_shape = ALEEnv::obs_shape();
    const auto N = static_cast<std::size_t>(obs_shape.w * obs_shape.h);
    const auto a = static_cast<float>(action) / num_actions;
    Observation obs(N, a);
    return obs;
}
auto encode_action_recurrent(Action action) -> Observation
{
    static const auto num_actions = static_cast<float>(ALEEnv::action_space().size());
    const auto obs_shape = encoded_obs_shape(ALEEnv::obs_shape(), true);
    const auto N = static_cast<std::size_t>(obs_shape.w * obs_shape.h);
    const auto a = static_cast<float>(action) / static_cast<float>(num_actions);
    Observation obs(N, a);
    return obs;
}

// Simple softmax schedule
auto get_softmax(int step) -> double
{
    if (step < 100000) {
        return 1.0;
    } else if (step < 200000) {
        return 0.5;
    }
    return 0.25;
}

}    // namespace

// Additional flag to choose whether to test or not
ABSL_FLAG(bool, test, false, "Test using human input.");
ABSL_FLAG(std::string, game_file_path, "", "Full path that the game file binary resides.");
ABSL_FLAG(int, noop_max, 30, "Maximum number of NOOPs to apply at reset.");
ABSL_FLAG(int, frame_skip, 4, "Number of frames to skip.");
ABSL_FLAG(bool, fire_reset, false, "Apply a fire command on reset.");
ABSL_FLAG(bool, episodic, false, "Reset environment on life loss.");
ABSL_FLAG(bool, episodic_pong, false, "Reset environment on life loss for pong.");
ABSL_FLAG(
    bool,
    max_frame,
    false,
    "States are the max between 2 consecutive frames (used for game with screen flickering)."
);

int main(int argc, char **argv)
{
    // parse flags
    parse_flags(argc, argv);
    Config config = initial_config();

    // Set ALE game properties
    ALEEnv::noop_max = absl::GetFlag(FLAGS_noop_max);
    ALEEnv::frame_skip = absl::GetFlag(FLAGS_frame_skip);
    ALEEnv::fire_reset = absl::GetFlag(FLAGS_fire_reset);
    ALEEnv::game_file_path = absl::GetFlag(FLAGS_game_file_path);
    ALEEnv::episodic = absl::GetFlag(FLAGS_episodic);
    ALEEnv::episodic_pong = absl::GetFlag(FLAGS_episodic_pong);
    ALEEnv::should_render = absl::GetFlag(FLAGS_test);
    ALEEnv::max_frame = absl::GetFlag(FLAGS_max_frame);

    // Need to load game first and then set the minimal action space
    {
        ALEEnv temp_env(0);
        ALEEnv::minimal_actions = temp_env.get_minimal_actions();
    }

    // Set specific values for the game
    config.observation_shape = ALEEnv::obs_shape();
    config.action_space = ALEEnv::action_space();
    config.action_channels = 1;
    config.num_players = 1;
    config.opponent_type = OpponentType::Self;
    config.action_representation_initial = encode_action_initial;
    config.action_representation_recurrent = encode_action_recurrent;
    config.visit_softmax_temperature = get_softmax;

    // Perform learning or testing
    if (absl::GetFlag(FLAGS_test)) {
        if (SDL_Init(SDL_INIT_VIDEO)) {
            std::cout << "Error initializing SDL" << std::endl;
            return 1;
        }
        int ret = play_test_model(config, game_factory<ALEEnv>);
        SDL_Quit();
        return ret;
    } else {
        return muzero::muzero(config, game_factory<ALEEnv>);
    }

    return 0;
}
