#include <SDL2/SDL.h>
#include <opencv2/core/types_c.h>

#include <ale_interface.hpp>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "muzero-cpp/abstract_game.h"
#include "muzero-cpp/config.h"
#include "muzero-cpp/default_flags.h"
#include "muzero-cpp/models.h"
#include "muzero-cpp/muzero.h"
#include "muzero-cpp/types.h"

using namespace muzero_cpp;
using namespace muzero_cpp::types;
using namespace muzero_cpp::muzero_config;

class ALEEnv : public AbstractGame {
public:
    ALEEnv(int seed) {
        was_real_done_ = true;
        lives_ = 0;
        rng_.seed(seed);
        env_.setInt("random_seed", seed);
        // Assumes the static variables are set before factory constructs
        env_.loadROM(game_file_path);
        env_.reset_game();
        if (should_render) { init_render(); }
        // If max frame, set both buffers to blank
        if (max_frame) {
            obs_buffer1_.insert(obs_buffer1_.end(), HEIGHT * WIDTH, 0);
            obs_buffer2_.insert(obs_buffer2_.end(), HEIGHT * WIDTH, 0);
        }
    }
    ALEEnv() = delete;
    ~ALEEnv() {
        if (should_render) { cleanup_render(); }
    }

    /**
     * Reset the environment for a new game.
     */
    Observation reset() override {
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
            if (env_.game_over()) { env_.reset_game(); }
        }
        // Check if we need to fire
        if (fire_reset) { env_.act((ale::Action)1); }
        lives_ = env_.lives();
        return get_obs();
    }

    /**
     * Apply the given action to the environment
     * @param action The action to send to the environment
     * @return A struct containing the observation, reward, and a flag indicating if the game is done
     */
    StepReturn step(Action action) override {
        assert(action < (int)minimal_actions.size());
        StepReturn step_return;
        float reward = 0;
        int lives = env_.lives();
        // Step N times collecting all rewards, and return Nth frame
        for (int i = 0; i < frame_skip; ++i) {
            reward += env_.act((ale::Action)minimal_actions[action]);
            step_return.done = env_.game_over();
            was_real_done_ = step_return.done;
            lives = env_.lives();
            // Store obs in buffer if using max frame
            if (max_frame && i == frame_skip - 2) { obs_buffer2_ = get_obs(); }
            if (max_frame && i == frame_skip - 1) { obs_buffer1_ = get_obs(); }

            // Check if we should set done because episodic life
            if (episodic && lives < lives_ && lives > 0) { step_return.done = true; }
            if (episodic_pong && reward < 0) { step_return.done = true; }
            // Break frame_skip early if done
            if (step_return.done) { break; }
        }
        // Max over the 2 frames in the buffer if using max frame
        if (max_frame) {
            step_return.observation.clear();
            std::transform(obs_buffer2_.begin(), obs_buffer2_.end(), obs_buffer1_.begin(),
                           std::back_inserter(step_return.observation),
                           [](float a, float b) { return std::max(a, b); });
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
    Player to_play() const override {
        return 0;
    }

    /**
     * Return the legal actions for the current environment state.
     * @returns Vector of legal action ids
     */
    std::vector<Action> legal_actions() const override {
        return ALEEnv::action_space();
    }

    /**
     * Get the entire action space
     * @returns Vector of legal action ids
     */
    static std::vector<Action> action_space() {
        std::vector<Action> actions;
        assert(minimal_actions.size() > 0);
        for (int a = 0; a < (int)minimal_actions.size(); ++a) {
            actions.push_back(a);
        }
        return actions;
    }

    std::vector<Action> get_minimal_actions() {
        std::vector<Action> minimal_actions;
        for (auto const& v : env_.getMinimalActionSet()) {
            minimal_actions.push_back(static_cast<Action>(v));
        }
        return minimal_actions;
    }

    // Get the observations shape
    static ObservationShape obs_shape() {
        return {1, HEIGHT, WIDTH};
    }

    /**
     * Returns an action given by an expert player/bot.
     * @returns An expert action which is legal
     */
    Action expert_action() override {
        auto actions = ALEEnv::action_space();
        std::uniform_int_distribution<> dist(0, actions.size() - 1);
        return actions[dist(this->rng_)];
    }

    /**
     * Returns a legal action given by human input.
     * @returns An action which is legal
     */
    Action human_to_action() override {
        auto actions = ALEEnv::action_space();
        std::uniform_int_distribution<> dist(0, actions.size() - 1);
        return actions[dist(this->rng_)];
    }

    /**
     * Render the environment for testing games.
     * This assumes an SDL context has already been created
     */
    void render() override {
        // Get screen values
        auto screen = env_.getScreen();
        std::vector<unsigned char> rgb_output_buffer;
        env_.getScreenRGB(rgb_output_buffer);
        // Resize
        cv::Mat image(screen.height(), screen.width(), CV_8UC3, rgb_output_buffer.data());
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(screen.width() * RENDER_SCALE, screen.height() * RENDER_SCALE),
                   cv::INTER_LINEAR);

        // Convert to SDL_Surface
        IplImage opencvimg_temp = cvIplImage(resized);
        IplImage* opencvimg = &opencvimg_temp;
        image_surface = SDL_CreateRGBSurfaceFrom((void*)opencvimg->imageData, opencvimg->width,
                                                 opencvimg->height, opencvimg->depth * opencvimg->nChannels,
                                                 opencvimg->widthStep, 0xff0000, 0x00ff00, 0x0000ff, 0);
        // Draw
        SDL_BlitSurface(image_surface, NULL, screen_surface, NULL);
        SDL_UpdateWindowSurface(win);
        SDL_FreeSurface(image_surface);
    }

    /**
     * Convert action to human readable string.
     * @param action The action to convert
     * @returns The string format of the action
     */
    std::string action_to_string(types::Action action) const override {
        return std::to_string(action);
    }

    inline static std::string game_file_path = "";    // Path of game ROM
    inline static int noop_max = 30;                  // Number of frames to NOOP at start
    inline static int frame_skip = 4;                 // Number of consecutive frames to skip
    inline static bool fire_reset = false;       // Flag to send fire action on reset (needed for some games)
    inline static bool episodic = false;         // Flag to tread end-of-life as end of episode
    inline static bool episodic_pong = false;    // Same as above, but specific to pong
    inline static bool should_render = false;    // Flag if we should render (used for testing)
    inline static bool max_frame = false;        // Flag if we should max pool 2 consecutive frames
    inline static std::vector<Action> minimal_actions;    //  (determined by ALE)

private:
    // Get screen from environment and convert to observation
    Observation get_obs() {
        return get_obs_bw();
    }

    Observation get_obs_bw() {
        // Get screen data from ale env
        auto screen = env_.getScreen();
        std::vector<unsigned char> grayscale_output_buffer;
        env_.getScreenGrayscale(grayscale_output_buffer);

        // Resize
        cv::Mat image(screen.height(), screen.width(), CV_8UC1, grayscale_output_buffer.data());
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(WIDTH, HEIGHT), cv::INTER_LINEAR);

        // Convert to normalized obs
        Observation obs;
        obs.reserve(resized.rows * resized.cols);
        for (int r = 0; r < resized.rows; ++r) {
            for (int c = 0; c < resized.cols; ++c) {
                obs.push_back((double)resized.at<uchar>(r, c) / 255);
            }
        }
        return obs;
    }

    Observation get_obs_rgb() {
        // Get screen data from ale env
        auto screen = env_.getScreen();
        std::vector<unsigned char> rgb_output_buffer;
        env_.getScreenRGB(rgb_output_buffer);

        // Resize
        cv::Mat image(screen.height(), screen.width(), CV_8UC3, rgb_output_buffer.data());
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(WIDTH, HEIGHT), cv::INTER_LINEAR);

        // Convert to normalized obs
        Observation obs(3 * resized.rows * resized.cols, 0);
        for (int r = 0; r < resized.rows; ++r) {
            for (int c = 0; c < resized.cols; ++c) {
                auto rgb = resized.at<cv::Vec3b>(r, c);
                for (int i = 0; i < 3; ++i) {
                    obs[(i * resized.rows * resized.cols) + (r * resized.cols + c)] = (double)rgb[i] / 255;
                }
            }
        }
        return obs;
    }

    // Initialize SDL assets for rendering (during testing)
    void init_render() {
        auto screen = env_.getScreen();
        win =
            SDL_CreateWindow("MuZero ALE", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                             screen.width() * RENDER_SCALE, screen.height() * RENDER_SCALE, SDL_WINDOW_SHOWN);
        screen_surface = SDL_GetWindowSurface(win);
    }

    // Cleanup initialized SDL assets
    void cleanup_render() {
        image_surface = NULL;
        SDL_DestroyWindow(win);
        win = NULL;
    }

    mutable ale::ALEInterface env_;        // Environmnet
    std::mt19937 rng_;                     // Source of RNG
    int lives_;                            // Number of lives outstanding
    bool was_real_done_;                   // Flag if environment actually ended
    Observation obs_buffer1_;              // Buffer for the even step before (if using max frame)
    Observation obs_buffer2_;              // Buffer for the odd step before (if using max frame)
    SDL_Window* win = NULL;                // Window to draw (if rendering)
    SDL_Surface* screen_surface = NULL;    // Surface for the window to draw (if rendering)
    SDL_Surface* image_surface = NULL;     // Surface for the state obs image to exist (if rendering)
    static const int WIDTH = 96;           // State observation width
    static const int HEIGHT = 96;          // State observation height
    static const int RENDER_SCALE = 4;     // Scaling of state observation to render window
};

MuZeroNetworkConfig network_config;

// Encode action as feature plane of values 1/action
Observation encode_action_initial(Action action) {
    static const int num_actions = ALEEnv::action_space().size();
    ObservationShape obs_shape = ALEEnv::obs_shape();
    Observation obs(obs_shape.w * obs_shape.h, (double)action / num_actions);
    return obs;
}
Observation encode_action_recurrent(Action action) {
    static const int num_actions = ALEEnv::action_space().size();
    ObservationShape obs_shape =
        muzero_cpp::model::RepresentationNetworkImpl::encoded_state_shape(ALEEnv::obs_shape(), true);
    Observation obs(obs_shape.w * obs_shape.h, (double)action / num_actions);
    return obs;
}

// Simple softmax schedule
double get_softmax(int step) {
    if (step < 100000) {
        return 1.0;
    } else if (step < 200000) {
        return 0.5;
    }
    return 0.25;
}

// Additional flag to choose whether to test or not
ABSL_FLAG(bool, test, false, "Test using human input.");
ABSL_FLAG(std::string, game_file_path, "", "Full path that the game file binary resides.");
ABSL_FLAG(int, noop_max, 30, "Maximum number of NOOPs to apply at reset.");
ABSL_FLAG(int, frame_skip, 4, "Number of frames to skip.");
ABSL_FLAG(bool, fire_reset, false, "Apply a fire command on reset.");
ABSL_FLAG(bool, episodic, false, "Reset environment on life loss.");
ABSL_FLAG(bool, episodic_pong, false, "Reset environment on life loss for pong.");
ABSL_FLAG(bool, max_frame, false,
          "States are the max between 2 consecutive frames (used for game with screen flickering).");

int main(int argc, char** argv) {
    // parse flags
    parse_flags(argc, argv);
    MuZeroConfig config = get_initial_config();

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
    config.network_config.normalize_hidden_states = true;
    config.action_channels = 1;
    config.num_players = 1;
    config.opponent_type = OpponentTypes::Self;
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
        return muzero(config, game_factory<ALEEnv>);
    }

    return 0;
}