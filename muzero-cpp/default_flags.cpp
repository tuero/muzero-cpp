#include "muzero-cpp/default_flags.h"

#include <string>

#include "absl/flags/flag.h"

// Default flag values with hints
ABSL_FLAG(int, seed, 0, "Seed for all sources of RNG.");
ABSL_FLAG(int, checkpoint_interval, 100, "Interval of learning steps to checkpoint model");
ABSL_FLAG(std::string, path, "/opt/muzero-cpp/", "Base path to store all checkpoints and metrics");
ABSL_FLAG(std::string, devices, "cpu", "List of devices to use to train and run inference");
ABSL_FLAG(bool, explicit_learning, false, "Block first device from inference");
ABSL_FLAG(int, num_actors, 1, "Number of actors to run inference");
ABSL_FLAG(int, num_evaluators, 1, "Number of evaluators to test learning progress");
ABSL_FLAG(int, initial_inference_batch_size, 1, "Batch size for initial inference");
ABSL_FLAG(int, recurrent_inference_batch_size, 1, "Batch size for recurrent inference");
ABSL_FLAG(int, initial_inference_threads, 1, "Number of threads to run inital inference");
ABSL_FLAG(int, recurrent_inference_threads, 1, "Number of threads to run recurrent inference");
ABSL_FLAG(int, max_training_steps, 1e6, "Maximum number of training steps");
ABSL_FLAG(bool, resume, false, "Flag to resume from most recent checkpoint");
ABSL_FLAG(int, stacked_observations, 0, "Maximum number previous observations to use");
ABSL_FLAG(bool, use_contractive_mapping, true, "Use contractive mapping");
ABSL_FLAG(int, max_moves, -1, "Max number of moves before sending partial game history");
ABSL_FLAG(int, num_simulations, 50, "Number of iterations to run MCTS per turn");
ABSL_FLAG(double, discount, 0.997, "Discount factor to apply");
ABSL_FLAG(double, train_selfplay_ratio, -1, "Number of training steps per self play step");
ABSL_FLAG(double, dirichlet_alpha, 0.3, "Dirichlet distribution alpha parameter");
ABSL_FLAG(double, dirichlet_epsilon, 0.25,
          "The fractional component of weighted sum for the Dirichlet noise");
ABSL_FLAG(double, pb_c_base, 19652, "PUCT c_base constant");
ABSL_FLAG(double, pb_c_init, 1.25, "PUCT c_init constant");
ABSL_FLAG(int, replay_buffer_size, 1e6, "Number of steps to store in the replay buffer");
ABSL_FLAG(int, batch_size, 128, "Batch size for training");
ABSL_FLAG(bool, reanalyze, false, "Flag to use reanalyze");
ABSL_FLAG(double, value_loss_weight, 0.25, "Value loss scaling to avoid overfitting");
ABSL_FLAG(int, min_sample_size, 256, "Minimum number of sample needed before training starts");
ABSL_FLAG(int, td_steps, 10, "Number of future td steps to take into account for future value");
ABSL_FLAG(int, num_unroll_steps, 5, "Number of steps to unroll for each sample");
ABSL_FLAG(double, per_alpha, 1, "Priority exponent");
ABSL_FLAG(double, per_beta, 1, "Correction for sampling bias");
ABSL_FLAG(double, per_epsilon, 0.01, "Epsilon added to error to avoid 0's");
ABSL_FLAG(double, per_beta_increment, 0.001, "How much to increment beta (caps at 1.0)");

namespace muzero_cpp {
using namespace muzero_config;

// Call the parser from absl
void parse_flags(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);
}

// Creates muzero config from the absl flags (default or user defined)
MuZeroConfig get_initial_config() {
    MuZeroConfig config;
    config.seed = absl::GetFlag(FLAGS_seed);
    config.checkpoint_interval = absl::GetFlag(FLAGS_checkpoint_interval);
    config.path = absl::GetFlag(FLAGS_path);
    config.devices = absl::GetFlag(FLAGS_devices);
    config.explicit_learning = absl::GetFlag(FLAGS_explicit_learning);
    config.num_actors = absl::GetFlag(FLAGS_num_actors);
    config.num_evaluators = absl::GetFlag(FLAGS_num_evaluators);
    config.initial_inference_batch_size = absl::GetFlag(FLAGS_initial_inference_batch_size);
    config.recurrent_inference_batch_size = absl::GetFlag(FLAGS_recurrent_inference_batch_size);
    config.initial_inference_threads = absl::GetFlag(FLAGS_initial_inference_threads);
    config.recurrent_inference_threads = absl::GetFlag(FLAGS_recurrent_inference_threads);
    config.max_training_steps = absl::GetFlag(FLAGS_max_training_steps);
    config.resume = absl::GetFlag(FLAGS_resume);
    config.stacked_observations = absl::GetFlag(FLAGS_stacked_observations);
    config.use_contractive_mapping = absl::GetFlag(FLAGS_use_contractive_mapping);
    config.max_moves = absl::GetFlag(FLAGS_max_moves);
    config.num_simulations = absl::GetFlag(FLAGS_num_simulations);
    config.discount = absl::GetFlag(FLAGS_discount);
    config.train_selfplay_ratio = absl::GetFlag(FLAGS_train_selfplay_ratio);
    config.dirichlet_alpha = absl::GetFlag(FLAGS_dirichlet_alpha);
    config.dirichlet_epsilon = absl::GetFlag(FLAGS_dirichlet_epsilon);
    config.pb_c_base = absl::GetFlag(FLAGS_pb_c_base);
    config.pb_c_init = absl::GetFlag(FLAGS_pb_c_init);
    config.replay_buffer_size = absl::GetFlag(FLAGS_replay_buffer_size);
    config.batch_size = absl::GetFlag(FLAGS_batch_size);
    config.reanalyze = absl::GetFlag(FLAGS_reanalyze);
    config.value_loss_weight = absl::GetFlag(FLAGS_value_loss_weight);
    config.min_sample_size = absl::GetFlag(FLAGS_min_sample_size);
    config.td_steps = absl::GetFlag(FLAGS_td_steps);
    config.num_unroll_steps = absl::GetFlag(FLAGS_num_unroll_steps);
    config.per_alpha = absl::GetFlag(FLAGS_per_alpha);
    config.per_beta = absl::GetFlag(FLAGS_per_beta);
    config.per_epsilon = absl::GetFlag(FLAGS_per_epsilon);
    config.per_beta_increment = absl::GetFlag(FLAGS_per_beta_increment);
    return config;
}

}    // namespace muzero_cpp