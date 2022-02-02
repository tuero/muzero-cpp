#include "muzero-cpp/default_flags.h"

#include <string>

#include "absl/flags/flag.h"
#include "muzero-cpp/types.h"
using namespace muzero_cpp::types;

// Default flag values with hints
ABSL_FLAG(int, seed, 0, "Seed for all sources of RNG.");
ABSL_FLAG(int, checkpoint_interval, 100, "Interval of learning steps to checkpoint model/buffer");
ABSL_FLAG(int, model_sync_interval, 100, "Interval of learning steps to sync the models");
ABSL_FLAG(std::string, path, "/opt/muzero-cpp/", "Base path to store all checkpoints and metrics");
ABSL_FLAG(std::string, devices, "cpu", "List of devices to use to train and run inference");
ABSL_FLAG(bool, explicit_learning, false, "Block first device from inference");
ABSL_FLAG(int, num_actors, 1, "Number of actors to run inference");
ABSL_FLAG(int, num_reanalyze_actors, 0, "Number of actors to run reanalyze");
ABSL_FLAG(int, num_evaluators, 1, "Number of evaluators to test learning progress");
ABSL_FLAG(int, initial_inference_batch_size, 1, "Batch size for initial inference");
ABSL_FLAG(int, recurrent_inference_batch_size, 1, "Batch size for recurrent inference");
ABSL_FLAG(int, initial_inference_threads, 1, "Number of threads to run inital inference");
ABSL_FLAG(int, recurrent_inference_threads, 1, "Number of threads to run recurrent inference");
ABSL_FLAG(int, max_training_steps, 1e6, "Maximum number of training steps");
ABSL_FLAG(bool, resume, false, "Flag to resume from most recent checkpoint");
ABSL_FLAG(int, stacked_observations, 0, "Maximum number previous observations to use");
ABSL_FLAG(double, value_upperbound, NINF_D, "Known upperbound for game value MCTS scaling");
ABSL_FLAG(double, value_lowerbound, INF_D, "Known lowerbound for game value MCTS scaling");
ABSL_FLAG(double, min_reward, -300, "Minimum possible reward");
ABSL_FLAG(double, max_reward, 300, "Maximum possible reward");
ABSL_FLAG(double, min_value, -300, "Minimum possible value");
ABSL_FLAG(double, max_value, 300, "Maximum possible value");
ABSL_FLAG(bool, use_contractive_mapping, true, "Use contractive mapping");
ABSL_FLAG(int, max_moves, -1, "Max number of moves before sending partial game history");
ABSL_FLAG(int, num_simulations, 50, "Number of iterations to run MCTS per turn");
ABSL_FLAG(double, discount, 0.997, "Discount factor to apply");
ABSL_FLAG(double, dirichlet_alpha, 0.3, "Dirichlet distribution alpha parameter");
ABSL_FLAG(double, dirichlet_epsilon, 0.25,
          "The fractional component of weighted sum for the Dirichlet noise");
ABSL_FLAG(double, pb_c_base, 19652, "PUCT c_base constant");
ABSL_FLAG(double, pb_c_init, 1.25, "PUCT c_init constant");
ABSL_FLAG(int, replay_buffer_size, 1e6, "Number of steps to store in the replay buffer");
ABSL_FLAG(int, reanalyze_buffer_size, 1e6, "Number of steps to store in the reanalyze buffer");
ABSL_FLAG(int, batch_size, 128, "Batch size for training");
ABSL_FLAG(double, train_reanalyze_ratio, 0,
          "Ratio of training sample to be from reanalyze (use 0 to not have reanalyze)");
ABSL_FLAG(double, value_loss_weight, 0.25, "Value loss scaling to avoid overfitting");
ABSL_FLAG(int, min_sample_size, 256, "Minimum number of sample needed before training starts");
ABSL_FLAG(int, td_steps, 10, "Number of future td steps to take into account for future value");
ABSL_FLAG(int, num_unroll_steps, 5, "Number of steps to unroll for each sample");
ABSL_FLAG(int, max_history_len, -1,
          "Maximum history length before sending to replay buffer. Use -1 to ignore.");
ABSL_FLAG(double, per_alpha, 1, "Priority exponent");
ABSL_FLAG(double, per_beta, 1, "Correction for sampling bias");
ABSL_FLAG(double, per_epsilon, 0.01, "Epsilon added to error to avoid 0's");
ABSL_FLAG(double, per_beta_increment, 0.001, "How much to increment beta (caps at 1.0)");

// Network config flags
ABSL_FLAG(double, learning_rate, 3e-4, "Learning rate");
ABSL_FLAG(double, l2_weight_decay, 1e-4, "L2 weight decay");
ABSL_FLAG(bool, downsample, false,
          "Flag to use downsample the input before passing to representation network");
ABSL_FLAG(bool, normalize_hidden_states, true, "Flag to scale encoded hidden state between [0,1]");
ABSL_FLAG(int, resnet_channels, 128,
          "Channels for each ResNet block in the representation, dynamics, and prediction network");
ABSL_FLAG(int, representation_blocks, 4, "Number of ResNet blocks in the representation network");
ABSL_FLAG(int, dynamics_blocks, 4, "Number of ResNet blocks in the dynamics network");
ABSL_FLAG(int, prediction_blocks, 4, "Number of ResNet blocks in the prediction network");
ABSL_FLAG(int, reward_reduced_channels, 16, "Number of reduced channels to reduce for reward head");
ABSL_FLAG(int, policy_reduced_channels, 16, "Number of reduced channels to reduce for policy head");
ABSL_FLAG(int, value_reduced_channels, 16, "Number of reduced channels to reduce for value head");
ABSL_FLAG(std::vector<std::string>, reward_head_layers, std::vector<std::string>({"64"}),
          "Comma separated list of layer sizes for reward head");
ABSL_FLAG(std::vector<std::string>, policy_head_layers, std::vector<std::string>({"64"}),
          "Comma separated list of layer sizes for policy head");
ABSL_FLAG(std::vector<std::string>, value_head_layers, std::vector<std::string>({"64"}),
          "Comma separated list of layer sizes for value head");

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
    config.model_sync_interval = absl::GetFlag(FLAGS_model_sync_interval);
    config.path = absl::GetFlag(FLAGS_path);
    config.devices = absl::GetFlag(FLAGS_devices);
    config.explicit_learning = absl::GetFlag(FLAGS_explicit_learning);
    config.num_actors = absl::GetFlag(FLAGS_num_actors);
    config.num_reanalyze_actors = absl::GetFlag(FLAGS_num_reanalyze_actors);
    config.num_evaluators = absl::GetFlag(FLAGS_num_evaluators);
    config.initial_inference_batch_size = absl::GetFlag(FLAGS_initial_inference_batch_size);
    config.recurrent_inference_batch_size = absl::GetFlag(FLAGS_recurrent_inference_batch_size);
    config.initial_inference_threads = absl::GetFlag(FLAGS_initial_inference_threads);
    config.recurrent_inference_threads = absl::GetFlag(FLAGS_recurrent_inference_threads);
    config.max_training_steps = absl::GetFlag(FLAGS_max_training_steps);
    config.resume = absl::GetFlag(FLAGS_resume);
    config.stacked_observations = absl::GetFlag(FLAGS_stacked_observations);
    config.value_upperbound = absl::GetFlag(FLAGS_value_upperbound);
    config.value_lowerbound = absl::GetFlag(FLAGS_value_lowerbound);
    config.min_reward = absl::GetFlag(FLAGS_min_reward);
    config.max_reward = absl::GetFlag(FLAGS_max_reward);
    config.min_value = absl::GetFlag(FLAGS_min_value);
    config.max_value = absl::GetFlag(FLAGS_max_value);
    config.use_contractive_mapping = absl::GetFlag(FLAGS_use_contractive_mapping);
    config.max_moves = absl::GetFlag(FLAGS_max_moves);
    config.num_simulations = absl::GetFlag(FLAGS_num_simulations);
    config.discount = absl::GetFlag(FLAGS_discount);
    config.dirichlet_alpha = absl::GetFlag(FLAGS_dirichlet_alpha);
    config.dirichlet_epsilon = absl::GetFlag(FLAGS_dirichlet_epsilon);
    config.pb_c_base = absl::GetFlag(FLAGS_pb_c_base);
    config.pb_c_init = absl::GetFlag(FLAGS_pb_c_init);
    config.replay_buffer_size = absl::GetFlag(FLAGS_replay_buffer_size);
    config.batch_size = absl::GetFlag(FLAGS_batch_size);
    config.train_reanalyze_ratio = absl::GetFlag(FLAGS_train_reanalyze_ratio);
    config.value_loss_weight = absl::GetFlag(FLAGS_value_loss_weight);
    config.min_sample_size = absl::GetFlag(FLAGS_min_sample_size);
    config.td_steps = absl::GetFlag(FLAGS_td_steps);
    config.num_unroll_steps = absl::GetFlag(FLAGS_num_unroll_steps);
    config.max_history_len = absl::GetFlag(FLAGS_max_history_len);
    config.per_alpha = absl::GetFlag(FLAGS_per_alpha);
    config.per_beta = absl::GetFlag(FLAGS_per_beta);
    config.per_epsilon = absl::GetFlag(FLAGS_per_epsilon);
    config.per_beta_increment = absl::GetFlag(FLAGS_per_beta_increment);

    // Get network config, and convert absl string vector into int vector
    config.network_config.learning_rate = absl::GetFlag(FLAGS_learning_rate);
    config.network_config.l2_weight_decay = absl::GetFlag(FLAGS_l2_weight_decay);
    config.network_config.downsample = absl::GetFlag(FLAGS_downsample);
    config.network_config.normalize_hidden_states = absl::GetFlag(FLAGS_normalize_hidden_states);
    config.network_config.resnet_channels = absl::GetFlag(FLAGS_resnet_channels);
    config.network_config.representation_blocks = absl::GetFlag(FLAGS_representation_blocks);
    config.network_config.dynamics_blocks = absl::GetFlag(FLAGS_dynamics_blocks);
    config.network_config.prediction_blocks = absl::GetFlag(FLAGS_prediction_blocks);
    config.network_config.reward_reduced_channels = absl::GetFlag(FLAGS_reward_reduced_channels);
    config.network_config.policy_reduced_channels = absl::GetFlag(FLAGS_policy_reduced_channels);
    config.network_config.value_reduced_channels = absl::GetFlag(FLAGS_value_reduced_channels);

    config.network_config.reward_head_layers.clear();
    for (const auto &r : absl::GetFlag(FLAGS_reward_head_layers)) {
        config.network_config.reward_head_layers.push_back(std::stoi(r));
    }
    config.network_config.policy_head_layers.clear();
    for (const auto &r : absl::GetFlag(FLAGS_policy_head_layers)) {
        config.network_config.policy_head_layers.push_back(std::stoi(r));
    }
    config.network_config.value_head_layers.clear();
    for (const auto &r : absl::GetFlag(FLAGS_value_head_layers)) {
        config.network_config.value_head_layers.push_back(std::stoi(r));
    }

    return config;
}

}    // namespace muzero_cpp