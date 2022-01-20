#ifndef MUZERO_CPP_CONFIG_H_
#define MUZERO_CPP_CONFIG_H_

#include <functional>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "muzero-cpp/types.h"

namespace muzero_cpp {
namespace muzero_config {

// Network Config
struct MuZeroNetworkConfig {
    double learning_rate = 3e-4;      // Learning rate
    double l2_weight_decay = 1e-4;    // L2 weight decay
    bool downsample = false;    // Flag to downsample the input before passing to the representation network
    bool normalize_hidden_states = true;    // Flag to scale encoded hidden state between [0,1]
    int resnet_channels = 128;              // Channels for each ResNet block in the representation network
    int representation_blocks = 4;          // Number of ResNet blocks in the representation network
    int dynamics_blocks = 4;                // Number of ResNet blocks in the dynamics network
    int prediction_blocks = 4;              // Number of ResNet blocks in the prediction network
    int reward_reduced_channels = 16;       // Number of reduced channels to reduce for reward head
    int policy_reduced_channels = 16;       // Number of reduced channels to reduce for policy head
    int value_reduced_channels = 16;        // Number of reduced channels to reduce for value head
    std::vector<int> reward_head_layers{64};    // Size of each layer for reward head
    std::vector<int> policy_head_layers{64};    // Size of each layer for policy head
    std::vector<int> value_head_layers{64};     // Size of each layer for value head
};

// Config containing necessary parameters for the muzero algorithm.
struct MuZeroConfig {
    // General
    int seed = 0;                              // Seed to use for all sourcse of RNG
    int checkpoint_interval = 100;             // Interval of training steps to checkpoint
    int model_sync_interval = 100;             // Interval of training steps to sync model weights
    std::string path = "/opt/muzero-cpp/";     // Base path for all things being stored
    std::string devices = "cpu:0";             // String of torch devices comma separated
    bool explicit_learning = false;            // Flag for first device to be blocked from inference
    int num_actors = 1;                        // Number of self-play actors
    int num_evaluators = 1;                    // Number of evaluators to test learning performance
    int initial_inference_batch_size = 1;      // Batch sized use for initial inference
    int recurrent_inference_batch_size = 1;    // Batch sized use for recurrent inference
    int initial_inference_threads = 1;         // Number of threads to perform initial inference
    int recurrent_inference_threads = 1;       // Number of threads to perform recurrent inference
    int max_training_steps = 100000;           // Maximum number of training steps to perform
    bool resume = false;                       // Flag to resume from last checkpoint

    // Game
    types::ObservationShape observation_shape;    // Observation shape (channel, height, width),
                                                  // use (1, 1, len) for 1D observations
    std::vector<types::Action>
        action_space;                // List of all possible actions (start with 0, ..., num_actions - 1)
    int action_channels;             // Number of channels the action are to be encoded as
    int num_players;                 // Number of players the game requires
    int stacked_observations = 0;    // Number of previous obs/actions to add to the current observation

    // Known upper/lower bound for game values (used for MCTS min/max scaling into [0, 1] range)
    double value_upperbound = types::NINF_D;    // Use types::NINF_D (-infinity) for general case, 1 for
                                                // example in 2-player board games
    double value_lowerbound = types::INF_D;     // Use types::INF_D (infinity) for general case, 0 for example
                                                // in 2-player board games

    // Value and reward categorical transformation
    double min_reward;                      // minimum possible reward
    double max_reward;                      // maximum possible reward
    double min_value;                       // minimum possible value
    double max_value;                       // maximum possible value
    bool use_contractive_mapping = true;    // Use contractive mapping (https://arxiv.org/abs/1805.11593)

    // Evaluate
    types::Player muzero_player = 0;       // Player number (turn) muzero begins to play
                                           // (0 to play first, 1 for second, ...)
    types::OpponentTypes opponent_type;    // Agent that muzero plays against during evaluation. This doesn't
                                           // influence training (self, random, expert)

    // Self play
    int max_moves = -1;                  // Maximum number of moves if selfplay game is not finished
                                         // (leave negative to ignore)
    int num_simulations = 50;            // Number of MCTS simulations per move
    double discount = 0.997;             // Discount factor for reward
    double train_selfplay_ratio = -1;    // Ratio of training steps per self play step
                                         // (leave negative to ignore)

    // Root prior dirichlet exploration noise
    double dirichlet_alpha = 0.3;       // Dirichlet distribution alpha parameter
    double dirichlet_epsilon = 0.25;    // The fractional component of weighted sum for the Dirichlet noise

    // PUCT constants
    double pb_c_base = 19652;    // PUCT c_base constant as defined in
                                 // (https://www.science.org/doi/10.1126/science.aar6404)
    double pb_c_init = 1.25;     // PUCT c_init constant as defined in
                                 // (https://www.science.org/doi/10.1126/science.aar6404)

    // Training
    int batch_size = 128;            // Samples per batch
    int min_sample_size = 256;       // Minimum samples needed to be stored before we can sample
    double value_loss_weight = 1;    // Scale value loss to avoid overfitting to the value function
    int td_steps = 10;               // Number of future td steps to take into account for future value
    int num_unroll_steps = 5;        // Number of steps to unroll for each sample
    int max_history_len = -1;    // Maximum size of history before sending to replay buffer. Use -1 for the
                                 // entire history to pushed as a single sample (instead of splitting up)
    bool reanalyze = false;      // Flag to run a reanalyze thread (See Appendix H : Reanalyze)

    // Replay buffer (Prioritized replay)
    int replay_buffer_size = 100000;      // Number of total samples to store
    double per_alpha = 1;                 // Priority exponent
    double per_beta = 1;                  // Correction for sampling bias
    double per_epsilon = 0.01;            // Epsilon added to error to avoid 0's
    double per_beta_increment = 0.001;    // How much to increment beta (caps at 1.0)

    // network
    MuZeroNetworkConfig network_config;

    // Action int value to tensor mapping (action_channels)
    types::ActionRepresentationFunction action_representation_initial;
    types::ActionRepresentationFunction action_representation_recurrent;

    // Defines a softmax temperature scheduler
    types::SoftmaxTemperatureFunction visit_softmax_temperature;

    // String representation of config for pretty printing
    std::string to_str() const {
        std::string output_str = "";
        output_str += "Network Config:\n";
        // Network config
        output_str += absl::StrFormat("\tLearning rate: %.5f\n", network_config.learning_rate);
        output_str += absl::StrFormat("\tL2 weight decay: %.5f\n", network_config.l2_weight_decay);
        output_str += absl::StrFormat("\tDownsample: %d\n", network_config.downsample);
        output_str +=
            absl::StrFormat("\tNormalize hidden states: %d\n", network_config.normalize_hidden_states);
        output_str += absl::StrFormat("\tResnet channels: %d\n", network_config.resnet_channels);
        output_str += absl::StrFormat("\tRepresentation blocks: %d\n", network_config.representation_blocks);
        output_str += absl::StrFormat("\tDynamics blocks: %d\n", network_config.dynamics_blocks);
        output_str += absl::StrFormat("\tPrediction blocks: %d\n", network_config.prediction_blocks);
        output_str +=
            absl::StrFormat("\tReward reduced channels: %d\n", network_config.reward_reduced_channels);
        output_str +=
            absl::StrFormat("\tPolicy reduced channels: %d\n", network_config.policy_reduced_channels);
        output_str +=
            absl::StrFormat("\tValue reduced channels: %d\n", network_config.value_reduced_channels);
        output_str += "\tReward head layer: { ";
        for (auto const& l : network_config.reward_head_layers) {
            output_str += std::to_string(l) + " ";
        }
        output_str += "}\n";
        output_str += "\tPolicy head layers: { ";
        for (auto const& l : network_config.reward_head_layers) {
            output_str += std::to_string(l) + " ";
        }
        output_str += "}\n";
        output_str += "\tValue head layers: { ";
        for (auto const& l : network_config.reward_head_layers) {
            output_str += std::to_string(l) + " ";
        }
        output_str += "}\n";
        // General
        output_str += "General\n";
        output_str += absl::StrFormat("\tSeed: %d\n", seed);
        output_str += absl::StrFormat("\tCheckpoint interval: %d\n", checkpoint_interval);
        output_str += absl::StrFormat("\tPath: %s\n", path);
        output_str += absl::StrFormat("\tDevices: %s\n", devices);
        output_str += absl::StrFormat("\tNumber of actors: %d\n", num_actors);
        output_str += absl::StrFormat("\tInitial inference batch size: %d\n", initial_inference_batch_size);
        output_str +=
            absl::StrFormat("\tRecurrent inference batch size: %d\n", recurrent_inference_batch_size);
        output_str += absl::StrFormat("\tInitial inference threads: %d\n", initial_inference_threads);
        output_str += absl::StrFormat("\tRecurrent inference threads: %d\n", recurrent_inference_threads);
        output_str += absl::StrFormat("\tMaximum training steps: %d\n", max_training_steps);
        output_str += absl::StrFormat("\tResume: %d\n", resume);

        // Game
        output_str += "Game\n";
        output_str += absl::StrFormat("\tObservation shape: { %d %d %d }\n", observation_shape.c,
                                      observation_shape.h, observation_shape.w);

        output_str += "\tAction space: { ";
        for (auto const& l : action_space) {
            output_str += std::to_string(l) + " ";
        }
        output_str += "}\n";

        output_str += absl::StrFormat("\tAction channels: %d\n", action_channels);
        output_str += absl::StrFormat("\tNumber of players: %d\n", num_players);
        output_str += absl::StrFormat("\tNumber of stacked observations: %d\n", stacked_observations);
        // Values
        output_str += "Values\n";
        output_str += absl::StrFormat("\tValue upperbound: %.2f\n", value_upperbound);
        output_str += absl::StrFormat("\tValue lowerbound: %.2f\n", value_lowerbound);
        output_str += absl::StrFormat("\tMin reward: %.2f\n", min_reward);
        output_str += absl::StrFormat("\tMax reward: %.2f\n", max_reward);
        output_str += absl::StrFormat("\tMin value: %.2f\n", min_value);
        output_str += absl::StrFormat("\tMax value: %.2f\n", max_value);
        output_str += absl::StrFormat("\tUse contractive mapping: %d\n", use_contractive_mapping);
        // Evaluate
        output_str += "Evaluate\n";
        output_str += absl::StrFormat("\tMuZero player: %d\n", muzero_player);
        output_str += absl::StrFormat("\tMuZero opponent type: %d\n", opponent_type);
        // Self play
        output_str += "Self Play\n";
        output_str += absl::StrFormat("\tMax moves: %d\n", max_moves);
        output_str += absl::StrFormat("\tNumber of simulations: %d\n", num_simulations);
        output_str += absl::StrFormat("\tDiscount: %.3f\n", discount);
        output_str += absl::StrFormat("\tTrain Selfplay ratio: %.3f\n", train_selfplay_ratio);
        // Self play
        output_str += "Dirichlet\n";
        output_str += absl::StrFormat("\tDirichlet alpha: %.3f\n", dirichlet_alpha);
        output_str += absl::StrFormat("\tDirichlet epsilon: %.3f\n", dirichlet_epsilon);
        // PUCT constants
        output_str += "PUCT\n";
        output_str += absl::StrFormat("\tpb_c_base: %.3f\n", pb_c_base);
        output_str += absl::StrFormat("\tpb_c_init: %.3f\n", pb_c_init);
        // Training
        output_str += "Training\n";
        output_str += absl::StrFormat("\tBatch size: %d\n", batch_size);
        output_str += absl::StrFormat("\tMin sample size: %d\n", min_sample_size);
        output_str += absl::StrFormat("\tValue loss weight: %.3f\n", value_loss_weight);
        output_str += absl::StrFormat("\tTD steps: %d\n", td_steps);
        output_str += absl::StrFormat("\tNum unroll steps: %d\n", num_unroll_steps);
        output_str += absl::StrFormat("\tMax history length: %d\n", max_history_len);
        output_str += absl::StrFormat("\tReanalyze: %d\n", reanalyze);
        // Replay buffer
        output_str += "Replay buffer\n";
        output_str += absl::StrFormat("\tReplay buffer size: %d\n", replay_buffer_size);
        output_str += absl::StrFormat("\tPER alpha: %.3f\n", per_alpha);
        output_str += absl::StrFormat("\tPER beta: %.3f\n", per_beta);
        output_str += absl::StrFormat("\tPER epsilon: %.5f\n", per_epsilon);
        output_str += absl::StrFormat("\tPER beta increment: %.5f\n", per_beta_increment);
        return output_str;
    }
};

}    // namespace muzero_config
}    // namespace muzero_cpp

#endif    // MUZERO_CPP_CONFIG_H_