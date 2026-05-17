#ifndef MUZERO_CONFIG_H_
#define MUZERO_CONFIG_H_

#include <muzero/types.h>

#include <nlohmann/json.hpp>

#include <format>
#include <string>
#include <vector>

namespace muzero {

// Network Config
struct ModelConfig {
    double learning_rate;                   // Learning rate
    double l2_weight_decay;                 // L2 weight decay
    bool downsample;                        // Flag to downsample the input before passing to the representation network
    bool normalize_hidden_states;           // Flag to scale encoded hidden state between [0,1]
    int resnet_channels;                    // Channels for each ResNet block in the representation network
    int representation_blocks;              // Number of ResNet blocks in the representation network
    int dynamics_blocks;                    // Number of ResNet blocks in the dynamics network
    int prediction_blocks;                  // Number of ResNet blocks in the prediction network
    int reward_reduced_channels;            // Number of reduced channels to reduce for reward head
    int policy_reduced_channels;            // Number of reduced channels to reduce for policy head
    int value_reduced_channels;             // Number of reduced channels to reduce for value head
    std::vector<int> reward_head_layers;    // Size of each layer for reward head
    std::vector<int> policy_head_layers;    // Size of each layer for policy head
    std::vector<int> value_head_layers;     // Size of each layer for value head
};

auto network_config_from_json(const nlohmann::json &config) -> ModelConfig;

/**
 * Get the encoded observation shape after processing through the representation network
 * This is required as the spatial shape changes through downsampling
 * @param obs_shape The original observation shape
 * @param downsample True if downsampling, false if not
 */
auto encoded_obs_shape(const ObservationShape &obs_shape, bool downsample) -> ObservationShape;

// Config containing necessary parameters for the muzero algorithm.
struct Config {
    // General
    int seed = 0;                              // Seed to use for all sourcse of RNG
    int checkpoint_interval = 100;             // Interval of training steps to checkpoint
    int model_sync_interval = 100;             // Interval of training steps to sync model weights
    std::string path = "/opt/muzero-cpp/";     // Base path for all things being stored
    std::string devices = "cpu:0";             // String of torch devices comma separated
    bool explicit_learning = false;            // Flag for first device to be blocked from inference
    int num_actors = 1;                        // Number of self-play actors
    int num_reanalyze_actors = 0;              // Number of reanalyze actors
    int num_evaluators = 1;                    // Number of evaluators to test learning performance
    int initial_inference_batch_size = 1;      // Batch sized use for initial inference
    int recurrent_inference_batch_size = 1;    // Batch sized use for recurrent inference
    int initial_inference_threads = 1;         // Number of threads to perform initial inference
    int recurrent_inference_threads = 1;       // Number of threads to perform recurrent inference
    int max_training_steps = 100000;           // Maximum number of training steps to perform
    bool resume = false;                       // Flag to resume from last checkpoint
    CheckpointStep testing_checkpoint =
        CheckpointStep::kMostRecentCheckpointStep;    // Which checkpoint to load for testing

    // Game
    ObservationShape observation_shape;    // Observation shape (channel, height, width),
                                           // use (1, 1, len) for 1D observations
    std::vector<Action> action_space;      // List of all possible actions (start with 0, ..., num_actions - 1)
    int action_channels;                   // Number of channels the action are to be encoded as
    int num_players;                       // Number of players the game requires
    int stacked_observations = 0;          // Number of previous obs/actions to add to the current observation

    // Known upper/lower bound for game values (used for MCTS min/max scaling into [0, 1] range)
    double value_upperbound = NINF_D;    // Use types::NINF_D (-infinity) for general case, 1 for
                                         // example in 2-player board games
    double value_lowerbound = INF_D;     // Use types::INF_D (infinity) for general case, 0 for example
                                         // in 2-player board games

    // Value and reward categorical transformation
    double min_reward = -300;               // minimum possible reward
    double max_reward = 300;                // maximum possible reward
    double min_value = -300;                // minimum possible value
    double max_value = 300;                 // maximum possible value
    bool use_contractive_mapping = true;    // Use contractive mapping (https://arxiv.org/abs/1805.11593)

    // Evaluate
    Player muzero_player = 0;      // Player number (turn) muzero begins to play
                                   // (0 to play first, 1 for second, ...)
    OpponentType opponent_type;    // Agent that muzero plays against during evaluation. This doesn't
                                   // influence training (self, random, expert)

    // Self play
    int max_moves = -1;          // Maximum number of moves if selfplay game is not finished
                                 // (leave negative to ignore)
    int num_simulations = 50;    // Number of MCTS simulations per move
    double discount = 0.997;     // Discount factor for reward

    // Root prior dirichlet exploration noise
    double dirichlet_alpha = 0.3;       // Dirichlet distribution alpha parameter
    double dirichlet_epsilon = 0.25;    // The fractional component of weighted sum for the Dirichlet noise

    // PUCT constants
    double pb_c_base = 19652;    // PUCT c_base constant as defined in
                                 // (https://www.science.org/doi/10.1126/science.aar6404)
    double pb_c_init = 1.25;     // PUCT c_init constant as defined in
                                 // (https://www.science.org/doi/10.1126/science.aar6404)

    // Training
    int batch_size = 128;                // Samples per batch
    int min_sample_size = 256;           // Minimum samples needed to be stored before we can sample
    double value_loss_weight = 1;        // Scale value loss to avoid overfitting to the value function
    int td_steps = 10;                   // Number of future td steps to take into account for future value
    int num_unroll_steps = 5;            // Number of steps to unroll for each sample
    int max_history_len = -1;            // Maximum size of history before sending to replay buffer. Use -1 for the
                                         // entire history to pushed as a single sample (instead of splitting up)
    double train_reanalyze_ratio = 0;    // Ratio training samples that come from reanalyze (0 to not have reanalzye)

    // Replay buffer (Prioritized replay)
    int replay_buffer_size = 100000;       // Number of total fresh self-play samples to store
    int reanalyze_buffer_size = 100000;    // Number of total samples to store for reanalyze
    double per_alpha = 1;                  // Priority exponent
    double per_beta = 1;                   // Correction for sampling bias
    double per_epsilon = 0.01;             // Epsilon added to error to avoid 0's
    double per_beta_increment = 0.001;     // How much to increment beta (caps at 1.0)

    // network
    ModelConfig model_config;

    // Action int value to tensor mapping (action_channels)
    ActionRepresentationFunction action_representation_initial;
    ActionRepresentationFunction action_representation_recurrent;

    // Defines a softmax temperature scheduler
    SoftmaxTemperatureFunction visit_softmax_temperature;

    // String representation of config for pretty printing
    [[nodiscard]] auto to_str() const -> std::string
    {
        std::string output_str = "";
        output_str += "Network Config:\n";
        // Network config
        output_str += std::format("\tLearning rate: {:.5f}\n", model_config.learning_rate);
        output_str += std::format("\tL2 weight decay: {:.5f}\n", model_config.l2_weight_decay);
        output_str += std::format("\tDownsample: {}\n", model_config.downsample);
        output_str += std::format("\tNormalize hidden states: {}\n", model_config.normalize_hidden_states);
        output_str += std::format("\tResnet channels: {}\n", model_config.resnet_channels);
        output_str += std::format("\tRepresentation blocks: {}\n", model_config.representation_blocks);
        output_str += std::format("\tDynamics blocks: {}\n", model_config.dynamics_blocks);
        output_str += std::format("\tPrediction blocks: {}\n", model_config.prediction_blocks);
        output_str += std::format("\tReward reduced channels: {}\n", model_config.reward_reduced_channels);
        output_str += std::format("\tPolicy reduced channels: {}\n", model_config.policy_reduced_channels);
        output_str += std::format("\tValue reduced channels: {}\n", model_config.value_reduced_channels);
        output_str += "\tReward head layer: { ";
        for (auto const &l : model_config.reward_head_layers) {
            output_str += std::to_string(l) + " ";
        }
        output_str += "}\n";
        output_str += "\tPolicy head layers: { ";
        for (auto const &l : model_config.reward_head_layers) {
            output_str += std::to_string(l) + " ";
        }
        output_str += "}\n";
        output_str += "\tValue head layers: { ";
        for (auto const &l : model_config.reward_head_layers) {
            output_str += std::to_string(l) + " ";
        }
        output_str += "}\n";
        // General
        output_str += "General\n";
        output_str += std::format("\tSeed: {}\n", seed);
        output_str += std::format("\tCheckpoint interval: {}\n", checkpoint_interval);
        output_str += std::format("\tModel sync interval: {}\n", model_sync_interval);
        output_str += std::format("\tPath: {}\n", path);
        output_str += std::format("\tDevices: {}\n", devices);
        output_str += std::format("\tExplicit learning: {}\n", explicit_learning);
        output_str += std::format("\tNumber of actors: {}\n", num_actors);
        output_str += std::format("\tNumber of reanalyze actors: {}\n", num_reanalyze_actors);
        output_str += std::format("\tNumber of evaluator actors: {}\n", num_evaluators);
        output_str += std::format("\tInitial inference batch size: {}\n", initial_inference_batch_size);
        output_str += std::format("\tRecurrent inference batch size: {}\n", recurrent_inference_batch_size);
        output_str += std::format("\tInitial inference threads: {}\n", initial_inference_threads);
        output_str += std::format("\tRecurrent inference threads: {}\n", recurrent_inference_threads);
        output_str += std::format("\tMaximum training steps: {}\n", max_training_steps);
        output_str += std::format("\tResume: {}\n", resume);

        // Game
        output_str += "Game\n";
        output_str += std::format(
            "\tObservation shape: {{{} {} {} }}\n",
            observation_shape.c,
            observation_shape.h,
            observation_shape.w
        );

        output_str += "\tAction space: { ";
        for (auto const &l : action_space) {
            output_str += std::to_string(l) + " ";
        }
        output_str += "}\n";

        output_str += std::format("\tAction channels: {}\n", action_channels);
        output_str += std::format("\tNumber of players: {}\n", num_players);
        output_str += std::format("\tNumber of stacked observations: {}\n", stacked_observations);
        // Values
        output_str += "Values\n";
        output_str += std::format("\tValue upperbound: {:.2f}\n", value_upperbound);
        output_str += std::format("\tValue lowerbound: {:.2f}\n", value_lowerbound);
        output_str += std::format("\tMin reward: {:.2f}\n", min_reward);
        output_str += std::format("\tMax reward: {:.2f}\n", max_reward);
        output_str += std::format("\tMin value: {:.2f}\n", min_value);
        output_str += std::format("\tMax value: {:.2f}\n", max_value);
        output_str += std::format("\tUse contractive mapping: {}\n", use_contractive_mapping);
        // Evaluate
        output_str += "Evaluate\n";
        output_str += std::format("\tMuZero player: {}\n", muzero_player);
        output_str += std::format("\tMuZero opponent type: {}\n", to_string(opponent_type));
        // Self play
        output_str += "Self Play\n";
        output_str += std::format("\tMax moves: {}\n", max_moves);
        output_str += std::format("\tNumber of simulations: {}\n", num_simulations);
        output_str += std::format("\tDiscount: {:.3f}\n", discount);
        // Self play
        output_str += "Dirichlet\n";
        output_str += std::format("\tDirichlet alpha: {:.3f}\n", dirichlet_alpha);
        output_str += std::format("\tDirichlet epsilon: {:.3f}\n", dirichlet_epsilon);
        // PUCT constants
        output_str += "PUCT\n";
        output_str += std::format("\tpb_c_base: {:.3f}\n", pb_c_base);
        output_str += std::format("\tpb_c_init: {:.3f}\n", pb_c_init);
        // Training
        output_str += "Training\n";
        output_str += std::format("\tBatch size: {}\n", batch_size);
        output_str += std::format("\tMin sample size: {}\n", min_sample_size);
        output_str += std::format("\tValue loss weight: {:.3f}\n", value_loss_weight);
        output_str += std::format("\tTD steps: {}\n", td_steps);
        output_str += std::format("\tNum unroll steps: {}\n", num_unroll_steps);
        output_str += std::format("\tMax history length: {}\n", max_history_len);
        output_str += std::format("\tTrain reanalyze ratio: {:.3f}\n", train_reanalyze_ratio);
        // Replay buffer
        output_str += "Replay buffer\n";
        output_str += std::format("\tReplay buffer size: {}\n", replay_buffer_size);
        output_str += std::format("\tReplay buffer size: {}\n", reanalyze_buffer_size);
        output_str += std::format("\tPER alpha: {:.3f}\n", per_alpha);
        output_str += std::format("\tPER beta: {:.3f}\n", per_beta);
        output_str += std::format("\tPER epsilon: {:.5f}\n", per_epsilon);
        output_str += std::format("\tPER beta increment: {:.5f}\n", per_beta_increment);
        return output_str;
    }
};

}    // namespace muzero

#endif    // MUZERO_CONFIG_H_
