#ifndef MUZERO_CPP_DEFAULT_FLAGS_H_
#define MUZERO_CPP_DEFAULT_FLAGS_H_

#include <string>
#include <vector>

#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "muzero-cpp/config.h"

// Declaration for external access of the flags
ABSL_DECLARE_FLAG(int, seed);
ABSL_DECLARE_FLAG(int, checkpoint_interval);
ABSL_DECLARE_FLAG(int, model_sync_interval);
ABSL_DECLARE_FLAG(std::string, path);
ABSL_DECLARE_FLAG(std::string, devices);
ABSL_DECLARE_FLAG(bool, explicit_learning);
ABSL_DECLARE_FLAG(int, num_actors);
ABSL_DECLARE_FLAG(int, num_reanalyze_actors);
ABSL_DECLARE_FLAG(int, num_evaluators);
ABSL_DECLARE_FLAG(int, initial_inference_batch_size);
ABSL_DECLARE_FLAG(int, recurrent_inference_batch_size);
ABSL_DECLARE_FLAG(int, initial_inference_threads);
ABSL_DECLARE_FLAG(int, recurrent_inference_threads);
ABSL_DECLARE_FLAG(int, max_training_steps);
ABSL_DECLARE_FLAG(bool, resume);
ABSL_DECLARE_FLAG(int, testing_checkpoint);
ABSL_DECLARE_FLAG(int, stacked_observations);
ABSL_DECLARE_FLAG(double, value_upperbound);
ABSL_DECLARE_FLAG(double, value_lowerbound);
ABSL_DECLARE_FLAG(double, min_reward);
ABSL_DECLARE_FLAG(double, max_reward);
ABSL_DECLARE_FLAG(double, min_value);
ABSL_DECLARE_FLAG(double, max_value);
ABSL_DECLARE_FLAG(bool, use_contractive_mapping);
ABSL_DECLARE_FLAG(int, max_moves);
ABSL_DECLARE_FLAG(int, num_simulations);
ABSL_DECLARE_FLAG(double, discount);
ABSL_DECLARE_FLAG(double, dirichlet_alpha);
ABSL_DECLARE_FLAG(double, dirichlet_epsilon);
ABSL_DECLARE_FLAG(double, pb_c_base);
ABSL_DECLARE_FLAG(double, pb_c_init);
ABSL_DECLARE_FLAG(int, replay_buffer_size);
ABSL_DECLARE_FLAG(int, reanalyze_buffer_size);
ABSL_DECLARE_FLAG(int, batch_size);
ABSL_DECLARE_FLAG(double, train_reanalyze_ratio);
ABSL_DECLARE_FLAG(double, value_loss_weight);
ABSL_DECLARE_FLAG(int, min_sample_size);
ABSL_DECLARE_FLAG(int, td_steps);
ABSL_DECLARE_FLAG(int, num_unroll_steps);
ABSL_DECLARE_FLAG(int, max_history_len);
ABSL_DECLARE_FLAG(double, per_alpha);
ABSL_DECLARE_FLAG(double, per_beta);
ABSL_DECLARE_FLAG(double, per_epsilon);
ABSL_DECLARE_FLAG(double, per_beta_increment);

namespace muzero_cpp {

/**
 * Parses the flags from cmd input and stores into the absl flags
 */
void parse_flags(int argc, char **argv);

/**
 * Creates muzero config from the absl flags (default or user defined)
 * @note not all values can be set (i.e. network definition, action_representation, etc.), 
 *       and so these should be manually set by the user (see examples.)
 * @returns muzero config with values set by flags
 */
muzero_config::MuZeroConfig get_initial_config();
}    // namespace muzero_cpp

#endif    // MUZERO_CPP_DEFAULT_FLAGS_H_