// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 
// Modifications made for the muzero network (tuero@ualberta.ca)

#ifndef MUZERO_CPP_VPRNET_H_
#define MUZERO_CPP_VPRNET_H_

#include <torch/torch.h>

#include <string>

#include "muzero-cpp/config.h"
#include "muzero-cpp/models.h"
#include "muzero-cpp/types.h"
#include "muzero-cpp/util.h"

namespace muzero_cpp {
class VPRNetModel {
public:
    // Struct holding loss info, used for reporting/updating PER
    struct LossInfo {
        LossInfo() {}
        LossInfo(double total, double value, double policy, double reward, const std::vector<int>& indices,
                 const std::vector<double>& errors)
            : total_loss(total),
              value(value),
              policy(policy),
              reward(reward),
              indices(indices),
              errors(errors) {}

        double total_loss = 0;         // Total scaled loss
        double value = 0;              // Unscaled value loss
        double policy = 0;             // Unscaled policy loss
        double reward = 0;             // Unscaled reward loss
        std::vector<int> indices;      // PER sample indices to update
        std::vector<double> errors;    // Errors to update PER
    };

    // Input for initial inference
    struct InitialInferenceInputs {
        types::Observation stacked_observation;
    };

    // Input for recurrent inference
    struct RecurrentInferenceInputs {
        types::Action action;
        types::Observation encoded_state;
    };

    // Output for both initial and recurrent inference
    struct InferenceOutputs {
        double value;                        // Predicted value
        double reward;                       // Predicted reward
        std::vector<double> policy;          // Predicted policy (network logits passed through softmax)
        types::Observation encoded_state;    // Encoded state
    };

    enum CheckpointStep { kMostRecentCheckpointStep = -1, kInvalidCheckpointStep = -2 };

    VPRNetModel(const muzero_config::MuZeroConfig& config, const std::string& device = "/cpu:0");

    // Move only, not copyable.
    VPRNetModel(VPRNetModel&& other) = default;
    VPRNetModel& operator=(VPRNetModel&& other) = default;
    VPRNetModel(const VPRNetModel&) = delete;
    VPRNetModel& operator=(const VPRNetModel&) = delete;

    /**
     * Pretty print torch model
     */
    void print() const;

    /**
     * Perform initial inference
     * @param inputs Batched stacked observations
     * @returns Predicted value, reward, policy, and encoded state 
     */
    std::vector<InferenceOutputs> InitialInference(std::vector<InitialInferenceInputs>& inputs);

    /**
     * Perform recurrent inference
     * @param inputs Batched stacked observations and actions taken
     * @returns Predicted value, reward, policy, and encoded state 
     */
    std::vector<InferenceOutputs> RecurrentInference(std::vector<RecurrentInferenceInputs>& inputs);

    /**
     * Perform a learning step for the given batch
     * @param inputs Input batch (non-tensor'd)
     * @return Loss info, used for metric logging
     */
    LossInfo Learn(types::Batch& inputs);

    /**
     * Save the model to a checkpoint model
     * @param step The step version to checkpoint as
     * @return Full path string of the model saved, used for model syncing
     */
    std::string SaveCheckpoint(int step);

    /**
     * Load model from checkpoint
     * @param step Which step version of the checkpoint to load
     */
    void LoadCheckpoint(int step);

    /**
     * Load model from checkpoint
     * @param path Directory containing the checkpoint
     */
    void LoadCheckpoint(const std::string& path);

    /**
     * Get the string named device
     * @return torch string device
     */
    const std::string Device() const {
        return device_;
    }

private:
    std::string device_;                    // string device name for torch device
    std::string path_;                      // path to checkpoint model
    muzero_config::MuZeroConfig config_;    // muzero config parameterizing model
    model::MuzeroNetwork model_;            // Instance of muzero network
    torch::optim::Adam model_optimizer_;    // torch optimizer (conducts l2 regularization as part of loss)
    torch::Device torch_device_;            // torch version of the device to store model on
    util::ValueEncoder value_encoder_;      // Encoder/decoder for values
    util::ValueEncoder reward_encoder_;     // Encoder/decoder for rewards
    types::ObservationShape encoded_obs_shape_;    // Observation shape to reshape flat vector inputs
    int initial_flat_size_;                        // Size of flat input for initial inference
    int recurrent_flat_size_;                      // Size of flat input for recurrent inference
    int action_flat_size_;                         // Size of flat action representation
    double value_loss_weight_;    // Scale the value loss to avoid overfitting (See Appendix H Reanalyze)
};

}    // namespace muzero_cpp

#endif    // MUZERO_CPP_VPRNET_H_