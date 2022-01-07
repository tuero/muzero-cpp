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

#ifndef MUZERO_CPP_VPRNET_EVALUATOR_H_
#define MUZERO_CPP_VPRNET_EVALUATOR_H_

#include <future>
#include <thread>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "muzero-cpp/device_manager.h"
#include "muzero-cpp/queue.h"
#include "muzero-cpp/types.h"
#include "muzero-cpp/util.h"
#include "muzero-cpp/vprnet.h"

namespace muzero_cpp {

// Modified evaluator class from open_spiel
// Essentially this acts as an interface for actors/evaluators to send queries (batched or not) to the model
class Evaluator {
public:
    virtual VPRNetModel::InferenceOutputs InitialInference(
        const types::Observation& stacked_observations) = 0;
    virtual VPRNetModel::InferenceOutputs RecurrentInference(const types::Observation& hidden_state,
                                                             types::Action action) = 0;
};

// Handles queries for the VPR model
class VPRNetEvaluator : public Evaluator {
public:
    /**
     * @param device_manager Pointer to device manager (holds the VPR models on devices)
     * @param batch_size_initial Batch size for initial inference queries
     * @param initial_inference_threads Number of threads running initial inference queries
     * @param batch_size_initial Batch size for recurrent inference queries
     * @param recurrent_inference_threads Number of threads running recurrent inference queries
     */
    explicit VPRNetEvaluator(DeviceManager* device_manager, int batch_size_initial,
                             int initial_inference_threads, int batch_size_recurrent,
                             int recurrent_inference_threads);
    ~VPRNetEvaluator();

    /**
     * Perform initial inference
     * @param stacked_observations The current observation + stacked historical observations + actions
     * @return The value, reward, policy, and encoded state during initial inference
     */
    VPRNetModel::InferenceOutputs InitialInference(const types::Observation& stacked_observations) override;

    /**
     * Perform recurrent inference
     * @param stacked_observations The current observation + stacked historical observations + actions
     * @param action The action to take
     * @return The value, reward, policy, and encoded state during initial inference
     */
    VPRNetModel::InferenceOutputs RecurrentInference(const types::Observation& hidden_state,
                                                     types::Action action) override;

private:
    // Runner to perform initial inference queries
    void InitialInferenceRunner();
    // Runner to perform recurrent inference queries
    void RecurrentInferenceRunner();

    DeviceManager& device_manager_;     // Reference to a device manager which holds the models on device
    const int batch_size_initial_;      // Batch size for initial inference
    const int batch_size_recurrent_;    // Batch size for recurrent inference

    // Struct for holding promised value for initial inference queries
    struct QueueItemInitial {
        VPRNetModel::InitialInferenceInputs inputs;
        std::promise<VPRNetModel::InferenceOutputs>* prom;
    };
    // Struct for holding promised value for recurrent inference queries
    struct QueueItemRecurrent {
        VPRNetModel::RecurrentInferenceInputs inputs;
        std::promise<VPRNetModel::InferenceOutputs>* prom;
    };

    ThreadedQueue<QueueItemInitial> queue_initial_;           // Queue for initial inference requests
    ThreadedQueue<QueueItemRecurrent> queue_recurrent_;       // Queue for recurrent inference requests
    util::StopToken stop_token_;                              // Stop token flag to signal to quit
    std::vector<std::thread> initial_inference_threads_;      // Threads for initial inference requests
    std::vector<std::thread> recurrent_inference_threads_;    // Threads for recurrent inference requests
    absl::Mutex initial_inference_queue_m_;                   // Mutex for the initial inference queue
    absl::Mutex recurrent_inference_queue_m_;                 // Mutex for the recurrent inference queue
};

}    // namespace muzero_cpp

#endif    // MUZERO_CPP_VPRNET_EVALUATOR_H_