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

#include "vprnet_evaluator.h"

#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

namespace muzero_cpp {
using namespace types;

VPRNetEvaluator::VPRNetEvaluator(
    DeviceManager* device_manager,     // holds the VPR models on devices
    int batch_size_initial,            // Batch size for initial inference queries
    int initial_inference_threads,     // Number of threads running initial inference queries
    int batch_size_recurrent,          // Batch size for recurrent inference queries
    int recurrent_inference_threads    // Number of threads running recurrent inference queries
    )
    : device_manager_(*device_manager),
      batch_size_initial_(batch_size_initial),
      batch_size_recurrent_(batch_size_recurrent),
      queue_initial_(batch_size_initial * initial_inference_threads * 4),
      queue_recurrent_(batch_size_recurrent * recurrent_inference_threads * 4) {
    // Don't bother spawning runners to batch requests if batch size <= 1.
    if (batch_size_initial_ <= 1) { initial_inference_threads = 0; }
    if (batch_size_recurrent_ <= 1) { recurrent_inference_threads = 0; }

    // Reserve space for each type of inference
    initial_inference_threads_.reserve(initial_inference_threads);
    for (int i = 0; i < initial_inference_threads; ++i) {
        initial_inference_threads_.emplace_back([this]() { this->InitialInferenceRunner(); });
    }
    recurrent_inference_threads_.reserve(recurrent_inference_threads);
    for (int i = 0; i < initial_inference_threads; ++i) {
        recurrent_inference_threads_.emplace_back([this]() { this->RecurrentInferenceRunner(); });
    }
}

// Ensure threads are closed
VPRNetEvaluator::~VPRNetEvaluator() {
    stop_token_.stop();
    // Clear the incoming queues
    queue_initial_.BlockNewValues();
    queue_recurrent_.BlockNewValues();
    queue_initial_.Clear();
    queue_recurrent_.Clear();
    // Stop the oustanding threads
    for (auto& t : initial_inference_threads_) {
        t.join();
    }
    for (auto& t : recurrent_inference_threads_) {
        t.join();
    }
}

// Perform initial inference
VPRNetModel::InferenceOutputs VPRNetEvaluator::InitialInference(const Observation& stacked_observations) {
    VPRNetModel::InitialInferenceInputs inputs = {stacked_observations};
    VPRNetModel::InferenceOutputs outputs;
    // Get output directly, or insert into queue and wait for batched input to be sent back
    if (batch_size_initial_ <= 1) {
        std::vector inference_inputs{inputs};
        outputs = device_manager_.Get(1)->InitialInference(inference_inputs)[0];
    } else {
        std::promise<VPRNetModel::InferenceOutputs> prom;
        std::future<VPRNetModel::InferenceOutputs> fut = prom.get_future();
        queue_initial_.Push(QueueItemInitial{inputs, &prom});
        outputs = fut.get();
    }
    return outputs;
}

// Perform recurrent inference
VPRNetModel::InferenceOutputs VPRNetEvaluator::RecurrentInference(const Observation& hidden_state,
                                                                  Action action) {
    VPRNetModel::RecurrentInferenceInputs inputs = {action, hidden_state};
    VPRNetModel::InferenceOutputs outputs;
    // Get output directly, or insert into queue and wait for batched input to be sent back
    if (batch_size_recurrent_ <= 1) {
        std::vector inference_inputs{inputs};
        outputs = device_manager_.Get(1)->RecurrentInference(inference_inputs)[0];
    } else {
        std::promise<VPRNetModel::InferenceOutputs> prom;
        std::future<VPRNetModel::InferenceOutputs> fut = prom.get_future();
        queue_recurrent_.Push(QueueItemRecurrent{inputs, &prom});
        outputs = fut.get();
    }
    return outputs;
}

// // Runner to perform initial inference queries
void VPRNetEvaluator::InitialInferenceRunner() {
    std::vector<VPRNetModel::InitialInferenceInputs> inputs;
    std::vector<std::promise<VPRNetModel::InferenceOutputs>*> promises;
    inputs.reserve(batch_size_initial_);
    promises.reserve(batch_size_initial_);
    while (!stop_token_.stop_requested()) {
        {
            // Only one thread at a time should be listening to the queue to maximize
            // batch size and minimize latency.
            absl::MutexLock lock(&initial_inference_queue_m_);
            absl::Time deadline = absl::InfiniteFuture();
            for (int i = 0; i < batch_size_initial_; ++i) {
                absl::optional<QueueItemInitial> item = queue_initial_.Pop(deadline);
                if (!item) {    // Hit the deadline.
                    break;
                }
                if (inputs.empty()) { deadline = absl::Now() + absl::Milliseconds(1); }
                inputs.push_back(item->inputs);
                promises.push_back(item->prom);
            }
        }

        // Almost certainly StopRequested.
        if (inputs.empty()) { continue; }

        // Get initial inference outputs from input batch
        std::vector<VPRNetModel::InferenceOutputs> outputs =
            device_manager_.Get(inputs.size())->InitialInference(inputs);
        // Send result to waiting promise/future connections
        for (int i = 0; i < (int)promises.size(); ++i) {
            promises[i]->set_value(outputs[i]);
        }
        inputs.clear();
        promises.clear();
    }
}

// Runner to perform recurrent inference queries
void VPRNetEvaluator::RecurrentInferenceRunner() {
    std::vector<VPRNetModel::RecurrentInferenceInputs> inputs;
    std::vector<std::promise<VPRNetModel::InferenceOutputs>*> promises;
    inputs.reserve(batch_size_recurrent_);
    promises.reserve(batch_size_recurrent_);
    while (!stop_token_.stop_requested()) {
        {
            // Only one thread at a time should be listening to the queue to maximize
            // batch size and minimize latency.
            absl::MutexLock lock(&recurrent_inference_queue_m_);
            absl::Time deadline = absl::InfiniteFuture();
            for (int i = 0; i < batch_size_recurrent_; ++i) {
                absl::optional<QueueItemRecurrent> item = queue_recurrent_.Pop(deadline);
                if (!item) {    // Hit the deadline.
                    break;
                }
                if (inputs.empty()) { deadline = absl::Now() + absl::Milliseconds(1); }
                inputs.push_back(item->inputs);
                promises.push_back(item->prom);
            }
        }

        // Almost certainly StopRequested.
        if (inputs.empty()) { continue; }

        // Get recurrent inference outputs from input batch
        std::vector<VPRNetModel::InferenceOutputs> outputs =
            device_manager_.Get(inputs.size())->RecurrentInference(inputs);
        // Send result to waiting promise/future connections
        for (int i = 0; i < (int)promises.size(); ++i) {
            promises[i]->set_value(outputs[i]);
        }
        inputs.clear();
        promises.clear();
    }
}

}    // namespace muzero_cpp
