#include "muzero-cpp/models.h"

#include <torch/torch.h>
#include <iostream>

#include "muzero-cpp/util.h"
#include "tests/test_macros.h"

namespace muzero_cpp {
using namespace types;
using namespace util;
namespace model {
namespace {

// ---------------------------- Representation Network ----------------------------

// Representation network without downsampling, check if we get expected output shape
void representation_network_test() {
    ObservationShape observation_shape{3, 96, 96};
    int batch_size = 4;
    int resnet_blocks = 4;
    int resnet_channels = 128;
    bool downsample = false;
    RepresentationNetwork representation_network(observation_shape.c, resnet_blocks, resnet_channels,
                                                 downsample);
    // Print for visual insepctions
    std::cout << representation_network << std::endl;
    // Ensure we get expected output shape
    torch::Tensor input =
        torch::rand({batch_size, observation_shape.c, observation_shape.h, observation_shape.w});
    torch::Tensor output = representation_network->forward(input);
    std::vector<int> expected_size{batch_size, resnet_channels, observation_shape.h, observation_shape.w};

    REQUIRE_EQUAL((int)expected_size.size(), (int)output.sizes().size());
    for (int i = 0; i < (int)expected_size.size(); ++i) {
        REQUIRE_EQUAL((int)expected_size[i], (int)output.size(i));
    }
    // Try to run backwards
    torch::Tensor loss = output.mean();
    loss.backward();
}

// Representation network with downsampling, check if we get expected output shape
void representation_network_downsample_test() {
    ObservationShape observation_shape{3, 96, 96};
    int batch_size = 4;
    int resnet_blocks = 4;
    int resnet_channels = 128;
    bool downsample = true;
    RepresentationNetwork representation_network(observation_shape.c, resnet_blocks, resnet_channels,
                                                 downsample);
    // Print for visual insepctions
    std::cout << representation_network << std::endl;
    // Ensure we get expected output shape
    torch::Tensor input =
        torch::rand({batch_size, observation_shape.c, observation_shape.h, observation_shape.w});
    torch::Tensor output = representation_network->forward(input);
    const std::vector<int> expected_size{batch_size, resnet_channels, (observation_shape.h + 16 - 1) / 16,
                                         (observation_shape.h + 16 - 1) / 16};

    REQUIRE_EQUAL((int)expected_size.size(), (int)output.sizes().size());
    for (int i = 0; i < (int)expected_size.size(); ++i) {
        REQUIRE_EQUAL((int)expected_size[i], (int)output.size(i));
    }
    // Try to run backwards
    torch::Tensor loss = output.mean();
    loss.backward();
}

// Check that the encoded state shape is expected
void representation_network_encoded_shape_test() {
    ObservationShape observation_shape{3, 96, 96};
    ObservationShape observation_shape_downsample{3, 6, 6};
    ObservationShape encoded_shape = RepresentationNetworkImpl::encoded_state_shape(observation_shape, false);
    ObservationShape encoded_shape_downsample =
        RepresentationNetworkImpl::encoded_state_shape(observation_shape, true);
    REQUIRE_TRUE(encoded_shape == observation_shape);
    REQUIRE_TRUE(encoded_shape_downsample == observation_shape_downsample);
}

// ---------------------------- Representation Network ----------------------------

// ---------------------------- Dynamics Network ----------------------------
// Check that the next encoded state and returned reward shape is expected
void dynamics_network_test() {
    ObservationShape encoded_state_shape{128, 20, 20};
    int action_channels = 2;
    int resnet_blocks = 4;
    int reward_reduced_channels = 2;
    int batch_size = 16;
    const std::vector<int> reward_head_layers{32, 32};
    int reward_support_size = 5;
    DynamicsNetwork dynamics_network(encoded_state_shape, action_channels, resnet_blocks,
                                     reward_reduced_channels, reward_head_layers, reward_support_size);
    // Print for visual insepctions
    std::cout << dynamics_network << std::endl;
    // Ensure we get expected output shape
    torch::Tensor input_obs =
        torch::rand({batch_size, encoded_state_shape.c, encoded_state_shape.h, encoded_state_shape.w});
    torch::Tensor input_action =
        torch::rand({batch_size, action_channels, encoded_state_shape.h, encoded_state_shape.w});
    torch::Tensor input = torch::cat({input_obs, input_action}, 1);
    // Concatenation is correct
    REQUIRE_TRUE(input.size(0) == batch_size && input.size(1) == encoded_state_shape.c + action_channels &&
                 input.size(2) == encoded_state_shape.h && input.size(3) == encoded_state_shape.w);

    DynamicsOutput output = dynamics_network->forward(input);
    // Check output action
    REQUIRE_EQUAL(output.reward.sizes().size(), 2);
    REQUIRE_TRUE(output.reward.size(0) == batch_size && output.reward.size(1) == reward_support_size);
    // Check output encoded_state
    REQUIRE_EQUAL(output.encoded_state.sizes().size(), 4);
    for (int i = 0; i < (int)input_obs.sizes().size(); ++i) {
        REQUIRE_EQUAL(input_obs.size(i), output.encoded_state.size(i));
    }
    // Try to run backwards
    torch::Tensor loss = output.reward.mean();
    loss.backward();
}

// ---------------------------- Dynamics Network ----------------------------

// ---------------------------- Prediction Network ----------------------------
// Check that the value and policy shape is expected
void prediction_network_test() {
    types::ObservationShape encoded_state_shape{128, 20, 20};
    int resnet_blocks = 4;
    int policy_reduced_channels = 32;
    int value_reduced_channels = 32;
    const std::vector<int> policy_head_layers{64, 64};
    const std::vector<int> value_head_layers{64, 64};
    int num_action_space = 4;
    int value_support_size = 5;
    int batch_size = 16;
    PredictionNetwork prediction_network(encoded_state_shape, resnet_blocks, policy_reduced_channels,
                                         value_reduced_channels, policy_head_layers, value_head_layers,
                                         num_action_space, value_support_size);
    // Print for visual insepctions
    std::cout << prediction_network << std::endl;
    // Ensure we get expected output shape
    torch::Tensor input =
        torch::rand({batch_size, encoded_state_shape.c, encoded_state_shape.h, encoded_state_shape.w});
    PredictionOutput output = prediction_network->forward(input);
    // Check output policy
    REQUIRE_EQUAL(output.policy.sizes().size(), 2);
    REQUIRE_TRUE(output.policy.size(0) == batch_size && output.policy.size(1) == num_action_space);
    // Check output value
    REQUIRE_EQUAL(output.value.sizes().size(), 2);
    REQUIRE_TRUE(output.value.size(0) == batch_size && output.value.size(1) == value_support_size);\
    // Try to run backwards
    torch::Tensor loss = output.policy.mean() + output.value.mean();
    loss.backward();
}
// ---------------------------- Prediction Network ----------------------------

// ---------------------------- MuZero Network ----------------------------
// Check for expected input for the entire muzero network
void muzero_network_test() {
    muzero_config::MuZeroConfig config;
    int batch_size = 16;
    config.network_config.normalize_hidden_states = true;
    config.use_contractive_mapping = true;
    config.min_reward = -1;
    config.max_reward = 10;
    config.min_value = -1;
    config.max_value = 100;
    config.observation_shape = {3, 20, 20};
    config.action_space = {0, 1, 2, 3, 4};
    config.stacked_observations = 4;
    config.action_channels = 2;
    config.network_config.representation_blocks = 4;
    config.network_config.dynamics_blocks = 4;
    config.network_config.prediction_blocks = 4;
    config.network_config.resnet_channels = 128;
    config.network_config.downsample = false;
    config.network_config.reward_reduced_channels = 32;
    config.network_config.value_reduced_channels = 32;
    config.network_config.policy_reduced_channels = 32;
    config.network_config.reward_head_layers = {64, 64};
    config.network_config.value_head_layers = {64, 64};
    config.network_config.policy_head_layers = {64, 64};
    MuzeroNetwork muzero_network(config);
    // Print for visual insepctions
    std::cout << muzero_network << std::endl;
    // Ensure we get expected output shape
    int stacked_channels = config.observation_shape.c +
                           (config.observation_shape.c * config.stacked_observations) +
                           (config.action_channels * config.stacked_observations);
    int reward_support_size =
        ValueEncoder::get_support_size(config.min_reward, config.max_reward, config.use_contractive_mapping);
    int value_support_size =
        ValueEncoder::get_support_size(config.min_value, config.max_value, config.use_contractive_mapping);
    torch::Tensor state =
        torch::rand({batch_size, stacked_channels, config.observation_shape.h, config.observation_shape.w});

    // Initial inference
    InferenceOutput init_inference_output = muzero_network->initial_inference(state);
    REQUIRE_EQUAL(init_inference_output.reward.sizes().size(), 2);
    REQUIRE_TRUE(init_inference_output.reward.size(0) == batch_size &&
                 init_inference_output.reward.size(1) == reward_support_size);
    REQUIRE_EQUAL(init_inference_output.value.sizes().size(), 2);
    REQUIRE_TRUE(init_inference_output.value.size(0) == batch_size &&
                 init_inference_output.value.size(1) == value_support_size);
    REQUIRE_EQUAL(init_inference_output.policy_logits.sizes().size(), 2);
    REQUIRE_TRUE(init_inference_output.policy_logits.size(0) == batch_size &&
                 init_inference_output.policy_logits.size(1) == (int)config.action_space.size());
    REQUIRE_EQUAL(init_inference_output.encoded_state.sizes().size(), 4);
    REQUIRE_TRUE(init_inference_output.encoded_state.size(0) == batch_size &&
                 init_inference_output.encoded_state.size(1) == config.network_config.resnet_channels &&
                 init_inference_output.encoded_state.size(2) == config.observation_shape.h &&
                 init_inference_output.encoded_state.size(3) == config.observation_shape.w);

    // Recurrent inference
    state = init_inference_output.encoded_state;
    torch::Tensor action = torch::rand({batch_size, config.action_channels, state.size(2), state.size(3)});
    InferenceOutput rec_inference_output = muzero_network->recurrent_inference(state, action);
    REQUIRE_EQUAL(rec_inference_output.reward.sizes().size(), 2);
    REQUIRE_TRUE(rec_inference_output.reward.size(0) == batch_size &&
                 rec_inference_output.reward.size(1) == reward_support_size);
    REQUIRE_EQUAL(rec_inference_output.value.sizes().size(), 2);
    REQUIRE_TRUE(rec_inference_output.value.size(0) == batch_size &&
                 rec_inference_output.value.size(1) == value_support_size);
    REQUIRE_EQUAL(rec_inference_output.policy_logits.sizes().size(), 2);
    REQUIRE_TRUE(rec_inference_output.policy_logits.size(0) == batch_size &&
                 rec_inference_output.policy_logits.size(1) == (int)config.action_space.size());
    REQUIRE_EQUAL(rec_inference_output.encoded_state.sizes().size(), 4);
    REQUIRE_TRUE(rec_inference_output.encoded_state.size(0) == batch_size &&
                 rec_inference_output.encoded_state.size(1) == config.network_config.resnet_channels &&
                 rec_inference_output.encoded_state.size(2) == config.observation_shape.h &&
                 rec_inference_output.encoded_state.size(3) == config.observation_shape.w);
}
// ---------------------------- MuZero Network ----------------------------

}    // namespace
}    // namespace model
}    // namespace muzero_cpp

int main() {
    muzero_cpp::model::representation_network_test();
    muzero_cpp::model::representation_network_downsample_test();
    muzero_cpp::model::representation_network_encoded_shape_test();
    muzero_cpp::model::dynamics_network_test();
    muzero_cpp::model::prediction_network_test();
    muzero_cpp::model::muzero_network_test();
}
