#include "models.h"

#include "value_encoder.h"

#include <cassert>

namespace muzero::model {

using namespace layers;

namespace {
// Scale hidden states to same range as activation input ([0,1])
// See MuZero Appendix G Training (https://arxiv.org/abs/1911.08265)
auto normalize_hidden_state(const torch::Tensor &encoded_state) -> torch::Tensor
{
    auto shape = encoded_state.sizes();
    torch::Tensor min_encoded_state =
        std::get<0>((encoded_state.view({-1, shape[1], shape[2] * shape[3]})).min(2, true)).unsqueeze(-1);
    torch::Tensor max_encoded_state =
        std::get<0>((encoded_state.view({-1, shape[1], shape[2] * shape[3]})).max(2, true)).unsqueeze(-1);
    torch::Tensor scale_encoded_state = max_encoded_state - min_encoded_state;
    scale_encoded_state.index_put_({scale_encoded_state < 1e-5}, 1e-5);    // Ensure no division by 0
    torch::Tensor encoded_state_normalized = (encoded_state - min_encoded_state) / scale_encoded_state;
    return encoded_state_normalized;
}
}    // namespace

// ------------------------- Representation Network -------------------------
// ResNet style representation network
RepresentationNetworkImpl::RepresentationNetworkImpl(
    int input_channels,     // Number of input channels (state observation)
    int resnet_blocks,      // Number of ResNet blocks
    int resnet_channels,    // Number of channels per ResNet Block
    bool downsample         // Flag whether to downsample (See the MuZero Appendix for details)
)
    : downsample_(downsample)
{
    // RedNet head
    if (downsample_) {
        resnet_head_->push_back(ResidualHeadDownsample(input_channels, resnet_channels));
    } else {
        resnet_head_->push_back(ResidualHead(input_channels, resnet_channels, "representation_"));
    }
    // ResNet body
    for (int i = 0; i < resnet_blocks; ++i) {
        resnet_layers_->push_back(ResidualBlock(resnet_channels, i));
    }
    register_module("representation_head", resnet_head_);
    register_module("representation_layers", resnet_layers_);
}

auto RepresentationNetworkImpl::forward(torch::Tensor x) -> torch::Tensor
{
    torch::Tensor output;
    // ResNet head (with optional downsampling)
    if (downsample_) {
        output = resnet_head_[0]->as<ResidualHeadDownsample>()->forward(x);
    } else {
        output = resnet_head_[0]->as<ResidualHead>()->forward(x);
    }
    // ResNet body
    for (int i = 0; i < (int)resnet_layers_->size(); ++i) {
        output = resnet_layers_[i]->as<ResidualBlock>()->forward(output);
    }
    return output;
}

// Get the observation shape the network outputs given the input
auto RepresentationNetworkImpl::encoded_state_shape(const ObservationShape &observation_shape, bool downsample)
    -> ObservationShape
{
    if (downsample) {
        return ResidualHeadDownsampleImpl::encoded_state_shape(observation_shape);
    } else {
        return ResidualHeadImpl::encoded_state_shape(observation_shape);
    }
}
// ------------------------- Representation Network -------------------------

// ---------------------------- Dynamics Network ----------------------------
// ResNet style Dynamics network
DynamicsNetworkImpl::DynamicsNetworkImpl(
    ObservationShape encoded_state_shape,          // Shape from representation function
    int action_channels,                           // Number of channels the encoded action plane
    int resnet_blocks,                             // Number of resnet blocks
    int reward_reduced_channels,                   // Number of reduced channels to use for conv1x1
                                                   // reward pass of encoded state
    const std::vector<int> &reward_head_layers,    // Layer sizes for reward head MLP
    int reward_support_size                        // encoded reward support size
)
    : reward_mlp_input_size_(reward_reduced_channels * encoded_state_shape.h * encoded_state_shape.w),
      resnet_head_(encoded_state_shape.c + action_channels, encoded_state_shape.c),
      conv1x1_reward_(conv1x1(encoded_state_shape.c, reward_reduced_channels)),
      reward_mlp_(
          reward_reduced_channels * encoded_state_shape.h * encoded_state_shape.w,
          reward_head_layers,
          reward_support_size,
          "dynamics_reward_"
      )
{
    // ResNet body
    for (int i = 0; i < resnet_blocks; ++i) {
        resnet_layers_->push_back(ResidualBlock(encoded_state_shape.c, i));
    }
    register_module("dynamics_resnet_head", resnet_head_);
    register_module("dynamics_resnet_body", resnet_layers_);
    register_module("dynamics_reward_conv", conv1x1_reward_);
    register_module("dynamics_reward_mlp", reward_mlp_);
}

// next_state, reward
auto DynamicsNetworkImpl::forward(torch::Tensor x) -> DynamicsOutput
{
    // ResNet to find next state
    torch::Tensor output = resnet_head_->forward(x);
    for (int i = 0; i < (int)resnet_layers_->size(); ++i) {
        output = resnet_layers_[i]->as<ResidualBlock>()->forward(output);
    }
    torch::Tensor next_state = output;
    // Get reward
    torch::Tensor reward = conv1x1_reward_->forward(output);
    reward = reward.view({-1, reward_mlp_input_size_});
    reward = reward_mlp_->forward(reward);
    return {.encoded_state = next_state, .reward = reward};
}
// ---------------------------- Dynamics Network ----------------------------

// --------------------------- Prediction Network ---------------------------
// AlphaZero style Prediction network
PredictionNetworkImpl::PredictionNetworkImpl(
    ObservationShape encoded_state_shape,          // Shape from representation function
    int resnet_blocks,                             // Number of resnet blocks
    int policy_reduced_channels,                   // Number of reduced channels to use for conv1x1
                                                   // policy pass of encoded state
    int value_reduced_channels,                    // Number of reduced channels to use for conv1x1
                                                   // value pass of encoded state
    const std::vector<int> &policy_head_layers,    // Layer sizes for policy head MLP
    const std::vector<int> &value_head_layers,     // Layer sizes for value head MLP
    int num_action_space,                          // Number of actions in action space
    int value_support_size                         // encoded value support size
)
    : policy_mlp_input_size_(policy_reduced_channels * encoded_state_shape.h * encoded_state_shape.w),
      value_mlp_input_size_(value_reduced_channels * encoded_state_shape.h * encoded_state_shape.w),
      conv1x1_policy_(conv1x1(encoded_state_shape.c, policy_reduced_channels)),
      conv1x1_value_(conv1x1(encoded_state_shape.c, value_reduced_channels)),
      policy_mlp_(
          policy_reduced_channels * encoded_state_shape.h * encoded_state_shape.w,
          policy_head_layers,
          num_action_space,
          "prediction_policy_"
      ),
      value_mlp_(
          value_reduced_channels * encoded_state_shape.h * encoded_state_shape.w,
          value_head_layers,
          value_support_size,
          "prediction_value_"
      )
{
    // ResNet body
    for (int i = 0; i < resnet_blocks; ++i) {
        resnet_layers_->push_back(ResidualBlock(encoded_state_shape.c, i));
    }
    register_module("prediction_resnet_body", resnet_layers_);
    register_module("prediction_policy_conv", conv1x1_policy_);
    register_module("prediction_value_conv", conv1x1_value_);
    register_module("prediction_policy_mlp", policy_mlp_);
    register_module("prediction_value_mlp", value_mlp_);
}

auto PredictionNetworkImpl::forward(torch::Tensor x) -> PredictionOutput
{
    // ResNet pass of encoded state
    for (int i = 0; i < (int)resnet_layers_->size(); ++i) {
        x = resnet_layers_[i]->as<ResidualBlock>()->forward(x);
    }
    // Get policy and value
    torch::Tensor policy_logits = conv1x1_policy_->forward(x);
    torch::Tensor value = conv1x1_value_->forward(x);
    policy_logits = policy_logits.view({-1, policy_mlp_input_size_});
    value = value.view({-1, value_mlp_input_size_});
    policy_logits = policy_mlp_->forward(policy_logits);
    value = value_mlp_->forward(value);
    return {.policy_logits = policy_logits, .value = value};
}
// --------------------------- Prediction Network ---------------------------

// ----------------------------- MuZero Network -----------------------------
// Muzero network which encapsulates representation, dynamics, and prediction networks
MuzeroNetworkImpl::MuzeroNetworkImpl(const Config &config)
    : normalize_hidden_state_(
          config.model_config.normalize_hidden_states
      ),    // Flag to normalize the hidden states to range [0, 1]
      reward_encoded_size_(
          ValueEncoder::get_support_size(
              config.min_reward,
              config.max_reward,
              config.use_contractive_mapping
          )
      ),    // Size of the reward encoding (see Appendix F Network
            // Architecture)
      value_encoded_size_(
          ValueEncoder::get_support_size(
              config.min_value,
              config.max_value,
              config.use_contractive_mapping
          )
      ),    // Size of the value encoding (see Appendix F Network
            // Architecture)
      initial_inference_channels_(
          (config.observation_shape.c * (config.stacked_observations + 1))
          + (config.stacked_observations * config.action_channels)
      ),    // Number of channels for initial inference (combination of stacked
            // observations + actions)
      encoded_observation_shape_(
          RepresentationNetworkImpl::encoded_state_shape(
              {.c = config.model_config.resnet_channels, .h = config.observation_shape.h, .w = config.observation_shape.w},
              config.model_config.downsample
          )
      ),    // Input shape to dynamics/prediction network
      representation_network_(
          initial_inference_channels_,
          config.model_config.representation_blocks,
          config.model_config.resnet_channels,
          config.model_config.downsample
      ),    // Internal representation network
      dynamics_network_(
          encoded_observation_shape_,
          config.action_channels,
          config.model_config.dynamics_blocks,
          config.model_config.reward_reduced_channels,
          config.model_config.reward_head_layers,
          reward_encoded_size_
      ),    // Internal dynamics network
      prediction_network_(
          encoded_observation_shape_,
          config.model_config.prediction_blocks,
          config.model_config.policy_reduced_channels,
          config.model_config.value_reduced_channels,
          config.model_config.policy_head_layers,
          config.model_config.value_head_layers,
          config.action_space.size(),
          value_encoded_size_
      )
{    // Internal prediction network
    register_module("representation_network", representation_network_);
    register_module("dynamics_network", dynamics_network_);
    register_module("prediction_network", prediction_network_);
}

// Get the number of required input channels for initial inference
// This the number of channels for the current observation + the number of channels of previous stacked *
// observations and actions
auto MuzeroNetworkImpl::get_initial_inference_channels() const -> int
{
    return initial_inference_channels_;
};

// Get the encoded observation shape the network expects
auto MuzeroNetworkImpl::get_encoded_observation_shape() const -> ObservationShape
{
    return encoded_observation_shape_;
}

// Get an encoded representation of the input observaton
auto MuzeroNetworkImpl::representation(torch::Tensor observation) -> torch::Tensor
{
    torch::Tensor encoded_state = representation_network_->forward(observation);
    // Scale hidden states to same range as activation input ([0,1])
    // See MuZero Appendix G Training (https://arxiv.org/abs/1911.08265)
    if (normalize_hidden_state_) {
        encoded_state = normalize_hidden_state(encoded_state);
    }
    return encoded_state;
}

// Use the dynamics network to get predictions for state after applying given action
auto MuzeroNetworkImpl::dynamics(torch::Tensor encoded_state, torch::Tensor action) -> DynamicsOutput
{
    torch::Tensor x = torch::cat({encoded_state, action}, 1);
    DynamicsOutput dynamics_output = dynamics_network_->forward(x);
    // Scale hidden states to same range as activation input ([0,1])
    // See MuZero Appendix G Training (https://arxiv.org/abs/1911.08265)
    if (normalize_hidden_state_) {
        dynamics_output.encoded_state = normalize_hidden_state(dynamics_output.encoded_state);
    }
    return dynamics_output;
}

// Use the prediction network to get value/policy predictions
PredictionOutput MuzeroNetworkImpl::prediction(torch::Tensor encoded_state)
{
    return prediction_network_->forward(encoded_state);
}

// Perform initial inference
auto MuzeroNetworkImpl::initial_inference(torch::Tensor observation) -> InferenceOutput
{
    torch::Tensor encoded_state = representation(observation);
    PredictionOutput pred_output = prediction(encoded_state);
    // Reward never used for initial reference, but ensure reward is 0 after decoding for consistency
    torch::Tensor reward = torch::log(
        torch::zeros({1, reward_encoded_size_})
            .scatter(1, torch::tensor({{reward_encoded_size_ / 2}}).to(torch::kLong), 1)
            .repeat({observation.size(0), 1})
            .to(observation.device())
    );
    return {
        .value = pred_output.value,
        .reward = reward,
        .policy_logits = pred_output.policy_logits,
        .encoded_state = encoded_state
    };
}

// Perform recurrent inference
auto MuzeroNetworkImpl::recurrent_inference(torch::Tensor encoded_state, torch::Tensor action) -> InferenceOutput
{
    DynamicsOutput dynamics_output = dynamics(encoded_state, action);
    PredictionOutput pred_output = prediction(dynamics_output.encoded_state);
    return {
        .value = pred_output.value,
        .reward = dynamics_output.reward,
        .policy_logits = pred_output.policy_logits,
        .encoded_state = dynamics_output.encoded_state
    };
}

// Compute the muzero network loss
auto MuzeroNetworkImpl::loss(
    torch::Tensor value,
    torch::Tensor reward,
    torch::Tensor policy_logits,
    torch::Tensor target_value,
    torch::Tensor target_reward,
    torch::Tensor target_policy
) -> LossOutput
{
    // Inidividual cross-entropy loss + l2 (handled by optimizer)
    torch::Tensor value_loss = torch::sum(-target_value * torch::log_softmax(value, 1), -1);
    torch::Tensor reward_loss = torch::sum(-target_reward * torch::log_softmax(reward, 1), -1);
    torch::Tensor policy_loss = torch::sum(-target_policy * torch::log_softmax(policy_logits, 1), -1);
    return {.value_loss = value_loss, .reward_loss = reward_loss, .policy_loss = policy_loss};
}
// ----------------------------- MuZero Network -----------------------------

}    // namespace muzero::model
