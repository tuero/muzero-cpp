#ifndef MUZERO_CPP_MODELS_H_
#define MUZERO_CPP_MODELS_H_

#include <torch/torch.h>

#include <string>
#include <vector>

#include "muzero-cpp/config.h"
#include "muzero-cpp/model_layers.h"
#include "muzero-cpp/types.h"

namespace muzero_cpp {
namespace model {

// Network output for inference
struct InferenceOutput {
    torch::Tensor value;
    torch::Tensor reward;
    torch::Tensor policy_logits;
    torch::Tensor encoded_state;
};

// Network output for the policy/value prediction
struct PredictionOutput {
    torch::Tensor policy;
    torch::Tensor value;
};

// Network output for the dynamics function
struct DynamicsOutput {
    torch::Tensor encoded_state;
    torch::Tensor reward;
};

// Loss information
struct LossOutput {
    torch::Tensor value_loss;
    torch::Tensor reward_loss;
    torch::Tensor policy_loss;
};

// ResNet style representation network
class RepresentationNetworkImpl : public torch::nn::Module {
public:
    RepresentationNetworkImpl(
        int input_channels,     // Number of input channels (state observation)
        int resnet_blocks,      // Number of ResNet blocks
        int resnet_channels,    // Number of channels per ResNet Block
        bool downsample         // Flag whether to downsample (See the MuZero Appendix for details)
    );
    torch::Tensor forward(torch::Tensor x);
    // Get the observation shape the network outputs given the input
    static types::ObservationShape encoded_state_shape(types::ObservationShape observation_shape,
                                                       bool downsample);

private:
    bool downsample_;                        // Flag to downsample the initial inference input
    torch::nn::ModuleList resnet_head_;      // Could be ResidualHead or ResidualHeadDownsample
    torch::nn::ModuleList resnet_layers_;    // Resnet body layers for the representation network
};
TORCH_MODULE(RepresentationNetwork);

// ResNet style Dynamics network
class DynamicsNetworkImpl : public torch::nn::Module {
public:
    DynamicsNetworkImpl(types::ObservationShape encoded_state_shape,    // Shape from representation function
                        int action_channels,            // Number of channels the encoded action plane
                        int resnet_blocks,              // Number of resnet blocks
                        int reward_reduced_channels,    // Number of reduced channels to use for conv1x1
                                                        // reward pass of encoded state
                        const std::vector<int> &reward_head_layers,    // Layer sizes for reward head MLP
                        int reward_support_size                        // encoded reward support size
    );
    DynamicsOutput forward(torch::Tensor x);

private:
    int reward_mlp_input_size_;                 // Flat input size for reward sub-module
    model_layers::ResidualHead resnet_head_;    // Resnet head
    torch::nn::Conv2d conv1x1_reward_;          // Conv pass before passing to reward mlp
    model_layers::MLP reward_mlp_;              // MLP network for predicted reward
    torch::nn::ModuleList resnet_layers_;       // Resnet body layers for the dynamics network
};
TORCH_MODULE(DynamicsNetwork);

// AlphaZero style Prediction network
class PredictionNetworkImpl : public torch::nn::Module {
public:
    PredictionNetworkImpl(
        types::ObservationShape encoded_state_shape,    // Shape from representation function
        int resnet_blocks,                              // Number of resnet blocks
        int policy_reduced_channels,                    // Number of reduced channels to use for conv1x1
                                                        // policy pass of encoded state
        int value_reduced_channels,                     // Number of reduced channels to use for conv1x1
                                                        // value pass of encoded state
        const std::vector<int> &policy_head_layers,     // Layer sizes for policy head MLP
        const std::vector<int> &value_head_layers,      // Layer sizes for value head MLP
        int num_action_space,                           // Number of actions in action space
        int value_support_size                          // encoded value support size
    );
    PredictionOutput forward(torch::Tensor x);

private:
    int policy_mlp_input_size_;              // Flat input size for policy sub-module
    int value_mlp_input_size_;               // Flat input size for value sub-module
    torch::nn::Conv2d conv1x1_policy_;       // Conv pass before passing to policy mlp
    torch::nn::Conv2d conv1x1_value_;        // Conv pass before passing to value mlp
    model_layers::MLP policy_mlp_;           // MLP network for predicted policy
    model_layers::MLP value_mlp_;            // MLP network for predicted value
    torch::nn::ModuleList resnet_layers_;    // Shared Resnet body layers for the prediction network
};
TORCH_MODULE(PredictionNetwork);

// Muzero network which encapsulates representation, dynamics, and prediction networks
class MuzeroNetworkImpl : public torch::nn::Module {
public:
    MuzeroNetworkImpl(const muzero_config::MuZeroConfig &config);

    /**
     * Perform initial inference
     * @param observation The raw flat observatioon from the environment
     * @return The predicted value, reward, policy, and the encoded representation of the observation
     */
    InferenceOutput initial_inference(torch::Tensor observation);

    /**
     * Perform recurrent inference
     * @param encoded_state An encoded state from a previous initial or recurrent inference
     * @param action The action to apply in the given encoded state
     * @return The predicted value, reward, policy, and the encoded representation
     *         of the next observation
     */
    InferenceOutput recurrent_inference(torch::Tensor encoded_state, torch::Tensor action);

    /**
     * Compute the muzero network loss
     * @note This uses the l^r, l^v, and l^p reward (See Section 3 MuZero Algorithm) except for L2 norm, as
     * that is handled by the PyTorch optimizer
     * @param value Predicted root value of observation
     * @param reward Predicted reward for transition
     * @param policy_logits Predicted policy at root
     * @param target_value Target value from sample return
     * @param target_reward Target reward from sample return
     * @param target_policy Target policy from search statistics
     * @return Non-scaled losses for the value, reward, and the policy
     */
    LossOutput loss(torch::Tensor value, torch::Tensor reward, torch::Tensor policy_logits,
                    torch::Tensor target_value, torch::Tensor target_reward, torch::Tensor target_policy);

    /**
     * Get the number of required input channels for initial inference
     * This the number of channels for the current observation + the number of channels of previous stacked
     * observations and actions
     * @return The number of channles for initial inference
     */
    int get_initial_inference_channels() const;

    /**
     * Get the encoded observation shape the network expects
     * @return The encoded observation shape the network expects
     */
    types::ObservationShape get_encoded_observation_shape() const;

private:
    // Get an encoded representation of the input observaton
    torch::Tensor representation(torch::Tensor observation);
    // Use the dynamics network to get predictions for state after applying given action
    DynamicsOutput dynamics(torch::Tensor encoded_state, torch::Tensor action);
    // Use the prediction network to get value/policy predictions
    PredictionOutput prediction(torch::Tensor encoded_state);

    bool normalize_hidden_state_;       // Flag to normalize the hidden states to range [0, 1]
    int reward_encoded_size_;           // Size of the reward encoding (see Appendix F Network Architecture)
    int value_encoded_size_;            // Size of the value encoding (see Appendix F Network Architecture)
    int initial_inference_channels_;    // Number of channels for initial inference (combination of stacked
                                        // observations + actions)
    types::ObservationShape encoded_observation_shape_;    // Input shape to dynamics/prediction network
    RepresentationNetwork representation_network_;         // Internal representation network
    DynamicsNetwork dynamics_network_;                     // Internal dynamics network
    PredictionNetwork prediction_network_;                 // Internal prediction network
};
TORCH_MODULE(MuzeroNetwork);

}    // namespace model
}    // namespace muzero_cpp

#endif    // MUZERO_CPP_MODELS_H_