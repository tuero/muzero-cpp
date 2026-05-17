#ifndef MUZERO_MODEL_LAYERS_H_
#define MUZERO_MODEL_LAYERS_H_

#include <muzero/types.h>

#include <torch/torch.h>

#include <string>
#include <vector>

namespace muzero::model::layers {

// Conv and pooling layer defaults
auto conv1x1(int in_channels, int out_channels) -> torch::nn::Conv2dOptions;
auto conv3x3(int in_channels, int out_channels, int stride = 1, int padding = 1, bool bias = false)
    -> torch::nn::Conv2dOptions;
auto avg_pool3x3(int kernel_size, int stride, int padding) -> torch::nn::AvgPool2dOptions;
auto bn(int num_filters) -> torch::nn::BatchNorm2dOptions;

// MLP
class MLPImpl : public torch::nn::Module {
public:
    /**
     * @param input_size Size of the input layer
     * @param layer_sizes Vector of sizes for each hidden layer
     * @param output_size Size of the output layer
     */
    MLPImpl(int input_size, const std::vector<int> &layer_sizes, int output_size, const std::string &name);
    auto forward(torch::Tensor x) -> torch::Tensor;

private:
    torch::nn::Sequential layers;
};
TORCH_MODULE(MLP);

// Main ResNet style residual block
class ResidualBlockImpl : public torch::nn::Module {
public:
    /**
     * @param num_channels Number of channels for the resnet block
     * @param layer_num Layer number id, used for pretty printing
     */
    ResidualBlockImpl(int num_channels, int layer_num);
    auto forward(torch::Tensor x) -> torch::Tensor;

private:
    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::BatchNorm2d batch_norm1;
    torch::nn::BatchNorm2d batch_norm2;
};
TORCH_MODULE(ResidualBlock);

/**
 * Initial input convolutional before ResNet residual blocks
 * Primary use is to take N channels and set to the expected number
 *  of channels for the rest of the resnet body
 */
class ResidualHeadImpl : public torch::nn::Module {
public:
    /**
     * @param input_channels Number of channels the head of the network receives
     * @param output_channels Number of output channels, should match the number of
     *                        channels used for the resnet body
     * @param name_prefix Used to ID the sub-module for pretty printing
     */
    ResidualHeadImpl(int input_channels, int output_channels, const std::string &name_prefix = "");
    auto forward(torch::Tensor x) -> torch::Tensor;
    // Get the observation shape the network outputs given the input
    static ObservationShape encoded_state_shape(ObservationShape observation_shape);

private:
    torch::nn::Conv2d conv;
    torch::nn::BatchNorm2d batch_norm;
};
TORCH_MODULE(ResidualHead);

/**
 * Downsampling network for image observations (MuZero Appendix)
 * Same as the above, but with additional downsampling for image style input
 * @note Result is a 16x deduction in both width/height dimensions
 *       (2x conv of stride 2, 2x pooling of stride 2)
 */
class ResidualHeadDownsampleImpl : public torch::nn::Module {
public:
    /**
     * @param input_channels Number of channels the head of the network receives
     * @param output_channels Number of output channels, should match the number of
     *                        channels used for the resnet body
     * @param name_prefix Used to ID the sub-module for pretty printing
     */
    ResidualHeadDownsampleImpl(int input_channels, int output_channels, const std::string &name_prefix = "");
    auto forward(torch::Tensor x) -> torch::Tensor;
    // Get the observation shape the network outputs given the input
    static ObservationShape encoded_state_shape(ObservationShape observation_shape);

private:
    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::AvgPool2d pooling1;
    torch::nn::AvgPool2d pooling2;
    torch::nn::ModuleList resblocks1;
    torch::nn::ModuleList resblocks2;
    torch::nn::ModuleList resblocks3;
};
TORCH_MODULE(ResidualHeadDownsample);

}    // namespace muzero::model::layers

#endif    // MUZERO_MODEL_LAYERS_H_
