#include "muzero-cpp/model_layers.h"

#include <cassert>

namespace muzero_cpp {
namespace model_layers {

// Create a conv1x1 layer using pytorch defaults
torch::nn::Conv2dOptions conv1x1(int in_channels, int out_channels) {
    return torch::nn::Conv2dOptions(in_channels, out_channels, 1)
        .stride(1)
        .padding(0)
        .bias(true)
        .dilation(1)
        .groups(1)
        .padding_mode(torch::kZeros);
}

// Create a conv3x3 layer using pytorch defaults
torch::nn::Conv2dOptions conv3x3(int in_channels, int out_channels, int stride, int padding, bool bias) {
    return torch::nn::Conv2dOptions(in_channels, out_channels, 3)
        .stride(stride)
        .padding(padding)
        .bias(bias)
        .dilation(1)
        .groups(1)
        .padding_mode(torch::kZeros);
}

// Create a avgerage pooling layer using pytorch defaults
torch::nn::AvgPool2dOptions avg_pool3x3(int kernel_size, int stride, int padding) {
    return torch::nn::AvgPool2dOptions(kernel_size).stride(stride).padding(padding);
}

// Create a batchnorm2d layer using pytorch defaults
torch::nn::BatchNorm2dOptions bn(int num_filters) {
    return torch::nn::BatchNorm2dOptions(num_filters)
        .eps(0.0001)
        .momentum(0.01)
        .affine(true)
        .track_running_stats(true);
}

// ------------------------------- MLP Network ------------------------------
// MLP
MLPImpl::MLPImpl(int input_size, const std::vector<int> &layer_sizes, int output_size,
                 const std::string &name) {
    std::vector<int> sizes = layer_sizes;
    sizes.insert(sizes.begin(), input_size);
    sizes.push_back(output_size);

    // Walk through adding layers
    for (int i = 0; i < (int)sizes.size() - 1; ++i) {
        layers->push_back("linear_" + std::to_string(i), torch::nn::Linear(sizes[i], sizes[i + 1]));
        if (i < (int)sizes.size() - 2) {
            layers->push_back("activation_" + std::to_string(i), torch::nn::ReLU());
        }
    }
    register_module(name + "mlp", layers);
}

torch::Tensor MLPImpl::forward(torch::Tensor x) {
    torch::Tensor output = layers->forward(x);
    return output;
}
// ------------------------------- MLP Network ------------------------------

// ------------------------------ ResNet Block ------------------------------
// Main ResNet style residual block
ResidualBlockImpl::ResidualBlockImpl(int num_channels, int layer_num)
    : conv1(conv3x3(num_channels, num_channels)),
      conv2(conv3x3(num_channels, num_channels)),
      batch_norm1(bn(num_channels)),
      batch_norm2(bn(num_channels)) {
    register_module("resnet_" + std::to_string(layer_num) + "_conv1", conv1);
    register_module("resnet_" + std::to_string(layer_num) + "_conv2", conv2);
    register_module("resnet_" + std::to_string(layer_num) + "_bn1", batch_norm1);
    register_module("resnet_" + std::to_string(layer_num) + "_bn2", batch_norm2);
}

torch::Tensor ResidualBlockImpl::forward(torch::Tensor x) {
    torch::Tensor residual = x.clone();
    torch::Tensor output = torch::relu(batch_norm1(conv1(x)));
    output = batch_norm2(conv2(output));
    output += residual;
    output = torch::relu(output);
    return output;
}
// ------------------------------ ResNet Block ------------------------------

// ------------------------------ ResNet Head -------------------------------
// Initial input convolutional before ResNet residual blocks
// Primary use is to take N channels and set to the expected number
//   of channels for the rest of the resnet body
ResidualHeadImpl::ResidualHeadImpl(int input_channels, int output_channels, const std::string &name_prefix)
    : conv(conv3x3(input_channels, output_channels)), batch_norm(bn(output_channels)) {
    register_module(name_prefix + "resnet_head_conv", conv);
    register_module(name_prefix + "resnet_head_bn", batch_norm);
}

torch::Tensor ResidualHeadImpl::forward(torch::Tensor x) {
    torch::Tensor output = torch::relu(batch_norm(conv(x)));
    return output;
}

// Shape doesn't change
types::ObservationShape ResidualHeadImpl::encoded_state_shape(types::ObservationShape observation_shape) {
    return observation_shape;
}
// ------------------------------ ResNet Head -------------------------------

// ---------------------------- Downsample Head -----------------------------
// Downsampling network for image observations (MuZero Appendix)
// Result is a 16x deduction in both width/height dimensions
//   (2x conv of stride 2, 2x pooling of stride 2)
ResidualHeadDownsampleImpl::ResidualHeadDownsampleImpl(int input_channels, int output_channels,
                                                       const std::string &name_prefix)
    : conv1(conv3x3(input_channels, output_channels / 2, 2, 1, false)),
      conv2(conv3x3(output_channels / 2, output_channels, 2, 1, false)),
      pooling1(avg_pool3x3(3, 2, 1)),
      pooling2(avg_pool3x3(3, 2, 1)) {
    for (int i = 0; i < 2; ++i) {
        resblocks1->push_back(ResidualBlock(output_channels / 2, i));
    }
    for (int i = 2; i < 5; ++i) {
        resblocks2->push_back(ResidualBlock(output_channels, i));
    }
    for (int i = 5; i < 8; ++i) {
        resblocks3->push_back(ResidualBlock(output_channels, i));
    }
    register_module(name_prefix + "resnet_head_downsample_conv1", conv1);
    register_module(name_prefix + "resnet_head_downsample_conv2", conv2);
    register_module(name_prefix + "resnet_head_downsample_pooling1", pooling1);
    register_module(name_prefix + "resnet_head_downsample_pooling2", pooling2);
    register_module(name_prefix + "resnet_head_downsample_resblock1", resblocks1);
    register_module(name_prefix + "resnet_head_downsample_resblock2", resblocks2);
    register_module(name_prefix + "resnet_head_downsample_resblock3", resblocks3);
}

torch::Tensor ResidualHeadDownsampleImpl::forward(torch::Tensor x) {
    torch::Tensor output = conv1(x);
    for (int i = 0; i < (int)resblocks1->size(); ++i) {
        output = resblocks1[i]->as<ResidualBlock>()->forward(output);
    }
    output = conv2(output);
    for (int i = 0; i < (int)resblocks2->size(); ++i) {
        output = resblocks2[i]->as<ResidualBlock>()->forward(output);
    }
    output = pooling1(output);
    for (int i = 0; i < (int)resblocks3->size(); ++i) {
        output = resblocks3[i]->as<ResidualBlock>()->forward(output);
    }
    output = pooling2(output);
    return output;
}

types::ObservationShape ResidualHeadDownsampleImpl::encoded_state_shape(
    types::ObservationShape observation_shape) {
    observation_shape.h = (observation_shape.h + 16 - 1) / 16;
    observation_shape.w = (observation_shape.w + 16 - 1) / 16;
    return observation_shape;
}

// ---------------------------- Downsample Head -----------------------------

}    // namespace model_layers
}    // namespace muzero_cpp