#include "muzero-cpp/model_layers.h"

#include <torch/torch.h>

#include <iostream>
#include <vector>

#include "tests/test_macros.h"

namespace muzero_cpp {
using namespace types;
namespace model_layers {
namespace {

// Standard MLP test
void mlp_test() {
    int input_size = 128;
    int output_size = 128;
    const std::vector<int> layers{64, 64};
    MLP mlp(input_size, layers, output_size, "test_net");
    // Print for visual insepctions
    std::cout << mlp << std::endl;
    // Ensure we get expected output shape
    torch::Tensor input = torch::rand({16, input_size});
    torch::Tensor output = mlp->forward(input);
    const std::vector<int> expected_size{16, input_size};
    REQUIRE_EQUAL(expected_size.size(), output.sizes().size());
    for (int i = 0; i < (int)expected_size.size(); ++i) {
        REQUIRE_EQUAL(expected_size[i], output.size(i));
    }
}

// Resnet block test (no downscaling)
void resnet_block_test() {
    int num_channels = 128;
    int layer_num = 4;
    ResidualBlock resnet_block(num_channels, layer_num);
    // Print for visual insepctions
    std::cout << resnet_block << std::endl;
    // Ensure we get expected output shape
    torch::Tensor input = torch::rand({16, num_channels, 96, 96});
    torch::Tensor output = resnet_block->forward(input);
    const std::vector<int> expected_size{16, num_channels, 96, 96};
    REQUIRE_EQUAL(expected_size.size(), output.sizes().size());
    for (int i = 0; i < (int)expected_size.size(); ++i) {
        REQUIRE_EQUAL(expected_size[i], output.size(i));
    }
}

// Resnet head test (no downscaling)
void resnet_head_test() {
    int input_channels = 3;
    int output_channels = 128;
    ResidualHead resenet_head(input_channels, output_channels, "test_");
    // Print for visual insepctions
    std::cout << resenet_head << std::endl;
    // Ensure we get expected output shape
    torch::Tensor input = torch::rand({16, input_channels, 96, 96});
    torch::Tensor output = resenet_head->forward(input);
    const std::vector<int> expected_size{16, output_channels, 96, 96};
    REQUIRE_EQUAL(expected_size.size(), output.sizes().size());
    for (int i = 0; i < (int)expected_size.size(); ++i) {
        REQUIRE_EQUAL(expected_size[i], output.size(i));
    }
}

// Resnet head test (with downscaling)
void resnet_head_downsample_test() {
    int input_channels = 3;
    int output_channels = 128;
    ResidualHeadDownsample resenet_head(input_channels, output_channels, "test_");
    // Print for visual insepctions
    std::cout << resenet_head << std::endl;
    // Ensure we get expected output shape
    torch::Tensor input = torch::rand({16, input_channels, 96, 96});
    torch::Tensor output = resenet_head->forward(input);
    const std::vector<int> expected_size{16, output_channels, 6, 6};
    REQUIRE_EQUAL(expected_size.size(), output.sizes().size());
    for (int i = 0; i < (int)expected_size.size(); ++i) {
        REQUIRE_EQUAL(expected_size[i], output.size(i));
    }
}

}    // namespace
}    // namespace model_layers
}    // namespace muzero_cpp

int main() {
    muzero_cpp::model_layers::mlp_test();
    muzero_cpp::model_layers::resnet_block_test();
    muzero_cpp::model_layers::resnet_head_test();
    muzero_cpp::model_layers::resnet_head_downsample_test();
}
