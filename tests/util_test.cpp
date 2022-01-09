#include "muzero-cpp/util.h"

#include <torch/torch.h>

#include "tests/test_macros.h"

namespace muzero_cpp {
namespace util {
namespace {

using namespace torch::indexing;

// Test decoding the support back to a scalar
void support_to_scalar_test() {
    ValueEncoder value_encoder(-22, 22);
    torch::Tensor tensor_scalars = torch::tensor({{0.0, 1.5, 21.25}, {0.0, -10.75, 2.0}});
    torch::Tensor tensor_support = value_encoder.encode(tensor_scalars);
    torch::Tensor tensor_truth = torch::tensor({
        {
            {0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000},
            {0.0000, 0.0000, 0.0000, 0.0000, 0.3896, 0.6104, 0.0000, 0.0000, 0.0000},
            {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0834, 0.9166},
        },
        {
            {0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000},
            {0.0000, 0.5549, 0.4451, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000},
            {0.0000, 0.0000, 0.0000, 0.0000, 0.2309, 0.7691, 0.0000, 0.0000, 0.0000},
        },
    });
    bool result = torch::allclose(tensor_support, tensor_truth, 1e-4, 1e-4);
    REQUIRE_TRUE(result);
}

// Test encoding the scalar into the support
void scalar_to_support_test() {
    ValueEncoder value_encoder(-22, 22);
    // support_to_scalar applies softmax, so inverse first -> log(x + eps) + c
    torch::Tensor tensor_support =
        torch::log(torch::tensor({
                       {
                           {0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000},
                           {0.0000, 0.0000, 0.0000, 0.0000, 0.3896, 0.6104, 0.0000, 0.0000, 0.0000},
                           {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0834, 0.9166},
                       },
                       {
                           {0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000},
                           {0.0000, 0.5549, 0.4451, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000},
                           {0.0000, 0.0000, 0.0000, 0.0000, 0.2309, 0.7691, 0.0000, 0.0000, 0.0000},
                       },
                   }) +
                   1e-7);
    torch::Tensor tensor_truth = torch::tensor({{0.0, 1.5, 21.25}, {0.0, -10.75, 2.0}});
    for (int i = 0; i < tensor_support.sizes()[1]; ++i) {
        torch::Tensor truth_slice = tensor_truth.index({Slice(), Slice(i, i + 1)});
        torch::Tensor tensor_scalar = value_encoder.decode(tensor_support.index({Slice(), i, Slice()}));
        bool result = torch::allclose(tensor_scalar, truth_slice, 1e-4, 1e-4);
        REQUIRE_TRUE(result);
    }
}

// Test encoding the scalar into the support
void scalar_to_support_probs_test() {
    ValueEncoder value_encoder(-22, 22);
    // support_to_scalar applies softmax, so inverse first -> log(x + eps) + c
    torch::Tensor tensor_support = torch::tensor({
        {
            {0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000},
            {0.0000, 0.0000, 0.0000, 0.0000, 0.3896, 0.6104, 0.0000, 0.0000, 0.0000},
            {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0834, 0.9166},
        },
        {
            {0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000},
            {0.0000, 0.5549, 0.4451, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000},
            {0.0000, 0.0000, 0.0000, 0.0000, 0.2309, 0.7691, 0.0000, 0.0000, 0.0000},
        },
    });
    torch::Tensor tensor_truth = torch::tensor({{0.0, 1.5, 21.25}, {0.0, -10.75, 2.0}});
    for (int i = 0; i < tensor_support.sizes()[1]; ++i) {
        torch::Tensor truth_slice = tensor_truth.index({Slice(), Slice(i, i + 1)});
        torch::Tensor tensor_scalar =
            value_encoder.decode(tensor_support.index({Slice(), i, Slice()}), false);
        bool result = torch::allclose(tensor_scalar, truth_slice, 1e-4, 1e-4);
        REQUIRE_TRUE(result);
    }
}

// Test that we get our original input after encode then decode
void one_to_one_mapping_test() {
    ValueEncoder value_encoder(-22, 22);
    torch::Tensor original_tensor = torch::rand({16, 5}, torch::kFloat32) * 10;
    torch::Tensor tensor_support = value_encoder.encode(original_tensor);
    // Apply inverse of softmax
    tensor_support = torch::log(tensor_support + 1e-7);
    for (int i = 0; i < original_tensor.sizes()[1]; ++i) {
        torch::Tensor original_slice = original_tensor.index({Slice(), Slice(i, i + 1)});
        torch::Tensor tensor_scalar = value_encoder.decode(tensor_support.index({Slice(), i, Slice()}));
        bool result = torch::allclose(tensor_scalar, original_slice, 1e-4, 1e-4);
        REQUIRE_TRUE(result);
    }
}

}    // namespace
}    // namespace util
}    // namespace muzero_cpp

int main() {
    muzero_cpp::util::support_to_scalar_test();
    muzero_cpp::util::scalar_to_support_test();
    muzero_cpp::util::scalar_to_support_probs_test();
    muzero_cpp::util::one_to_one_mapping_test();
}
