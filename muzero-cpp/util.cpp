#include "muzero-cpp/util.h"

#include <cassert>
#include <cmath>
#include <numeric>

namespace muzero_cpp {
namespace util {
namespace {

const float epsilon = 0.001;

// Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
torch::Tensor contractive_mapping(const torch::Tensor &x) {
    return torch::sign(x) * (torch::sqrt(torch::abs(x) + 1) - 1) + epsilon * x;
}

// Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
torch::Tensor inverse_contractive_mapping(const torch::Tensor &x) {
    return torch::sign(x) *
           (torch::pow(((torch::sqrt(1 + 4 * epsilon * (torch::abs(x) + 1 + epsilon)) - 1) / (2 * epsilon)),
                       2) -
            1);
}
}    // namespace

// Sample N values according to a parameterized Dirichlet distribution
std::vector<double> sample_dirichlet(double alpha, int num_samples, std::mt19937 &rng) {
    // Dirichlet distribution sample can be generating {X_1, ..., X_n} ~ Gamma(alpha, 1),
    //  then normalizing
    assert(num_samples > 0);
    std::vector<double> samples;
    samples.reserve(num_samples);
    std::gamma_distribution<double> gamma_dist(alpha, 1.0);
    for (int i = 0; i < num_samples; ++i) {
        samples.push_back(gamma_dist(rng));
    }
    double sum = std::reduce(samples.begin(), samples.end());
    for (auto &sample : samples) {
        sample /= sum;
    }
    return samples;
}

ValueEncoder::ValueEncoder(double min_value, double max_value, bool use_contractive_mapping)
    : min_value_(min_value), max_value_(max_value), use_contractive_mapping_(use_contractive_mapping) {
    assert(min_value_ < max_value_);
    if (use_contractive_mapping_) {
        max_value_ = contractive_mapping(torch::tensor({max_value_})).item<double>();
        min_value_ = contractive_mapping(torch::tensor({min_value_})).item<double>();
    }
    support_size_ = (int)(std::ceil(max_value_) - std::floor(min_value_) + 1);
    step_size_ = (max_value_ - min_value_) / (support_size_ - 1);
};

// Encode the tensor of values into the categorical transformation
// Input shape: x [batch_size]
//              x [batch_size, num_rollout_steps]
torch::Tensor ValueEncoder::encode(const torch::Tensor &x) const {
    torch::Tensor reduction = use_contractive_mapping_ ? contractive_mapping(x) : x;
    reduction = reduction.clamp_(min_value_, max_value_).unsqueeze(-1);
    torch::Tensor above_min = reduction - min_value_;
    torch::Tensor num_steps = above_min / step_size_;
    torch::Tensor lower_step = num_steps.floor();
    torch::Tensor upper_mod = num_steps - lower_step;
    torch::Tensor lower_mod = 1.0 - upper_mod;
    lower_step = lower_step.to(torch::kLong);
    // Center of encoding is 0, left side are negatives, right side are positive (equally distributed
    // encoding)
    // We can get the transformation by considering both upper/lower sides
    torch::Tensor upper_step = lower_step + 1;
    torch::Tensor step_range = torch::arange(0, support_size_).to(torch::kLong).to(x.device());
    torch::Tensor lower_encoding = lower_step.eq(step_range).to(torch::kFloat) * lower_mod;
    torch::Tensor upper_encoding = upper_step.eq(step_range).to(torch::kFloat) * upper_mod;
    return lower_encoding + upper_encoding;
}

// Decode the tensor of categorical representation into the associated values
// Input_shape: x [batch_size, num_rollout_steps]
torch::Tensor ValueEncoder::decode(const torch::Tensor &logits, bool take_softmax) const {
    torch::Tensor probabilities = (take_softmax) ? torch::softmax(logits, 1) : logits;
    torch::Tensor support = torch::arange(0, support_size_, probabilities.device())
                                .expand(probabilities.sizes())
                                .to(torch::kFloat32);
    torch::Tensor x = (torch::sum(support * probabilities, 1, true) * step_size_) + min_value_;
    return use_contractive_mapping_ ? inverse_contractive_mapping(x) : x;
}

// Get the support size of the encoded values (number of items in the categorical transformation)
// Static version of the below
int ValueEncoder::get_support_size(double min_value, double max_value, bool use_contractive_mapping) {
    if (use_contractive_mapping) {
        max_value = contractive_mapping(torch::tensor({max_value})).item<double>();
        min_value = contractive_mapping(torch::tensor({min_value})).item<double>();
    }
    return (int)(std::ceil(max_value) - std::floor(min_value) + 1);
}

// Get the support size of the encoded values (number of items in the categorical transformation)
int ValueEncoder::get_support_size() const {
    return support_size_;
}

}    // namespace util
}    // namespace muzero_cpp