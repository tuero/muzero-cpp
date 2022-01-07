#ifndef MUZERO_CPP_UTIL_H_
#define MUZERO_CPP_UTIL_H_

#include <torch/torch.h>

#include <atomic>
#include <cassert>
#include <random>

namespace muzero_cpp {
namespace util {

/**
 * Sample N values according to a parameterized Dirichlet distribution
 * @param alpha The alpha parameter for the Dirichlet distribution
 * @param num_samples The number of elements to sample
 * @param rng The source of randomness
 * @return Vector of samples following the parameterized Dirichlet distribution
 */
std::vector<double> sample_dirichlet(double alpha, int num_samples, std::mt19937 &rng);

// Encode/decode reward and values using an invertible categorical transformation.
class ValueEncoder {
public:
    /**
     * @param min_value Minimum possible value (without contraction mapping)
     * @param max_value Maximum possible value (without contraction mapping)
     * @param use_contractive_mapping Flag to use contraction mapping (See Appendix F Network Architecture)
     */
    ValueEncoder(double min_value, double max_value, bool use_contractive_mapping = true);
    ValueEncoder() = delete;

    /**
     * Encode the tensor of values into the categorical transformation
     * @param x The input tensor to encode ([batch_size] or [batch_size, num_rollout_steps])
     * @return The encoded tensor
     */
    torch::Tensor encode(const torch::Tensor &x) const;

    /**
     * Decode the tensor of categorical representation into the associated values
     * @param logits The input tensor to decode ([batch_size, num_rollout_steps])
     * @param take_softmax Flag to take softmax of the logits. Defaulted to true as we assume network output
     * unnormalized logits, but flag is given here for testing for invertibility of encode/decode
     * @return The decoded tensor
     */
    torch::Tensor decode(const torch::Tensor &logits, bool take_softmax = true) const;

    /**
     * Get the support size of the encoded values (number of items in the categorical transformation)
     * @return support size
     */
    int get_support_size() const;

    /**
     * Get the support size of the encoded values (number of items in the categorical transformation)
     * @note This is a static version
     * @param min_value Minimum possible value (without contraction mapping)
     * @param max_value Maximum possible value (without contraction mapping)
     * @param use_contractive_mapping Flag to use contraction mapping (See Appendix F Network Architecture)
     * @return support size
     */
    static int get_support_size(double min_value, double max_value, bool use_contractive_mapping);

private:
    double min_value_;
    double max_value_;
    bool use_contractive_mapping_;
    int support_size_;
    double step_size_;
};

// std::stop_token like flag class to signal for threads
class StopToken {
public:
    StopToken() : flag_(false) {}
    void stop() {
        flag_ = true;
    }
    bool stop_requested() const {
        return flag_;
    }

private:
    std::atomic<bool> flag_;
};

}    // namespace util
}    // namespace muzero_cpp

#endif    // MUZERO_CPP_UTIL_H_