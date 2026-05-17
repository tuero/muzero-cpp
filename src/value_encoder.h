#ifndef MUZERO_VALUE_ENCODER_H_
#define MUZERO_VALUE_ENCODER_H_

#include <torch/torch.h>

namespace muzero {

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
    [[nodiscard]] auto encode(const torch::Tensor &x) const -> torch::Tensor;

    /**
     * Decode the tensor of categorical representation into the associated values
     * @param logits The input tensor to decode ([batch_size, num_rollout_steps])
     * @param take_softmax Flag to take softmax of the logits. Defaulted to true as we assume network output
     * unnormalized logits, but flag is given here for testing for invertibility of encode/decode
     * @return The decoded tensor
     */
    [[nodiscard]] auto decode(const torch::Tensor &logits, bool take_softmax = true) const -> torch::Tensor;

    /**
     * Get the support size of the encoded values (number of items in the categorical transformation)
     * @return support size
     */
    [[nodiscard]] auto get_support_size() const -> int;

    /**
     * Get the support size of the encoded values (number of items in the categorical transformation)
     * @note This is a static version
     * @param min_value Minimum possible value (without contraction mapping)
     * @param max_value Maximum possible value (without contraction mapping)
     * @param use_contractive_mapping Flag to use contraction mapping (See Appendix F Network Architecture)
     * @return support size
     */
    static auto get_support_size(double min_value, double max_value, bool use_contractive_mapping) -> int;

private:
    double min_value_;
    double max_value_;
    bool use_contractive_mapping_;
    int support_size_;
    double step_size_;
};

}    // namespace muzero

#endif    // MUZERO_VALUE_ENCODER_H_
