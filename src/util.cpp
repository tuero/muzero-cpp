#include "util.h"

#include <cassert>
#include <vector>

namespace muzero {

// Sample N values according to a parameterized Dirichlet distribution
auto sample_dirichlet(double alpha, int num_samples, std::mt19937 &rng) -> std::vector<double>
{
    // Dirichlet distribution sample can be generating {X_1, ..., X_n} ~ Gamma(alpha, 1),
    //  then normalizing
    assert(num_samples > 0);
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(num_samples));
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

}    // namespace muzero
