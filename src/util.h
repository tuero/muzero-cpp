#ifndef MUZERO_UTIL_H_
#define MUZERO_UTIL_H_

#include <random>
#include <vector>

namespace muzero {

/**
 * Sample N values according to a parameterized Dirichlet distribution
 * @param alpha The alpha parameter for the Dirichlet distribution
 * @param num_samples The number of elements to sample
 * @param rng The source of randomness
 * @return Vector of samples following the parameterized Dirichlet distribution
 */
auto sample_dirichlet(double alpha, int num_samples, std::mt19937 &rng) -> std::vector<double>;

}    // namespace muzero

#endif    // MUZERO_UTIL_H_
