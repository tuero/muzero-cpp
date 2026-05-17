#ifndef MUZERO_BATCH_H_
#define MUZERO_BATCH_H_

#include <muzero/types.h>

#include <vector>

namespace muzero {

// Replay buffer
struct Batch {
    std::vector<double> priorities;
    std::vector<int> indices;
    std::vector<Action> actions;
    Observation stacked_observations;
    std::vector<double> target_rewards;
    std::vector<double> target_values;
    std::vector<double> target_policies;
    std::vector<double> gradient_scale;
    int num_samples = 0;

    // Append 2 batches together (used for concatenating batch from self-play and reanalyze)
    auto operator+=(const Batch &other) -> Batch &
    {
        priorities.insert(priorities.end(), other.priorities.begin(), other.priorities.end());
        indices.insert(indices.end(), other.indices.begin(), other.indices.end());
        actions.insert(actions.end(), other.actions.begin(), other.actions.end());
        stacked_observations
            .insert(stacked_observations.end(), other.stacked_observations.begin(), other.stacked_observations.end());
        target_rewards.insert(target_rewards.end(), other.target_rewards.begin(), other.target_rewards.end());
        target_values.insert(target_values.end(), other.target_values.begin(), other.target_values.end());
        target_policies.insert(target_policies.end(), other.target_policies.begin(), other.target_policies.end());
        gradient_scale.insert(gradient_scale.end(), other.gradient_scale.begin(), other.gradient_scale.end());
        num_samples += other.num_samples;
        return *this;
    }
};

}    // namespace muzero

#endif    // MUZERO_BATCH_H_
