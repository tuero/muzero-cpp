#include <muzero/config.h>

#include "models.h"

#include <nlohmann/json.hpp>

#include <format>
#include <iostream>
#include <vector>

namespace muzero {

namespace {

void check_config_key_exits(const nlohmann::json &config, const std::string &key)
{
    if (!config.contains(key)) {
        std::cerr << std::format("model config json should contain an entry '{}'", key) << std::endl;
        std::exit(1);
    }
}

}    // namespace

auto encoded_obs_shape(const ObservationShape &obs_shape, bool downsample) -> ObservationShape
{
    return model::RepresentationNetworkImpl::encoded_state_shape(obs_shape, downsample);
}

auto network_config_from_json(const nlohmann::json &config) -> ModelConfig
{
    // Check for valid json
    check_config_key_exits(config, "learning_rate");
    check_config_key_exits(config, "l2_weight_decay");
    check_config_key_exits(config, "downsample");
    check_config_key_exits(config, "normalize_hidden_states");
    check_config_key_exits(config, "resnet_channels");
    check_config_key_exits(config, "representation_blocks");
    check_config_key_exits(config, "dynamics_blocks");
    check_config_key_exits(config, "prediction_blocks");
    check_config_key_exits(config, "reward_reduced_channels");
    check_config_key_exits(config, "policy_reduced_channels");
    check_config_key_exits(config, "value_reduced_channels");
    check_config_key_exits(config, "reward_head_layers");
    check_config_key_exits(config, "policy_head_layers");
    check_config_key_exits(config, "value_head_layers");

    return {
        .learning_rate = config["learning_rate"].template get<double>(),
        .l2_weight_decay = config["l2_weight_decay"].template get<double>(),
        .downsample = config["downsample"].template get<bool>(),
        .normalize_hidden_states = config["normalize_hidden_states"].template get<bool>(),
        .resnet_channels = config["resnet_channels"].template get<int>(),
        .representation_blocks = config["representation_blocks"].template get<int>(),
        .dynamics_blocks = config["dynamics_blocks"].template get<int>(),
        .prediction_blocks = config["prediction_blocks"].template get<int>(),
        .reward_reduced_channels = config["reward_reduced_channels"].template get<int>(),
        .policy_reduced_channels = config["policy_reduced_channels"].template get<int>(),
        .value_reduced_channels = config["value_reduced_channels"].template get<int>(),
        .reward_head_layers = config["reward_head_layers"].template get<std::vector<int>>(),
        .policy_head_layers = config["policy_head_layers"].template get<std::vector<int>>(),
        .value_head_layers = config["value_head_layers"].template get<std::vector<int>>(),
    };
}

}    // namespace muzero
