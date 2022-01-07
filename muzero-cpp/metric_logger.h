#ifndef MUZERO_CPP_METRIC_LOGGER_H_
#define MUZERO_CPP_METRIC_LOGGER_H_

#include <memory>

#include "muzero-cpp/config.h"
#include "muzero-cpp/shared_stats.h"
#include "muzero-cpp/util.h"

namespace muzero_cpp {

/**
 * Continuously logs stats to tensorboard
 * @param config Muzero config
 * @param shared_stats Common self play and training stats
 * @param stop Stop token, used to terminate the actor
 */
void metric_logger(const muzero_config::MuZeroConfig& config, std::shared_ptr<SharedStats> shared_stats,
                   util::StopToken* stop);

}    // namespace muzero_cpp

#endif    // MUZERO_CPP_METRIC_LOGGER_H_