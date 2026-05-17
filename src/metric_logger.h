#ifndef MUZERO_METRIC_LOGGER_H_
#define MUZERO_METRIC_LOGGER_H_

#include <muzero/config.h>

#include "shared_stats.h"
#include "stop_token.h"

#include <memory>

namespace muzero {

/**
 * Continuously logs stats to tensorboard
 * @param config Muzero config
 * @param shared_stats Common self play and training stats
 * @param stop Stop token, used to terminate the actor
 */
void metric_logger(const Config &config, std::shared_ptr<SharedStats> shared_stats, std::shared_ptr<StopToken> stop);

}    // namespace muzero

#endif    // MUZERO_METRIC_LOGGER_H_
