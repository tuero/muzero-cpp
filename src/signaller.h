#ifndef MUZERO_SIGNALLER_H_
#define MUZERO_SIGNALLER_H_

#include "stop_token.h"

#include <memory>

namespace muzero {

/**
 * Create and install a signal handler
 * On SIGINT, token will request stop, and all objects storing it will call their exit code
 */
std::shared_ptr<StopToken> signal_installer();

/**
 * Create and install a signal handler
 * On SIGINT, token will request stop, and all objects storing it will call their exit code
 */
void signal_installer(std::shared_ptr<StopToken> stop_token);

}    // namespace muzero

#endif    // MUZERO_SIGNALLER_H_
