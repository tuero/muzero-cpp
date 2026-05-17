#ifndef MUZERO_STOP_TOKEN_H_
#define MUZERO_STOP_TOKEN_H_

#include <atomic>

namespace muzero {

// std::stop_token like flag class to signal for threads
class StopToken {
public:
    StopToken()
        : flag_(false)
    {}
    void stop()
    {
        flag_ = true;
    }
    [[nodiscard]] auto stop_requested() const -> bool
    {
        return flag_;
    }

private:
    std::atomic<bool> flag_;
};

}    // namespace muzero

#endif    // MUZERO_STOP_TOKEN_H_
