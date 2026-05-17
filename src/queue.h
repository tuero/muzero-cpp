// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MUZERO_THREAD_SAFE_QUEUE_H_
#define MUZERO_THREAD_SAFE_QUEUE_H_

#include <absl/synchronization/mutex.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <absl/types/optional.h>

#include <queue>

namespace muzero {

// A threadsafe-queue.
template <class T>
class ThreadedQueue {
public:
    explicit ThreadedQueue(int max_size)
        : max_size_(max_size)
    {}

    // Add an element to the queue.
    auto Push(const T &value) -> bool
    {
        return Push(value, absl::InfiniteDuration());
    }
    auto Push(const T &value, absl::Duration wait) -> bool
    {
        return Push(value, absl::Now() + wait);
    }
    auto Push(const T &value, absl::Time deadline) -> bool
    {
        absl::MutexLock lock(&m_);
        if (block_new_values_) {
            return false;
        }
        while ((int)q_.size() >= max_size_) {
            if (absl::Now() > deadline || block_new_values_) {
                return false;
            }
            cv_.WaitWithDeadline(&m_, deadline);
        }
        q_.push(value);
        cv_.Signal();
        return true;
    }

    auto Pop() -> absl::optional<T>
    {
        return Pop(absl::InfiniteDuration());
    }
    auto Pop(absl::Duration wait) -> absl::optional<T>
    {
        return Pop(absl::Now() + wait);
    }
    auto Pop(absl::Time deadline) -> absl::optional<T>
    {
        absl::MutexLock lock(&m_);
        while (q_.empty()) {
            if (absl::Now() > deadline || block_new_values_) {
                return absl::nullopt;
            }
            cv_.WaitWithDeadline(&m_, deadline);
        }
        T val = q_.front();
        q_.pop();
        cv_.Signal();
        return val;
    }

    auto Empty() -> bool
    {
        absl::MutexLock lock(&m_);
        return q_.empty();
    }

    void Clear()
    {
        absl::MutexLock lock(&m_);
        while (!q_.empty()) {
            q_.pop();
        }
    }

    auto Size() -> int
    {
        absl::MutexLock lock(&m_);
        return static_cast<int>(q_.size());
    }

    // Causes pushing new values to fail. Useful for shutting down the queue.
    void BlockNewValues()
    {
        absl::MutexLock lock(&m_);
        block_new_values_ = true;
        cv_.SignalAll();
    }

private:
    bool block_new_values_ = false;
    int max_size_;
    std::queue<T> q_;
    absl::Mutex m_;
    absl::CondVar cv_;
};

}    // namespace muzero

#endif    // MUZERO_THREAD_SAFE_QUEUE_H_
