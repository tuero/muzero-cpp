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

#ifndef MUZERO_DEVICE_MANAGER_H_
#define MUZERO_DEVICE_MANAGER_H_

#include "vprnet.h"

#include <absl/synchronization/mutex.h>

#include <vector>

namespace muzero {

// Keeps track of a bunch of VPRNet models, intended to be one per device, and
// gives them out based on usage. When you request a device you specify how much
// work you're going to give it, which is assumed done once the loan is
// returned.
class DeviceManager {
public:
    void AddDevice(model::VPRNetModel model)
    {    // Not thread safe.
        devices.emplace_back(Device{.model = std::move(model)});
        multiple_devices_ = devices.size() > 1;
    }

    // Acts as a pointer to the model, but lets the manager know when you're done.
    class DeviceLoan {
    public:
        // DeviceLoan is not public constructible and is move only.
        DeviceLoan(DeviceLoan &&other) = default;
        DeviceLoan &operator=(DeviceLoan &&other) = default;
        DeviceLoan(const DeviceLoan &) = delete;
        DeviceLoan &operator=(const DeviceLoan &) = delete;

        ~DeviceLoan()
        {
            manager_->Return(device_id_, requests_);
        }
        auto operator->() -> model::VPRNetModel *
        {
            return model_;
        }

    private:
        DeviceLoan(DeviceManager *manager, model::VPRNetModel *model, int device_id, int requests)
            : manager_(manager), model_(model), device_id_(device_id), requests_(requests)
        {}
        DeviceManager *manager_;
        model::VPRNetModel *model_;
        int device_id_;
        int requests_;
        friend DeviceManager;
    };

    // Gives the device with the fewest outstanding requests.
    auto Get(int requests, int device_id = -1) -> DeviceLoan
    {
        absl::MutexLock lock(&m_);
        if (device_id < 0) {
            // The starting device changes depending on if we are allowed to
            // use the first device or not.
            device_id = 0 + (learning_ && multiple_devices_);
            for (std::size_t i = 1 + (learning_ && multiple_devices_); i < devices.size(); ++i) {
                if (devices[i].requests < devices[static_cast<std::size_t>(device_id)].requests) {
                    device_id = static_cast<int>(i);
                }
            }
        }
        devices[static_cast<std::size_t>(device_id)].requests += requests;
        return {this, &devices[static_cast<std::size_t>(device_id)].model, device_id, requests};
    }

    // A member to ensure that when device:0 is learning and there are
    // multiple devices available, that device:0 does not take on any
    // inference requests from the actors and evaluators. These inference
    // requests should be dealt with by the other available devices.
    void SetLearning(bool value)
    {
        learning_ = value;
    }

    [[nodiscard]] auto Count() const -> int
    {
        return static_cast<int>(devices.size());
    }

private:
    void Return(int device_id, int requests)
    {
        absl::MutexLock lock(&m_);
        devices[static_cast<std::size_t>(device_id)].requests -= requests;
    }

    struct Device {
        model::VPRNetModel model;
        int requests = 0;
    };

    bool learning_ = false;
    bool multiple_devices_ = false;
    std::vector<Device> devices;
    absl::Mutex m_;
};

}    // namespace muzero

#endif    // MUZERO_DEVICE_MANAGER_H_
