#include "muzero-cpp/muzero.h"

#include <torch/torch.h>

#include <algorithm>
#include <csignal>
#include <filesystem>
#include <memory>
#include <thread>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "muzero-cpp/config.h"
#include "muzero-cpp/device_manager.h"
#include "muzero-cpp/learner.h"
#include "muzero-cpp/metric_logger.h"
#include "muzero-cpp/queue.h"
#include "muzero-cpp/replay_buffer.h"
#include "muzero-cpp/self_play.h"
#include "muzero-cpp/shared_stats.h"
#include "muzero-cpp/types.h"
#include "muzero-cpp/util.h"
#include "muzero-cpp/vprnet.h"

namespace muzero_cpp {

using namespace muzero_config;
using namespace types;
using namespace util;
using namespace buffer;

namespace {
// Stop token and signaling for exiting cleanly
StopToken stop;
void signal_handler(int s) {
    (void)s;
    if (stop.stop_requested()) {
        exit(1);
    } else {
        stop.stop();
    }
}

void signal_installer() {
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = signal_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, nullptr);
}
}    // namespace

// Use the model to play a test game
bool play_test_model(const MuZeroConfig& config,
                     std::function<std::unique_ptr<AbstractGame>(int)> game_factory) {
    // Set torch seed
    torch::manual_seed(config.seed);

    // Create models on devices
    DeviceManager device_manager;
    for (const absl::string_view& device : absl::StrSplit(config.devices, ',')) {
        device_manager.AddDevice(VPRNetModel(config, std::string(device)));
    }

    if (device_manager.Count() == 0) {
        std::cerr << "No devices specified?" << std::endl;
        return false;
    }

    // Sync all models so that they have the same weights
    {
        for (int i = 0; i < device_manager.Count(); ++i) {
            device_manager.Get(0, i)->LoadCheckpoint(VPRNetModel::kMostRecentCheckpointStep);
        }
    }

    // Model evaluator and shared stats necessary to call the self play function
    // Since we are only running in 1 thread, we set batch sizes/threads to 1
    // Maybe for compute intensive games we want to manually set this, but most cases have the user being
    // asked for input after each move.
    auto vpr_eval = std::make_shared<VPRNetEvaluator>(&device_manager, 1, 1, 1, 1);
    auto shared_stats = std::make_shared<SharedStats>();

    // Play game
    self_play_test(config, game_factory(config.seed), vpr_eval, shared_stats, &stop);

    return true;
}

// Train a model using the muzero algorithm
bool muzero(const MuZeroConfig& config, std::function<std::unique_ptr<AbstractGame>(int)> game_factory) {
    // Signal installer on stop token
    signal_installer();

    // Set torch seed
    torch::manual_seed(config.seed);

    // Display current config values
    std::cout << "Using config:" << std::endl;
    std::cout << config.to_str() << std::endl;

    // Create directories if they dont exists
    const std::string shared_stats_path = absl::StrCat(config.path, "/metrics/");
    std::filesystem::create_directories(config.path);
    std::filesystem::create_directories(shared_stats_path);

    // Create models on devices
    DeviceManager device_manager;
    for (const absl::string_view& device : absl::StrSplit(config.devices, ',')) {
        device_manager.AddDevice(VPRNetModel(config, std::string(device)));
    }

    { device_manager.Get(0)->print(); }

    // Check for some things which wouldn't make sense. This isn't robust for all cases.
    // Some input may crash if you try something that shouldn't make senese.
    // Nothing to train and act on
    if (device_manager.Count() < 2) {
        std::cerr << "Need to specify at least 2 devices." << std::endl;
        return false;
    }
    // Can't have 0 samples being taken from reananlyze
    if (config.train_reanalyze_ratio == 1) {
        std::cerr << "Train reanalyze ratio should be in range [0, 1)." << std::endl;
        return false;
    }
    // If we want to use reananlyze, we probably should have some reananlyze actors
    if (config.train_reanalyze_ratio > 0 and config.num_reanalyze_actors == 0) {
        std::cerr << "If using reananlyze then set config.num_reanalyze_actors > 0." << std::endl;
        return false;
    }

    std::cout << "Using " << device_manager.Count() << " devices." << std::endl;

    // Set first device off-limits from inference if requested
    // This is to ensure that actors/evaluator are using a target network and not the immediate fresh weights
    // for stability
    device_manager.SetLearning(true);

    // Sync all models so that they have the same weights
    {
        // If not resuming then we checkpoint current starting point
        if (!config.resume) { device_manager.Get(0)->SaveCheckpoint(VPRNetModel::kMostRecentCheckpointStep); }
        for (int i = 0; i < device_manager.Count(); ++i) {
            device_manager.Get(0, i)->LoadCheckpoint(VPRNetModel::kMostRecentCheckpointStep);
        }
    }

    // Shared evaluator
    // Add to batch size for # of evaluators + actors
    int total_actors = config.num_actors + config.num_reanalyze_actors;
    int initial_inference_batch_size =
        std::max(1, std::min(config.initial_inference_batch_size, total_actors + 1));
    int recurrent_inference_batch_size =
        std::max(1, std::min(config.recurrent_inference_batch_size, total_actors + 1));
    int initial_inference_threads =
        std::max(1, std::min(config.initial_inference_threads, (1 + total_actors + 1) / 2));
    int recurrent_inference_threads =
        std::max(1, std::min(config.initial_inference_threads, (1 + total_actors + 1) / 2));
    auto vpr_eval = std::make_shared<VPRNetEvaluator>(
        &device_manager, initial_inference_batch_size, initial_inference_threads,
        recurrent_inference_batch_size, recurrent_inference_threads);

    // Shared queue for passing self played games from actors to learner to store
    ThreadedQueue<GameHistory> trajectory_queue(config.replay_buffer_size);

    // Shared stats, load if we are resuming
    auto shared_stats = std::make_shared<SharedStats>();
    shared_stats->set_path(shared_stats_path);
    if (config.resume) {
        shared_stats->load(VPRNetModel::kMostRecentCheckpointStep);
        std::cout << "Resuming at training step: " << shared_stats->get_training_step() << std::endl;
    }

    // Self play actors
    std::vector<std::thread> actors;
    for (int i = 0; i < config.num_actors; ++i) {
        actors.push_back(std::thread(self_play_actor, std::cref(config), game_factory(config.seed + i), i,
                                     &trajectory_queue, vpr_eval, shared_stats, &stop));
    }
    std::cout << "Spawned " << config.num_actors << " actors." << std::endl;

    // Evaluator
    std::vector<std::thread> evaluators;
    for (int i = 0; i < config.num_evaluators; ++i) {
        evaluators.push_back(std::thread(self_play_evaluator, std::cref(config),
                                         game_factory(config.seed + i), i, vpr_eval, shared_stats, &stop));
    }
    std::cout << "Spawned " << 1 << " evaluator." << std::endl;

    // Metrics loggger
    std::vector<std::thread> metric_loggers;
    for (int i = 0; i < 1; ++i) {
        metric_loggers.push_back(std::thread(metric_logger, std::cref(config), shared_stats, &stop));
    }
    std::cout << "Spawned " << 1 << " metrics tracker." << std::endl;

    // Reanalyze
    auto replay_buffer =
        std::make_shared<PrioritizedReplayBuffer>(config, config.replay_buffer_size, "replay_buffer");
    auto reanalyze_buffer =
        std::make_shared<PrioritizedReplayBuffer>(config, config.reanalyze_buffer_size, "reanalyze_buffer");
    std::vector<std::thread> reanalyse_threads;
    for (int i = 0; i < config.num_reanalyze_actors; ++i) {
        reanalyse_threads.push_back(std::thread(reanalyze_actor, std::cref(config), reanalyze_buffer, i,
                                                vpr_eval, shared_stats, &stop));
    }
    std::cout << "Spawned " << config.num_reanalyze_actors << " reanalyse actors." << std::endl;

    // Start to learn
    std::cout << "Starting learning." << std::endl;
    if (config.resume) { replay_buffer->load(); }
    learn(config, &device_manager, replay_buffer, reanalyze_buffer, &trajectory_queue, shared_stats, &stop);

    // Empty the queue so that the actors can exit.
    trajectory_queue.BlockNewValues();
    trajectory_queue.Clear();

    // Join threads
    std::cout << "\nJoining all the threads." << std::endl;
    for (auto& t : reanalyse_threads) {
        t.join();
    }
    for (auto& t : actors) {
        t.join();
    }
    for (auto& t : evaluators) {
        t.join();
    }
    for (auto& t : metric_loggers) {
        t.join();
    }

    // Before we exit, save buffer
    std::cout << "Saving replay buffer." << std::endl;
    replay_buffer->save();

    // Save shared stats
    std::cout << "Saving shared stats." << std::endl;
    shared_stats->save(VPRNetModel::kMostRecentCheckpointStep);

    std::cout << "Exiting cleanly." << std::endl;
    return true;
}

}    // namespace muzero_cpp