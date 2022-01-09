# Source files
set(MUZERO_CPP_SOURCES
    config.h
    default_flags.h 
    default_flags.cpp
    device_manager.h
    learner.h 
    learner.cpp
    mcts.h 
    mcts.cpp
    metric_logger.h
    metric_logger.cpp
    model_layers.h
    model_layers.cpp
    models.h 
    models.cpp
    muzero.h 
    muzero.cpp
    queue.h 
    replay_buffer.h 
    replay_buffer.cpp 
    self_play.h 
    self_play.cpp
    shared_stats.h 
    types.h
    types.cpp
    util.h
    util.cpp
    vprnet_evaluator.h 
    vprnet_evaluator.cpp 
    vprnet.h
    vprnet.cpp 
)

# Create library object for the project
add_library(muzero_cpp STATIC ${MUZERO_CPP_SOURCES})
target_compile_options(muzero_cpp PUBLIC 
    -Wall -Wextra 
    $<$<CONFIG:RELEASE>:-O3> $<$<CONFIG:RELEASE>:-DNDEBUG>
    $<$<CONFIG:DEBUG>:-g> $<$<CONFIG:DEBUG>:-DDEBUG>
)

target_include_directories(muzero_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(muzero_cpp ${TORCH_LIBRARIES})

# Internal source files can use #include "muzero-cpp/..."
include_directories(..)
