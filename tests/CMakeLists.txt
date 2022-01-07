include_directories(..)

add_executable(mcts_test mcts_test.cpp $<TARGET_OBJECTS:muzero_cpp>)
target_link_libraries(mcts_test "${TORCH_LIBRARIES}")
add_test(mcts_test mcts_test)

add_executable(model_layers_test model_layers_test.cpp $<TARGET_OBJECTS:muzero_cpp>)
target_link_libraries(model_layers_test "${TORCH_LIBRARIES}")
add_test(model_layers_test model_layers_test)

add_executable(models_test models_test.cpp $<TARGET_OBJECTS:muzero_cpp>)
target_link_libraries(models_test "${TORCH_LIBRARIES}")
add_test(models_test models_test)

add_executable(util_test util_test.cpp $<TARGET_OBJECTS:muzero_cpp>)
target_link_libraries(util_test "${TORCH_LIBRARIES}")
add_test(util_test util_test)

add_executable(replay_buffer_test replay_buffer_test.cpp $<TARGET_OBJECTS:muzero_cpp>)
target_link_libraries(replay_buffer_test "${TORCH_LIBRARIES}")
add_test(replay_buffer_test replay_buffer_test)

add_executable(shared_stats_test shared_stats_test.cpp $<TARGET_OBJECTS:muzero_cpp>)
target_link_libraries(shared_stats_test "${TORCH_LIBRARIES}")
add_test(shared_stats_test shared_stats_test)