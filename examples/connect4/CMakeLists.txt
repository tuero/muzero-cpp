include_directories(../..)

add_executable(muzero_connect4 muzero_connect4.cpp connect4.h $<TARGET_OBJECTS:muzero_cpp>)
target_link_libraries(muzero_connect4 "${TORCH_LIBRARIES}")
