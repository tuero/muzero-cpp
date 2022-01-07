include_directories(../..)

add_executable(muzero_tictactoe muzero_tictactoe.cpp tictactoe.h $<TARGET_OBJECTS:muzero_cpp>)
target_link_libraries(muzero_tictactoe "${TORCH_LIBRARIES}")
