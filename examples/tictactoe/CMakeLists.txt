include_directories(../..)

add_executable(muzero_tictactoe muzero_tictactoe.cpp tictactoe.h)
target_link_libraries(muzero_tictactoe PUBLIC muzero_cpp)
