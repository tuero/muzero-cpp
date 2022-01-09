include_directories(../..)

add_executable(muzero_connect4 muzero_connect4.cpp connect4.h)
target_link_libraries(muzero_connect4 PUBLIC muzero_cpp)
