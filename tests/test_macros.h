#ifndef MUZERO_CPP_TESTS_MACRO_H_
#define MUZERO_CPP_TESTS_MACRO_H_

#include <iostream>
#include <string>
#include <cmath>


#define REQUIRE_TRUE(condition)                                 \
{                                                               \
    if(!(condition))                                            \
    {                                                           \
        std::cerr << std::string( __FILE__ )                    \
                   + std::string( ":" )                         \
                   + std::to_string( __LINE__ )                 \
                   + std::string( " in " )                      \
                   + std::string( __PRETTY_FUNCTION__ )         \
        << std::endl;                                           \
        std::exit(1);                                           \
    }                                                           \
}


#define REQUIRE_FALSE(condition)                                \
{                                                               \
    REQUIRE_TRUE(!condition)                                    \
}


#define REQUIRE_EQUAL(x, y)                                     \
{                                                               \
    if((x) != (y))                                              \
    {                                                           \
        std::cerr << std::string( __FILE__ )                    \
                   + std::string( ":" )                         \
                   + std::to_string( __LINE__ )                 \
                   + std::string( " in " )                      \
                   + std::string( __PRETTY_FUNCTION__ )         \
                   + std::string( ": " )                        \
                   + std::to_string( ( x ) )                    \
                   + std::string( " != " )                      \
                   + std::to_string( ( y ) )                    \
        << std::endl;                                           \
        std::exit(1);                                           \
    }                                                           \
}

#define REQUIRE_NEQUAL(x, y)                                    \
{                                                               \
    if((x) == (y))                                              \
    {                                                           \
        std::cerr << std::string( __FILE__ )                    \
                   + std::string( ":" )                         \
                   + std::to_string( __LINE__ )                 \
                   + std::string( " in " )                      \
                   + std::string( __PRETTY_FUNCTION__ )         \
                   + std::string( ": " )                        \
                   + std::to_string( ( x ) )                    \
                   + std::string( " != " )                      \
                   + std::to_string( ( y ) )                    \
        << std::endl;                                           \
        std::exit(1);                                           \
    }                                                           \
}


#define REQUIRE_NEAR(x, y, tol)                                 \
{                                                               \
    if(std::abs((x) - (y)) >= (tol))                            \
    {                                                           \
        std::cerr << std::string( __FILE__ )                    \
                   + std::string( ":" )                         \
                   + std::to_string( __LINE__ )                 \
                   + std::string( " in " )                      \
                   + std::string( __PRETTY_FUNCTION__ )         \
                   + std::string( ": |" )                       \
                   + std::to_string( ( x ) )                    \
                   + std::string( " - " )                       \
                   + std::to_string( ( y ) )                    \
                   + std::string( "|  < " )                     \
                   + std::to_string( ( tol ) )                  \
        << std::endl;                                           \
        std::exit(1);                                           \
    }                                                           \
}

#endif // MUZERO_CPP_TESTS_MACRO_H_