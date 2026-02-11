cmake_minimum_required(VERSION 3.15)
project(ChessEngine)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
if(MSVC)
    add_compile_options(/O2 /W4 /Ot /Ob2 /arch:AVX2)
    add_link_options(/LTCG)
else()
    add_compile_options(-O3 -march=native -Wall -Wextra)
endif()
set(CORE_SOURCES
        src/cores/position.cpp
        src/cores/movegen.cpp
        src/cores/attacks.cpp
        src/cores/zobrist.cpp
        src/uci/uci.cpp
)

set(CORE_HEADERS
        src/cores/position.h
        src/cores/movegen.h
        src/cores/attacks.h
        src/cores/zobrist.h
        src/cores/types.h
        src/cores/bitboard.h
        src/cores/move.h
        src/cores/invariants.h
)

add_library(chess_core STATIC ${CORE_SOURCES} ${CORE_HEADERS})
target_include_directories(chess_core PUBLIC src)
add_executable(ChessEngine
        main.cpp
        tests/perfts.cpp
        tests/perfts.h
        src/uci/uci.h
)

target_link_libraries(ChessEngine chess_core)
set_target_properties(ChessEngine PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
