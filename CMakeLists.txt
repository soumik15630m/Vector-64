cmake_minimum_required(VERSION 3.15)
project(ChessEngine)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Single-config generators default to no build type, which silently skips
# NDEBUG and leaves asserts in the hot path. Default to Release.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(ENGINE_NATIVE "Tune codegen for the build machine (-march=native)" ON)
# Profile-guided optimization. Configure with -DENGINE_PGO=generate, build,
# run `bench 12`, then reconfigure with -DENGINE_PGO=use and rebuild.
set(ENGINE_PGO "off" CACHE STRING "PGO mode: off | generate | use")

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
if(MSVC)
    add_compile_options(/O2 /W4 /Ot /Ob2 /arch:AVX2)
    add_link_options(/LTCG)
else()
    add_compile_options(-O3 -Wall -Wextra)
    if(ENGINE_NATIVE)
        add_compile_options(-march=native)
    endif()
    if(ENGINE_PGO STREQUAL "generate")
        add_compile_options(-fprofile-generate)
        add_link_options(-fprofile-generate)
    elseif(ENGINE_PGO STREQUAL "use")
        add_compile_options(-fprofile-use -fprofile-correction -Wno-missing-profile)
        add_link_options(-fprofile-use)
    endif()
    if(WIN32)
        # MinGW: avoid depending on the toolchain's runtime DLLs at run time
        add_link_options(-static)
    endif()
endif()

set(CORE_SOURCES
        src/cores/position.cpp
        src/cores/movegen.cpp
        src/cores/attacks.cpp
        src/cores/zobrist.cpp
        src/uci/uci.cpp
        src/search/move_ordering.cpp
        src/search/evaluator.cpp
        src/search/search.cpp
        src/nnue/nnue.cpp
        src/nnue/nnue_adapter.cpp
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
        src/search/transposition_table.h
        src/search/move_ordering.h
        src/search/evaluator.h
        src/search/search.h
        src/nnue/nnue.h
        src/nnue/nnue_adapter.h
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

add_executable(test_zobrist tests/test_zobrist.cpp)
target_link_libraries(test_zobrist chess_core)
set_target_properties(test_zobrist PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
