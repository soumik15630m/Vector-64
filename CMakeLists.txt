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

# Perft-only optimization: a precomputed enemy king-danger map speeds up perft
# (~+3%) because it amortizes over every king move, but it regresses real
# search (king moves are rare there), so it is off by default. Enable only for
# perft benchmarking: -DOPT_KING_DANGER=ON.
option(OPT_KING_DANGER "Precompute enemy king-danger map (perft-only win)" OFF)
if(OPT_KING_DANGER)
    add_compile_definitions(ENGINE_KING_DANGER)
endif()
# Profile-guided optimization. Configure with -DENGINE_PGO=generate, build,
# run `bench 12`, then reconfigure with -DENGINE_PGO=use and rebuild.
set(ENGINE_PGO "off" CACHE STRING "PGO mode: off | generate | use")

# LTO/IPO is on by default but can be disabled (e.g. for sanitizer builds).
option(ENGINE_LTO "Enable link-time / interprocedural optimization" ON)
if(ENGINE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

if(MSVC)
    add_compile_options(/O2 /W4 /Ot /Ob2 /arch:AVX2)
    # Deep search recursion (up to MAX_PLY plies, plus singular same-ply
    # re-searches, each frame carrying large move buffers) overflows MSVC's
    # 1 MB default stack. Give it 32 MB; on Windows this PE value also seeds
    # std::thread worker stacks, so it protects real deep searches too.
    add_link_options(/STACK:33554432)
    if(ENGINE_LTO)
        add_link_options(/LTCG)
    endif()
else()
    add_compile_options(-O3 -Wall -Wextra)
    if(EMSCRIPTEN)
        # The browser build links with shared memory (pthreads), and wasm-ld
        # refuses to mix objects built with the atomics feature and objects
        # built without it -- so every translation unit needs these, not just
        # the final link. -msimd128 lets the compiler vectorise the portable
        # scalar NNUE kernels. Set here rather than via CXXFLAGS/-DCMAKE_CXX_FLAGS
        # because emcmake re-invokes cmake and drops those.
        add_compile_options(-pthread -msimd128)
    elseif(ENGINE_NATIVE)
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
        # Same deep-recursion stack headroom as the MSVC build (the PE default
        # stack also seeds std::thread workers on Windows).
        add_link_options(-Wl,--stack,33554432)
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
        src/search/syzygy.cpp
        src/nnue/network.cpp
        src/datagen/datagen.cpp
        # Vendored Syzygy tablebase prober (Fathom, MIT). Built as C++ (see the
        # LANGUAGE property below) so it uses std::atomic / std::mutex on every
        # toolchain, avoiding MSVC's opt-in C11 atomics.
        src/syzygy/tbprobe.c
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
        src/search/syzygy.h
        src/nnue/halfka.h
        src/nnue/network.h
        src/datagen/datagen.h
)

add_library(chess_core STATIC ${CORE_SOURCES} ${CORE_HEADERS})
# Build Fathom's C source with the C++ compiler: it then selects std::atomic /
# std::mutex, which every toolchain supports without extra flags.
set_source_files_properties(src/syzygy/tbprobe.c PROPERTIES LANGUAGE CXX)
target_include_directories(chess_core PUBLIC src)
# Fathom's own sources use quoted includes (tbconfig.h, stdendian.h, tbchess.c)
# resolved relative to this directory.
target_include_directories(chess_core PRIVATE src/syzygy)

add_executable(ChessEngine
        app/main.cpp
        tests/cpp/perfts.cpp
        tests/cpp/perfts.h
)
target_include_directories(ChessEngine PRIVATE tests/cpp)
target_link_libraries(ChessEngine chess_core)
set_target_properties(ChessEngine PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ---------------------------------------------------------------------------
# NNUE binary: ChessEngine-nnue, with the default net baked into the executable
# (Stockfish-style, via GNU-as .incbin). Built only when the net exists and the
# toolchain is GNU/Clang (the embed uses GNU-as directives; MSVC/macOS embedding
# would need a different section/symbol convention). The classical ChessEngine
# above is unchanged. Override the net with -DEMBED_NNUE=/path/to/net.nnue.
set(EMBED_NNUE "${CMAKE_SOURCE_DIR}/nets/stk-vector-64.nnue"
        CACHE FILEPATH "Net embedded into the ChessEngine-nnue binary")
if(EXISTS "${EMBED_NNUE}" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|MSVC")
        set(STK_EMBED_NNUE_PATH "${EMBED_NNUE}")
        set(_nnue_embed src/nnue/embedded_net.cpp)
        if(MSVC)
                # Windows resource (RCDATA); accessed via FindResource.
                enable_language(RC)
                configure_file(src/nnue/embedded_net.rc.in
                        ${CMAKE_BINARY_DIR}/embedded_net.rc @ONLY)
                # The resource only references the net by path, so make cannot
                # see that dependency by itself: state it, or a changed net
                # silently keeps the previously embedded bytes.
                set_source_files_properties(${CMAKE_BINARY_DIR}/embedded_net.rc
                        PROPERTIES OBJECT_DEPENDS "${EMBED_NNUE}")
                list(APPEND _nnue_embed ${CMAKE_BINARY_DIR}/embedded_net.rc)
        else()
                # GNU-as .incbin (ELF: Linux, Mach-O: macOS, PE-COFF: MinGW).
                enable_language(ASM)
                configure_file(src/nnue/embedded_net.S.in
                        ${CMAKE_BINARY_DIR}/embedded_net.S @ONLY)
                # .incbin pulls the net in at assemble time; make cannot see
                # that, so declare it or a changed net is silently ignored.
                set_source_files_properties(${CMAKE_BINARY_DIR}/embedded_net.S
                        PROPERTIES OBJECT_DEPENDS "${EMBED_NNUE}")
                list(APPEND _nnue_embed ${CMAKE_BINARY_DIR}/embedded_net.S)
        endif()
        add_executable(ChessEngine-nnue
                app/main.cpp
                tests/cpp/perfts.cpp
                tests/cpp/perfts.h
                ${_nnue_embed}
        )
        target_compile_definitions(ChessEngine-nnue PRIVATE STK_EMBED_NNUE)
        target_include_directories(ChessEngine-nnue PRIVATE tests/cpp)
        target_link_libraries(ChessEngine-nnue chess_core)
        set_target_properties(ChessEngine-nnue PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
        message(STATUS "NNUE binary: ChessEngine-nnue embeds ${EMBED_NNUE}")
else()
        message(STATUS "NNUE binary: skipped (net '${EMBED_NNUE}' missing or unsupported toolchain)")
endif()

# ---------------------------------------------------------------------------
# Visualizer support library. Kept out of chess_core on purpose: the tournament
# binaries must gain nothing from it (no extra code, no future third-party
# link deps). Only the visualizer binary and its tests link this.
# ---------------------------------------------------------------------------
add_library(chess_viz STATIC
        src/viz/probe.cpp
        src/viz/probe.h
        src/viz/session.cpp
        src/viz/session.h
        src/viz/wire.cpp
        src/viz/wire.h
        src/viz/terminal.cpp
        src/viz/terminal.h
)
target_link_libraries(chess_viz PUBLIC chess_core)

# Header-only MIT dependencies, pinned and fetched as single files (no git
# history, no build system of their own). They are attached to chess_viz only:
# the tournament binaries never see them.
option(ENGINE_VIZ "Build the NNUE visualizer binary (needs network on first configure)" ON)
if(ENGINE_VIZ)
    include(FetchContent)
    FetchContent_Declare(stk_json
            URL https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp
            URL_HASH SHA256=9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6
            DOWNLOAD_NO_EXTRACT TRUE)
    FetchContent_Declare(stk_httplib
            URL https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.18.3/httplib.h
            URL_HASH SHA256=a0a0c13dc086663863dbe6730a19716f7d3744904b6205496e8301750814dbd5
            DOWNLOAD_NO_EXTRACT TRUE)
    FetchContent_MakeAvailable(stk_json stk_httplib)
    target_include_directories(chess_viz PUBLIC
            ${stk_json_SOURCE_DIR} ${stk_httplib_SOURCE_DIR})
    target_compile_definitions(chess_viz PUBLIC STK_VIZ_DEPS)

    # --- ChessEngine-viz --------------------------------------------------
    # Embeds the default net (so it visualizes a real net out of the box) and,
    # when built, the single-file UI bundle. Both use the same .incbin / RCDATA
    # mechanism as ChessEngine-nnue.
    set(_viz_sources
            src/viz/main_viz.cpp
            src/viz/server.cpp
            src/viz/server.h
            src/viz/embedded_ui.cpp
            src/viz/embedded_ui.h)
    set(_viz_defs "")

    set(STK_UI_BUNDLE "${CMAKE_SOURCE_DIR}/ui/dist/index.html"
            CACHE FILEPATH "Single-file UI bundle embedded into ChessEngine-viz")

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|MSVC")
        if(EXISTS "${EMBED_NNUE}")
            set(STK_EMBED_NNUE_PATH "${EMBED_NNUE}")
            list(APPEND _viz_defs STK_EMBED_NNUE)
            if(MSVC)
                enable_language(RC)
                configure_file(src/nnue/embedded_net.rc.in
                        ${CMAKE_BINARY_DIR}/viz_embedded_net.rc @ONLY)
                set_source_files_properties(${CMAKE_BINARY_DIR}/viz_embedded_net.rc
                        PROPERTIES OBJECT_DEPENDS "${EMBED_NNUE}")
                list(APPEND _viz_sources ${CMAKE_BINARY_DIR}/viz_embedded_net.rc)
            else()
                enable_language(ASM)
                configure_file(src/nnue/embedded_net.S.in
                        ${CMAKE_BINARY_DIR}/viz_embedded_net.S @ONLY)
                set_source_files_properties(${CMAKE_BINARY_DIR}/viz_embedded_net.S
                        PROPERTIES OBJECT_DEPENDS "${EMBED_NNUE}")
                list(APPEND _viz_sources ${CMAKE_BINARY_DIR}/viz_embedded_net.S)
            endif()
            list(APPEND _viz_sources src/nnue/embedded_net.cpp)
        endif()

        if(EXISTS "${STK_UI_BUNDLE}")
            set(STK_EMBED_UI_PATH "${STK_UI_BUNDLE}")
            list(APPEND _viz_defs STK_EMBED_UI)
            if(MSVC)
                enable_language(RC)
                configure_file(src/viz/embedded_ui.rc.in
                        ${CMAKE_BINARY_DIR}/embedded_ui.rc @ONLY)
                set_source_files_properties(${CMAKE_BINARY_DIR}/embedded_ui.rc
                        PROPERTIES OBJECT_DEPENDS "${STK_UI_BUNDLE}")
                list(APPEND _viz_sources ${CMAKE_BINARY_DIR}/embedded_ui.rc)
            else()
                enable_language(ASM)
                configure_file(src/viz/embedded_ui.S.in
                        ${CMAKE_BINARY_DIR}/embedded_ui.S @ONLY)
                # Rebuild the embed whenever the bundle changes -- otherwise
                # `npm run build` appears to do nothing.
                set_source_files_properties(${CMAKE_BINARY_DIR}/embedded_ui.S
                        PROPERTIES OBJECT_DEPENDS "${STK_UI_BUNDLE}")
                list(APPEND _viz_sources ${CMAKE_BINARY_DIR}/embedded_ui.S)
            endif()
            message(STATUS "Visualizer: embedding UI bundle ${STK_UI_BUNDLE}")
        else()
            message(STATUS "Visualizer: no UI bundle yet (serving placeholder); build ui/ to create ${STK_UI_BUNDLE}")
        endif()

        add_executable(ChessEngine-viz ${_viz_sources})
        target_link_libraries(ChessEngine-viz chess_viz)
        if(_viz_defs)
            target_compile_definitions(ChessEngine-viz PRIVATE ${_viz_defs})
        endif()
        if(WIN32)
            # ws2_32: sockets. shell32: ShellExecute, used to open the browser
            # without spawning a command processor.
            target_link_libraries(ChessEngine-viz ws2_32 shell32)
        endif()
        find_package(Threads REQUIRED)
        target_link_libraries(ChessEngine-viz Threads::Threads)
        set_target_properties(ChessEngine-viz PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    endif()
endif()

add_executable(test_zobrist tests/cpp/test_zobrist.cpp)
target_link_libraries(test_zobrist chess_core)
set_target_properties(test_zobrist PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

add_executable(test_viz tests/cpp/test_viz.cpp)
target_link_libraries(test_viz chess_viz)
set_target_properties(test_viz PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")

# ---------------------------------------------------------------------------
# Test suite (ctest). Each check is a named test with a single clear goal.
# ---------------------------------------------------------------------------
enable_testing()

# Perft — exact node counts for the six reference positions, each depth its own
# test, so a movegen regression points at the exact position and depth.
function(add_perft_test name fen depth expected)
    add_test(NAME "perft.${name}"
             COMMAND ChessEngine --perft-one "${fen}" ${depth} ${expected})
endfunction()

set(FEN_START  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
set(FEN_KIWI   "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
set(FEN_POS3   "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")
set(FEN_POS4   "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
set(FEN_POS5   "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8")
set(FEN_POS6   "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10")

add_perft_test(startpos_d1 "${FEN_START}" 1 20)
add_perft_test(startpos_d2 "${FEN_START}" 2 400)
add_perft_test(startpos_d3 "${FEN_START}" 3 8902)
add_perft_test(startpos_d4 "${FEN_START}" 4 197281)
add_perft_test(startpos_d5 "${FEN_START}" 5 4865609)
add_perft_test(kiwipete_d1 "${FEN_KIWI}" 1 48)
add_perft_test(kiwipete_d2 "${FEN_KIWI}" 2 2039)
add_perft_test(kiwipete_d3 "${FEN_KIWI}" 3 97862)
add_perft_test(kiwipete_d4 "${FEN_KIWI}" 4 4085603)
add_perft_test(kiwipete_d5 "${FEN_KIWI}" 5 193690690)
add_perft_test(position3_d1 "${FEN_POS3}" 1 14)
add_perft_test(position3_d2 "${FEN_POS3}" 2 191)
add_perft_test(position3_d3 "${FEN_POS3}" 3 2812)
add_perft_test(position3_d4 "${FEN_POS3}" 4 43238)
add_perft_test(position3_d5 "${FEN_POS3}" 5 674624)
add_perft_test(position3_d6 "${FEN_POS3}" 6 11030083)
add_perft_test(position4_d1 "${FEN_POS4}" 1 6)
add_perft_test(position4_d2 "${FEN_POS4}" 2 264)
add_perft_test(position4_d3 "${FEN_POS4}" 3 9467)
add_perft_test(position4_d4 "${FEN_POS4}" 4 422333)
add_perft_test(position4_d5 "${FEN_POS4}" 5 15833292)
add_perft_test(position5_d1 "${FEN_POS5}" 1 44)
add_perft_test(position5_d2 "${FEN_POS5}" 2 1486)
add_perft_test(position5_d3 "${FEN_POS5}" 3 62379)
add_perft_test(position5_d4 "${FEN_POS5}" 4 2103487)
add_perft_test(position6_d1 "${FEN_POS6}" 1 46)
add_perft_test(position6_d2 "${FEN_POS6}" 2 2079)
add_perft_test(position6_d3 "${FEN_POS6}" 3 89890)
add_perft_test(position6_d4 "${FEN_POS6}" 4 3894594)

# State / evaluation invariants.
add_test(NAME "consistency.nnue_simd_bitexact" COMMAND test_zobrist nnue)
add_test(NAME "consistency.make_unmake_and_movegen" COMMAND test_zobrist games)

# Visualizer telemetry reports the engine's real arithmetic: the captured eval
# equals the ordinary eval path, and each layer's attribution sums back to that
# layer's output.
add_test(NAME "viz.telemetry_exact" COMMAND test_viz frame)
# The session driver keeps the board and the reported move list in agreement,
# and self-play games actually reach terminal states.
add_test(NAME "viz.session_selfplay" COMMAND test_viz session)
# A search is deterministic, so without root-move variety self-play replays one
# identical game from the same position forever -- a silent failure that only
# shows up as "why is every game the same?". Random openings hide it, so this
# runs with them off.
add_test(NAME "viz.selfplay_variety" COMMAND test_viz variety)
# Rows are written as numbered shards, which is what the training pipeline
# globs; a resumed run must continue the sequence without a gap and without
# regenerating the games it resumed past (which would duplicate rows).
add_test(NAME "viz.datagen_shards" COMMAND test_viz shards)
# The wire format is binary: a renamed field would fail silently at runtime, so
# the encoder must keep producing something the UI decoder understands. CI pairs
# this with the TypeScript side (see the viz-ui job).
add_test(NAME "viz.wire_fixture" COMMAND test_viz fixture "${CMAKE_BINARY_DIR}/viz_frame.bin")

# Search regression fingerprint.
find_program(BASH_EXECUTABLE bash)
if(BASH_EXECUTABLE)
    add_test(NAME "search.bench_signature"
             COMMAND "${BASH_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/bench_signature.sh" $<TARGET_FILE:ChessEngine>)
endif()
