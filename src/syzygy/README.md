# Vendored: Fathom (Syzygy tablebase prober)

These files are a verbatim vendoring of **Fathom**, the Syzygy tablebase
probing library, used under the MIT license (see `LICENSE`).

- Upstream: https://github.com/jdart1/Fathom
- Commit: `c9c6fef0dddc05d2e242c183acf5833149ab676d`
- Files: `tbprobe.c`, `tbprobe.h`, `tbchess.c`, `tbconfig.h`, `stdendian.h`

## Build notes

`tbprobe.c` is compiled **as C++** (see `set_source_files_properties(... LANGUAGE
CXX)` in the top-level `CMakeLists.txt`). Fathom then selects `std::atomic` /
`std::mutex`, which every toolchain supports without extra flags — this avoids
MSVC's opt-in C11 `<stdatomic.h>` support.

These files are excluded from `clang-format` (CI) and `clang-tidy`
(`.clang-tidy` here disables all checks) since they are third-party.

The engine wrapper lives in `src/search/syzygy.{h,cpp}`; do not call Fathom
directly from elsewhere. Tablebase *data* files (`.rtbw` / `.rtbz`) are not part
of the repo — point the `SyzygyPath` UCI option at a directory containing them.
