#ifndef DATAGEN_DATAGEN_H
#define DATAGEN_DATAGEN_H

namespace Datagen {

// Native self-play data generation: `ChessEngine datagen [options]`.
// Reuses the engine's own search + NNUE (all SIMD / incremental accumulator /
// fast paths), so it is much faster than driving the engine over UCI from
// Python. argv[1] is expected to be "datagen"; returns a process exit code.
int run(int argc, char **argv);

} // namespace Datagen

#endif
