#ifndef VIZ_PROBE_H
#define VIZ_PROBE_H

#include "../cores/position.h"
#include "../nnue/network.h"

#include <cstdint>
#include <string>
#include <vector>

// Telemetry extraction for the NNUE visualizer.
//
// This layer is pure data: it reads a position and a loaded net and produces a
// VizFrame describing exactly what the network computed. It performs no I/O and
// holds no state, so it is trivially testable and safe to call from any thread.
//
// It reports the engine's real numbers -- every field is an exact value taken
// from the same forward pass the search would run, never an approximation.
//
// IMPORTANT: capture() rebuilds an accumulator from scratch and runs a full
// forward pass. It is strictly out-of-band tooling and must never be called
// from negamax/quiescence, which would change node counts and the bench
// signature.
namespace Viz {

// One active input feature: which piece, where it sits, and which feature
// column it drives. Lets the UI draw a ray from a board square into the
// accumulator.
struct ActiveFeature {
  int square = 0;         // 0..63, real board square
  int orientedSquare = 0; // 0..63, after the perspective flip/mirror
  int pieceColor = 0;     // Core::Color
  int pieceType = 0;      // Core::PieceType
  int pieceKind = 0; // 0..10, colour-relative kind (friendly P..Q, enemy P..K)
  int featureIndex = 0; // 0..22527
};

// The perspective's own king is the bucket anchor and is not itself a feature.
struct PerspectiveInput {
  int kingSquare = 0;
  int kingBucket = 0; // 0..31
  bool mirrored = false;
  std::vector<ActiveFeature> features;
};

// A weighted edge: how strongly source neuron `index` drove its target.
struct Contribution {
  int index = 0;
  int32_t value = 0; // weight * activation, exact integer
};

// A complete snapshot of one evaluation.
//
// Layout note: the dense stack is stm-relative. `accUs` is the side-to-move
// perspective and `accThem` the opponent's, matching the order the forward pass
// concatenates them into `l1in` (us in [0,512), them in [512,1024)).
struct VizFrame {
  std::string fen;
  int sideToMove = 0;

  PerspectiveInput white;
  PerspectiveInput black;

  // Exact network internals.
  std::vector<int16_t> accUs;   // HIDDEN (1024)
  std::vector<int16_t> accThem; // HIDDEN (1024)
  std::vector<uint8_t> l1in;    // HIDDEN (1024), post-pairwise activations
  std::vector<uint8_t> l1out;   // Arch::L1 (16)
  std::vector<uint8_t> l2out;   // Arch::L2 (32)

  int bucket = 0;     // PSQT/dense bucket actually used, 0..7
  int psqt = 0;       // PSQT side-output term (cp)
  int positional = 0; // dense-stack term (cp)
  int eval = 0;       // psqt + positional, stm-relative (cp)

  // Attribution -- weight * activation, the exact integers the layer summed.
  // outContrib[j]          = outw[j]  * l2out[j]                (L2)
  // l2Contrib[o*L1 + j]    = l2w[...] * l1out[j]                (L2 * L1)
  // l1Top: per L1 neuron, its `l1TopK` strongest sources from l1in, ordered by
  // |contribution| descending and stored neuron-major (L1 * l1TopK).
  std::vector<int32_t> outContrib;
  std::vector<int32_t> l2Contrib;
  int l1TopK = 0;
  std::vector<Contribution> l1Top;
};

// Capture a frame for `pos` using `net`. `net` must be loaded; `l1TopK` is
// clamped to [0, HIDDEN]. Rebuilds the accumulator, so the caller needs no
// search state and any position can be probed directly.
VizFrame capture(const Core::Position &pos, const NNUE::Network &net,
                 int l1TopK = 12);

} // namespace Viz

#endif
