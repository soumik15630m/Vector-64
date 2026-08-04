// ChessEngine-viz: the NNUE live visualizer ("Vector Scope").
//
// Runs the real engine and serves a browser UI that renders what its network is
// actually computing. Deliberately a separate binary from the UCI engines so
// those stay lean -- no HTTP server, no UI bytes, no extra link dependencies.
#include "../cores/attacks.h"
#include "../cores/zobrist.h"
#include "server.h"
#include "session.h"

#ifdef STK_EMBED_NNUE
#include "../nnue/embedded_net.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

namespace {

void usage() {
  std::printf(
      "ChessEngine-viz -- NNUE live visualizer\n\n"
      "  --port <n>        HTTP port (default 7777)\n"
      "  --net <file>      net to load (default: the embedded net)\n"
      "  --nodes <n>       nodes per move (default 20000)\n"
      "  --threads <n>     search threads (default 1)\n"
      "  --hash <mb>       transposition table size (default 32)\n"
      "  --delay <ms>      pause between self-play moves (default 300)\n"
      "  --seed <n>        self-play opening seed\n"
      "  --no-browser      do not open a browser window\n"
      "  --headless        serve without opening a browser and exit on SIGINT\n"
      "  -h, --help        this message\n\n"
      "Binds 127.0.0.1 only: the visualizer is unauthenticated and exposes\n"
      "engine control, so it is a local tool by construction.\n");
}

bool next_int(int argc, char **argv, int &i, int &out) {
  if (i + 1 >= argc)
    return false;
  try {
    out = std::stoi(argv[++i]);
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    Core::Attacks::init();
    Core::Zobrist::init();

    Viz::Config cfg;
    Viz::ServerOptions opts;
    std::string netPath;

    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-h" || a == "--help") {
        usage();
        return 0;
      } else if (a == "--port") {
        if (!next_int(argc, argv, i, opts.port))
          return std::fprintf(stderr, "viz: --port needs a number\n"), 2;
      } else if (a == "--nodes") {
        if (!next_int(argc, argv, i, cfg.nodes))
          return std::fprintf(stderr, "viz: --nodes needs a number\n"), 2;
      } else if (a == "--threads") {
        if (!next_int(argc, argv, i, cfg.threads))
          return std::fprintf(stderr, "viz: --threads needs a number\n"), 2;
      } else if (a == "--hash") {
        if (!next_int(argc, argv, i, cfg.hashMb))
          return std::fprintf(stderr, "viz: --hash needs a number\n"), 2;
      } else if (a == "--delay") {
        if (!next_int(argc, argv, i, cfg.moveDelayMs))
          return std::fprintf(stderr, "viz: --delay needs a number\n"), 2;
      } else if (a == "--seed") {
        int s = 0;
        if (!next_int(argc, argv, i, s))
          return std::fprintf(stderr, "viz: --seed needs a number\n"), 2;
        cfg.seed = static_cast<uint64_t>(s);
      } else if (a == "--net") {
        if (i + 1 >= argc)
          return std::fprintf(stderr, "viz: --net needs a path\n"), 2;
        netPath = argv[++i];
      } else if (a == "--no-browser" || a == "--headless") {
        opts.openBrowser = false;
      } else {
        std::fprintf(stderr, "viz: unknown argument '%s'\n", a.c_str());
        usage();
        return 2;
      }
    }

    Viz::Session session(cfg);

    // Explicit --net wins; otherwise fall back to the net baked into this
    // binary. Without either, the engine still plays (classical eval) but there
    // is no network to visualize, so say so plainly.
    bool loaded = false;
    if (!netPath.empty()) {
      loaded = session.load_net(netPath);
      if (!loaded)
        std::fprintf(stderr, "viz: could not load net '%s'\n", netPath.c_str());
    }
#ifdef STK_EMBED_NNUE
    if (!loaded) {
      loaded = session.load_net_buffer(NNUE::embedded_net_data(),
                                       NNUE::embedded_net_size());
      if (loaded)
        std::printf("viz: using the embedded net (%zu bytes)\n",
                    NNUE::embedded_net_size());
    }
#endif
    if (!loaded)
      std::fprintf(stderr,
                   "viz: WARNING no net loaded -- the engine will use the "
                   "classical evaluation and there is nothing to visualize\n");

    session.start();
    const int rc = Viz::run_server(session, opts);
    session.stop();
    return rc;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[FATAL] %s\n", e.what());
    return 2;
  } catch (...) {
    std::fprintf(stderr, "[FATAL] unknown exception\n");
    return 2;
  }
}
