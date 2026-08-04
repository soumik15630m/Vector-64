#ifndef VIZ_TERMINAL_H
#define VIZ_TERMINAL_H

#include "session.h"

namespace Viz {

// Plain-text live view: the board, the evaluation, the candidate moves and the
// search counters, redrawn in place. For running the engine on a machine with
// no browser, over ssh, or simply preferring a console. Returns when the
// session stops.
int run_terminal(Session &session);

} // namespace Viz

#endif
