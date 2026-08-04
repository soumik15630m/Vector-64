#ifndef VIZ_WIRE_H
#define VIZ_WIRE_H

#include "session.h"

#include <string>

// Wire format between the engine and the visualizer UI.
//
// A state message is one binary blob:
//
//   [uint32 LE headerLen][headerLen bytes of UTF-8 JSON][raw buffers ...]
//
// The JSON header describes the session (game, search telemetry, architecture
// constants, scalar frame values) and carries a `frame.buffers` table listing
// each raw array's name, element type and length, in payload order. The client
// slices them straight out of the ArrayBuffer.
//
// Sending the bulk arrays as raw little-endian binary rather than JSON numbers
// or base64 keeps a full frame around 5 KB instead of ~25 KB, and the client
// gets typed arrays with no parsing.
namespace Viz {

// Serialize a snapshot into the framed binary message described above.
std::string encode_state(const Snapshot &s);

// One compact JSON line per frame for session recording: the numbers worth
// keeping, without the multi-kilobyte activation buffers.
std::string encode_record(const Snapshot &s);

// Handle a control command (JSON object) against `session`.
// Returns a JSON response body; `httpStatus` is set to 200 or 400.
std::string handle_control(Session &session, const std::string &body,
                           int &httpStatus);

// Static description of the loaded net for the inspector: weight histograms and
// per-bucket statistics, plus the king-bucket map. Computed on demand.
std::string encode_net_info(const NNUE::Network &net, bool loaded);

} // namespace Viz

#endif
