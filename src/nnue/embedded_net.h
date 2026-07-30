#ifndef NNUE_EMBEDDED_NET_H
#define NNUE_EMBEDDED_NET_H

#include <cstddef>

// The default net baked into the NNUE build (target ChessEngine-nnue). The net
// bytes are embedded at build time -- GNU-as `.incbin` on ELF (Linux) and
// Mach-O (macOS), a Windows RCDATA resource on MSVC -- and reached through
// these two accessors, defined per-platform in embedded_net.cpp (compiled into
// the NNUE target only). The classical build never references them.
namespace NNUE {

const unsigned char *embedded_net_data();
std::size_t embedded_net_size();

} // namespace NNUE

#endif
