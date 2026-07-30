#ifndef UCI_H
#define UCI_H

#include <cstddef>

namespace UCI {

// `embeddedNet` (if non-null) is the default net baked into the NNUE binary;
// main() passes it from the embedded image. The classical binary passes null
// and defaults to the material+psqt eval.
int run(const unsigned char *embeddedNet = nullptr,
        std::size_t embeddedNetSize = 0);

} // namespace UCI

#endif
