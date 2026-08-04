#ifndef VIZ_EMBEDDED_UI_H
#define VIZ_EMBEDDED_UI_H

#include <cstddef>

// The visualizer UI is a single self-contained HTML file baked into the binary
// the same way the default net is (GNU-as .incbin / MSVC RCDATA), so the tool
// ships as one executable with no asset directory to lose.
//
// Returns nullptr / 0 when the binary was built without a bundle; the server
// then serves a placeholder page and the engine stream still works.
namespace Viz {

const char *embedded_ui_data();
std::size_t embedded_ui_size();

} // namespace Viz

#endif
