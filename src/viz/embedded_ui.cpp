#include "embedded_ui.h"

// Platform accessors for the embedded UI bundle. Mirrors
// src/nnue/embedded_net.cpp: MSVC uses a Windows RCDATA resource, every other
// toolchain uses two labels bracketing a GNU-as .incbin.
//
// Without -DSTK_EMBED_UI the accessors report "no bundle" and the server falls
// back to a built-in placeholder page.

#if !defined(STK_EMBED_UI)

namespace Viz {
const char *embedded_ui_data() { return nullptr; }
std::size_t embedded_ui_size() { return 0; }
} // namespace Viz

#elif defined(_MSC_VER)

#include <windows.h>

namespace Viz {
namespace {
HRSRC find_res() {
  // MAKEINTRESOURCEW(10) == the wide RT_RCDATA.
  return FindResourceW(nullptr, L"STK_EMBEDDED_UI", MAKEINTRESOURCEW(10));
}
} // namespace

const char *embedded_ui_data() {
  return static_cast<const char *>(
      LockResource(LoadResource(nullptr, find_res())));
}

std::size_t embedded_ui_size() {
  return static_cast<std::size_t>(SizeofResource(nullptr, find_res()));
}
} // namespace Viz

#else

extern "C" {
extern const char stk_embedded_ui_data[];
extern const char stk_embedded_ui_end[];
}

namespace Viz {

const char *embedded_ui_data() { return stk_embedded_ui_data; }

std::size_t embedded_ui_size() {
  return static_cast<std::size_t>(stk_embedded_ui_end - stk_embedded_ui_data);
}

} // namespace Viz

#endif
