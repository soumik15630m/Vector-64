#ifndef INVARIANTS_H
#define INVARIANTS_H

#include <cassert>

namespace Core {

#ifndef NDEBUG
#define ASSERT_CONSISTENCY(pos) \
do { \
if (!(pos).is_ok()) { \
assert(false && "Position consistency check failed"); \
} \
} while (0)
#else
#define ASSERT_CONSISTENCY(pos) do {} while (0)
#endif

}

#endif
