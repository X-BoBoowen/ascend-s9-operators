#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <vector>

#define SET(param) tiling.set_##param(param)
#define ALIGN32(mem) ((mem) / 32u * 32u)
#define CEIL(x, align_num) (((x) + (align_num) - 1) / (align_num) * (align_num))
#define FLOOR(x, align_num) ((x) / (align_num) * (align_num))
#define DEBUG_OUTPUT 1

using std::max;
using std::min;
using namespace ge;