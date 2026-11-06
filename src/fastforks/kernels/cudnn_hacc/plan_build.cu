// One half-compute conv plan in one host call: what `_backend.op_graph` + the first heuristic config + `plan` do
// through ~45 ctypes calls (~0.15 ms per geometry), here ~0.07 ms (most of it cuDNN's heuristic). Host code only; it
// calls the cuDNN backend API through the function pointers it is given (the libcudnn PyTorch loaded), so it links
// against no cuDNN. The attribute numbers are those of `_backend` (the cuDNN 9 ABI).
#include <stdint.h>
#include <string.h>

typedef int (*create_t)(int, void **);
typedef int (*destroy_t)(void *);
typedef int (*finalize_t)(void *);
typedef int (*set_t)(void *, int, int, int64_t, const void *);
typedef int (*get_t)(void *, int, int, int64_t, int64_t *, void *);
struct ffh_api {
  create_t create;
  destroy_t destroy;
  finalize_t finalize;
  set_t set;
  get_t get;
};

enum { ENGINE = 2, ENGINECFG = 3, ENGINEHEUR = 4, EXECUTION_PLAN = 5, OP_FWD = 10, OP_BWD = 12, GRAPH = 15, TENSOR = 17 };
enum { T_HANDLE = 0, T_DATA_TYPE = 1, T_BOOLEAN = 2, T_INT64 = 3, T_FLOAT = 4, T_HEUR_MODE = 8, T_DESC = 15, T_NOTE = 19 };
enum { X, Y, OP, GRAPH_, HEUR, CFG, PLAN, KEEP };  // the descriptors this call creates
#define CK(e)              \
  do {                     \
    int s_ = (e);          \
    if (s_) return s_;     \
  } while (0)

static int tensor(const ffh_api *a, void **t, int64_t uid, int nd, const int64_t *dims, const int64_t *strides) {
  int32_t half = 2;
  int64_t alignment = 16;
  bool virt = false;
  CK(a->create(TENSOR, t));
  CK(a->set(*t, 906, T_INT64, 1, &uid));
  CK(a->set(*t, 901, T_DATA_TYPE, 1, &half));
  CK(a->set(*t, 900, T_INT64, 1, &alignment));
  CK(a->set(*t, 902, T_INT64, nd, dims));
  CK(a->set(*t, 903, T_INT64, nd, strides));
  CK(a->set(*t, 907, T_BOOLEAN, 1, &virt));
  return a->finalize(*t);
}

static int build(const ffh_api *a, void *handle, int nd, const int64_t *shapes, void *w, void *conv, int dgrad,
                 int mode, const int64_t *skip, int nskip, void **keep, int64_t *workspace, int64_t *engine) {
  CK(tensor(a, &keep[X], 1, nd, shapes, shapes + nd));
  CK(tensor(a, &keep[Y], 3, nd, shapes + 2 * nd, shapes + 3 * nd));
  float one = 1.f, zero = 0.f;
  CK(a->create(dgrad ? OP_BWD : OP_FWD, &keep[OP]));
  void *op = keep[OP];
  if (dgrad) {  // the data gradient: dy = the transposed conv's input, dx = its output
    CK(a->set(op, 708, T_DESC, 1, &conv));
    CK(a->set(op, 709, T_DESC, 1, &w));
    CK(a->set(op, 711, T_DESC, 1, &keep[X]));
    CK(a->set(op, 710, T_DESC, 1, &keep[Y]));
    CK(a->set(op, 706, T_FLOAT, 1, &one));
    CK(a->set(op, 707, T_FLOAT, 1, &zero));
  } else {
    CK(a->set(op, 702, T_DESC, 1, &conv));
    CK(a->set(op, 704, T_DESC, 1, &keep[X]));
    CK(a->set(op, 703, T_DESC, 1, &w));
    CK(a->set(op, 705, T_DESC, 1, &keep[Y]));
    CK(a->set(op, 700, T_FLOAT, 1, &one));
    CK(a->set(op, 701, T_FLOAT, 1, &zero));
  }
  CK(a->finalize(op));
  CK(a->create(GRAPH, &keep[GRAPH_]));
  CK(a->set(keep[GRAPH_], 800, T_HANDLE, 1, &handle));
  CK(a->set(keep[GRAPH_], 801, T_DESC, 1, &op));
  CK(a->finalize(keep[GRAPH_]));
  CK(a->create(ENGINEHEUR, &keep[HEUR]));
  int32_t heur_mode = mode;
  CK(a->set(keep[HEUR], 201, T_DESC, 1, &keep[GRAPH_]));
  CK(a->set(keep[HEUR], 200, T_HEUR_MODE, 1, &heur_mode));
  CK(a->finalize(keep[HEUR]));
  CK(a->create(ENGINECFG, &keep[CFG]));
  int64_t n = 0;
  CK(a->get(keep[HEUR], 202, T_DESC, 1, &n, &keep[CFG]));  // only the first config is finalized
  if (n < 1) return -1;
  void *eng = 0;
  CK(a->create(ENGINE, &eng));
  int64_t index = -1, nnotes = 0;
  int32_t notes[32];
  int st = a->get(keep[CFG], 300, T_DESC, 1, &n, &eng);
  if (!st) st = a->get(eng, 1301, T_INT64, 1, &n, &index);
  if (!st) st = a->get(eng, 1305, T_NOTE, 32, &nnotes, notes);
  a->destroy(eng);
  if (st) return st;
  for (int i = 0; i < nskip; i++)
    if (index == skip[i]) return -1;
  for (int64_t i = 0; i < nnotes; i++)
    if (notes[i] == 0) return -1;  // needs runtime compilation
  *engine = index;
  CK(a->create(EXECUTION_PLAN, &keep[PLAN]));
  CK(a->set(keep[PLAN], 400, T_HANDLE, 1, &handle));
  CK(a->set(keep[PLAN], 401, T_DESC, 1, &keep[CFG]));
  CK(a->finalize(keep[PLAN]));
  return a->get(keep[PLAN], 402, T_INT64, 1, &n, workspace);
}

// The plan of the first config of heuristic `mode` for one conv (`dgrad`: the transposed conv x -> y) on the cached
// weight tensor `w` and conv descriptor `conv`. `shapes` = x dims, x strides, y dims, y strides (`nd` each).
// Returns 0 with keep = {x, y, op, graph, 0, cfg, plan} (the caller owns them), else nonzero with all of them
// destroyed: a cuDNN status, or -1 if the first config is not usable (no config, an engine in `skip`, runtime
// compilation, or no plan), where the caller's full query decides.
extern "C" int ffh_build(const ffh_api *a, void *handle, int nd, const int64_t *shapes, void *w, void *conv,
                         int dgrad, int mode, const int64_t *skip, int nskip, void **keep, int64_t *workspace,
                         int64_t *engine) {
  memset(keep, 0, KEEP * sizeof(void *));
  int st = build(a, handle, nd, shapes, w, conv, dgrad, mode, skip, nskip, keep, workspace, engine);
  for (int i = KEEP - 1; i >= 0; i--) {
    if (keep[i] && (st || i == HEUR)) {
      a->destroy(keep[i]);
      keep[i] = 0;
    }
  }
  return st;
}
