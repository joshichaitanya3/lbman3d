// PR VII-a — NVSHMEM build/link scaffolding.
//
// This translation unit exists to prove that the `LBM_ENABLE_NVSHMEM` build
// path is wired correctly end-to-end:
//   * `find_package(NVSHMEM)` resolves the CUDA-major-matched config,
//   * this file compiles with `-rdc=true` (set per-source in
//     `src/cuda/CMakeLists.txt`),
//   * the final link runs the intermediate device-link step
//     (`CUDA_RESOLVE_DEVICE_SYMBOLS`) so `libnvshmem_device` symbols get
//     resolved by nvlink, and
//   * `libnvshmem_host` survives `--as-needed` linking.
//
// No NVSHMEM initialisation, no exchanges. That work lands in VII-c and
// VII-e/f/g respectively; the current file is scaffold only. See
// `src/cuda/CLAUDE.md` → "NVSHMEM multi-GPU roadmap" for the PR sequence
// and design invariants.

#include <nvshmem.h>
#include <nvshmemx.h>

namespace lbm::nvshmem_stub {

// Device-side reference. `nvshmem_my_pe()` inside a __global__ kernel is
// what actually pulls a symbol out of libnvshmem_device — without a
// device-side call, `-rdc=true` on this TU would be dead weight and the
// device-link step could compile away. The kernel is never launched from
// production code; VII-c will wire an equivalent sanity check post-init.
__global__ void PingPeKernel(int* out) {
    if (out != nullptr && threadIdx.x == 0 && blockIdx.x == 0) {
        *out = nvshmem_my_pe();
    }
}

// Host-side reference. The linker default on this platform is
// `--as-needed`, which drops a shared library if no symbol from it is
// referenced by the executable. Having this wrapper reference
// `nvshmem_my_pe()` on the host side is what keeps `libnvshmem_host.so`
// on the final link line even though nothing calls this function yet.
extern "C" int LbmNvshmemStubHostRef() {
    return nvshmem_my_pe();
}

} // namespace lbm::nvshmem_stub
