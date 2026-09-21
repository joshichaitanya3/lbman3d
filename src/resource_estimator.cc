// Standalone memory-budget and SLURM-generation tool for lbman3d.
//
// Computes host-RAM and device-DRAM requirements from the field inventory in
// FluidFields, QTensorFields, and DeviceFields.  No project physics headers are
// included — field counts are taken directly from the documented tables in
// src/cuda/CLAUDE.md so the tool never goes stale from a compile-time params.h.
//
// Build-mode flags (GPU/MPI/NVSHMEM), compiled-in grid dims (nx/ny/nz), and
// OMP thread count are baked in by CMake via cmake/resource_estimator_config.h.in.
//
// Usage:
//   resource_estimator NX NY NZ            CPU single-rank memory estimate
//   resource_estimator N                   CPU cube  (NX = NY = NZ = N)
//   resource_estimator gpu NX NY NZ        GPU: device DRAM + host snapshot
//   resource_estimator gpu N               GPU cube
//   resource_estimator slurm [NX NY NZ]    Generate SLURM job script
//   resource_estimator slurm [N]           Same with cube domain
//   resource_estimator --list-clusters     List known cluster configurations
//
// Slurm options:
//   --cluster <id>           perlmutter | generic (default: perlmutter)
//   --account <acct>         HPC project account number
//   --ranks <N>              Total MPI ranks (default: 1)
//   --omp-threads <T>        OMP threads/rank — CPU builds only
//                            (default: kNumOMPThreads from compiled params.h)
//   --time <HH:MM:SS>        Walltime override (default: cluster maximum)
//   --gpu-variant <40g|80g>  GPU memory tier — GPU builds only (default: 40g)
//   --output <file>          Output path (default: job.slurm)

#include "resource_estimator_config.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// ── Field inventory ──────────────────────────────────────────────────────────
// Doubles per cell for each logical group, matching FluidFields / QTensorFields
// / DeviceFields.  `sym_heap` marks fields that go on the NVSHMEM symmetric
// heap (halo-exchanged between ranks).

struct FieldGroup {
    const char* label;
    const char* note;
    int         dpc;       // doubles per cell
    bool        sym_heap;  // on NVSHMEM symmetric heap?
};

// D3Q15 has 15 velocity directions; Q-tensor has 5 independent components.
static constexpr FieldGroup kGroups[] = {
    { "LBM: f, f_new",
      "D3Q15 populations, double-buffered",          30, true  },
    { "LBM: rho, ux, uy, uz",
      "macroscopic moments",                          4, false },
    { "LBM: fx, fy, fz",
      "body force (written by QTensorSolver)",        3, false },
    { "Q: q (current)",
      "5-comp. traceless-symmetric tensor",           5, true  },
    { "Q: q_new (scratch)",
      "forward-Euler buffer, swapped each step",      5, false },
    { "Q: Sigma (passive symmetric)",
      "symmetric nematic stress, 5 components",       5, true  },
    { "Q: Tau (passive antisymmetric)",
      "antisymmetric (torque) stress, 3 components",  3, true  },
};
static constexpr int kNumGroups = static_cast<int>(sizeof kGroups / sizeof kGroups[0]);

static int totalDpc() {
    int t = 0;
    for (int i = 0; i < kNumGroups; ++i) t += kGroups[i].dpc;
    return t;  // 55
}
static int symDpc() {
    int t = 0;
    for (int i = 0; i < kNumGroups; ++i)
        if (kGroups[i].sym_heap) t += kGroups[i].dpc;
    return t;  // 43
}

// ── Cluster configuration ────────────────────────────────────────────────────
//
// To add a new cluster:
//  1. Add a NodeSpec for each partition (CPU, GPU) describing cores, RAM, GPUs.
//  2. Add a PartitionSpec with the SLURM queue name, optional -C constraint,
//     and maximum walltime string.
//  3. Add a ClusterSpec entry in kClusters with module setup strings.
//     Module strings are newline-separated; nullptr or "" = omit that section.
//
// The `setup_mpi` and `setup_nvshmem` lines are appended only when the binary
// was built with MPI or NVSHMEM respectively — you do not need to guard them.

struct NodeSpec {
    int    cores_per_node;
    double ram_gib;         // total host DRAM (GiB)
    int    gpus_per_node;   // 0 = CPU-only partition
    double gpu_mem_gib;     // per-GPU HBM (GiB); 0 if no GPUs
};

struct PartitionSpec {
    const char* queue;          // SLURM -q value
    const char* constraint;     // SLURM -C value (nullptr = omit directive)
    NodeSpec    node;
    const char* max_walltime;   // "HH:MM:SS" — used as default --time
};

struct ClusterSpec {
    const char*   id;           // CLI key passed to --cluster
    const char*   display_name;
    const char*   setup_base_cpu;   // module/env lines for CPU builds
    const char*   setup_base_gpu;   // module/env lines for GPU builds
    const char*   setup_mpi;        // extra lines appended if MPI build
    const char*   setup_nvshmem;    // extra lines appended if NVSHMEM build
    PartitionSpec cpu;
    PartitionSpec gpu;              // gpu.node.gpus_per_node == 0 → no GPU partition
    const char*   gpu_80g_constraint; // SLURM -C for 80 GB variant (nullptr = none)
};

static const ClusterSpec kClusters[] = {
    {
        // ── NERSC Perlmutter (HPE Cray EX) ──────────────────────────────────
        // CPU nodes: 2× AMD EPYC 7763 (128 cores), 512 GiB DDR4, 1× Slingshot NIC
        // GPU nodes: 1× AMD EPYC 7763 (64 cores), 4× NVIDIA A100, 256 GiB DDR4
        //            4× Slingshot NICs, PCIe 4.0 GPU↔CPU, NVLink between GPUs
        "perlmutter",
        "NERSC Perlmutter (HPE Cray EX)",
        // setup_base_cpu
        "module load PrgEnv-gnu",
        // setup_base_gpu
        "module load PrgEnv-gnu\nmodule load cudatoolkit",
        // setup_mpi (appended for MPI builds)
        "module load cray-mpich",
        // setup_nvshmem (appended for NVSHMEM builds)
        "module load nvshmem",
        // CPU partition
        { "regular", "cpu", { 128, 512.0, 0, 0.0 }, "24:00:00" },
        // GPU partition — default A100 40 GB
        { "regular", "gpu", {  64, 256.0, 4, 40.0 }, "24:00:00" },
        // 80 GB HBM2e variant constraint
        "gpu&hbm80g",
    },
    {
        // ── Generic HPC cluster (user-editable placeholder) ──────────────────
        // Fill in cores_per_node, ram_gib, gpus_per_node, gpu_mem_gib,
        // queue names, and module setup for your cluster.
        "generic",
        "Generic HPC cluster",
        "# TODO: module load <compiler>",
        "# TODO: module load <compiler> <cuda>",
        "# TODO: module load <mpi>",
        "# TODO: module load <nvshmem>",
        { "regular", nullptr, { 128, 256.0,  0, 0.0 }, "24:00:00" },
        { "regular", nullptr, {   0,   0.0,  0, 0.0 }, "24:00:00" },
        nullptr,
    },
};
static constexpr int kNumClusters = static_cast<int>(sizeof kClusters / sizeof kClusters[0]);

static const ClusterSpec* findCluster(const char* name) {
    for (int i = 0; i < kNumClusters; ++i)
        if (std::strcmp(kClusters[i].id, name) == 0)
            return &kClusters[i];
    return nullptr;
}

// ── Formatting ───────────────────────────────────────────────────────────────

static std::string fmtBytes(double bytes) {
    const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    int u = 0;
    double v = bytes;
    while (v >= 1024.0 && u < 4) { v /= 1024.0; ++u; }
    char buf[32];
    if (v >= 100.0)      snprintf(buf, sizeof buf, "%.1f %s", v, units[u]);
    else if (v >= 10.0)  snprintf(buf, sizeof buf, "%.2f %s", v, units[u]);
    else                 snprintf(buf, sizeof buf, "%.3f %s", v, units[u]);
    return buf;
}

static double groupBytes(long long cells, int dpc) {
    return static_cast<double>(cells) * static_cast<double>(dpc) * 8.0;
}

static void printSep() {
    printf("  %s\n",
           "--------------------------------------------------------------------");
}

static void printGroupRow(int i, long long cells, bool annotate_sym = false) {
    const FieldGroup& g = kGroups[i];
    std::string sz = fmtBytes(groupBytes(cells, g.dpc));
    printf("  %-34s  %2d d/cell  %11s", g.label, g.dpc, sz.c_str());
    if (annotate_sym && g.sym_heap) printf("   [sym-heap]");
    printf("\n");
}

// ── CPU estimate ─────────────────────────────────────────────────────────────

static void cpuEstimate(long long nx, long long ny, long long nz) {
    long long V   = nx * ny * nz;
    int       dpc = totalDpc();
    double    tot = groupBytes(V, dpc);

    printf("\nResource estimate — %lld x %lld x %lld  [CPU, single rank]\n",
           nx, ny, nz);
    printf("  %lld cells  |  %d doubles/cell  |  %d B/cell\n\n",
           V, dpc, dpc * 8);
    printf("  %-34s  %s    %s\n", "Field group", "doubles", "Size");
    printSep();
    for (int i = 0; i < kNumGroups; ++i)
        printGroupRow(i, V, false);
    printSep();
    printf("  %-34s  %2d d/cell  %11s\n",
           "Total host RAM", dpc, fmtBytes(tot).c_str());
    printf("\n");
}

// ── GPU estimate ─────────────────────────────────────────────────────────────

static void gpuEstimate(long long nx, long long ny, long long nz) {
    long long V    = nx * ny * nz;
    int       dpc  = totalDpc();
    int       sdpc = symDpc();
    double    dev  = groupBytes(V, dpc);
    double    host = groupBytes(V, dpc);
    double    sym  = groupBytes(V, sdpc);
    double    reg  = groupBytes(V, dpc - sdpc);

    printf("\nResource estimate — %lld x %lld x %lld  [GPU + host, single rank]\n",
           nx, ny, nz);
    printf("  %lld cells  |  %d doubles/cell  |  %d B/cell\n\n",
           V, dpc, dpc * 8);

    printf("  Device DRAM:\n");
    printf("  %-34s  %s    %s\n", "Field group", "doubles", "Size");
    printSep();
    for (int i = 0; i < kNumGroups; ++i)
        printGroupRow(i, V, true);
    printSep();
    printf("  %-34s  %2d d/cell  %11s\n",
           "Device total", dpc, fmtBytes(dev).c_str());

    printf("\n  Host snapshot (FluidFields + QTensorFields):\n");
    printf("  %-34s  %2d d/cell  %11s\n",
           "Host total", dpc, fmtBytes(host).c_str());

    printSep();
    printf("  %-34s             %11s\n",
           "Grand total (device + host)", fmtBytes(dev + host).c_str());

    printf("\n  NVSHMEM symmetric-heap hint (LBM_ENABLE_NVSHMEM, per-PE):\n");
    printf("    sym-heap  %2d d/cell: %s"
           "  (d_f/f_new, d_q, d_Sigma, d_Tau)\n",
           sdpc, fmtBytes(sym).c_str());
    printf("    regular   %2d d/cell: %s"
           "  (d_q_new, moments, body force)\n",
           dpc - sdpc, fmtBytes(reg).c_str());

    // Per-card fit table
    struct Card { const char* label; double usable_gib; };
    Card cards[] = {
        { " 8 GB", 7.2  },
        {"16 GB", 14.5  },
        {"24 GB", 22.0  },
        {"40 GB", 38.0  },
        {"48 GB", 45.0  },
        {"80 GB", 74.0  },
    };
    int ncards = static_cast<int>(sizeof cards / sizeof cards[0]);

    printf("\n  Device DRAM vs. GPU cards (usable = total - ~0.8 GiB overhead):\n");
    for (int k = 0; k < ncards; ++k) {
        double usable = cards[k].usable_gib * (1024.0 * 1024.0 * 1024.0);
        long long max_cells = static_cast<long long>(usable / (static_cast<double>(dpc) * 8.0));
        long long cube      = static_cast<long long>(std::cbrt(static_cast<double>(max_cells)));
        printf("    %-6s  usable ~%4.1f GiB  max cells %9lld  max cube ~%lld^3  %s\n",
               cards[k].label, cards[k].usable_gib, max_cells, cube,
               dev <= usable ? "fits" : "too large");
    }
    printf("\n");
}

// ── SLURM generation ─────────────────────────────────────────────────────────

static int ceilDiv(int a, int b) { return (a + b - 1) / b; }

struct SlurmOpts {
    const char* cluster;
    const char* account;
    int         ranks;
    int         omp_threads;   // per MPI rank
    const char* walltime;      // nullptr = use cluster default
    bool        gpu_80g;
    const char* output_file;   // nullptr or "" = stdout
};

// Write newline-separated module lines; skips null or empty strings.
static void writeModuleLines(FILE* f, const char* lines) {
    if (!lines || lines[0] == '\0') return;
    const char* p = lines;
    while (*p) {
        const char* end = p;
        while (*end && *end != '\n') ++end;
        if (end > p) fprintf(f, "%.*s\n", static_cast<int>(end - p), p);
        if (*end == '\n') ++end;
        p = end;
    }
}

// Returns the largest integer <= cap that divides n exactly.
static int largestDivisorAtMost(int n, int cap) {
    int best = 1;
    for (int d = 1; d <= cap; ++d)
        if (n % d == 0) best = d;
    return best;
}

// Returns the smallest integer >= floor that divides n exactly.
static int smallestDivisorAtLeast(int n, int floor_val) {
    for (int d = floor_val; d <= n; ++d)
        if (n % d == 0) return d;
    return n;
}

static bool generateSlurm(long long nx, long long ny, long long nz,
                           const SlurmOpts& opts) {
    const ClusterSpec* cs = findCluster(opts.cluster);
    if (!cs) {
        fprintf(stderr, "error: unknown cluster '%s'\n", opts.cluster);
        fprintf(stderr, "       known clusters:");
        for (int i = 0; i < kNumClusters; ++i)
            fprintf(stderr, " %s", kClusters[i].id);
        fprintf(stderr, "\n");
        return false;
    }

    const bool is_gpu     = (LBM_BUILD_GPU != 0);
    const bool is_mpi     = (LBM_BUILD_MPI != 0);
    const bool is_nvshmem = (LBM_BUILD_NVSHMEM != 0);

    const PartitionSpec& part = is_gpu ? cs->gpu : cs->cpu;
    const NodeSpec&      node = part.node;

    if (is_gpu && node.gpus_per_node == 0) {
        fprintf(stderr, "warning: cluster '%s' has no GPU partition defined; "
                        "falling back to CPU partition\n", opts.cluster);
    }

    // Warn when the user requests multiple ranks but the binary has no MPI/GPU.
    // A single-rank build launched over N nodes via srun runs N independent
    // copies with no communication — almost certainly not what is wanted.
    bool effective_mpi = is_gpu || is_mpi;
    int  effective_ranks = opts.ranks;
    if (opts.ranks > 1 && !effective_mpi) {
        fprintf(stderr,
            "warning: --ranks %d requested but this binary was built without MPI\n"
            "         A single-rank CPU build runs only one process regardless of node count.\n"
            "         The SLURM script will request 1 node / 1 task.\n"
            "         Rebuild with -DLBM_ENABLE_MPI=ON to use multiple ranks.\n",
            opts.ranks);
        effective_ranks = 1;
    }

    // ── Node / task arithmetic ───────────────────────────────────────────────
    int tasks_per_node, nodes;
    if (is_gpu && node.gpus_per_node > 0) {
        // Each MPI rank owns exactly one GPU.
        tasks_per_node = node.gpus_per_node;
        nodes = ceilDiv(effective_ranks, tasks_per_node);
    } else {
        // CPU: pack as many MPI ranks as possible given the OMP thread count.
        int t = std::max(1, opts.omp_threads);
        tasks_per_node = std::max(1, node.cores_per_node / t);
        nodes = ceilDiv(effective_ranks, tasks_per_node);
    }

    // ── OMP efficiency check (CPU builds only) ───────────────────────────────
    // Warn when kNumOMPThreads does not divide cores_per_node evenly, i.e.
    // some cores will sit idle.  Suggest the nearest divisors so the user can
    // update params.h::kNumOMPThreads before the next build.
    bool omp_efficient = true;
    if (!is_gpu && node.cores_per_node > 0) {
        int used = tasks_per_node * opts.omp_threads;
        if (used != node.cores_per_node) {
            omp_efficient = false;
            int idle = node.cores_per_node - used;
            fprintf(stderr,
                "warning: kNumOMPThreads=%d → %d tasks/node × %d threads = %d/%d cores used "
                "(%d idle)\n",
                opts.omp_threads, tasks_per_node, opts.omp_threads,
                used, node.cores_per_node, idle);
            int suggest_lo = largestDivisorAtMost(node.cores_per_node, opts.omp_threads);
            int suggest_hi = smallestDivisorAtLeast(node.cores_per_node, opts.omp_threads + 1);
            fprintf(stderr,
                "         For full core utilisation set kNumOMPThreads=%d",
                suggest_lo);
            if (suggest_hi != suggest_lo && suggest_hi <= node.cores_per_node)
                fprintf(stderr, " or %d", suggest_hi);
            fprintf(stderr,
                " in params.h (both divide %d evenly)\n", node.cores_per_node);
        }
    }

    // ── Memory fit check and auto-scaling ────────────────────────────────────
    // Each MPI rank holds 1/N of the domain; for GPU, device and host each
    // hold one copy.  We compute the total domain footprint and then derive
    // per-rank usage from the effective rank count.
    long long V          = nx * ny * nz;
    int       dpc        = totalDpc();
    double    total_mem  = groupBytes(V, dpc);  // full domain across all ranks

    // Per-rank budget — the binding constraint for scaling.
    // For GPU: device DRAM per rank is the usual bottleneck; host snapshot
    // also eats one copy of the domain from CPU RAM, so we take the tighter
    // of the two limits.  For CPU: host RAM divided equally among tasks.
    static constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    double gpu_mem_gib = opts.gpu_80g && cs->gpu_80g_constraint ? 80.0
                                                                 : node.gpu_mem_gib;
    double device_budget = is_gpu ? gpu_mem_gib * kGiB : 1e300;
    double host_budget   = (node.ram_gib / static_cast<double>(std::max(1, tasks_per_node)))
                           * kGiB;
    double budget_per_rank = is_gpu ? std::min(device_budget, host_budget) : host_budget;

    // Keep 10 % headroom so we don't sit right at the hardware limit.
    double usable_per_rank = budget_per_rank * 0.9;

    // Minimum ranks to fit the domain with headroom.
    int min_ranks = static_cast<int>(std::ceil(total_mem / usable_per_rank));
    min_ranks     = std::max(1, min_ranks);

    // Auto-scale: if the requested rank count is too low, bump it up to the
    // next clean node boundary so #SBATCH -N is always an integer.
    // Only possible for builds that actually support multiple ranks (MPI/GPU).
    // For single-rank CPU builds, emit a warning instead.
    int original_ranks = effective_ranks;
    bool auto_scaled   = false;
    if (effective_ranks < min_ranks) {
        if (!effective_mpi) {
            // Cannot auto-scale: no parallelism compiled in.  Warn and suggest.
            int suggested = ceilDiv(min_ranks, tasks_per_node) * tasks_per_node;
            fprintf(stderr,
                "warning: domain needs %s but single-rank budget is %s usable\n"
                "         Rebuild with -DLBM_ENABLE_MPI=ON and pass --ranks %d\n"
                "         to distribute the domain across %d node%s.\n",
                fmtBytes(total_mem).c_str(),
                fmtBytes(usable_per_rank).c_str(),
                suggested,
                suggested / tasks_per_node,
                suggested / tasks_per_node != 1 ? "s" : "");
        } else {
            auto_scaled     = true;
            // Round up to the next full set of nodes.
            effective_ranks = ceilDiv(min_ranks, tasks_per_node) * tasks_per_node;
            nodes           = effective_ranks / tasks_per_node;
            fprintf(stderr,
                "info: domain needs %s total; %s usable/rank → minimum %d ranks\n"
                "      auto-scaled from %d to %d ranks (%d node%s × %d tasks/node)\n"
                "      Pass --ranks %d to suppress this message.\n",
                fmtBytes(total_mem).c_str(),
                fmtBytes(usable_per_rank).c_str(),
                min_ranks,
                original_ranks, effective_ranks,
                nodes, nodes != 1 ? "s" : "", tasks_per_node,
                effective_ranks);
        }
    }

    double mem_per_rank = total_mem / static_cast<double>(effective_ranks);

    // ── Constraint / walltime ────────────────────────────────────────────────
    const char* constraint = part.constraint;
    if (is_gpu && opts.gpu_80g && cs->gpu_80g_constraint)
        constraint = cs->gpu_80g_constraint;
    const char* wt =
        (opts.walltime && opts.walltime[0]) ? opts.walltime : part.max_walltime;

    // ── Build mode label ─────────────────────────────────────────────────────
    const char* mode_label =
        is_nvshmem ? "CUDA+NVSHMEM+MPI" :
        (is_gpu && is_mpi) ? "CUDA+MPI" :
        is_gpu  ? "CUDA" :
        is_mpi  ? "CPU+MPI" : "CPU (single-rank)";

    // ── Open output ──────────────────────────────────────────────────────────
    FILE* f     = stdout;
    bool  own_f = false;
    if (opts.output_file && opts.output_file[0]) {
        f = fopen(opts.output_file, "w");
        if (!f) {
            fprintf(stderr, "error: cannot open '%s' for writing\n", opts.output_file);
            return false;
        }
        own_f = true;
    }

    // ── Header comment ───────────────────────────────────────────────────────
    fprintf(f, "#!/bin/bash\n");
    fprintf(f, "# lbman3d SLURM job script — auto-generated by resource_estimator\n");
    fprintf(f, "# Cluster  : %s\n", cs->display_name);
    fprintf(f, "# Grid     : %lld x %lld x %lld  (%lld cells)\n", nx, ny, nz, V);
    fprintf(f, "# Build    : %s\n", mode_label);
    fprintf(f, "# Memory   : %s total  |  %s/rank  (budget: %.1f %s/rank)\n",
            fmtBytes(total_mem).c_str(),
            fmtBytes(mem_per_rank).c_str(),
            is_gpu ? gpu_mem_gib : node.ram_gib / tasks_per_node,
            is_gpu ? "GiB GPU" : "GiB CPU RAM");
    fprintf(f, "# Layout   : %d rank%s  →  %d node%s × %d task%s/node%s\n",
            effective_ranks, effective_ranks != 1 ? "s" : "",
            nodes,           nodes           != 1 ? "s" : "",
            tasks_per_node,  tasks_per_node  != 1 ? "s" : "",
            auto_scaled ? "  [ranks auto-scaled to fit domain]" : "");
    if (!is_gpu) {
        int used = tasks_per_node * opts.omp_threads;
        fprintf(f, "# OMP      : %d thread%s/rank  (%d/%d cores/node used%s)\n",
                opts.omp_threads, opts.omp_threads != 1 ? "s" : "",
                used, node.cores_per_node,
                omp_efficient ? "" : " — see warning above");
        if (!omp_efficient) {
            int suggest_lo = largestDivisorAtMost(node.cores_per_node, opts.omp_threads);
            int suggest_hi = smallestDivisorAtLeast(node.cores_per_node, opts.omp_threads + 1);
            fprintf(f, "#            To use all %d cores, set kNumOMPThreads=%d",
                    node.cores_per_node, suggest_lo);
            if (suggest_hi != suggest_lo && suggest_hi <= node.cores_per_node)
                fprintf(f, " or %d", suggest_hi);
            fprintf(f, " in params.h\n");
        }
    }
    fprintf(f, "#\n");

    // ── SBATCH directives ────────────────────────────────────────────────────
    fprintf(f, "#SBATCH --job-name=lbman3d_%lldx%lldx%lld\n", nx, ny, nz);
    if (opts.account && opts.account[0])
        fprintf(f, "#SBATCH -A %s\n", opts.account);
    if (constraint)
        fprintf(f, "#SBATCH -C %s\n", constraint);
    fprintf(f, "#SBATCH -q %s\n", part.queue);
    fprintf(f, "#SBATCH -N %d\n", nodes);
    fprintf(f, "#SBATCH --ntasks-per-node=%d\n", tasks_per_node);
    if (is_gpu && node.gpus_per_node > 0)
        fprintf(f, "#SBATCH --gpus-per-task=1\n");
    if (!is_gpu && opts.omp_threads > 1)
        fprintf(f, "#SBATCH --cpus-per-task=%d\n", opts.omp_threads);
    fprintf(f, "#SBATCH -t %s\n", wt);
    fprintf(f, "#SBATCH --output=lbman3d-%%j.out\n");
    fprintf(f, "\n");

    // ── Module setup ─────────────────────────────────────────────────────────
    writeModuleLines(f, is_gpu ? cs->setup_base_gpu : cs->setup_base_cpu);
    if (is_mpi)     writeModuleLines(f, cs->setup_mpi);
    if (is_nvshmem) writeModuleLines(f, cs->setup_nvshmem);
    fprintf(f, "\n");

    // ── Environment / launch ─────────────────────────────────────────────────
    if (!is_gpu && opts.omp_threads > 1)
        fprintf(f, "export OMP_NUM_THREADS=%d\n\n", opts.omp_threads);

    if (is_gpu && node.gpus_per_node > 0)
        fprintf(f, "srun --ntasks-per-node=%d --gpus-per-task=1 ./build/main\n",
                tasks_per_node);
    else if (is_mpi)
        fprintf(f, "srun ./build/main\n");
    else
        fprintf(f, "./build/main\n");

    if (own_f) {
        fclose(f);
        printf("SLURM script written → %s\n", opts.output_file);
        printf("  Cluster  : %s\n", cs->display_name);
        printf("  Grid     : %lld x %lld x %lld  |  Build: %s\n",
               nx, ny, nz, mode_label);
        printf("  Memory   : %s total  |  %s/rank\n",
               fmtBytes(total_mem).c_str(), fmtBytes(mem_per_rank).c_str());
        printf("  Layout   : %d rank%s  →  %d node%s × %d task%s/node%s\n",
               effective_ranks, effective_ranks != 1 ? "s" : "",
               nodes,           nodes           != 1 ? "s" : "",
               tasks_per_node,  tasks_per_node  != 1 ? "s" : "",
               auto_scaled ? "  [auto-scaled]" : "");
    }

    return true;
}

// ── Cluster list ─────────────────────────────────────────────────────────────

static void listClusters() {
    printf("Known clusters (pass to --cluster):\n\n");
    for (int i = 0; i < kNumClusters; ++i) {
        const ClusterSpec& c = kClusters[i];
        printf("  %-16s  %s\n", c.id, c.display_name);
        const NodeSpec& cn = c.cpu.node;
        if (cn.cores_per_node > 0)
            printf("    CPU  %-14s  %d cores  %.0f GiB RAM   queue=%-10s  max=%s\n",
                   c.cpu.constraint ? c.cpu.constraint : "(no constraint)",
                   cn.cores_per_node, cn.ram_gib, c.cpu.queue, c.cpu.max_walltime);
        const NodeSpec& gn = c.gpu.node;
        if (gn.gpus_per_node > 0) {
            printf("    GPU  %-14s  %d GPUs (%.0f GiB/GPU)   queue=%-10s  max=%s\n",
                   c.gpu.constraint ? c.gpu.constraint : "(no constraint)",
                   gn.gpus_per_node, gn.gpu_mem_gib, c.gpu.queue, c.gpu.max_walltime);
            if (c.gpu_80g_constraint)
                printf("    GPU  %-14s  80 GiB variant\n", c.gpu_80g_constraint);
        }
        printf("\n");
    }
}

// ── Usage ────────────────────────────────────────────────────────────────────

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s NX NY NZ              CPU single-rank memory estimate\n"
        "  %s N                     CPU cube  (NX = NY = NZ = N)\n"
        "  %s gpu NX NY NZ          GPU device-DRAM + host snapshot\n"
        "  %s gpu N                 GPU cube\n"
        "  %s slurm [NX NY NZ]      Generate SLURM job script\n"
        "  %s slurm [N]             Same with cube domain\n"
        "  %s --list-clusters       List known cluster configurations\n"
        "\n"
        "Slurm options (all optional):\n"
        "  --cluster <id>           Target cluster  (compiled default: \"%s\")\n"
        "  --account <acct>         HPC project account  (compiled default: \"%s\")\n"
        "  --ranks <N>              Total MPI ranks  (default: 1)\n"
        "  --omp-threads <T>        OMP threads/rank — CPU builds only\n"
        "                           (compiled default: kNumOMPThreads=%d)\n"
        "  --time <HH:MM:SS>        Walltime override  (default: cluster maximum)\n"
        "  --gpu-variant <40g|80g>  GPU memory tier — GPU builds only  (default: 40g)\n"
        "  --output <file>          Output path  (default: job.slurm)\n"
        "\n"
        "Compiled-in grid (used when NX/NY/NZ are omitted in slurm mode):\n"
        "  NX=%d  NY=%d  NZ=%d\n"
        "Build flags: GPU=%d  MPI=%d  NVSHMEM=%d\n",
        prog, prog, prog, prog, prog, prog, prog,
        LBM_DEFAULT_CLUSTER[0] ? LBM_DEFAULT_CLUSTER : "perlmutter",
        LBM_DEFAULT_ACCOUNT,
        LBM_COMPILED_OMP_THREADS > 0 ? LBM_COMPILED_OMP_THREADS : 1,
        LBM_COMPILED_NX, LBM_COMPILED_NY, LBM_COMPILED_NZ,
        LBM_BUILD_GPU, LBM_BUILD_MPI, LBM_BUILD_NVSHMEM);
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc >= 2 && std::strcmp(argv[1], "--list-clusters") == 0) {
        listClusters();
        return 0;
    }
    if (argc >= 2 && (std::strcmp(argv[1], "--help") == 0 ||
                      std::strcmp(argv[1], "-h") == 0)) {
        usage(argv[0]);
        return 0;
    }
    if (argc < 2) { usage(argv[0]); return 1; }

    // Parse mode: first positional token may be "gpu" or "slurm".
    bool gpu_mode   = false;
    bool slurm_mode = false;
    int  first      = 1;

    if      (std::strcmp(argv[1], "gpu")   == 0) { gpu_mode   = true; first = 2; }
    else if (std::strcmp(argv[1], "slurm") == 0) { slurm_mode = true; first = 2; }

    // ── slurm mode ───────────────────────────────────────────────────────────
    if (slurm_mode) {
        const char* default_cluster = LBM_DEFAULT_CLUSTER;
        if (default_cluster[0] == '\0') default_cluster = "perlmutter";

        int default_omp = (LBM_COMPILED_OMP_THREADS > 0) ? LBM_COMPILED_OMP_THREADS : 1;

        SlurmOpts opts{};
        opts.cluster     = default_cluster;
        opts.account     = LBM_DEFAULT_ACCOUNT;
        opts.ranks       = 1;
        opts.omp_threads = default_omp;
        opts.walltime    = nullptr;
        opts.gpu_80g     = false;
        opts.output_file = "job.slurm";

        long long nx = LBM_COMPILED_NX, ny = LBM_COMPILED_NY, nz = LBM_COMPILED_NZ;
        int pos_count = 0;

        for (int i = first; i < argc; ++i) {
            // Helper: consume the next argv token as a required argument.
            auto requireNext = [&](const char* flag) -> const char* {
                if (i + 1 >= argc) {
                    fprintf(stderr, "error: %s requires an argument\n", flag);
                    exit(1);
                }
                return argv[++i];
            };

            if      (std::strcmp(argv[i], "--cluster")     == 0)
                opts.cluster     = requireNext("--cluster");
            else if (std::strcmp(argv[i], "--account")     == 0)
                opts.account     = requireNext("--account");
            else if (std::strcmp(argv[i], "--ranks")       == 0)
                opts.ranks       = std::atoi(requireNext("--ranks"));
            else if (std::strcmp(argv[i], "--omp-threads") == 0)
                opts.omp_threads = std::atoi(requireNext("--omp-threads"));
            else if (std::strcmp(argv[i], "--time")        == 0)
                opts.walltime    = requireNext("--time");
            else if (std::strcmp(argv[i], "--output")      == 0)
                opts.output_file = requireNext("--output");
            else if (std::strcmp(argv[i], "--gpu-variant") == 0) {
                const char* v = requireNext("--gpu-variant");
                opts.gpu_80g = (std::strcmp(v, "80g")  == 0 ||
                                std::strcmp(v, "80gb") == 0 ||
                                std::strcmp(v, "80GB") == 0);
            }
            else if (argv[i][0] != '-') {
                // Positional: NX, NY, NZ (or single N for cube)
                long long v = std::atoll(argv[i]);
                if      (pos_count == 0) nx = v;
                else if (pos_count == 1) ny = v;
                else if (pos_count == 2) nz = v;
                ++pos_count;
            }
            else {
                fprintf(stderr, "error: unknown option '%s'\n", argv[i]);
                usage(argv[0]);
                return 1;
            }
        }

        if (pos_count == 1) { ny = nx; nz = nx; }  // cube shorthand

        if (nx <= 0 || ny <= 0 || nz <= 0) {
            fprintf(stderr,
                "error: grid dimensions not available.\n"
                "       Pass NX NY NZ explicitly, or rebuild after setting\n"
                "       LBM_PARAMS_DIR so CMake can read nx/ny/nz from params.h.\n");
            return 1;
        }

        return generateSlurm(nx, ny, nz, opts) ? 0 : 1;
    }

    // ── cpu / gpu estimate mode ──────────────────────────────────────────────
    int nargs = argc - first;
    long long nx = 0, ny = 0, nz = 0;
    if (nargs == 1) {
        nx = ny = nz = std::atoll(argv[first]);
    } else if (nargs == 3) {
        nx = std::atoll(argv[first]);
        ny = std::atoll(argv[first + 1]);
        nz = std::atoll(argv[first + 2]);
    } else {
        usage(argv[0]);
        return 1;
    }

    if (nx <= 0 || ny <= 0 || nz <= 0) {
        fprintf(stderr, "error: all dimensions must be positive integers\n");
        return 1;
    }

    if (gpu_mode)
        gpuEstimate(nx, ny, nz);
    else
        cpuEstimate(nx, ny, nz);

    return 0;
}
