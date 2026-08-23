#ifndef LBM_AN_SIM_CONFIG_ATTRS_H_
#define LBM_AN_SIM_CONFIG_ATTRS_H_

#include <cstdint>
#include <string>
#include <vector>
#include <params.h>
#include <sim_config.h>
#include "boundary.h"
#include "boundary_names.h"
#include "qtensor_types.h"
#include "hdf5_internals.h"
#include "git_commit.h"

// ─────────────────────────────────────────────────────────────────────────────
// Sim-config attribute contract for VTKHDF frames.
//
// Every lbm_*.vtkhdf and disclinations_*.vtkhdf gets these attributes stamped
// on its /VTKHDF group at write time (StampSimConfigAttributes). find_defects
// reads them on the first frame it opens (ReadSimConfigAttributes) and
// validates against its own compile-time Params + SimBC.
//
// The set is intentionally the whole runtime-relevant Params surface plus BC
// slot names plus the git commit — "snapshot the whole simulation state".
// Hard/soft split (what makes find_defects error vs. warn) lives in the
// validator, not here — this header only defines what's on the wire.
// ─────────────────────────────────────────────────────────────────────────────

namespace SimConfigAttr {

// Attribute name constants — one string per attribute, referenced by both
// writer and reader so a rename is a single-point edit.
inline constexpr const char* kBCName             = "BCName";
inline constexpr const char* kXLo_QBC            = "XLo_QBC";
inline constexpr const char* kXLo_UBC            = "XLo_UBC";
inline constexpr const char* kXHi_QBC            = "XHi_QBC";
inline constexpr const char* kXHi_UBC            = "XHi_UBC";
inline constexpr const char* kYLo_QBC            = "YLo_QBC";
inline constexpr const char* kYLo_UBC            = "YLo_UBC";
inline constexpr const char* kYHi_QBC            = "YHi_QBC";
inline constexpr const char* kYHi_UBC            = "YHi_UBC";
inline constexpr const char* kZLo_QBC            = "ZLo_QBC";
inline constexpr const char* kZLo_UBC            = "ZLo_UBC";
inline constexpr const char* kZHi_QBC            = "ZHi_QBC";
inline constexpr const char* kZHi_UBC            = "ZHi_UBC";
inline constexpr const char* kQAdvection         = "QAdvection";
inline constexpr const char* kGitCommit          = "GitCommit";

inline constexpr const char* kNx                 = "nx";
inline constexpr const char* kNy                 = "ny";
inline constexpr const char* kNz                 = "nz";
inline constexpr const char* kDX                 = "DX";
inline constexpr const char* kDY                 = "DY";
inline constexpr const char* kDZ                 = "DZ";
inline constexpr const char* kDT                 = "DT";
inline constexpr const char* kTAUF               = "TAUF";
inline constexpr const char* kA                  = "A";
inline constexpr const char* kB                  = "B";
inline constexpr const char* kC                  = "C";
inline constexpr const char* kL                  = "L";
inline constexpr const char* kALPHA              = "ALPHA";
inline constexpr const char* kLAMBDA             = "LAMBDA";
inline constexpr const char* kGAMMA              = "GAMMA";
inline constexpr const char* kMU                 = "MU";
inline constexpr const char* kNOISE              = "NOISE";
inline constexpr const char* kNumSteps           = "kNumSteps";
inline constexpr const char* kSaveInterval       = "kSaveInterval";

// ── Low-level HDF5 attribute helpers (attach to any hid_t) ──────────────────

inline void write_string_attr(hid_t obj, const char* name, const std::string& value) {
    H5Datatype atype{ H5Tcopy(H5T_C_S1), "H5Tcopy" };
    H5Tset_size(atype, value.size() > 0 ? value.size() : 1);
    H5Tset_strpad(atype, H5T_STR_NULLPAD);
    H5Tset_cset(atype, H5T_CSET_ASCII);
    H5Dataspace aspace{ H5Screate(H5S_SCALAR), "H5Screate" };
    H5Attribute attr{
        H5Acreate2(obj, name, atype, aspace, H5P_DEFAULT, H5P_DEFAULT), name };
    const char* buf = value.empty() ? "" : value.data();
    if (H5Awrite(attr, atype, buf) < 0)
        throw std::runtime_error(std::string("H5Awrite failed: ") + name);
}

template <typename T>
inline void write_scalar_attr(hid_t obj, const char* name, T value) {
    H5Dataspace aspace{ H5Screate(H5S_SCALAR), "H5Screate" };
    H5Attribute attr{
        H5Acreate2(obj, name, H5Native<T>::id(), aspace,
                   H5P_DEFAULT, H5P_DEFAULT), name };
    if (H5Awrite(attr, H5Native<T>::id(), &value) < 0)
        throw std::runtime_error(std::string("H5Awrite failed: ") + name);
}

inline std::string read_string_attr(hid_t obj, const char* name) {
    H5Attribute attr{ H5Aopen(obj, name, H5P_DEFAULT), name };
    H5Datatype atype{ H5Aget_type(attr), "H5Aget_type" };
    const size_t sz = H5Tget_size(atype);
    std::string out(sz, '\0');
    if (H5Aread(attr, atype, out.data()) < 0)
        throw std::runtime_error(std::string("H5Aread failed: ") + name);
    // strip any trailing NUL padding
    while (!out.empty() && out.back() == '\0') out.pop_back();
    return out;
}

template <typename T>
inline T read_scalar_attr(hid_t obj, const char* name) {
    H5Attribute attr{ H5Aopen(obj, name, H5P_DEFAULT), name };
    T value{};
    if (H5Aread(attr, H5Native<T>::id(), &value) < 0)
        throw std::runtime_error(std::string("H5Aread failed: ") + name);
    return value;
}

inline std::string QAdvectionName(Advection a) {
    return (a == Advection::Centred) ? "Centred" : "Upwind";
}

// ── Stamp: called by writers once per file at construction time ─────────────

template <typename BC>
inline void StampSimConfigAttributes(hid_t root) {
    using namespace Params;
    write_string_attr(root, kBCName, std::string(BC::name));
    write_string_attr(root, kXLo_QBC, BCName<typename BC::XLo::QBC>::get());
    write_string_attr(root, kXLo_UBC, BCName<typename BC::XLo::UBC>::get());
    write_string_attr(root, kXHi_QBC, BCName<typename BC::XHi::QBC>::get());
    write_string_attr(root, kXHi_UBC, BCName<typename BC::XHi::UBC>::get());
    write_string_attr(root, kYLo_QBC, BCName<typename BC::YLo::QBC>::get());
    write_string_attr(root, kYLo_UBC, BCName<typename BC::YLo::UBC>::get());
    write_string_attr(root, kYHi_QBC, BCName<typename BC::YHi::QBC>::get());
    write_string_attr(root, kYHi_UBC, BCName<typename BC::YHi::UBC>::get());
    write_string_attr(root, kZLo_QBC, BCName<typename BC::ZLo::QBC>::get());
    write_string_attr(root, kZLo_UBC, BCName<typename BC::ZLo::UBC>::get());
    write_string_attr(root, kZHi_QBC, BCName<typename BC::ZHi::QBC>::get());
    write_string_attr(root, kZHi_UBC, BCName<typename BC::ZHi::UBC>::get());
    // Fully-qualify against ::identifiers (the sim_config.h/params.h globals),
    // since the SimConfigAttr namespace's string constants shadow them here.
    write_string_attr(root, kQAdvection, QAdvectionName(::kQAdvection));
    write_string_attr(root, kGitCommit, std::string(::kGitCommit));

    write_scalar_attr<int32_t>(root, kNx, nx);
    write_scalar_attr<int32_t>(root, kNy, ny);
    write_scalar_attr<int32_t>(root, kNz, nz);
    write_scalar_attr<double>(root, kDX, DX);
    write_scalar_attr<double>(root, kDY, DY);
    write_scalar_attr<double>(root, kDZ, DZ);
    write_scalar_attr<double>(root, kDT, DT);
    write_scalar_attr<double>(root, kTAUF, TAUF);
    write_scalar_attr<double>(root, kA, A);
    write_scalar_attr<double>(root, kB, B);
    write_scalar_attr<double>(root, kC, C);
    write_scalar_attr<double>(root, kL, L);
    write_scalar_attr<double>(root, kALPHA, ALPHA);
    write_scalar_attr<double>(root, kLAMBDA, LAMBDA);
    write_scalar_attr<double>(root, kGAMMA, GAMMA);
    write_scalar_attr<double>(root, kMU, MU);
    write_scalar_attr<double>(root, kNOISE, NOISE);
    write_scalar_attr<int32_t>(root, kNumSteps, ::kNumSteps);
    write_scalar_attr<int32_t>(root, kSaveInterval, ::kSaveInterval);
}

// ── Read: struct populated by ImageDataReader on open ───────────────────────

struct SimConfigSnapshot {
    std::string bc_name;
    std::string xlo_qbc, xlo_ubc, xhi_qbc, xhi_ubc;
    std::string ylo_qbc, ylo_ubc, yhi_qbc, yhi_ubc;
    std::string zlo_qbc, zlo_ubc, zhi_qbc, zhi_ubc;
    std::string qadvection;
    std::string git_commit;
    int32_t nx = 0, ny = 0, nz = 0;
    double DX = 0, DY = 0, DZ = 0, DT = 0;
    double TAUF = 0, A = 0, B = 0, C = 0, L = 0;
    double ALPHA = 0, LAMBDA = 0, GAMMA = 0, MU = 0, NOISE = 0;
    int32_t kNumSteps = 0, kSaveInterval = 0;
};

inline SimConfigSnapshot ReadSimConfigAttributes(hid_t root) {
    SimConfigSnapshot s;
    s.bc_name    = read_string_attr(root, kBCName);
    s.xlo_qbc    = read_string_attr(root, kXLo_QBC);
    s.xlo_ubc    = read_string_attr(root, kXLo_UBC);
    s.xhi_qbc    = read_string_attr(root, kXHi_QBC);
    s.xhi_ubc    = read_string_attr(root, kXHi_UBC);
    s.ylo_qbc    = read_string_attr(root, kYLo_QBC);
    s.ylo_ubc    = read_string_attr(root, kYLo_UBC);
    s.yhi_qbc    = read_string_attr(root, kYHi_QBC);
    s.yhi_ubc    = read_string_attr(root, kYHi_UBC);
    s.zlo_qbc    = read_string_attr(root, kZLo_QBC);
    s.zlo_ubc    = read_string_attr(root, kZLo_UBC);
    s.zhi_qbc    = read_string_attr(root, kZHi_QBC);
    s.zhi_ubc    = read_string_attr(root, kZHi_UBC);
    s.qadvection = read_string_attr(root, kQAdvection);
    s.git_commit = read_string_attr(root, kGitCommit);

    s.nx           = read_scalar_attr<int32_t>(root, kNx);
    s.ny           = read_scalar_attr<int32_t>(root, kNy);
    s.nz           = read_scalar_attr<int32_t>(root, kNz);
    s.DX           = read_scalar_attr<double>(root, kDX);
    s.DY           = read_scalar_attr<double>(root, kDY);
    s.DZ           = read_scalar_attr<double>(root, kDZ);
    s.DT           = read_scalar_attr<double>(root, kDT);
    s.TAUF         = read_scalar_attr<double>(root, kTAUF);
    s.A            = read_scalar_attr<double>(root, kA);
    s.B            = read_scalar_attr<double>(root, kB);
    s.C            = read_scalar_attr<double>(root, kC);
    s.L            = read_scalar_attr<double>(root, kL);
    s.ALPHA        = read_scalar_attr<double>(root, kALPHA);
    s.LAMBDA       = read_scalar_attr<double>(root, kLAMBDA);
    s.GAMMA        = read_scalar_attr<double>(root, kGAMMA);
    s.MU           = read_scalar_attr<double>(root, kMU);
    s.NOISE        = read_scalar_attr<double>(root, kNOISE);
    s.kNumSteps    = read_scalar_attr<int32_t>(root, kNumSteps);
    s.kSaveInterval= read_scalar_attr<int32_t>(root, kSaveInterval);
    return s;
}

// ── Validate: compare a snapshot against the current build's Params + BC ────
// Returns a list of (name, expected, got) mismatches per bucket.

struct Mismatch {
    std::string field;
    std::string expected;
    std::string actual;
};

struct ValidationReport {
    std::vector<Mismatch> hard;   // fields that change defect analysis output
    std::vector<Mismatch> soft;   // fields that don't, but a mismatch is a red flag
    bool git_commit_mismatch = false;
    std::string expected_git_commit;
    std::string actual_git_commit;

    bool ok() const { return hard.empty() && soft.empty(); }
};

// Append a mismatch if `expected != actual`, formatted for user output.
template <typename T>
inline void push_if_neq(std::vector<Mismatch>& out, const char* field,
                        const T& expected, const T& actual) {
    if (!(expected == actual)) {
        out.push_back({field,
                       compat::format("{}", expected),
                       compat::format("{}", actual)});
    }
}

template <typename BC>
inline ValidationReport ValidateAgainstBuild(const SimConfigSnapshot& s) {
    using namespace Params;
    ValidationReport r;

    // Hard: 6 QBCs (defect analysis walks Q ghosts), grid dims, LdG constants,
    // lattice spacings — all directly affect finder/analyzer output.
    push_if_neq(r.hard, kXLo_QBC, BCName<typename BC::XLo::QBC>::get(), s.xlo_qbc);
    push_if_neq(r.hard, kXHi_QBC, BCName<typename BC::XHi::QBC>::get(), s.xhi_qbc);
    push_if_neq(r.hard, kYLo_QBC, BCName<typename BC::YLo::QBC>::get(), s.ylo_qbc);
    push_if_neq(r.hard, kYHi_QBC, BCName<typename BC::YHi::QBC>::get(), s.yhi_qbc);
    push_if_neq(r.hard, kZLo_QBC, BCName<typename BC::ZLo::QBC>::get(), s.zlo_qbc);
    push_if_neq(r.hard, kZHi_QBC, BCName<typename BC::ZHi::QBC>::get(), s.zhi_qbc);
    push_if_neq(r.hard, kNx, static_cast<int32_t>(nx), s.nx);
    push_if_neq(r.hard, kNy, static_cast<int32_t>(ny), s.ny);
    push_if_neq(r.hard, kNz, static_cast<int32_t>(nz), s.nz);
    push_if_neq(r.hard, kDX, DX, s.DX);
    push_if_neq(r.hard, kDY, DY, s.DY);
    push_if_neq(r.hard, kDZ, DZ, s.DZ);
    push_if_neq(r.hard, kA, A, s.A);
    push_if_neq(r.hard, kB, B, s.B);
    push_if_neq(r.hard, kC, C, s.C);
    push_if_neq(r.hard, kL, L, s.L);

    // Soft: 6 UBCs (only touch LBM streaming — irrelevant to a static (S, n̂)
    // snapshot), plus every non-defect Params dial.
    push_if_neq(r.soft, kXLo_UBC, BCName<typename BC::XLo::UBC>::get(), s.xlo_ubc);
    push_if_neq(r.soft, kXHi_UBC, BCName<typename BC::XHi::UBC>::get(), s.xhi_ubc);
    push_if_neq(r.soft, kYLo_UBC, BCName<typename BC::YLo::UBC>::get(), s.ylo_ubc);
    push_if_neq(r.soft, kYHi_UBC, BCName<typename BC::YHi::UBC>::get(), s.yhi_ubc);
    push_if_neq(r.soft, kZLo_UBC, BCName<typename BC::ZLo::UBC>::get(), s.zlo_ubc);
    push_if_neq(r.soft, kZHi_UBC, BCName<typename BC::ZHi::UBC>::get(), s.zhi_ubc);
    push_if_neq(r.soft, kDT, DT, s.DT);
    push_if_neq(r.soft, kTAUF, TAUF, s.TAUF);
    push_if_neq(r.soft, kALPHA, ALPHA, s.ALPHA);
    push_if_neq(r.soft, kLAMBDA, LAMBDA, s.LAMBDA);
    push_if_neq(r.soft, kGAMMA, GAMMA, s.GAMMA);
    push_if_neq(r.soft, kMU, MU, s.MU);
    push_if_neq(r.soft, kNOISE, NOISE, s.NOISE);
    push_if_neq(r.soft, kNumSteps, static_cast<int32_t>(::kNumSteps), s.kNumSteps);
    push_if_neq(r.soft, kSaveInterval, static_cast<int32_t>(::kSaveInterval), s.kSaveInterval);
    push_if_neq(r.soft, kQAdvection, QAdvectionName(::kQAdvection), s.qadvection);

    // BCName label: informational, not part of hard/soft — a rename with
    // identical slot types is meaningful (someone reorganized presets) but
    // doesn't change output; keep it out of the hard bucket but flag it soft.
    push_if_neq(r.soft, kBCName, std::string(BC::name), s.bc_name);

    // Git commit: reported separately (not part of hard/soft — it's meta).
    // Empty expected/actual means "not built from a git checkout" — silently accepted.
    // Global ::kGitCommit is the hash from generated/git_commit.h; the bare
    // name here would shadow to SimConfigAttr::kGitCommit (the attribute-name
    // string "GitCommit"). Same shadowing pattern as ::kQAdvection above.
    r.expected_git_commit = ::kGitCommit;
    r.actual_git_commit   = s.git_commit;
    r.git_commit_mismatch =
        !r.expected_git_commit.empty() && !r.actual_git_commit.empty() &&
        r.expected_git_commit != r.actual_git_commit;
    return r;
}

} // namespace SimConfigAttr

#endif // LBM_AN_SIM_CONFIG_ATTRS_H_
