#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

#include <params.h>
#include <sim_config.h>
#include "format_compat.h"
#include "analysis_fields.h"
#include "qtensor_fields.h"
#include "vtkhdf_reader.h"
#include "disclination_io.h"
#include "sim_config_attrs.h"
#include "mpi/mpi_context.h"
#include "boundary.h"
#include "boundary_names.h"

#include "analysis/defect_fields.h"
#include "analysis/defect_finder.h"
#include "analysis/defect_analyzer.h"

// ─────────────────────────────────────────────────────────────────────────────
// find_defects: standalone binary that runs the defect detection + analysis
// pipeline on lbm_*.vtkhdf frames produced by the simulation, without needing
// to re-run the simulation. Emits matching disclinations_*.vtkhdf files.
//
// Uses the compile-time SimBC (from sim_config.h) and Params (from params.h)
// — must be rebuilt to match the run being reprocessed. Every frame carries
// the stamped sim config as HDF5 attributes; a mismatch against the current
// build's compile-time values is a hard error (see --force to override).
// ─────────────────────────────────────────────────────────────────────────────

namespace {

struct CLI {
    std::string input_dir  = "data";
    std::string output_dir;   // filled from input_dir if left empty
    std::string single_frame;
    bool with_beta = true;
    bool force = false;
};

void PrintUsage() {
    std::cerr <<
        "Usage: find_defects [options]\n"
        "  --input-dir DIR      directory holding lbm_*.vtkhdf (default: data)\n"
        "  --output-dir DIR     where to write disclinations_*.vtkhdf (default: input-dir)\n"
        "  --single-frame PATH  process just one lbm_*.vtkhdf and exit\n"
        "  --no-beta            skip β computation (skips Q reconstruction too)\n"
        "  --force              downgrade hard sim-config mismatches to warnings\n"
        "  -h, --help           show this message\n";
}

bool ParseCLI(int argc, char* argv[], CLI& cli) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need_val = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "find_defects: " << name << " requires an argument\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") {
            PrintUsage();
            std::exit(0);
        } else if (a == "--input-dir") {
            cli.input_dir = need_val("--input-dir");
        } else if (a == "--output-dir") {
            cli.output_dir = need_val("--output-dir");
        } else if (a == "--single-frame") {
            cli.single_frame = need_val("--single-frame");
        } else if (a == "--no-beta") {
            cli.with_beta = false;
        } else if (a == "--force") {
            cli.force = true;
        } else {
            std::cerr << "find_defects: unknown argument: " << a << "\n";
            PrintUsage();
            return false;
        }
    }
    if (cli.output_dir.empty()) cli.output_dir = cli.input_dir;
    return true;
}

struct Frame {
    std::filesystem::path path;
    int step = 0;
    int digit_width = 1;   // width of the digit run in the input filename;
                           // reused verbatim for the output name so
                           // disclinations_*.vtkhdf and lbm_*.vtkhdf match
                           // digit-for-digit even when the sim's kNumSteps
                           // would have chosen a wider padding than a
                           // partial-run enumeration would derive from the
                           // max step index alone.
};

// Enumerate lbm_XXXXX.vtkhdf files in `dir`, extract the step number, sort by
// step ascending. Anything that doesn't match the pattern is skipped
// silently (e.g. disclinations_*.vtkhdf files sitting in the same dir).
std::vector<Frame> EnumerateFrames(const std::string& dir) {
    static const std::regex kPattern(R"(^lbm_(\d+)\.vtkhdf$)");
    std::vector<Frame> out;
    if (!std::filesystem::is_directory(dir)) {
        std::cerr << "find_defects: not a directory: " << dir << "\n";
        return out;
    }
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        std::smatch m;
        if (!std::regex_match(name, m, kPattern)) continue;
        Frame f{entry.path(), std::stoi(m[1].str()),
                static_cast<int>(m[1].str().size())};
        out.push_back(f);
    }
    std::sort(out.begin(), out.end(),
              [](const Frame& a, const Frame& b) { return a.step < b.step; });
    return out;
}

// Prints the mismatches and returns whether we should proceed. Hard buckets
// are fatal unless --force; soft buckets are always warnings.
bool ReportAndCheck(const SimConfigAttr::ValidationReport& r,
                    const std::string& file_path,
                    bool force) {
    for (const auto& m : r.soft) {
        std::cerr << compat::format(
            "find_defects: [warn] soft mismatch in {}: {} expected \"{}\", got \"{}\"\n",
            file_path, m.field, m.expected, m.actual);
    }
    if (r.git_commit_mismatch) {
        std::cerr << compat::format(
            "find_defects: [warn] git commit mismatch in {}: build={}, file={}\n",
            file_path, r.expected_git_commit, r.actual_git_commit);
    }
    if (r.hard.empty()) return true;

    for (const auto& m : r.hard) {
        std::cerr << compat::format(
            "find_defects: [{}] HARD mismatch in {}: {} expected \"{}\", got \"{}\"\n",
            force ? "warn" : "error", file_path, m.field, m.expected, m.actual);
    }
    if (force) {
        std::cerr << "find_defects: --force in effect, continuing despite hard mismatches\n";
        return true;
    }
    std::cerr << "find_defects: rebuild with a matching params.h / sim_config.h, "
                 "or pass --force to proceed anyway.\n";
    return false;
}

}  // namespace

int main(int argc, char* argv[]) {
    CLI cli;
    if (!ParseCLI(argc, argv, cli)) return 2;

    // Serial-only for now (defect finder is not MPI-parallel yet; see the
    // #ifndef LBM_ENABLE_MPI guard in active_nematic.h::Export). The reader
    // is MPI-capable so flipping this to an MPI build is a one-line change
    // once the finder catches up.
    const MPIContext ctx(periodicity_by_axis<SimBC>);
    const LocalGrid grid = ctx.MakeLocalGrid();

    // Enumerate work. --single-frame takes precedence and skips discovery.
    std::vector<Frame> frames;
    if (!cli.single_frame.empty()) {
        static const std::regex kPattern(R"(^lbm_(\d+)\.vtkhdf$)");
        std::filesystem::path p{cli.single_frame};
        std::smatch m;
        const std::string name = p.filename().string();
        int step = 0;
        int width = 1;
        if (std::regex_match(name, m, kPattern)) {
            step  = std::stoi(m[1].str());
            width = static_cast<int>(m[1].str().size());
        }
        frames.push_back({p, step, width});
    } else {
        frames = EnumerateFrames(cli.input_dir);
        if (frames.empty()) {
            std::cerr << "find_defects: no lbm_*.vtkhdf files under " << cli.input_dir << "\n";
            return 1;
        }
    }

    // State reused across frames — allocated once, overwritten per frame.
    AnalysisFields  af(grid);
    QTensorFields   qf(grid);
    DefectFields    df(periodicity_by_axis<SimBC>);
    DefectFinder<SimBC>   finder;
    DefectAnalyzer<SimBC> da;

    bool validated_config = false;

    for (const Frame& frame : frames) {
        const std::string in_path  = frame.path.string();
        const std::string out_path = compat::format(
            "{}/disclinations_{:0{}}.vtkhdf",
            cli.output_dir, frame.step, frame.digit_width);

        try {
            ImageDataReader reader(in_path, ctx);

            // Validate the sim config once (on the first frame). Every frame
            // carries the same values — a single check is enough. A file
            // written by a pre-attribute build has no BCName attribute; that
            // throws inside ReadSimConfig. --force accepts unstamped files.
            if (!validated_config) {
                try {
                    const auto snap = reader.ReadSimConfig();
                    const auto report = SimConfigAttr::ValidateAgainstBuild<SimBC>(snap);
                    if (!ReportAndCheck(report, in_path, cli.force)) return 3;
                } catch (const std::exception& e) {
                    if (!cli.force) {
                        std::cerr <<
                            "find_defects: this file is missing sim-config attributes "
                            "(pre-attribute build?): " << e.what() << "\n"
                            "Pass --force to proceed anyway "
                            "(validation is skipped, correctness is up to you).\n";
                        return 3;
                    }
                    std::cerr << "find_defects: [warn] no sim-config attributes on "
                              << in_path << " — --force in effect, skipping validation.\n";
                }
                validated_config = true;
            }

            reader.ReadScalarField("order", af.order_.data(), grid);
            reader.ReadVectorField("director", af.director_.data(), grid, 3);

            if (cli.with_beta) {
                // Reconstruct Q from (S, n̂) on the uniaxial subset — exact
                // where β samples Q (ring radius ≥ 2 lattice units, well
                // outside the defect core). See OrderDirectorToQtensor.
                OrderDirectorToQtensor(af, qf);
            }

            finder.FindDefects(af, df);
            if (cli.with_beta) {
                da.AnalyzeDefects(df, qf);
            }

            WriteDisclinationsVTKHDF<SimBC>(df, out_path, ctx);
            std::cout << compat::format(
                "find_defects: {} → {} ({} disclinations)\n",
                in_path, out_path, df.disclinations.size());
        } catch (const std::exception& e) {
            std::cerr << compat::format(
                "find_defects: failed on {}: {}\n", in_path, e.what());
            return 4;
        }
    }
    return 0;
}
