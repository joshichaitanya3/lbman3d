#include "defect_fields.h"
#include <params.h>

using namespace Params;


DefectFields::DefectFields(std::array<int, 3> periodic) :
    periodic_by_axis(periodic)
{
    const int nfx = nx - (periodic[0] ? 0 : 0); // along-normal: always nx
    const int nfy = ny - (periodic[1] ? 0 : 0);
    const int nfz = nz - (periodic[2] ? 0 : 0);
    (void)nfx; (void)nfy; (void)nfz; // silence -Wunused when only tangent counts matter

    // Tangent extents: the plaquette sits between vertices i and i+1, so
    // there are n plaquettes when i+1 wraps (periodic) and n-1 otherwise.
    const int tx = periodic[0] ? nx : nx - 1;
    const int ty = periodic[1] ? ny : ny - 1;
    const int tz = periodic[2] ? nz : nz - 1;

    nfx_x = nx;  nfx_y = ty;  nfx_z = tz;
    nfy_x = tx;  nfy_y = ny;  nfy_z = tz;
    nfz_x = tx;  nfz_y = ty;  nfz_z = nz;

    n_def_x = static_cast<FaceId>(nfx_x) * static_cast<FaceId>(nfx_y) * static_cast<FaceId>(nfx_z);
    n_def_y = static_cast<FaceId>(nfy_x) * static_cast<FaceId>(nfy_y) * static_cast<FaceId>(nfy_z);
    n_def_z = static_cast<FaceId>(nfz_x) * static_cast<FaceId>(nfz_y) * static_cast<FaceId>(nfz_z);

    def_x.assign(static_cast<size_t>(n_def_x), 0);
    def_y.assign(static_cast<size_t>(n_def_y), 0);
    def_z.assign(static_cast<size_t>(n_def_z), 0);
}


bool FlipN(
    double n1x, double n1y, double n1z,
    double n2x, double n2y, double n2z
) {
    const double n1_dot_n2 = n1x*n2x + n1y*n2y + n1z*n2z;
    return (n1_dot_n2 <= 0.0);
}
