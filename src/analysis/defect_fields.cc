#include "defect_fields.h"
#include <params.h>

using namespace Params;


DefectFields::DefectFields() :
    def_x(nx     * (ny - 1) * (nz - 1), 0),
    def_y((nx-1) * ny       * (nz - 1), 0),
    def_z((nx-1) * (ny - 1) * nz      , 0)
{}



bool FlipN(
    double n1x,
    double n1y,
    double n1z,
    double n2x,
    double n2y,
    double n2z
) {
    double n1_dot_n2 = n1x*n2x + n1y*n2y + n1z*n2z;
    return (n1_dot_n2 <= 0.0);
}