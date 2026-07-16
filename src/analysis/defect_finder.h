#ifndef LBM_AN_ANALYSIS_DEFECT_FINDER_H
#define LBM_AN_ANALYSIS_DEFECT_FINDER_H

#include "defect_fields.h"
#include "analysis_fields.h"
#include "qtensor_fields.h"
#include <params.h>
#include <ranges>
#include <vector>
#include "vtkhdf_writer.h"
#include "boundary.h"
#include "offsets.h"

using namespace Params;


template<typename BC>
class DefectFinder {
    void ComputeWindingNumbers(const QTensorFields& qf, const AnalysisFields& af, DefectFields& df);

    void BuildConnectivityGraph(DefectFields& df);

    void IsolateDisclinationsFromGraph(DefectFields& df);

public:

    DefectFinder() = default;
    void FindDefects(const QTensorFields& qf, const AnalysisFields& af, DefectFields& df) {
        std::fill(df.def_x.begin(), df.def_x.end(), 0);
        std::fill(df.def_y.begin(), df.def_y.end(), 0);
        std::fill(df.def_z.begin(), df.def_z.end(), 0);
        df.disclinations.clear();
        ComputeWindingNumbers(qf, af, df);
        BuildConnectivityGraph(df);
        IsolateDisclinationsFromGraph(df);
    }

};

#include "defect_finder.tpp"

#endif // LBM_AN_ANALYSIS_DEFECT_FINDER_H
