#ifndef LBM_AN_ANALYSIS_DEFECT_FINDER_H
#define LBM_AN_ANALYSIS_DEFECT_FINDER_H

#include "defect_fields.h"
#include "analysis_fields.h"
#include "qtensor_fields.h"
#include "params.h"
#include <ranges>
#include <vector>
#include "vtkhdf_writer.h"
#include "boundary.h"
#include "grid.h"

using namespace Params;


template<typename BC>
class DefectFinder {
    Grid<BC> grid_;

    // Re-use the offset rules of Q-tensor for the defect finding
    int  Xoff(int x, int s) const { return grid_.QXoff(x, s); }
    int  Yoff(int y, int s) const { return grid_.QYoff(y, s); }
    int  Zoff(int z, int s) const { return grid_.QZoff(z, s); }
    
    void ComputeWindingNumbers(const QTensorFields& qf, const AnalysisFields& af, DefectFields& df);
    
    void BuildConnectivityGraph(DefectFields& df);

    void IsolateDisclinationsFromGraph(DefectFields& df);

public:

    explicit DefectFinder(Grid<BC> grid);
    void FindDefects(const QTensorFields& qf, const AnalysisFields& af, DefectFields& df) {
        std::fill(df.def_x_data.begin(), df.def_x_data.end(), 0);
        std::fill(df.def_y_data.begin(), df.def_y_data.end(), 0);
        std::fill(df.def_z_data.begin(), df.def_z_data.end(), 0);
        df.disclinations.clear();
        ComputeWindingNumbers(qf, af, df);
        BuildConnectivityGraph(df);
        IsolateDisclinationsFromGraph(df);
    }

};

#include "defect_finder.tpp"

#endif // LBM_AN_ANALYSIS_DEFECT_FINDER_H
