module GriddedFunctionPlotsExt

using GriddedFunctions
using RecipesBase

import GriddedFunctions: AbstractGriddedFunction, grid, gridaxes, ncontinuousdims, dimnames

# `values` clashes with Base.values; qualify explicitly throughout.

# Return the axis label for dimension `d` of point type `T`.
# Uses type `T`'s dimension names when available; falls back to "xd" for plain Tuples.
function _axis_label(gf::AbstractGriddedFunction, d::Int)
    T = eltype(grid(gf))
    names = dimnames(T)
    if (d <= length(names)) && (names[d] isa Symbol)
        string(names[d])
    else
        "x$d"
    end
end

# ─── 1D ContinuousGrid ───────────────────────────────────────────────────────
# Default: line plot

@recipe function f(gf::AbstractGriddedFunction{TY, 1}) where TY
    ax = only(gridaxes(grid(gf)))
    xlabel --> _axis_label(gf, 1)

    # default to scatter if the gridded function is discrete
    if ncontinuousdims(gf) == 0
        seriestype --> :scatter
    end

    collect(ax), vec(GriddedFunctions.values(gf))
end

# ─── 2D ContinuousGrid ───────────────────────────────────────────────────────
# Default: surface plot. Override with `seriestype=:heatmap` for a 2D view.
#
# Note on matrix orientation: GriddedFunction stores values[i_x, i_y], but
# Plots.jl interprets z[row, col] as z[i_y, i_x] (row = y, col = x). The
# transpose corrects this so the x and y axes match their grid counterparts.

@recipe function f(gf::AbstractGriddedFunction{TY, 2}) where TY
    g = grid(gf)
    ax1, ax2 = gridaxes(g)
    seriestype --> :surface
    xlabel --> _axis_label(gf, 1)
    ylabel --> _axis_label(gf, 2)
    collect(ax1), collect(ax2), GriddedFunctions.values(gf)'
end

end # module GriddedFunctionPlotsExt
