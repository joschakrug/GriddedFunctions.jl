module GriddedFunctionPlotsExt

using GriddedFunctions
using RecipesBase

import GriddedFunctions: ContinuousGrid, DiscreteGrid, grid, gridaxes

# `values` clashes with Base.values; qualify explicitly throughout.

# Return the axis label for dimension `d` of point type `T`.
# Uses the struct field name when available; falls back to "xd" for plain Tuples.
function _axis_label(::Type{T}, d::Int) where T
    names = fieldnames(T)
    if (d <= length(names)) && (names[d] isa Symbol)
        string(names[d])
    else
        "x$d"
    end
end

# ─── 1D ContinuousGrid ───────────────────────────────────────────────────────
# Default: line plot

@recipe function f(gf::GriddedFunction{TX, TY, 1, <:ContinuousGrid}) where {TX, TY}
    ax = only(gridaxes(grid(gf)))
    xlabel --> _axis_label(TX, 1)
    collect(ax), vec(GriddedFunctions.values(gf))
end

# ─── 1D DiscreteGrid ─────────────────────────────────────────────────────────
# Default: scatter plot

@recipe function f(gf::GriddedFunction{TX, TY, 1, <:DiscreteGrid}) where {TX, TY}
    ax = only(gridaxes(grid(gf)))
    seriestype --> :scatter
    xlabel --> _axis_label(TX, 1)
    collect(ax), vec(GriddedFunctions.values(gf))
end

# ─── 2D ContinuousGrid ───────────────────────────────────────────────────────
# Default: surface plot. Override with `seriestype=:heatmap` for a 2D view.
#
# Note on matrix orientation: GriddedFunction stores values[i_x, i_y], but
# Plots.jl interprets z[row, col] as z[i_y, i_x] (row = y, col = x). The
# transpose corrects this so the x and y axes match their grid counterparts.

@recipe function f(gf::GriddedFunction{TX, TY, 2, <:ContinuousGrid}) where {TX, TY}
    g = grid(gf)
    ax1, ax2 = gridaxes(g)
    seriestype --> :surface
    xlabel --> _axis_label(TX, 1)
    ylabel --> _axis_label(TX, 2)
    collect(ax1), collect(ax2), GriddedFunctions.values(gf)'
end

end # module GriddedFunctionPlotsExt
