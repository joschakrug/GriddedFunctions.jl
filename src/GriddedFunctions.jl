module GriddedFunctions

import Interpolations

include("axes.jl")
include("grids.jl")
include("functions.jl")
include("views.jl")

export Axis, LinearAxis, DiscreteAxis, Grid, gridaxes, inbounds, grid
export GridView, source
export GriddedFunction, GriddedFunctionView, GFInterpolation
export interpolate, xmap, xmap!, points, maxpoint

end # module GriddedFunctions
