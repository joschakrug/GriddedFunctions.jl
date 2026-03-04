module GriddedFunctions

import Interpolations

include("axes.jl")
include("grids.jl")
include("functions.jl")

export Axis, LinearAxis, DiscreteAxis, Grid, gridaxes, inbounds
export GriddedFunction, GFInterpolation, interpolate, xmap, xmap!, points, maxpoint

end # module GriddedFunctions
