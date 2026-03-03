module GriddedFunctions

import Interpolations

# include("triangular.jl")
include("grids.jl")
include("functions.jl")

export Axis, LinearAxis, DiscreteAxis, Grid, gridaxes
export GriddedFunction, GFInterpolation, interpolate, xmap, xmap!, points, maxpoint

end # module GriddedFunctions
