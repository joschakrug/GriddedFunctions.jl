module GriddedFunctions

import Interpolations

# include("triangular.jl")
include("grids.jl")
include("functions.jl")

export Axis, LinearAxis, DiscreteAxis, Grid
export GriddedFunction, interpolate, xmap!, points, maxpoint

end # module GriddedFunctions
