module GriddedFunctions

import Interpolations

include("axes.jl")
include("grids.jl")
include("functions.jl")
include("views.jl")

export approximately, exactly, inrange
export Axis, LinearAxis, DiscreteAxis, onaxisapprox
export AbstractGrid, Grid, gridaxes, inbounds
export AbstractGriddedFunction, GriddedFunction, GFInterpolation
export interpolate, argmap, argmap!, points, maxpoint, fvalues, grid
export SubAxis, SubGrid, SubGriddedFunction, subset

end # module GriddedFunctions
