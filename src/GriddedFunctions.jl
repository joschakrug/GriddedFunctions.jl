module GriddedFunctions

import Interpolations

include("axes.jl")
include("grids.jl")
include("functions.jl")
include("views.jl")

export Axis, LinearAxis, DiscreteAxis, AbstractGrid, Grid, gridaxes, inbounds
export GriddedFunction, GFInterpolation
export interpolate, argmap, argmap!, points, maxpoint, fvalues, grid
export SubGrid, SubGriddedFunction

end # module GriddedFunctions
