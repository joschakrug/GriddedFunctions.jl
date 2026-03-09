module GriddedFunctions

import Interpolations

include("axes.jl")
include("grids.jl")
include("functions.jl")
include("views.jl")

export Axis, LinearAxis, DiscreteAxis, AbstractGrid, Grid, gridaxes, inbounds, grid
export GriddedFunction, GFInterpolation
export interpolate, xmap, xmap!, points, maxpoint
export SubGrid, SubGriddedFunction

end # module GriddedFunctions
