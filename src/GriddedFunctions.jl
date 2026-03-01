module GriddedFunctions

using Interpolations

# include("triangular.jl")
include("grids.jl")
include("functions.jl")

export Axis, LinearAxis, DiscreteAxis, Grid
export GriddedFunction, GriddedFunctionInterpolation

end # module GriddedFunctions
