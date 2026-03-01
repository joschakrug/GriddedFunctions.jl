
# longer run to do: add a Domain (potentially linked to a GridSection type)
# to allow for functions that are not defined over the entire grid

"""
    GriddedFunction{TX, TY, D, G} <: AbstractArray{TY, D}

A function of type `TY` defined over a `D`-dimensional grid of type `G`.

Values are stored as an `Array{TY}` with one entry per grid point. `TX` is the
type of a grid point (a `Tuple` of axis element types).

# Constructors

    GriddedFunction(g::Grid, values::Array)

Construct from a pre-computed array of values. `size(values)` must match
`size(g)`.

    GriddedFunction(T::Type, g::Grid, f::Function)

Construct by evaluating `f` at every grid point. The return type of `f` must
be convertible to `T`.
"""
mutable struct GriddedFunction{TX, TY <: Real, D, G <: Grid{TX, D}} <: AbstractArray{TY, D}
    grid::G
    values::Array{TY}

    function GriddedFunction(g::Grid, values)
        @assert size(g) == size(values)
        D = ndims(g)
        new{eltype(g), eltype(values), D, typeof(g)}(g, values)
    end
end

function GriddedFunction(T::Type, g::Grid, f::Function)
    val = Array{T}(undef, size(g))
    for I in eachindex(val)
        val[I] = f(g[I]...)
    end
    GriddedFunction(g, val)
end

function GriddedFunction(T::Type, g::Grid, u::UndefInitializer)
    val = Array{T}(u, size(g))
    GriddedFunction(g, val)
end

"""
    Grid(gf::GriddedFunction)

Return the grid over which `gf` is defined.
"""
Grid(gf::GriddedFunction) = gf.grid

"Return the array of function values of `gf` at each grid point."
values(gf::GriddedFunction) = gf.values

Base.eltype(::Type{GriddedFunction{TX, TY}}) where {TX, TY} = TY
Base.size(gf::GriddedFunction) = size(Grid(gf))

function Base.getindex(gf::GriddedFunction{TX, TY, D}, I::Vararg{Int, D}) where {TX, TY, D}
    values(gf)[I...]
end

function Base.setindex!(gf::GriddedFunction{TX, TY, D}, v::TY, I::Vararg{Int, D}) where {TX, TY, D}
    gf.values[I...] = v
end

"""
    pairs(gf::GriddedFunction)

Return an iterator over `(grid_point => function_value)` pairs of `gf`.

Each element is a `Pair` whose key is a grid point (a scalar for 1-D grids,
a `Tuple` of axis values for higher-dimensional grids) and whose value is the
corresponding function value stored in `gf`.

# Examples

```{julia}
g  = Grid(LinearAxis(range(0.0, 1.0; length=5)))
gf = GriddedFunction(g, x -> x^2, Float64)

for (x, fx) in pairs(gf)
    println("f(\$x) = \$fx")
end
# f(0.0) = 0.0
# f(0.25) = 0.0625
# ...
```

For multi-dimensional grids the grid point is a `Tuple`:

```{julia}
g2  = Grid(LinearAxis(range(0.0, 1.0; length=3)), LinearAxis(range(0.0, 1.0; length=3)))
gf2 = GriddedFunction(g2, (x, y) -> x + y, Float64)

for ((x, y), fxy) in pairs(gf2)
    println("f(\$x, \$y) = \$fxy")
end
```
"""
Base.pairs(gf::GriddedFunction) = (g => v for (g, v) in zip(Grid(gf), values(gf)))

function Base.:+(gfa::GriddedFunction{TX, TYA, D, G}, gfb::GriddedFunction{TX, TYB, D, G}) where {TX, TYA, TYB, D, G}
    @assert Grid(gfa) === Grid(gfb)
    GriddedFunction(Grid(gfa), values(gfa) + values(gfb))
end

function Base.:-(gfa::GriddedFunction{TX, TYA, D, G}, gfb::GriddedFunction{TX, TYB, D, G}) where {TX, TYA, TYB, D, G}
    @assert Grid(gfa) === Grid(gfb)
    GriddedFunction(Grid(gfa), values(gfa) - values(gfb))
end

function Base.:*(gfa::GriddedFunction{TX, TYA, D, G}, gfb::GriddedFunction{TX, TYB, D, G}) where {TX, TYA, TYB, D, G}
    @assert Grid(gfa) === Grid(gfb)
    GriddedFunction(Grid(gfa), values(gfa) .* values(gfb))
end

function Base.:/(gfa::GriddedFunction{TX, TYA, D, G}, gfb::GriddedFunction{TX, TYB, D, G}) where {TX, TYA, TYB, D, G}
    @assert Grid(gfa) === Grid(gfb)
    GriddedFunction(Grid(gfa), values(gfa) ./ values(gfb))
end

function Base.:+(gf::GriddedFunction, c::Real)
    GriddedFunction(Grid(gf), values(gf) .+ c)
end

function Base.:-(gf::GriddedFunction, c::Real)
    GriddedFunction(Grid(gf), values(gf) .- c)
end

function Base.:*(gf::GriddedFunction, c::Real)
    GriddedFunction(Grid(gf), values(gf) .* c)
end

function Base.:/(gf::GriddedFunction, c::Real)
    GriddedFunction(Grid(gf), values(gf) ./ c)
end

Base.:+(c::Real, gf::GriddedFunction) = gf + c
Base.:-(c::Real, gf::GriddedFunction) = GriddedFunction(Grid(gf), c .- values(gf))
Base.:*(c::Real, gf::GriddedFunction) = gf * c
Base.:/(c::Real, gf::GriddedFunction) = GriddedFunction(Grid(gf), c ./ values(gf))

"""
    map(f, gf::GriddedFunction)

Apply scalar function `f` elementwise to every value of `gf` and return a new
`GriddedFunction` on the same grid.

# Examples

```{julia}
map(log,          gf)
map(exp,          gf)
map(x -> x^2,    gf)
map(x -> 1/(1+x), gf)
```
"""
Base.map(f, gf::GriddedFunction) = GriddedFunction(Grid(gf), map(f, values(gf)))

"""
    map!(f, gf::GriddedFunction)

Apply scalar function `f` elementwise to every value of `gf` in place, mutating
`gf` and returning it.
"""
function Base.map!(f, gf::GriddedFunction)
    map!(f, values(gf))
    gf
end

Base.log(gf::GriddedFunction) = map(log, gf)
Base.exp(gf::GriddedFunction) = map(exp, gf)
Base.:^(gf::GriddedFunction, x::Real) = map(v -> v^x, gf)

"""
    continuousview(gf, discretex)

Return a view of `gf`'s value array with all continuous dimensions open and
the discrete dimensions fixed at the grid indices given by `discretex`.
"""
function continuousview(
        gf::GriddedFunction{TX, TY, D, MixedGrid{TX, D, DC, DD, CG, DG}},
        discretex::CartesianIndex{DD}
    ) where {TX, TY, D, DC, DD, CG, DG}
    # view of the values array with continuous dimensions open, discrete dimensions fixed
    view(values(gf), ntuple(_ -> :, Val(DC))..., Tuple(discretex)...)
end

"""
    GriddedFunctionInterpolation{TX, TY, D, DD, G, SITP}

A pre-computed interpolation of a [`GriddedFunction`](@ref) over a grid of
type `G`.

For each combination of discrete axis values, a scaled interpolation object of
type `SITP` is built over the continuous subgrid once at construction time.
Subsequent calls to [`evaluate`](@ref) look up the appropriate interpolation
and evaluate it at the continuous components of the query point, avoiding
repeated allocation.

`DD` is the number of discrete axes (0 for a purely continuous grid, in which
case `interpolations` is a 0-dimensional array holding the single interpolation
object).

# Constructors

    GriddedFunctionInterpolation(gf, interpmode)

Construct from any [`GriddedFunction`](@ref) defined over a `ContinuousGrid`
or a `MixedGrid`. `interpmode` is passed directly to
`Interpolations.interpolate`.
"""
struct GriddedFunctionInterpolation{TX, TY, D, DD, G <: Grid{TX, D}, SITP}
    gf::GriddedFunction{TX, TY, D, G}
    interpolations::Array{SITP, DD}
end

function GriddedFunctionInterpolation(
        gf::GriddedFunction{TX, TY, D, MixedGrid{TX, D, DC, DD, CG, DG}},
        interpmode::Interpolations.InterpolationType
    ) where {TX, TY, D, DC, DD, CG, DG}
    g = Grid(gf)

    itps = map(CartesianIndices(discretegrid(g))) do discretex
        cont_view = continuousview(gf, discretex)
        scale(interpolate(cont_view, interpmode), map(range, continuousaxes(g))...)
    end

    GriddedFunctionInterpolation(gf, itps)
end

function GriddedFunctionInterpolation(
        gf::GriddedFunction{TX, TY, D, ContinuousGrid{TX, D, A}},
        interpmode::Interpolations.InterpolationType
    ) where {TX, TY, D, A}
    g = Grid(gf)
    sitp = scale(interpolate(values(gf), interpmode), map(range, gridaxes(g))...)
    GriddedFunctionInterpolation(gf, fill(sitp))
end

GriddedFunction(gfi::GriddedFunctionInterpolation) = gfi.gf
Grid(gfi::GriddedFunctionInterpolation) = Grid(GriddedFunction(gfi))

"""
    interpolate(
        gf::GriddedFunction, interpmode::Interpolations.InterpolationType = BSpline(Linear())
    )

Returns a `GriddedFunctionInterpolation` object based on `gf`, using
`Interpolations.interpolate` under the hood to compute a scaled interpolation
between all points on the continuous part of the grid of `gf`.
"""
function interpolate(
        gf::GriddedFunction, interpmode::Interpolations.InterpolationType = BSpline(Linear())
    )
    GriddedFunctionInterpolation(gf, interpmode)
end

"""
    evaluate(gitp::GriddedFunctionInterpolation, x)

Evaluate the interpolated function at point `x`.

Looks up the pre-computed interpolation for the discrete components of `x`,
then evaluates it at the continuous components. `x` must be a tuple of type
`TX` with the continuous components first, followed by the discrete components.
"""
function evaluate(gitp::GriddedFunctionInterpolation{TX, TY, D, DD}, x::TX) where {TX, TY, D, DD}
    disc_idx = searchdiscrete(Grid(gitp), x)
    x_cont = ntuple(i -> x[i], Val(D - DD))
    gitp.interpolations[disc_idx](x_cont...)
end


# make interpolated function callable
(gitp::GriddedFunctionInterpolation)(x::Vararg) = evaluate(gitp, x)
