
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
mutable struct GriddedFunction{TX, TY, D, G <: Grid{TX, D}} <: AbstractArray{TY, D}
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
    grid(gf::GriddedFunction)

Return the grid over which `gf` is defined.
"""
grid(gf::GriddedFunction) = gf.grid

"Return the array of function values of `gf` at each grid point."
values(gf::GriddedFunction) = gf.values

"""
    points(gf::GriddedFunction{TX, TY})

A generator iterating over all points of `gf`, returning each point as a
`Pair{TX, TY}` object. 
"""
points(gf::GriddedFunction) = (x => y for (x, y) in zip(grid(gf), values(gf)))

"""
    maxpoint(gf::GriddedFunction)

Find the maximum point of `gf` and return it as a `Pair{TX, TY}` object.

_Note_: For consistency with the [`points`](@ref) function, this returns the
maximising \$x\$ point first and the maximum \$y\$ value second. This differs
from the behaviour of the `Base.findmax` function that returns the \$y\$ value
first and its index second.
"""
function maxpoint(gf::GriddedFunction)
    (y, I) = findmax(gf)
    (grid(gf)[I] => y)
end

Base.eltype(::Type{GriddedFunction{TX, TY}}) where {TX, TY} = TY
Base.size(gf::GriddedFunction) = size(grid(gf))

"""
    Base.similar(gf::GriddedFunction)

Create an uninitialised `GriddedFunction` over the same grid as `gf`, and with
the same element type.
"""
Base.similar(gf::GriddedFunction{TX, TY}) where {TX, TY} = GriddedFunction(TY, grid(gf), undef)

function Base.getindex(gf::GriddedFunction{TX, TY, D}, I::Vararg{Int, D}) where {TX, TY, D}
    values(gf)[I...]
end

function Base.setindex!(gf::GriddedFunction{TX, TY, D}, v::TY, I::Vararg{Int, D}) where {TX, TY, D}
    gf.values[I...] = v
end

function Base.:+(gfa::GriddedFunction{TX, TYA, D, G}, gfb::GriddedFunction{TX, TYB, D, G}) where {TX, TYA, TYB, D, G}
    @assert grid(gfa) === grid(gfb)
    GriddedFunction(grid(gfa), values(gfa) + values(gfb))
end

function Base.:-(gfa::GriddedFunction{TX, TYA, D, G}, gfb::GriddedFunction{TX, TYB, D, G}) where {TX, TYA, TYB, D, G}
    @assert grid(gfa) === grid(gfb)
    GriddedFunction(grid(gfa), values(gfa) - values(gfb))
end

function Base.:*(gfa::GriddedFunction{TX, TYA, D, G}, gfb::GriddedFunction{TX, TYB, D, G}) where {TX, TYA, TYB, D, G}
    @assert grid(gfa) === grid(gfb)
    GriddedFunction(grid(gfa), values(gfa) .* values(gfb))
end

function Base.:/(gfa::GriddedFunction{TX, TYA, D, G}, gfb::GriddedFunction{TX, TYB, D, G}) where {TX, TYA, TYB, D, G}
    @assert grid(gfa) === grid(gfb)
    GriddedFunction(grid(gfa), values(gfa) ./ values(gfb))
end

function Base.:+(gf::GriddedFunction, c::Real)
    GriddedFunction(grid(gf), values(gf) .+ c)
end

function Base.:-(gf::GriddedFunction, c::Real)
    GriddedFunction(grid(gf), values(gf) .- c)
end

function Base.:*(gf::GriddedFunction, c::Real)
    GriddedFunction(grid(gf), values(gf) .* c)
end

function Base.:/(gf::GriddedFunction, c::Real)
    GriddedFunction(grid(gf), values(gf) ./ c)
end

Base.:+(c::Real, gf::GriddedFunction) = gf + c
Base.:-(c::Real, gf::GriddedFunction) = GriddedFunction(grid(gf), c .- values(gf))
Base.:*(c::Real, gf::GriddedFunction) = gf * c
Base.:/(c::Real, gf::GriddedFunction) = GriddedFunction(grid(gf), c ./ values(gf))

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
Base.map(f, gf::GriddedFunction) = GriddedFunction(grid(gf), map(f, values(gf)))

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
    xmap!(f, gf::GriddedFunction)

Apply function `f` to every element of `gf` in place.

Unlike `Base.map!`, `f` does _not_ take the existing value of `gf` at each point
as its argument. Instead, it takes the grid value at each point of the grid (i.e.
the function's 'x' value) as its argument.
"""
function xmap!(f, gf::GriddedFunction)
    g = grid(gf)
    v = values(gf)
    for I in eachindex(g)
        v[I] = f(g[I])
    end
    gf
end

"""
    xmap(f, gf::GriddedFunction)

Like [`xmap`](@ref) but generates a new gridded function over the same grid as
`gf`.
"""
function xmap(f, gf::GriddedFunction)
    gf_new = similar(gf)
    xmap!(f, gf_new)
end

"""
    continuousview(gf, idiscrete)

Return a view of `gf`'s value array with all continuous dimensions open and
the discrete dimensions fixed at the grid indices given by `idiscrete`.
"""
function continuousview(
        gf::GriddedFunction{TX, TY, D, MixedGrid{TX, D, DC, DD, CG, DG}},
        idiscrete::CartesianIndex{DD}
    ) where {TX, TY, D, DC, DD, CG, DG}
    # view of the values array with continuous dimensions open, discrete dimensions fixed
    view(values(gf), ntuple(_ -> :, Val(DC))..., Tuple(idiscrete)...)
end

"""
    GFInterpolation{TX, TY, D, DD, G, SITP}

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

    GFInterpolation(gf, interpmode)

Construct from any [`GriddedFunction`](@ref) defined over a `ContinuousGrid`
or a `MixedGrid`. `interpmode` is passed directly to
`Interpolations.interpolate`.
"""
struct GFInterpolation{TX, TY, D, DD, G <: Grid{TX, D}, SITP}
    gf::GriddedFunction{TX, TY, D, G}
    interpolations::Array{SITP, DD}
end

function GFInterpolation(
        gf::GriddedFunction{TX, TY, D, MixedGrid{TX, D, DC, DD, CG, DG}},
        interpmode::Interpolations.InterpolationType
    ) where {TX, TY, D, DC, DD, CG, DG}
    g = grid(gf)

    itps = map(CartesianIndices(discretegrid(g))) do discretex
        cont_view = continuousview(gf, discretex)
        Interpolations.scale(Interpolations.interpolate(cont_view, interpmode), map(range, continuousaxes(g))...)
    end

    GFInterpolation(gf, itps)
end

function GFInterpolation(
        gf::GriddedFunction{TX, TY, D, ContinuousGrid{TX, D, A}},
        interpmode::Interpolations.InterpolationType
    ) where {TX, TY, D, A}
    g = grid(gf)
    sitp = Interpolations.scale(Interpolations.interpolate(values(gf), interpmode), map(range, gridaxes(g))...)
    GFInterpolation(gf, fill(sitp))
end

griddedfunction(gfitp::GFInterpolation) = gfitp.gf
grid(gfitp::GFInterpolation) = grid(griddedfunction(gfitp))

"""
    interpolate(
        gf::GriddedFunction, interpmode::Interpolations.InterpolationType = BSpline(Linear())
    )

Returns a [`GFInterpolation`](@ref) object based on `gf`, using
`Interpolations.interpolate` under the hood to compute a scaled interpolation
between all points on the continuous part of the grid of `gf`.
"""
function interpolate(
        gf::GriddedFunction, interpmode::Interpolations.InterpolationType = Interpolations.BSpline(Interpolations.Linear())
    )
    GFInterpolation(gf, interpmode)
end

"""
    evaluate(gfitp::GFInterpolation, x)

Evaluate the interpolated function `gitp` at point `x`.

Looks up the pre-computed interpolation for the discrete components of `x`,
then evaluates it at the continuous components. `x` must be of `gfitp`'s domain
type `TX`.
"""
function evaluate(gfitp::GFInterpolation{TX, TY, D, DD}, x::TX) where {TX, TY, D, DD}
    g = grid(gfitp)
    idx_disc = finddiscrete(g, x)
    x_cont, _ = decompose(g, x)
    gfitp.interpolations[idx_disc](x_cont...)
end

# make interpolated function callable
(gfitp::GFInterpolation{TX})(x::Vararg) where TX = evaluate(gfitp, TX(x))
(gfitp::GFInterpolation{TX})(x::TX) where TX = evaluate(gfitp, x)
