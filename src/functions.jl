
# longer run to do: add a Domain (potentially linked to a GridSection type)
# to allow for functions that are not defined over the entire grid

"""
    AbstractGriddedFunction{TY, D} <: AbstractArray{TY, D}

Abstract supertype for gridded functions. A function with values of type `TY`
defined over a `D`-dimensional grid.

Each subtype must implement:

- [`grid`](@ref) — return the grid
- [`values`](@ref) — return the AbstractArray of values
- [`gridtype`](@ref)
"""
abstract type AbstractGriddedFunction{TY, D} <: AbstractArray{TY, D} end

"""
    grid(gf::AbstractGriddedFunction)

Return the grid over which `gf` is defined.
"""
function grid end

"""
    fvalues(gf::AbstractGriddedFunction)

Return the mutable array of function values of `gf`.
"""
function fvalues end

"""
    gridtype(GF::Type{AbstractGriddedFunction})

Return the type of the grid underlying the concrete gridded function type `GF`.
"""
function gridtype end

Base.size(agf::AbstractGriddedFunction) = size(grid(agf))

function Base.getindex(gf::AbstractGriddedFunction{TY, D}, I::Vararg{Int, D}) where {TY, D}
    fvalues(gf)[I...]
end

function Base.setindex!(gf::AbstractGriddedFunction{TY, D}, v::TY, I::Vararg{Int, D}) where {TY, D}
    fvalues(gf)[I...] = v
end

ncontinuousdims(::Type{GF}) where GF <: AbstractGriddedFunction = ncontinuousdims(gridtype(GF))
ndiscretedims(::Type{GF}) where GF <: AbstractGriddedFunction = ndiscretedims(gridtype(GF))
ncontinuousdims(gf::AbstractGriddedFunction) = ncontinuousdims(typeof(gf))
ndiscretedims(gf::AbstractGriddedFunction) = ndiscretedims(typeof(gf))

"""
    points(gf::AbstractGriddedFunction{TY})

A generator iterating over all points of `gf`, returning each point as a
`Pair` of a grid point and its corresponding functoin value.
"""
points(gf::AbstractGriddedFunction) = (x => y for (x, y) in zip(grid(gf), fvalues(gf)))

"""
    maxpoint(gf::AbstractGriddedFunction)

Find the maximum point of `gf` and return it as a `Pair{TX, TY}` object.

_Note_: For consistency with the [`points`](@ref) function, this returns the
maximising \$x\$ point first and the maximum \$y\$ value second. This differs
from the behaviour of the `Base.findmax` function that returns the \$y\$ value
first and its index second.
"""
function maxpoint(gf::AbstractGriddedFunction)
    (y, I) = findmax(gf)
    (grid(gf)[I] => y)
end

"""
    map!(f, gf::AbstractGriddedFunction)

Apply scalar function `f` elementwise to every value of `gf` in place, mutating
`gf` and returning it.
"""
function Base.map!(f, gf::AbstractGriddedFunction)
    map!(f, fvalues(gf))
    gf
end

"""
    argmap!(f, gf::AbstractGriddedFunction)

Apply function `f` to every element of `gf` in place.

Unlike `Base.map!`, `f` does _not_ take the existing value of `gf` at each point
as its argument. Instead, it takes the grid value at each point of the grid (i.e.
the function's 'x' value) as its argument.
"""
function argmap!(f, gf::AbstractGriddedFunction)
    g = grid(gf)
    v = fvalues(gf)
    for I in eachindex(g)
        v[I] = f(g[I])
    end
    gf
end

"""
    argmap(f, gf::AbstractGriddedFunction)

Like [`argmap`](@ref) but generates a new gridded function over the same grid as
`gf`.
"""
function argmap(f, gf::AbstractGriddedFunction)
    gf_new = similar(gf)
    argmap!(f, gf_new)
end

"""
    GriddedFunction{TY, D, G} <: AbstractGriddedFunction{TY, D}

A function with values of type `TY` defined over a `D`-dimensional grid of
type `G`. Values are stored as a mutable `Array{TY}` with one entry per grid
point.

# Constructors

    GriddedFunction(g::Grid, values::Array)

Construct from a pre-computed array of values. `size(values)` must match
`size(g)`.

    GriddedFunction(T::Type, g::Grid, f::Function)

Construct by evaluating `f` at every grid point. `f` is called with the
individual axis coordinates as separate arguments. The return type must be
convertible to `T`.

    GriddedFunction(T::Type, g::Grid, undef)

Construct an uninitialised `GriddedFunction` over `g` with element type `T`.
"""
mutable struct GriddedFunction{TY, D, G <: Grid} <: AbstractGriddedFunction{TY, D}
    grid::G
    values::Array{TY}

    function GriddedFunction(g::Grid, values)
        @assert size(g) == size(values)
        D = ndims(g)
        new{eltype(values), D, typeof(g)}(g, values)
    end
end

function GriddedFunction(T::Type, g::Grid, f::Function)
    val = Array{T}(undef, size(g))
    for I in eachindex(val)
        val[I] = f(totuple(g[I], g)...)
    end
    GriddedFunction(g, val)
end

function GriddedFunction(T::Type, g::Grid, u::UndefInitializer)
    val = Array{T}(u, size(g))
    GriddedFunction(g, val)
end

grid(gf::GriddedFunction) = gf.grid
fvalues(gf::GriddedFunction) = gf.values
gridtype(::Type{GriddedFunction{TY, D, G}}) where {TY, D, G} = G

# - [ ] think about generalisation of all GriddedFunction methods that
#       produce a _copy_ of the current GriddedFunction to an
#       AbstractGriddedFunction (that may also be a view, for example)

"""
    Base.similar(gf::GriddedFunction)

Create an uninitialised `GriddedFunction` over the same grid as `gf`, and with
the same element type.
"""
Base.similar(gf::GriddedFunction{TY}) where TY = GriddedFunction(TY, grid(gf), undef)

"""
    gfa + gfb
    gfa - gfb
    gfa * gfb
    gfa / gfb
    gf + c  (and c + gf, gf - c, etc.)

Elementwise arithmetic on [`GriddedFunction`](@ref) objects. Both operands
must be defined on the same grid object (checked with `===`). Mixed
`GriddedFunction`/scalar operations broadcast the scalar across all grid
points. All operations return a new `GriddedFunction` on the same grid.
"""
function Base.:+(gfa::GriddedFunction{TYA, D, G}, gfb::GriddedFunction{TYB, D, G}) where {TYA, TYB, D, G}
    @assert grid(gfa) === grid(gfb)
    GriddedFunction(grid(gfa), fvalues(gfa) + fvalues(gfb))
end

function Base.:-(gfa::GriddedFunction{TYA, D, G}, gfb::GriddedFunction{TYB, D, G}) where {TYA, TYB, D, G}
    @assert grid(gfa) === grid(gfb)
    GriddedFunction(grid(gfa), fvalues(gfa) - fvalues(gfb))
end

function Base.:*(gfa::GriddedFunction{TYA, D, G}, gfb::GriddedFunction{TYB, D, G}) where {TYA, TYB, D, G}
    @assert grid(gfa) === grid(gfb)
    GriddedFunction(grid(gfa), fvalues(gfa) .* fvalues(gfb))
end

function Base.:/(gfa::GriddedFunction{TYA, D, G}, gfb::GriddedFunction{TYB, D, G}) where {TYA, TYB, D, G}
    @assert grid(gfa) === grid(gfb)
    GriddedFunction(grid(gfa), fvalues(gfa) ./ fvalues(gfb))
end

function Base.:+(gf::GriddedFunction, c::Real)
    GriddedFunction(grid(gf), fvalues(gf) .+ c)
end

function Base.:-(gf::GriddedFunction, c::Real)
    GriddedFunction(grid(gf), fvalues(gf) .- c)
end

function Base.:*(gf::GriddedFunction, c::Real)
    GriddedFunction(grid(gf), fvalues(gf) .* c)
end

function Base.:/(gf::GriddedFunction, c::Real)
    GriddedFunction(grid(gf), fvalues(gf) ./ c)
end

Base.:+(c::Real, gf::GriddedFunction) = gf + c
Base.:-(c::Real, gf::GriddedFunction) = GriddedFunction(grid(gf), c .- fvalues(gf))
Base.:*(c::Real, gf::GriddedFunction) = gf * c
Base.:/(c::Real, gf::GriddedFunction) = GriddedFunction(grid(gf), c ./ fvalues(gf))

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
Base.map(f, gf::GriddedFunction) = GriddedFunction(grid(gf), map(f, fvalues(gf)))

"""
    log(gf)
    exp(gf)
    gf ^ x

Elementwise `log`, `exp`, or power applied to every value of `gf`. Returns a
new `GriddedFunction` on the same grid.
"""
Base.log(gf::AbstractGriddedFunction) = map(log, gf)
Base.exp(gf::AbstractGriddedFunction) = map(exp, gf)
Base.:^(gf::AbstractGriddedFunction, x::Real) = map(v -> v^x, gf)

"""
    GFInterpolation{GF, DD, SITP}

A pre-computed interpolation of a [`GriddedFunction`](@ref).

For each combination of discrete axis values, a scaled interpolation object of
type `SITP` is built over the continuous subgrid once at construction time.
Subsequent calls to [`evaluate`](@ref) look up the appropriate interpolation
and evaluate it at the continuous components of the query point, avoiding
repeated allocation.

`DD` is the number of discrete axes (0 for a purely continuous grid, in which
case `interpolations` is a 0-dimensional array holding the single interpolation
object).

# Constructor

    GFInterpolation(gf, interpmode)

Construct from any [`GriddedFunction`](@ref). `interpmode` is passed directly
to `Interpolations.interpolate`. Prefer the [`interpolate`](@ref) convenience
wrapper.
"""
struct GFInterpolation{GF <: AbstractGriddedFunction, DD, SITP}
    gf::GF
    interpolations::Array{SITP, DD}

    function GFInterpolation(gf::GF, interpmode::ITP) where {
            GF <: AbstractGriddedFunction,
            ITP <: Interpolations.InterpolationType
        }

        itps = map(discreteindices(grid(gf))) do I
            subgf = continuousview(gf, I)
            itp = Interpolations.interpolate(fvalues(subgf), interpmode)
            Interpolations.scale(itp, map(range, gridaxes(grid(subgf))))
        end

        new{GF, ndiscretedims(GF), eltype(itps)}(gf, itps)
    end
end

"""
    griddedfunction(gfitp::GFInterpolation)

Return the source [`GriddedFunction`](@ref) from which `gfitp` was built.
"""
griddedfunction(gfitp::GFInterpolation) = gfitp.gf
grid(gfitp::GFInterpolation) = grid(griddedfunction(gfitp))

"""
    interpolate(gf::AbstractGriddedFunction, interpmode = BSpline(Linear()))

Returns a [`GFInterpolation`](@ref) object based on `gf`, using
`Interpolations.interpolate` under the hood to compute a scaled interpolation
between all points on the continuous part of the grid of `gf`.
"""
function interpolate(
        gf::AbstractGriddedFunction,
        interpmode = Interpolations.BSpline(Interpolations.Linear())
    )
    GFInterpolation(gf, interpmode)
end

"""
    evaluate(gfitp::GFInterpolation, x)

Evaluate the interpolated function `gfitp` at point `x`.

Looks up the pre-computed interpolation for the discrete components of `x`,
then evaluates it at the continuous components. `x` must be a valid point type
for the underlying grid (i.e. `eltype(grid(gfitp))`).
"""
function evaluate(gfitp::GFInterpolation, x)
    g = grid(gfitp)
    t_cont, t_disc = decompose(x, g)
    I_disc = finddiscrete(t_disc, g)
    gfitp.interpolations[I_disc](t_cont...)
end

# make interpolated function callable

_aspoint(x::Tuple, g::AbstractGrid) = topoint(x, g)
_aspoint(x::Tuple{T}, ::AbstractGrid{T}) where T = only(x)

(gfitp::GFInterpolation)(x::Vararg) = evaluate(gfitp, _aspoint(x, grid(gfitp)))
