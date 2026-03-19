"""
    dimnames(::Type{T}) where T

Returns the names of the different dimensions as per data type `T`. Can be
overloaded for any type `T` in order to support axis selection by name for
any [`AbstractGrid{T}`](@ref) and its subtypes.

Defaults to an empty tuple for all types that do not have dimension names
specified. Defaults to the field names of `T` if `T` is a named tuple.
"""
dimnames(::Type{T}) where T <: Any = ()
dimnames(::Type{T}) where T <: NamedTuple = fieldnames(T)
dimnames(::Type{T}, d) where T = dimnames(T)[d]

"""
    dimnum(::Type{T}, name::Symbol) where T

Return the number of dimension `name` in type `T`, as specified by
[`dimnames`](@ref).
"""
dimnum(::Type{T}, name::Symbol) where T = findfirst(==(name), dimnames(T))

"""
    AbstractGrid{T, D} <: AbstractArray{T, D}

Abstract supertype for all grid types. A grid spanned by `D` axes where each
point is of type `T`. The grid may contain both continuous and discrete axes
(but continuous axes have to come first).

Each subtype of Grid needs to implement at least the following methods:

- [`gridaxes`](@ref)
- [`topoint`](@ref)
- [`totuple`](@ref)
- [`ncontinuousdims`](@ref)
"""
abstract type AbstractGrid{T, D} <: AbstractArray{T, D} end

"""
    gridaxes(g::AbstractGrid, d = nothing)

Return the tuple of axes spanning `g` (if `d = nothing`) or the `d`th axis of
`g` (if `d` is specified). `d` can be either the number of the axis or, if
the grid is based on a named type (see [`dimnames`](@ref)), the name of the
respective axis.

This method must be implemented by any concrete type implementing the
[`AbstractGrid`](@ref) interface.]
`gridaxes(g, d)` defaults to `gridaxes(g)[d]` if no method specific to
the concrete type of `g` is defined.
"""
function gridaxes end
gridaxes(g::AbstractGrid, d::Int) = gridaxes(g)[d]
gridaxes(g::AbstractGrid{T}, d::Symbol) where T = gridaxes(g, dimnum(T, d))

"""
    topoint(t::NTuple{D}, g::AbstractGrid{T, D}) -> x::T

Return the point `x` corresponding to the tuple of axis values `t` on grid `g`.

This method must be implemented by any concrete type implementing the
[`AbstractGrid`](@ref) interface.
"""
function topoint end

"""
    totuple(x::T, g::AbstractGrid{T, D}) -> t::NTuple{D}

Return the tuple of axis values `t` corresponding to point `x` on grid `g`.

This method must be implemented by any concrete type implementing the
[`AbstractGrid`](@ref) interface.
"""
function totuple end

"""
    ncontinuousdims(G::Type{<: AbstractGrid})
    ncontinuousdims(g::G) where G <: AbstractGrid

Return the number of continuous dimensions of concrete grid type `G`.

This method must be implemented by any concrete type implementing the
[`AbstractGrid`](@ref) interface.
"""
function ncontinuousdims end

"""
    ndiscretedims(G::Type{<: AbstractGrid})
    ndiscretedims(g::G) where G <: AbstractGrid

Return the number of discrete dimensions of concrete grid type `G`.
"""
ndiscretedims(G::Type{<: AbstractGrid{T, D}}) where {T, D} = D - ncontinuousdims(G)

ncontinuousdims(g::AbstractGrid) = ncontinuousdims(typeof(g))
ndiscretedims(g::AbstractGrid) = ndiscretedims(typeof(g))

"""
    find(g::AbstractGrid{T, D}, x::T) where {T, D}

Find the index of point `x` on grid `g`.

If `x` is wrapped in
[`approximately`](@ref), inexact values along continuous dimensions will be
tolerated and the index of the closest value on the respective axes will be
returned. Otherwise, throw an error if `x` is not exactly on the grid.

Returns the CartesianIndex of `x` on the grid if `D > 1` and an integer
index if `D = 1`.
"""
function find(x::T, g::AbstractGrid{T, D}) where {T, D}
    t = totuple(x, g)
    CartesianIndex(
        ntuple(
            d -> find(t[d], gridaxes(g, d)), Val(D)
        )
    )
end

function find(x::Approximator{T}, g::AbstractGrid{T, D}) where {T, D}
    t = totuple(value(x), g)
    CartesianIndex(
        ntuple(
            d -> find(approximately(t[d]), gridaxes(g, d)), Val(D)
        )
    )
end

function find(x::T, g::AbstractGrid{T, 1}) where T
    t = totuple(x, g)
    find(only(t), only(gridaxes(g)))
end

function find(x::Approximator{T}, g::AbstractGrid{T, 1}) where T
    t = totuple(value(x), g)
    find(approximately(only(t)), only(gridaxes(g)))
end

function Base.getindex(g::AbstractGrid{T, D}, I::Vararg{Int, D}) where {T, D}
    t = ntuple(d -> gridaxes(g, d)[I[d]], Val(D))
    topoint(t, g)
end

function Base.getindex(g::AbstractGrid{T, 1}, i::Int) where T
    t = (only(gridaxes(g))[i],)
    topoint(t, g)
end

Base.eltype(::Type{<: AbstractGrid{T}}) where T = T
Base.size(g::AbstractGrid{T, D}) where {T, D} = ntuple(d -> length(gridaxes(g, d)), Val(D))

"""
    continuousaxes(g::AbstractGrid)

Return the tuple of continuous axes of `g`. Returns an empty tuple
if `g` does not have any continuous axes.
"""
function continuousaxes(g::G) where {T, D, G <: AbstractGrid{T, D}}
    ntuple(d -> gridaxes(g, d), ncontinuousdims(G))
end

"""
    discreteaxes(g::AbstractGrid)

Return the tuple of discrete axes of `g`. Returns an empty tuple
if `g` does not have any discrete axes.
"""
function discreteaxes(g::G) where {T, D, G <: AbstractGrid{T, D}}
    ntuple(d -> gridaxes(g, d + ncontinuousdims(G)), ndiscretedims(G))
end

"""
    discreteindices(g::AbstractGrid)

Return a `CartesianIndices` object iterating over all index combinations of the
discrete axes of `g`. For a purely continuous grid (no discrete axes) this
returns a 0-dimensional `CartesianIndices{0}` whose single element is
`CartesianIndex()`.
"""
function discreteindices(g::G) where {G <: AbstractGrid}
    ax = discreteaxes(g)
    CartesianIndices(ntuple(d -> length(ax[d]), Val(ndiscretedims(G))))
end

"""
    finddiscrete(td::Tuple, g::AbstractGrid{T})

Return the Cartesian index of a tuple of discrete axis coordinates on the
discrete component of a grid.

# Example

```julia
g = Grid(LinearAxis(range(0, 5, length = 8)), DiscreteAxis([0, 1, 2]))
t_cont, t_disc = decompose((2.5, 1), g)
finddiscrete(t_disc, g) # == 2
```
"""
function finddiscrete(td::Tuple, g::G) where {T, D, G <: AbstractGrid{T, D}}
    CartesianIndex(ntuple(d -> begin
            dd = d + ncontinuousdims(G)
            find(td[d], gridaxes(g, dd))
        end, ndiscretedims(G)
    ))
end

"""
    decompose(x::T, g::AbstractGrid{T})

Decompose point `x` into a tuple of continuous axis coordinates and a tuple of
discrete axis coordinates.

# Example

```julia
cont, disc = decompose(x, g)
```
"""
function decompose(x::T, g::G) where {T, D, G <: AbstractGrid{T, D}}
    t = totuple(x, g)
    (t[1:ncontinuousdims(G)], t[(ncontinuousdims(G) + 1):D])
end

"""
    inbounds(x::T, g::AbstractGrid{T}) where T

Check whether point `x` lies within the valid domain of grid `g`.

- For each **continuous** axis the corresponding coordinate of `x` must lie
  within the closed interval `[minimum(ax), maximum(ax)]`.
- For each **discrete** axis the coordinate of `x` must equal an axis point
  exactly (i.e. `coord in ax`).

Returns `true` if all components satisfy their respective conditions,
`false` otherwise.  Unlike [`Base.in`](@ref), continuous coordinates are only
required to be in-range, not to coincide with an exact grid point.

# Examples

```julia
g = Grid(LinearAxis(range(0.0, 1.0; length=100)), DiscreteAxis([0, 1]))
inbounds((0.42, 1), g)   # true  – 0.42 is in [0,1] and 1 is on the discrete axis
inbounds((1.5,  0), g)   # false – 1.5 is out of range
inbounds((0.5,  2), g)   # false – 2 is not on the discrete axis
```
"""
function inbounds(x::T, g::AbstractGrid{T}) where T
    t_cont, t_disc = decompose(x, g)
    cont_ok = all(minimum(ax) <= t_cont[dc] <= maximum(ax) for (dc, ax) in enumerate(continuousaxes(g)))
    cont_ok && all(t_disc[dd] in ax for (dd, ax) in enumerate(discreteaxes(g)))
end

"""
    Grid{T, D, A, DC} <: AbstractGrid{T, D}

A concrete regular grid spanned by `D` axes where each point is of type `T`.
All continuous axes must come before all discrete axes.

A custom point type `T` must support:
- `convert(T, t::NTuple{D})` — construct a point from a tuple of axis values
- `x[d]` — retrieve the `d`-th axis component of a point `x`

# Constructors

    Grid(axes::Vararg{Axis})
    Grid(T::Type, axes::Vararg{Axis})

Construct from any combination of [`LinearAxis`](@ref) and [`DiscreteAxis`](@ref).
Without an explicit type argument `T` defaults to `Tuple{eltype(ax₁), eltype(ax₂), …}`.
Throws an error if any discrete axis precedes a continuous axis.

    Grid(T::Type; name = axis, ...)

Construct from named keyword arguments. The axes are reordered to match
`dimnames(T)`, so the keywords may be given in any order. Requires
[`dimnames`](@ref) to be defined for `T`.

    Grid(; name = axis, ...)

Construct from named keyword arguments with no explicit point type. The axes
are taken in the order the keywords are given and `T` defaults to
`NamedTuple{names, Tuple{eltype(ax₁), …}}` where `names` are the keyword
names. Points of the resulting grid are `NamedTuple`s whose fields match the
axis names.
"""
struct Grid{T, D, A <: NTuple{D, Axis}, DC} <: AbstractGrid{T, D}
    axes::A

    function Grid{T}(axes::NTuple{D, Axis}) where {T, D}
        DC = 0
        for d in 1:D
            if axes[d] isa ContinuousAxis
                d == (DC + 1) ? DC += 1 : error("Continuous axes need to come first when defining a grid")
            end
        end

        new{T, D, typeof(axes), DC}(axes)
    end
end

Grid(T::Type, axes::Vararg{Axis}) = Grid{T}(axes)
function Grid(axes::Vararg{Axis})
    T = Tuple{map(eltype, axes)...}
    Grid{T}(axes)
end

function Grid(T::Type; kwargs...)
    if (dimnames(T) == ()) || (length(dimnames(T)) != length(kwargs))
        error("Named dimensions of type $T do not correspond to grid axis specification.")
    end
    names = dimnames(T)
    axes = ntuple(d -> kwargs[names[d]], Val(length(names)))
    Grid{T}(axes)
end

function Grid(; kwargs...)
    axes = Tuple(values(kwargs))
    T = NamedTuple{keys(kwargs), Tuple{map(eltype, axes)...}}
    Grid{T}(axes)
end

gridaxes(g::Grid) = g.axes
topoint(t::NTuple{D, Any}, ::Grid{T, D}) where {T, D} = convert(T, t)
totuple(x::T, ::Grid{T, D}) where {T, D} = ntuple(d -> x[d], Val(D))
ncontinuousdims(::Type{Grid{T, D, A, DC}}) where {T, D, A, DC} = DC

"""
    Base.in(x::T, g::Grid{T, D})

Return `true` if `x` is a point on grid `g`, i.e. every component of
`x` lies on its corresponding axis.

If `x` is wrapped in
[`approximately`](@ref), inexact values along continuous dimensions will be
tolerated and the index of the closest value on the respective axes will be
returned. Otherwise, return false if `x` is not exactly on the grid.
"""
function Base.in(x::T, g::Grid{T, D}) where {T, D}
    t = totuple(x, g)
    all(t[d] in gridaxes(g, d) for d in 1:D)
end

function Base.in(x::Approximator{T}, g::Grid{T, D}) where {T, D}
    t = totuple(T(x), g)
    all(approximately(t[d]) in gridaxes(g, d) for d in 1:D)
end
