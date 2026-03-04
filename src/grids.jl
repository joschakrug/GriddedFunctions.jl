"""
    Grid{T, D} <: AbstractArray{T, D}

Abstract supertype for all grid types. A grid spanned by `D` axes where each
point is a combination of type `T` of its individual axis coordinates
(typically a `Tuple` or a `Tuple`-like type).

A point type `T` must support construction based on a generator (e.g.
`T(i for i in 1:D)`).

Each subtype of Grid needs to implement at least the following methods:

- [`gridaxes`](@ref)
- [`decompose`](@ref)
- [`finddiscrete`](@ref)
- [`continuousgrid`]
- [`discretegrid`]
"""
abstract type Grid{T, D} <: AbstractArray{T, D} end

"""
    gridaxes(g::Grid, d = nothing)

Return the tuple of axes spanning `g` (if `d = nothing`) or the `d`th axis of
`g` (if `d` is specified).
"""
function gridaxes end

"""
    decompose(g::Grid{T}, x::T) where T

Return a tuple of the continuous and discrete components of point `x`
on grid `g`. Components are tuples of the respective coordinates irrespective
of grid point type `T`.

# Example

```{julia}
continuous, discrete = decompose(grid, x)
```
"""
function decompose end

"""
    finddiscrete(g::Grid{T}, x::T)

Return the Cartesian index of the discrete components of point `x` on the discrete
axes of `g`. Return `CartesianIndex()` if `g` does not have any discrete axes.

For a `ContinuousGrid` this is always `CartesianIndex()`. For a `MixedGrid` it is an
`CartesianIndex{DD}` giving the position of each discrete component on its axis,
found via `searchsortedfirst`. Throws an error if any discrete component of
`x` is outside the range of its axis.
"""
function finddiscrete end

Base.in(x::T, g::Grid{T, D}) where {T, D} = all(x[d] in gridaxes(g, d) for d in 1:D)

"""
    find(g::Grid{T, D}, x::T) where {T, D}

Find the exact index of point `x` on grid `g`. Throw an error if it is not
on the grid.

Returns the CartesianIndex of `x` on the grid if `D > 1` and an integer
index if `D = 1`.
"""
function find(g::Grid{T, D}, x::T) where {T, D}
    CartesianIndex(ntuple(d -> find(gridaxes(g, d), x[d]), Val(D)))
end

find(g::Grid{T, 1}, x::T) where T = find(only(gridaxes(g)), x[1])

Base.eltype(::Type{<: Grid{T}}) where T = T
Base.size(g::Grid{T, D}) where {T, D} = ntuple(d -> length(gridaxes(g, d)), D)

function Base.getindex(g::Grid{T, D}, I::Vararg{Int, D}) where {T, D}
    T(gridaxes(g, d)[I[d]] for d in 1:D)
end

function Base.getindex(g::Grid{T, 1}, I::Int) where T
    T(gridaxes(g, 1)[I])
end

"""
    continuousaxes(g::Grid)

Return the tuple of continuous axes of `g`. Returns an empty `ContinuousGrid()`
if `g` does not have any continuous axes.
"""
continuousaxes(g::Grid) = gridaxes(continuousgrid(g))

"""
    discreteaxes(g::Grid)

Return the tuple of discrete axes of `g`. Returns an empty `DiscreteGrid()`
if `g` does not have any continuous axes.
"""
discreteaxes(g::Grid) = gridaxes(discretegrid(g))

"""
    inbounds(x::T, g::Grid{T}) where T

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
function inbounds(x::T, g::Grid{T}) where T
    x_cont, x_disc = decompose(g, x)
    cont_ok = all(minimum(ax) <= x_cont[d] <= maximum(ax) for (d, ax) in enumerate(continuousaxes(g)))
    disc_ok = all(x_disc[d] in ax for (d, ax) in enumerate(discreteaxes(g)))
    cont_ok && disc_ok
end


"""
    ContinuousGrid{T, D, A <: NTuple{D, ContinuousAxis}} <: Grid{T, D}

A grid spanned by `D` continuous axes. `T` is the type of a grid point
(defaults to a `Tuple` of the axis element types).

# Constructors

    ContinuousGrid(axes::NTuple{D, ContinuousAxis})
    ContinuousGrid(axes::ContinuousAxis...)
"""
struct ContinuousGrid{T, D, A <: NTuple{D, ContinuousAxis}} <: Grid{T, D}
    axes::A

    function ContinuousGrid{T}(axes::A) where {T, D, A <: NTuple{D, ContinuousAxis}}
        @assert fieldtypes(T) === Tuple(eltype(AX) for AX in A.parameters)
        new{T, D, A}(axes)
    end
end

function ContinuousGrid(axes::NTuple{D, ContinuousAxis}) where D
    T = Tuple{(eltype(a) for a in axes)...}
    ContinuousGrid{T}(axes)
end

ContinuousGrid(axes::Vararg{ContinuousAxis}) = ContinuousGrid(axes)

gridaxes(g::ContinuousGrid) = g.axes
gridaxes(g::ContinuousGrid, d) = gridaxes(g)[d]

continuousgrid(g::ContinuousGrid) = g
discretegrid(::ContinuousGrid) = DiscreteGrid()
decompose(::ContinuousGrid{T, D}, x::T) where {T, D} = (ntuple(d -> x[d], Val(D)), ())
finddiscrete(::ContinuousGrid, _) = CartesianIndex()

"""
    DiscreteGrid{T, D, A <: NTuple{D, DiscreteAxis}} <: Grid{T, D}

A grid spanned by `D` discrete axes. `T` is the type of a grid point
(defaults to a `Tuple` of the axis element types).

# Constructors

    DiscreteGrid(axes::NTuple{D, DiscreteAxis})
    DiscreteGrid(axes::DiscreteAxis...)
"""
struct DiscreteGrid{T, D, A <: NTuple{D, DiscreteAxis}} <: Grid{T, D}
    axes::A

    function DiscreteGrid{T}(axes::A) where {T, D, A <: NTuple{D, DiscreteAxis}}
        @assert fieldtypes(T) === Tuple(eltype(AX) for AX in A.parameters)
        new{T, D, A}(axes)
    end
end

function DiscreteGrid(axes::NTuple{D, DiscreteAxis}) where D
    T = Tuple{(eltype(a) for a in axes)...}
    DiscreteGrid{T}(axes)
end

DiscreteGrid(axes::Vararg{DiscreteAxis}) = DiscreteGrid(axes)

gridaxes(g::DiscreteGrid) = g.axes
gridaxes(g::DiscreteGrid, d) = gridaxes(g)[d]

continuousgrid(::DiscreteGrid) = ContinuousGrid()
discretegrid(g::DiscreteGrid) = g
decompose(::DiscreteGrid{T, D}, x::T) where {T, D} = ((), ntuple(d -> x[d], Val(D)))
finddiscrete(g::DiscreteGrid{T}, x::T) where T = find(g, x)

"""
    MixedGrid{T, D, DC, DD, CG <: ContinuousGrid, DG <: DiscreteGrid} <: Grid{T, D}

A grid with `DC` continuous axes followed by `DD` discrete axes (`D = DC + DD`),
stored internally as a `ContinuousGrid` and a `DiscreteGrid`. `T` is the type
of a grid point (a `Tuple` of all axis element types, continuous first).

# Constructor

    MixedGrid(continuous::ContinuousGrid, discrete::DiscreteGrid)

Prefer the [`Grid`](@ref) factory function, which selects the correct subtype
automatically.
"""
struct MixedGrid{T, D, DC, DD, CG <: ContinuousGrid, DG <: DiscreteGrid} <: Grid{T, D}
    continuous::CG
    discrete::DG

    function MixedGrid{T}(continuous::ContinuousGrid{TC, DC, AC}, discrete::DiscreteGrid{TD, DD, AD}) where {T, TC, TD, DC, DD, AC, AD}
        A = tuple(Tuple(AX for AX in AC.parameters)..., Tuple(AX for AX in AD.parameters)...)
        @assert fieldtypes(T) === Tuple(eltype(AX) for AX in A)
        D = DC + DD
        new{T, D, DC, DD, typeof(continuous), typeof(discrete)}(continuous, discrete)
    end
end

function MixedGrid(continuous::ContinuousGrid{TC, DC, CA}, discrete::DiscreteGrid{TD, DD, DA}) where {TC, TD, DC, DD, CA, DA}
    all_axes = (gridaxes(continuous)..., gridaxes(discrete)...)
    T = Tuple{(eltype(a) for a in all_axes)...}
    MixedGrid{T}(continuous, discrete)
end

continuousgrid(g::MixedGrid) = g.continuous
discretegrid(g::MixedGrid) = g.discrete

gridaxes(g::MixedGrid) = (continuousaxes(g)..., discreteaxes(g)...)
gridaxes(g::MixedGrid, d) = gridaxes(g)[d]

"""
    Grid(T::Type = Tuple, axes::Vararg{Axis})

Construct a regular grid from a sequence of axes. Continuous axes
must precede discrete axes. Returns a `ContinuousGrid`, `DiscreteGrid`, or
`MixedGrid` depending on the axis types provided.
"""
function Grid(T::Type, axs::Vararg{Axis})
    discrete = DiscreteAxis[]
    continuous = ContinuousAxis[]

    firstdiscrete = false

    for a in axs
        if (a isa ContinuousAxis) & !firstdiscrete
            push!(continuous, a)
        elseif a isa DiscreteAxis
            push!(discrete, a)
            firstdiscrete = true
        else
            error("Invalid configuration of axes. Continuous axes must be added first.")
        end
    end

    if isempty(discrete)
        ContinuousGrid{T}(Tuple(continuous))
    elseif isempty(continuous)
        DiscreteGrid{T}(Tuple(discrete))
    else
        MixedGrid{T}(ContinuousGrid(Tuple(continuous)), DiscreteGrid(Tuple(discrete)))
    end
end

function Grid(axs::Vararg{Axis})
    T = Tuple{(eltype(a) for a in axs)...}
    Grid(T, axs...)
end

function decompose(::MixedGrid{T, D, DC}, x::T) where {T, D, DC}
    (ntuple(d -> x[d], Val(DC)), ntuple(d -> x[d + DC], Val(D - DC)))
end

function finddiscrete(g::MixedGrid{T, D, DC, DD}, x::T) where {T, D, DC, DD}
    _, x_disc = decompose(g, x)
    find(discretegrid(g), x_disc)
end
