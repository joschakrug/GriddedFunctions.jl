
"""
    Axis{T} <: AbstractVector{T}

A grid axis with axis element type `T`.

Each subtype of the abstract type `Axis{T}` needs to define at least the
following methods:

- `Base.length`
- `Base.getindex`
- `Base.in`
- [`find`](@ref)
"""
abstract type Axis{T} <: AbstractVector{T} end

Base.eltype(::Type{Axis{T}}) where T = T
Base.size(ax::A) where {A <: Axis} = (length(ax),)

"""
    find(ax::Axis{T}, x::T) where T

Return the (integer) index of value `x` on axis `ax` and raise an error if `x`
does not correspond to a point on `ax`.
"""
function find end

"""
    ContinuousAxis{T} <: Axis{T}

A grid axis representing a continuous range of values of type `T`.

Each subtype of the abstract type `ContinuousAxis{T}` needs to define the
additional methods:

- `Base.minimum`
- `Base.maximum`
"""
abstract type ContinuousAxis{T} <: Axis{T} end

"""
    LinearAxis{T, S <: StepRangeLen{T}} <: ContinuousAxis{T}

A linearly scaled continuous grid axis representing values of type `T`.

`S` is just an auxiliary parameter to ensure type stability.

# Constructor

    LinearAxis(r::StepRangeLen{T})

Construct a linear axis from a step range `r`. Use `range(start, stop; length=n)`
or `range(start, stop; step=s)` to create suitable ranges.
"""
struct LinearAxis{T, S <: StepRangeLen{T}} <: ContinuousAxis{T}
    range::S
end

Base.range(lax::LinearAxis) = lax.range

for f in (:length, :getindex, :minimum, :maximum, :iterate)
    @eval Base.$f(lax::LinearAxis, args...) = Base.$f(range(lax), args...)
end

function Base.in(x::T, axis::LinearAxis{T}) where T
    x in range(axis)
end

# function _bestguessindex(lax::LinearAxis{T}, x::T) where T
#     (x - minimum(lax)) / (maximum(lax) - minimum(lax)) * (length(lax) - 1) + 1
# end

function find(lax::LinearAxis{T}, x::T) where T
    i = searchsortedlast(range(lax), x)
    @assert lax[i] ≈ x
    i
end


"""
    DiscreteAxis{T} <: Axis{T}

An axis representing discrete but ordered values of type `T`.

# Constructor

    DiscreteAxis(points::Vector{T})

Construct a discrete axis from a sorted vector of points. Throws an error if
`points` is not in ascending order.
"""
struct DiscreteAxis{T} <: Axis{T}
    points::Vector{T}

    function DiscreteAxis(points::Vector{T}) where T
        if all(points[i] < points[i + 1] for i in 1:(length(points) - 1))
            new{T}(points)
        else
            error("Points on axis need to be unique and stored in ascending order.")
        end
    end
end

"Return the underlying sorted vector of points of `dax`."
points(dax::DiscreteAxis) = dax.points

Base.length(dax::DiscreteAxis) = length(points(dax))
Base.getindex(dax::DiscreteAxis, i) = getindex(points(dax), i)
Base.in(value, dax::DiscreteAxis) = in(value, points(dax))

function find(dax::DiscreteAxis{T}, x::T) where T
    pts = points(dax)

    if pts[1] <= x <= pts[end]
        searchsortedfirst(pts, x)
    else
        error("$x not on axis.")
    end
end

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

decompose(::ContinuousGrid, x) = (x, ())
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

decompose(::DiscreteGrid, x) = ((), x)
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

"Return the `ContinuousGrid` component of `g`."
continuousgrid(g::MixedGrid) = g.continuous

"Return the `DiscreteGrid` component of `g`."
discretegrid(g::MixedGrid) = g.discrete

"Return the tuple of continuous axes of `g`."
continuousaxes(g::MixedGrid) = gridaxes(continuousgrid(g))

"Return the tuple of discrete axes of `g`."
discreteaxes(g::MixedGrid) = gridaxes(discretegrid(g))

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
