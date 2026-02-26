
"""
    Axis{T} <: AbstractVector{T}

A grid axis with axis element type `T`.

Each subtype of the abstract type `Axis{T}` needs to define at least the
following methods:

- `Base.length`
- `Base.getindex`
- `search` (return the unique index of a value `x` if it is on the axis)
"""
abstract type Axis{T} <: AbstractVector{T} end

Base.eltype(::Type{Axis{T}}) where T = T
Base.size(ax::A) where {A <: Axis} = (length(ax),)

"Search the index of value `x` on axis `ax` and raise an error if it is not on the axis."
search(ax::Axis{T}, x::T) where T = search(ax, x)

"""
    ContinuousAxis{T} <: Axis{T}

A grid axis representing a continuous range of values of type `T`.

Each subtype of the abstract type `ContinuousAxis{T}` needs to define the
additional methods:

- `Base.minimum`
- `Base.maximum`
"""
abstract type ContinuousAxis{T} <: Axis{T} end

function Base.in(value::T, axis::ContinuousAxis{T}) where T
    value >= minimum(axis) && value <= maximum(axis)
end

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

Base.length(lax::LinearAxis) = length(range(lax))
Base.getindex(lax::LinearAxis, i) = getindex(range(lax), i)
Base.minimum(lax::LinearAxis) = minimum(range(lax))
Base.maximum(lax::LinearAxis) = maximum(range(lax))
Base.iterate(lax::LinearAxis) = iterate(range(lax))
Base.iterate(lax::LinearAxis, state) = iterate(range(lax), state)

function _bestguessindex(lax::LinearAxis{T}, x::T) where T
    (x - minimum(lax)) / (maximum(lax) - minimum(lax)) * (length(lax) - 1) + 1
end

function search(lax::LinearAxis{T}, x::T) where T
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
        if issorted(points)
            new{T}(points)
        else
            error("Points on axis need to be stored in ascending order.")
        end
    end
end

"Return the underlying sorted vector of points of `dax`."
points(dax::DiscreteAxis) = dax.points

Base.length(dax::DiscreteAxis) = length(points(dax))
Base.getindex(dax::DiscreteAxis, i) = getindex(points(dax), i)
Base.in(value, dax::DiscreteAxis) = in(value, points(dax))

function search(dax::DiscreteAxis{T}, x::T) where T
    pts = points(dax)

    if (x >= pts[1]) & (x <= pts[end])
        searchsortedfirst(pts, x)
    else
        error("$x not on axis.")
    end
end

"""
    Grid{T, D} <: AbstractArray{T, D}

Abstract supertype for all grid types. A grid spanned by `D` axes where each
point is a tuple of type `T`.

Each subtype of Grid needs to implement at least the following methods:

- `axes`: return a `D`-dimensional tuple of axes
"""
abstract type Grid{T, D} <: AbstractArray{T, D} end

Base.in(x::T, g::Grid{T}) where T = all(x[d] in axes(g, d) for d in 1:D)

"""
    search(g::Grid{T, D}, x::T) where {T, D}

Search the exact index of point `x` on grid `g`. Throw an error if it is not
on the grid.
"""
function search(g::Grid{T, D}, x::T) where {T, D}
    ntuple(d -> search(axes(g, d), x[d]), Val(D))
end

"""
    ContinuousGrid{T, D, A <: NTuple{D, ContinuousAxis}} <: Grid{T, D}

A grid spanned by `D` continuous axes. `T` is the type of a grid point
(a `Tuple` of the axis element types).

# Constructors

    ContinuousGrid(axes::NTuple{D, ContinuousAxis})
    ContinuousGrid(axes::ContinuousAxis...)
"""
struct ContinuousGrid{T, D, A <: NTuple{D, ContinuousAxis}} <: Grid{T, D}
    axes::A

    function ContinuousGrid(axes::NTuple{D, ContinuousAxis}) where D
        T = Tuple{(eltype(a) for a in axes)...}
        new{T, D, typeof(axes)}(axes)
    end
end

ContinuousGrid(axes::Vararg{ContinuousAxis}) = ContinuousGrid(axes)

"Return the tuple of axes spanning `g`."
axes(g::ContinuousGrid) = g.axes

"""
    DiscreteGrid{T, D, A <: NTuple{D, DiscreteAxis}} <: Grid{T, D}

A grid spanned by `D` discrete axes. `T` is the type of a grid point
(a `Tuple` of the axis element types).

# Constructors

    DiscreteGrid(axes::NTuple{D, DiscreteAxis})
    DiscreteGrid(axes::DiscreteAxis...)
"""
struct DiscreteGrid{T, D, A <: NTuple{D, DiscreteAxis}} <: Grid{T, D}
    axes::A

    function DiscreteGrid(axes::NTuple{D, DiscreteAxis}) where D
        T = Tuple{(eltype(a) for a in axes)...}
        new{T, D, typeof(axes)}(axes)
    end
end

DiscreteGrid(axes::Vararg{DiscreteAxis}) = DiscreteGrid(axes)

"Return the tuple of axes spanning `g`."
axes(g::DiscreteGrid) = g.axes

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

    function MixedGrid(continuous::ContinuousGrid{TC, DC, CA}, discrete::DiscreteGrid{TD, DD, DA}) where {TC, TD, DC, DD, CA, DA}
        all_axes = (axes(continuous)..., axes(discrete)...)
        T = Tuple{(eltype(a) for a in all_axes)...}
        D = DC + DD
        new{T, D, DC, DD, typeof(continuous), typeof(discrete)}(continuous, discrete)
    end
end

"Return the `ContinuousGrid` component of `g`."
continuousgrid(g::MixedGrid) = g.continuous

"Return the `DiscreteGrid` component of `g`."
discretegrid(g::MixedGrid) = g.discrete

"Return the tuple of continuous axes of `g`."
continuousaxes(g::MixedGrid) = axes(continuousgrid(g))

"Return the tuple of discrete axes of `g`."
discreteaxes(g::MixedGrid) = axes(discretegrid(g))

"Return the tuple of all axes spanning `g` (continuous first, then discrete)."
axes(g::MixedGrid) = (continuousaxes(g)..., discreteaxes(g)...)

"""
    Grid(axes::Vararg{Axis})

Construct the appropriate grid subtype from a sequence of axes. Continuous axes
must precede discrete axes. Returns a `ContinuousGrid`, `DiscreteGrid`, or
`MixedGrid` depending on the axis types provided.
"""
function Grid(axs::Vararg{Axis})
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
        ContinuousGrid(Tuple(continuous))
    elseif isempty(continuous)
        DiscreteGrid(Tuple(discrete))
    else
        MixedGrid(ContinuousGrid(Tuple(continuous)), DiscreteGrid(Tuple(discrete)))
    end
end

"Return the `d`-th axis of `g`."
axes(g::Grid, d) = axes(g)[d]

Base.eltype(::Type{<:Grid{T}}) where T = T
Base.size(g::Grid{T, D}) where {T, D} = ntuple(d -> length(axes(g, d)), D)

function Base.getindex(g::Grid{T, D}, I::Vararg{Int, D}) where {T, D}
    Tuple(axes(g, d)[I[d]] for d in 1:D)
end

Base.getindex(g::Grid{T, 1}, i::Int) where T = only(axes(g))[i]

"""
    discretecomponent(g::MixedGrid, x)

Return the discrete components of grid point `x` as a tuple (i.e. elements
`DC+1` through `D`).
"""
function discretecomponent(g::MixedGrid{T, D, DC}, x::T) where {T, D, DC}
    x[(DC + 1):D]
end

"""
    continuouscomponent(g::MixedGrid, x)

Return the continuous components of grid point `x` as a tuple (i.e. the first
`DC` elements).
"""
function continuouscomponent(g::MixedGrid{T, D, DC}, x::T) where {T, D, DC}
    x[1:DC]
end

"""
    searchdiscrete(g::Grid, x)

Return the index tuple into the discrete axes of `g` for point `x`.

For a `ContinuousGrid` this is always `()`. For a `MixedGrid` it is an
`NTuple{DD, Int}` giving the position of each discrete component on its axis,
found via `searchsortedfirst`. Throws an error if any discrete component of
`x` is outside the range of its axis.
"""
searchdiscrete(::ContinuousGrid, _) = ()

function searchdiscrete(g::MixedGrid{T, D, DC, DD}, x::T) where {T, D, DC, DD}
    search(discretegrid(g), discretecomponent(g, x))
end

searchdiscrete(g::DiscreteGrid{T}, x::T) where T = search(g, x)


