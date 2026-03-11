"""
    Axis{T} <: AbstractVector{T}

A grid axis with axis element type `T`.

Each subtype of the abstract type `Axis{T}` needs to define at least the
following methods:

- `Base.length`
- `Base.getindex` (for integers as well as for unit ranges)
- `Base.in`
- [`find`](@ref)
"""
abstract type Axis{T} <: AbstractVector{T} end

Base.eltype(::Type{Axis{T}}) where T = T
Base.size(ax::A) where {A <: Axis} = (length(ax),)

"""
    find(x::T, ax::Axis{T}) where T

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

function Base.getindex(ax::LAX, rng::UnitRange{Int}) where {LAX <: LinearAxis}
    LAX(range(ax[minimum(rng)], ax[maximum(rng)], length = length(rng)))
end

function Base.in(x::T, axis::LinearAxis{T}) where T
    x in range(axis)
end

# function _bestguessindex(lax::LinearAxis{T}, x::T) where T
#     (x - minimum(lax)) / (maximum(lax) - minimum(lax)) * (length(lax) - 1) + 1
# end

"""
    find(x::T, lax::LinearAxis{T})

Return the index of `x` on `lax`. Uses an approximate equality check (`≈`);
throws an `AssertionError` if no grid point matches.
"""
function find(x::T, lax::LinearAxis{T}) where T
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
Base.getindex(dax::DiscreteAxis, rng::AbstractVector{Int}) = DiscreteAxis(points(dax)[rng])
Base.getindex(dax::DiscreteAxis, i) = getindex(points(dax), i)
Base.minimum(ax::DiscreteAxis) = ax[1]
Base.maximum(ax::DiscreteAxis) = ax[end]
Base.in(x, dax::DiscreteAxis) = in(x, points(dax))

"""
    find(x::T, dax::DiscreteAxis{T})

Return the index of `x` on `dax` using `searchsortedfirst`. Throws an error
if `x` is outside the range of `dax`.
"""
function find(x::T, dax::DiscreteAxis{T}) where T
    pts = points(dax)

    if pts[1] <= x <= pts[end]
        searchsortedfirst(pts, x)
    else
        error("$x not on axis.")
    end
end
