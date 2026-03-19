"""
    Approximator{T}

A wrapper around type `T` to allow for different dispatching depending on
whether a value should be matched exactly or whether the nearest neighbouring
value on a grid or axis should be matched.

Do not construct directly, instead, use [`approximately`](@ref).
"""
struct Approximator{T}
    x::T
end

(::Type{T})(x::Approximator{T}) where T = x.x
Base.convert(::Type{T}, x::Approximator{T}) where T = T(x)

"""
    approximately(x::T) where T

Create an approximation wrapper around `x`.

Typically used when subsetting axis values where one does not want to match
on the exact value of `x` but on the value next to `x` on an axis.

# Examples

```julia
ax = LinearAxis(range(0., 10., 11))
find(5.01, ax)   # error: only 5.0 is on axis
find(approximately(5.01), ax)   # returns index 6
```
"""
approximately(x::T) where T = Approximator{T}(x)

"Return the wrapped value of an [`Approximator`](@ref)."
value(x::Approximator) = x.x


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

Return the (integer) index of value `x` on axis `ax`.

If `x` is wrapped in [`approximately`](@ref), tolerate inexact matches: As long
as `x` is within the bounds of `ax`, return the index of the value closest to
`x` on the axis. Otherwise, throw an error.
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

function Base.in(x::Approximator{T}, axis::LinearAxis{T}) where T
    minimum(axis) <= T(x) <= maximum(axis)
end

function _bestguessindex(x::T, lax::LinearAxis{T}) where T
    (x - minimum(lax)) / (maximum(lax) - minimum(lax)) * (length(lax) - 1) + 1
end

function find(x::T, lax::LinearAxis{T}) where T
    i = round(Int, _bestguessindex(x, lax))
    isapprox(lax[i], x) ? i : error("$x is not on $lax")
end

function find(x::Approximator{T}, lax::LinearAxis{T}) where T
    bestguess = _bestguessindex(value(x), lax)
    1 <= bestguess <= length(lax) || error("$x is outside the bounds of $lax")
    round(Int, bestguess)
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
Base.in(x::Union{T, Approximator{T}}, dax::DiscreteAxis{T}) where T = in(x, points(dax))

function find(x::Union{T, Approximator{T}}, dax::DiscreteAxis{T}) where T
    searchsortedonly(points(dax), x)
end

"Return first occurence of item in collection, error if not available."
function searchsortedonly(collection, item)
    for (index, curitem) in enumerate(collection)
        (curitem == item) && return index
        (curitem > item) && break
    end
    error("$x not on $sa")
end
