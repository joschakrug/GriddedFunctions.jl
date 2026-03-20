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
    Continuous

Defines the `Continuous` trait of [`Axis`](@ref).

If the `iscontinuous` method returns `Continuous()` for an `Axis` type,
this axis is treated as continuous. Any axis with the continuous trait
needs to implement the following methods:

- [`bestguessindex`](@ref)
"""
struct Continuous end

"See [`Continuous`](@ref)"
struct Discrete end

"""
    Axis{T} <: AbstractVector{T}

A grid axis with axis element type `T`.

Each subtype of the abstract type `Axis{T}` needs to define at least the
following methods:

- `Base.length`
- `Base.getindex` (for integers as well as for unit ranges)
- [`points`](@ref)
- [`iscontinuous`](@ref)
"""
abstract type Axis{T} <: AbstractVector{T} end

Base.eltype(::Type{Axis{T}}) where T = T
Base.size(ax::Axis) = (length(ax),)
Base.minimum(ax::Axis) = ax[1]
Base.maximum(ax::Axis) = ax[end]

"""
    points(ax::Axis)

Return the vector of points on axis `ax`.
"""
function points end

"""
    iscontinuous(::Type{A}) where A <: Axis

`Continuous()` if `A` is continuous, `Discrete()` if not.

This function identifies the `Continuous` trait.
"""
function iscontinuous end
iscontinuous(::Type{<:Axis}) = Discrete()

"""
    bestguessindex(x::T, ax::Axis{T})

Compute the best guess for the index of `x` on `ax` based on the structure of
`ax`. Any axis type with the [`Continuous`](@ref) trait needs to implement this
method.

For a `LinearAxis`, the best guess can be computed as
`(x - minimum(ax))/(maximum(ax)-minimum(ax)) * length(ax) + 1`. For other types
of continuous axes, other formulas may apply.
"""
function bestguessindex end

"""
    find(x::T, ax::Axis{T}) where T

Return the (integer) index of value `x` on axis `ax`.

If `x` is wrapped in [`approximately`](@ref), tolerate inexact matches on
continuous axes: As long
as `x` is within the bounds of `ax`, return the index of the value closest to
`x` on the axis. Otherwise, throw an error.
"""
find(x::Union{T, Approximator{T}}, ax::A) where {T, A <: Axis{T}} =
    find(iscontinuous(A), x, ax)

find(::Discrete, x::Union{T, Approximator{T}}, ax::Axis{T}) where T =
    searchsortedonly(points(ax), x)

function find(::Continuous, x::T, ax::Axis{T}) where T
    i = round(Int, bestguessindex(x, ax))
    isapprox(ax[i], x) ? i : error("$x is not on $ax")
end

function find(::Continuous, x::Approximator{T}, ax::Axis{T}) where T
    bestguess = bestguessindex(value(x), ax)
    1 <= bestguess <= length(ax) || error("$x is outside the bounds of $ax")
    round(Int, bestguess)
end

"""
    onaxisapprox(x::T, y::T, ax::Axis{T}; steptol = 0.5) where T

True if the difference between `x` and `y` on `ax` is smaller than `steptol`
times the local axis step size.

This is valuable when, for example, a `GriddedFunction` has a discontinuity
at a given point (e.g. 0) but that point is not exactly on the axis (e.g.
because the underlying range only includes -0.01 and 0.01, not 0. itself).

`steptol` defaults to 0.5, which ensures that only the value of `x` closest to
`y` on `ax` returns true.
"""
onaxisapprox(x::T, y::T, ax::A; steptol = 0.5) where {T, A <: Axis{T}} =
    onaxisapprox(iscontinuous(A), x, y, ax, steptol)

onaxisapprox(::Continuous, x, y, ax, steptol) =
    abs(bestguessindex(x, ax) - bestguessindex(y, ax)) < steptol

onaxisapprox(::Discrete, x, y, ax, steptol) = isapprox(x, y)

"""
    LinearAxis{T, S <: StepRangeLen{T}} <: Axis{T}

A linearly scaled continuous grid axis representing values of type `T`.

`S` is just an auxiliary parameter to ensure type stability.

# Constructor

    LinearAxis(r::StepRangeLen{T})

Construct a linear axis from a step range `r`. Use `range(start, stop; length=n)`
or `range(start, stop; step=s)` to create suitable ranges.
"""
struct LinearAxis{T, S <: StepRangeLen{T}} <: Axis{T}
    range::S
end

Base.show(io::IO, ::Type{<:LinearAxis{T}}) where T = print(io, "LinearAxis{$T}")

iscontinuous(::Type{<:LinearAxis}) = Continuous()

Base.range(lax::LinearAxis) = lax.range
points(lax::LinearAxis) = collect(range(lax))

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

# implement the Continuous trait interface

function bestguessindex(x::T, lax::LinearAxis{T}) where T
    (x - minimum(lax)) / (maximum(lax) - minimum(lax)) * (length(lax) - 1) + 1
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

points(dax::DiscreteAxis) = dax.points

Base.length(dax::DiscreteAxis) = length(points(dax))
Base.getindex(dax::DiscreteAxis, rng::AbstractVector{Int}) = DiscreteAxis(points(dax)[rng])
Base.getindex(dax::DiscreteAxis, i) = getindex(points(dax), i)
Base.minimum(ax::DiscreteAxis) = ax[1]
Base.maximum(ax::DiscreteAxis) = ax[end]
Base.in(x::Union{T, Approximator{T}}, dax::DiscreteAxis{T}) where T = in(x, points(dax))

"Return first occurence of item in collection, error if not available."
function searchsortedonly(collection, item)
    for (index, curitem) in enumerate(collection)
        (curitem == item) && return index
        (curitem > item) && break
    end
    error("$x not on $sa")
end
