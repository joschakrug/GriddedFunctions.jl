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

"""
    exactly(x::Approximator{T}) where T

Return the exact value wrapped by [`Approximator`](@ref) `x`.

This reverses [`approximately`](@ref).

# Examples

```julia
exactly(approximately(5.01)) === 5.01   # true
```
"""
exactly(x::Approximator) = x.x
exactly(x::T) where T = x

Base.convert(::Type{T}, x::Approximator{T}) where T = exactly(x)
Base.show(io::IO, x::Approximator) = print(io, "~ $(exactly(x))")

"""
    SelectionRange{T}

A wrapper around a `(min, max)` pair of type `T` for selecting a contiguous
range of axis values when subsetting.

Do not construct directly, instead, use [`inrange`](@ref).
"""
struct SelectionRange{T}
    min::T
    max::T
end

"Return the minimum value of a [`SelectionRange`](@ref)."
rangemin(r::SelectionRange) = r.min

"Return the maximum value of a [`SelectionRange`](@ref)."
rangemax(r::SelectionRange) = r.max

Base.show(io::IO, r::SelectionRange) = print(io, "$(rangemin(r)) .. $(rangemax(r))")

"""
    inrange(min::T, max::T) where T

Create a selection range from `min` to `max` (inclusive).

Typically used when subsetting axis values to select all axis points between
`min` and `max`.

# Examples

```julia
ax = LinearAxis(range(0., 10., 11))
subset(ax, inrange(2., 6.))   # SubAxis covering values 2 through 6
```
"""
inrange(min::T, max::T) where T = SelectionRange{T}(min, max)

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

- [`points`](@ref)

It can optionally overwrite the following default methods:

- `Base.length` - defaults to length of the axis' point vector
- `Base.getindex` (for integers as well as for unit ranges) -
  passed through to the axis' vector of points by default
- [`iscontinuous`](@ref) - defaults to `Discrete()`
"""
abstract type Axis{T} <: AbstractVector{T} end

Base.show(io::IO, ::Type{A}) where {T, A <: Axis{T}} = print(io, "$(nameof(A)){$T}")
Base.show(io::IO, ::MIME"text/plain", ::Type{A}) where {A <: Axis} =
    get(io, :compact, false) ? show(io, A) : print(io, "$A")

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

Base.show(io::IO, ax::A) where {T, A <: Axis{T}} =
    print(io, "$(nameof(A))($(points(ax)))")
Base.showarg(io::IO, ::A, ::Bool) where {T, A <: Axis{T}} =
    print(io, "$(nameof(A)){$T}")

Base.size(ax::Axis) = (length(ax),)
Base.minimum(ax::Axis) = ax[1]
Base.maximum(ax::Axis) = ax[end]
Base.length(ax::Axis) = length(points(ax))
Base.getindex(ax::Axis, i) = getindex(points(ax), i)

Base.in(x::Union{T, Approximator{T}}, ax::A) where {T, A <: Axis{T}} =
    _inaxis(iscontinuous(A), x, ax)

_inaxis(::Discrete, x::T, ax::Axis{T}) where T = in(x, points(ax))
_inaxis(::Discrete, x::Approximator{T}, ax::Axis{T}) where T =
    _inaxis(Discrete(), exactly(x), ax)
_inaxis(::Continuous, x::T, ax::Axis{T}) where T = in(x, points(ax))
_inaxis(::Continuous, x::Approximator{T}, ax::Axis{T}) where T =
    minimum(ax) <= exactly(x) <= maximum(ax)

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
    searchsortedonly(points(ax), exactly(x))

function find(::Continuous, x::T, ax::Axis{T}) where T
    i = round(Int, bestguessindex(x, ax))
    isapprox(ax[i], x) ? i : error("$x is not on $ax")
end

function find(::Continuous, x::Approximator{T}, ax::Axis{T}) where T
    bestguess = bestguessindex(exactly(x), ax)
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

iscontinuous(::Type{<:LinearAxis}) = Continuous()

Base.range(lax::LinearAxis) = lax.range
points(lax::LinearAxis) = collect(range(lax))

Base.show(io::IO, ax::LinearAxis{T}) where T =
    print(io, "LinearAxis($(range(ax)))")
function Base.show(io::IO, ::MIME"text/plain", ax::LinearAxis{T}) where T
    compact = get(io, :compact, false)
    compact ? show(io, ax) : begin
        Base.showarg(io, ax, true)
        print(io, "($(range(ax)))")
    end
end

for f in (:length, :getindex, :minimum, :maximum, :iterate)
    @eval Base.$f(lax::LinearAxis, args...) = Base.$f(range(lax), args...)
end

function Base.getindex(ax::LAX, rng::UnitRange{Int}) where {LAX <: LinearAxis}
    LAX(range(ax[minimum(rng)], ax[maximum(rng)], length = length(rng)))
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
Base.getindex(dax::DiscreteAxis, rng::AbstractVector{Int}) = DiscreteAxis(points(dax)[rng])

"Return first occurence of item in collection, error if not available."
function searchsortedonly(collection, item)
    for (index, curitem) in enumerate(collection)
        (curitem == item) && return index
        (curitem > item) && break
    end
    error("$x not on $sa")
end
