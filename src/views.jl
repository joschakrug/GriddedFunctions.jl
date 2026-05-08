"""
    Selector{T}

Type alias for the set of valid axis selectors:

- `T` — a scalar value of axis element type `T`; fixes the axis to that value,
  collapsing it from the view.
- `Approximator{T}` - an approximation wrapper around a scalar value of axis
  element type `T`; fixes the axis to that value, collapsing it from the view
- `SelectionRange{T}` — created via [`inrange`](@ref); restricts the axis to the
  index range spanning the given `(min, max)` values (free dimension).
- `Colon` (`:`) — keep the full axis unchanged (free dimension).
"""
const Selector{T} = Union{T, SelectionRange{T}, Colon, Approximator{T}}

"""
    SubAxis{T, I, A} <: Axis{T}

A restricted or fixed view of a source axis of type `A <: Axis{T}`.

- When `I <: AbstractUnitRange{Int}`: a free sub-axis covering the given index range
  of the source axis. Even a singleton range (`i:i`) retains the dimension.
- When `I == Int`: a fixed sub-axis collapsed to a single source index.
  The dimension it represents is dropped from any [`SubGrid`](@ref) that uses it.

Construct via [`subset`](@ref) on an axis rather than directly.
"""
struct SubAxis{T, I <: Union{Int, AbstractUnitRange{Int}}, A <: Axis{T}} <: Axis{T}
    source::A
    indices::I
end

function SubAxis(source::A, ::Colon) where {T, A <: Axis{T}}
    I = typeof(1:length(source))
    SubAxis{T, I, A}(source, 1:length(source))
end

function SubAxis(source::A, i::Int) where {T, A <: Axis{T}}
    (1 <= i <= length(source)) ||
        error("$source does not have index $i")
    SubAxis{T, Int, A}(source, i)
end

function SubAxis(source::A, r::AbstractUnitRange{Int}) where {T, A <: Axis{T}}
    (first(r) >= 1 && last(r) <= length(source)) ||
        error("Index range $r out of bounds for axis of length $(length(source))")
    SubAxis{T, typeof(r), A}(source, r)
end

"Return the source axis underlying `sa`."
source(sa::SubAxis) = sa.source

"Return the index or index range into the source axis that `sa` covers."
indices(sa::SubAxis) = sa.indices

"Return the source-axis index corresponding to sub-axis index `i`."
sourceindex(sa::SubAxis, i::Int) = indices(sa)[i]

"Return the sub-axis index corresponding to source-axis index `i`."
subindex(sa::SubAxis, i::Int) = searchsortedfirst(indices(sa), i)

"""
    isfixed(::Type{SA}) where SA <: SubAxis
    isfixed(ax::SA)

True if subaxis type `SA` implies that the axis is fixed to a scalar value.
"""
isfixed(::Type{<:SubAxis{<:Any, Int}}) = true
isfixed(::Type{<:SubAxis}) = false
isfixed(sa::SubAxis) = isfixed(typeof(sa))

# implement the Axis interface

Base.length(sa::SubAxis) = length(indices(sa))
Base.getindex(sa::SubAxis, i::Int) = source(sa)[indices(sa)[i]]
points(sa::SubAxis) = points(source(sa))[indices(sa)]

# only works if the source axis can be represented as a range as well
Base.range(sa::SubAxis{<:Any, <:AbstractUnitRange}) =
    range(source(sa))[indices(sa)]

# implement the Continuous trait interface (where applicable)

iscontinuous(::Type{<:SubAxis{<:Any, <:AbstractUnitRange, A}}) where A =
    iscontinuous(A)
bestguessindex(x::T, sa::SubAxis{T}) where T =
    bestguessindex(x, source(sa)) - first(indices(sa)) + 1

"""
    SubGrid{T, D, DS, GS, DC} <: AbstractGrid{T, D}

A view of a `DS`-dimensional source grid that restricts and/or fixes some of
its axes, yielding a `D`-dimensional view (`D ≤ DS`).

- **Free axes** are kept in full (`:`) or restricted to a subrange — they
  contribute one dimension each to the view, even if the subrange has length 1.
- **Fixed axes** are collapsed to a single value, reducing the dimensionality
  from `DS` to `D`. `getindex` on the view still returns a complete `T`
  (all `DS` components) with the fixed-axis values embedded at their original
  positions.

`DC` is the number of continuous dimensions in the view.

# Constructors

    SubGrid(g::AbstractGrid{T, DS}, subaxes::NTuple{DS, SubAxis})

Low-level constructor. Build from a tuple of [`SubAxis`](@ref) objects obtained
via [`subset`](@ref) on each axis.

    SubGrid(g::AbstractGrid{T, DS}, inds...)

Convenience constructor. Create a subgrid of `g`, selecting indices `inds`.
(Roughly equivalent to `Base.view`.)

Use [`subset`](@ref) on the grid for selection based on axis values and named
keyword selection.

# Examples

```julia
g = Grid(LinearAxis(range(0.0, 1.0; length=11)),
         LinearAxis(range(0.0, 2.0; length=6)),
         DiscreteAxis([10, 20, 30]))

sg  = SubGrid(g, :, :, 20)    # fix discrete axis → 2-D view
sg[3, 2]                       # returns (0.2, 0.4, 20)

sg2 = SubGrid(g, 1:6, :, :)   # restrict first axis to indices 1:6

sg3 = SubGrid(g, 3:3, :, :)   # size-1 first axis — NOT collapsed (still 3-D)
```
"""
struct SubGrid{T, D, DS, GS <: AbstractGrid{T, DS}, SA <: NTuple{DS, SubAxis}} <: AbstractGrid{T, D}
    source::GS
    axes::SA

    function SubGrid(src::GS, subaxes::SA) where {T, DS, GS <: AbstractGrid{T, DS}, SA <: NTuple{DS, SubAxis}}
        for ds in 1:DS
            source(subaxes[ds]) === gridaxes(src, ds) ||
                error("$(subaxes[ds]) is not a subaxis of source grid axis $ds")
        end
        D = DS - count(ds -> isfixed(fieldtype(SA, ds)), 1:DS)
        new{T, D, DS, GS, SA}(src, subaxes)
    end
end

function SubGrid(src::GS, inds::Vararg{Any, DS}) where {T, DS, GS <: AbstractGrid{T, DS}}
    axs = ntuple(ds -> SubAxis(gridaxes(src, ds), inds[ds]), Val(DS))
    SubGrid(src, axs)
end

"Return the source grid underlying sub-grid `g`."
source(g::SubGrid) = g.source

"""
    subaxes(g::SubGrid, ds = nothing)


Return the tuple of [`SubAxis`](@ref) objects of subgrid `g`.

If `ds` is specified, return only the `SubAxis` corresponding to dimension `ds`
of the source grid.
"""
subaxes(g::SubGrid) = g.axes
subaxes(g::SubGrid, ds) = subaxes(g)[ds]

"""
    isfixed(::Type{SG}, ds) where {SG <: SubGrid}
    isfixed(sg::SG, ds)

Return `true` at compile time if source dimension `ds` of `SubGrid` type `SG` is fixed.
"""
isfixed(::Type{SubGrid{T, D, DS, GS, SA}}, ds) where {T, D, DS, GS, SA} =
    isfixed(fieldtype(SA, ds))
isfixed(g::SubGrid, ds) = isfixed(typeof(g), ds)

"Return the source index at which source dimension `ds` is fixed in sub-grid `g`."
fixedat(g::SubGrid, ds) = only(indices(subaxes(g, ds)))

"Return the axis value to which source dimension `ds` is fixed in sub-grid `g`."
function fixedto(g::SubGrid, ds)
    sa = subaxes(g, ds)
    only(source(sa)[indices(sa)])
end

"""
    subdim(::Type{SG}, ds) where SG <: SubGrid
    subdim(g::SG, ds)

Return the view dimension corresponding to source dimension `ds` in subgrid type
`SG`.
"""
function subdim(::Type{SG}, ds) where SG <: SubGrid
    if !isfixed(SG, ds)
        return ds - sum(isfixed(SG, dsprev) for dsprev in 1:ds)
    else
        error("Dimension $ds is fixed for subgrid type $SG")
    end
end
subdim(g::SubGrid, ds) = subdim(typeof(g), ds)

"""
    sourcedim(::Type{SG}, d) where SG <: SubGrid
    sourcedim(g::SG, d)

Return the source dimension index corresponding to view dimension `d` of subgrid
type `SG`.
"""
function sourcedim(::Type{SG}, d) where {T, D, DS, SG <: SubGrid{T, D, DS}}
    for ds in 1:DS
        !isfixed(SG, ds) && (subdim(SG, ds) == d) && (return ds)
    end
    error("SubGrid type $SG does not have free dimension $d")
end
sourcedim(g::SubGrid, d) = sourcedim(typeof(g), d)

"Return the source-grid `CartesianIndex` corresponding to sub-grid index `I`."
function sourceindex(g::SubGrid{T, D, DS}, I::CartesianIndex{D}) where {T, D, DS}
    SG = typeof(g)
    CartesianIndex(
        ntuple(ds -> begin
            if isfixed(SG, ds)
                fixedat(g, ds)
            else
                sourceindex(subaxes(g, ds), I[subdim(SG, ds)])
            end
        end, Val(DS))
    )
end

"""
Return the subgrid `CartesianIndex` corresponding to source grid index `I`.

If `strict = false`, ignore axes that are fixed in the subgrid
"""
function subindex(
        g::SG,
        I::CartesianIndex{DS}; strict = false
    )::CartesianIndex{D} where {T, D, DS, SG <: SubGrid{T, D, DS}}

    if strict && !all(I[ds] == fixedat(g, ds) for ds in 1:DS if isfixed(SG, ds))
        error("Source index $I is not in subgrid")
    end
    CartesianIndex(
        Tuple(
            subindex(subaxes(g, ds), I[ds]) for ds in 1:DS if !isfixed(SG, ds)
        )
    )
end

dimnames(::Type{SG}) where {T, D, SG <: SubGrid{T, D}} =
    ntuple(d -> dimnames(T, sourcedim(SG, d)), Val(D))
dimnum(::Type{SG}, name::Symbol) where {T, SG <: SubGrid{T}} = subdim(SG, dimnum(T, name))

# implement AbstractGrid interface

function gridaxes(g::SG) where {T, D, SG <: SubGrid{T, D}}
    ntuple(d -> (ds = sourcedim(SG, d); subaxes(g, ds)), Val(D))
end

gridaxes(g::SubGrid, d::Int) = subaxes(g, sourcedim(g, d))

function topoint(t::NTuple{D, Any}, g::SG) where {T, D, DS, SG <: SubGrid{T, D, DS}}
    convert(
        T,
        ntuple(
            ds -> isfixed(SG, ds) ? fixedto(g, ds) : t[subdim(SG, ds)],
            Val(DS)
        )
    )
end

function totuple(x::T, ::SG) where {T, D, DS, SG <: SubGrid{T, D, DS}}
    Tuple(x[ds] for ds in 1:DS if !isfixed(SG, ds))
end

function ncontinuousdims(::Type{SubGrid{T, D, DS, GS, SA}}) where {T, D, DS, GS, SA}
    ncontinuousdims(GS) > 0 ?
        ncontinuousdims(GS) - sum(isfixed(SubGrid{T, D, DS, GS, SA}, ds) for ds in 1:ncontinuousdims(GS)) :
        0
end

# --- SubGriddedFunction -------------------------------------------

"""
    SubGriddedFunction{TY, D, GF, VV, GV} <: AbstractGriddedFunction{TY, D}

A view of a [`GriddedFunction`](@ref) (or another `AbstractGriddedFunction`)
that restricts and/or fixes some of its axes, analogous to [`SubGrid`](@ref)
for grids. The underlying value array is a non-copying `SubArray`.

Fixed axes (scalar selectors) drop their dimension from both the value array and
the sub-grid, so the view appears as a lower-dimensional function. Axes restricted
to a singleton range (e.g. `3:3`) are retained as size-1 dimensions.

# Constructor

    SubGrid(gf::AbstractGriddedFunction{T, DS}, inds...)

Convenience constructor. Create a subfunction of `gf`, selecting indices `inds`.
(Roughly equivalent to `Base.view`.)

Use [`subset`](@ref) on `gf` for selection based on axis values and named
keyword selection.

# Examples

```julia
g  = Grid(LinearAxis(range(0.0, 1.0; length=11)),
          LinearAxis(range(0.0, 2.0; length=6)),
          DiscreteAxis([10, 20, 30]))
gf = GriddedFunction(Float64, g, (x, y, z) -> x + y + z)

sgf  = SubGriddedFunction(gf, :, :, 20)    # fix discrete axis → 2-D view
sgf2 = SubGriddedFunction(gf, 1:6, :, :)   # restrict first axis to indices 1:6

# use subset to select on values
sgf3 = subset(gf, :, :, 20)

# works also with keyword selection
sgf4 = subset(gf, z = 20)
```
"""
struct SubGriddedFunction{TY, D, GF, VV, GV} <: AbstractGriddedFunction{TY, D}
    source::GF
    values::VV
    subgrid::GV

    function SubGriddedFunction(src::GF, inds::Vararg{Any, DS}) where {
            T, DS, GF <: AbstractGriddedFunction{T, DS}
        }
        sg = SubGrid(grid(src), inds...)
        SG = typeof(sg)
        vals = view(fvalues(src), ntuple(ds -> indices(subaxes(sg, ds)), Val(DS))...)
        new{eltype(GF), ndims(SG), GF, typeof(vals), SG}(src, vals, sg)
    end
end

grid(sgf::SubGriddedFunction) = sgf.subgrid
fvalues(sgf::SubGriddedFunction) = sgf.values
gridtype(::Type{SubGriddedFunction{TY, D, GF, VV, GV}}) where {TY, D, GF, VV, GV} = GV

"""
    continuousview(source::AbstractGriddedFunction, I_disc::CartesianIndex)

Return a [`SubGriddedFunction`](@ref) of `source` that fixes all discrete
axes at the discrete index combination `I_disc` and keeps all continuous axes
free. Used internally by [`GFInterpolation`](@ref) to slice out the
continuous subgrid for each discrete point.
"""
function continuousview(src::AbstractGriddedFunction, I_disc::CartesianIndex)
    DC = ncontinuousdims(src)
    DD = ndiscretedims(src)
    DS = DC + DD
    inds = ntuple(ds -> ds <= DC ? (:) : I_disc[ds - DC], Val(DS))
    SubGriddedFunction(src, inds...)
end

"""
    subset(ax::Axis{T}, selector)

Return a [`SubAxis`](@ref) of `ax` for the given [`Selector`](@ref):

- `:` — full axis (free)
- `inrange(from, to)::SelectionRange{T}` — restrict to the index range spanning
  values `from` through `to` (free)
- `x::T` — fix the axis to the single source index of value `x` (collapsed)
"""
subset(ax::Axis, ::Colon) = SubAxis(ax, :)
subset(ax::Axis{T}, x::Union{T, Approximator{T}}) where T = SubAxis(ax, find(x, ax))
function subset(ax::Axis{T}, r::SelectionRange{T}) where T
    from, to = rangemin(r), rangemax(r)
    ifrom, ito = 0, length(ax)
    for (i, x) in enumerate(ax)
        (x <= from) && (ifrom = i)
        (x <= to) && (ito = i)
        (x >= to) && return SubAxis(ax, ifrom:ito)
    end
    error("$r is not entirely on axis $ax")
end

"""
    subset(g::AbstractGrid, selectors...)
    subset(g::AbstractGrid; name = selector, ...)

Create a [`SubGrid`](@ref) of `g` from the given [`Selector`](@ref)s.

Pass one selector per axis of `g`:

- `:` — keep the full axis unchanged (free)
- `inrange(from, to)` — restrict to the index range spanning values `from` through `to` (free)
- scalar — fix the axis to that value, collapsing it from the view

The keyword form accepts a named subset of axes (requires [`dimnames`](@ref) on the
element type of `g`). Any axis not mentioned defaults to `:` (unrestricted).

# Examples

```julia
g = Grid(x = LinearAxis(range(0.0, 1.0; length=11)),
         y = LinearAxis(range(0.0, 2.0; length=6)),
         z = DiscreteAxis([10, 20, 30]))

subset(g, :, :, 20)                    # fix :z → 2-D SubGrid
subset(g, inrange(0.2, 0.8), :, :)     # restrict :x by value range
subset(g, x = approximately(0.51))     # restrict :x to value closest to 0.51
subset(g; z = 20)                      # fix :z by name
subset(g; z = 20, x = inrange(0.2, 0.8))  # mixed, any order
```
"""
function subset(g::AbstractGrid{T, DS}, selectors::Vararg{Selector, DS}) where {T, DS}
    subaxes = ntuple(ds -> subset(gridaxes(g, ds), selectors[ds]), Val(DS))
    SubGrid(g, subaxes)
end

function subset(g::G; kwargs...) where {T, DS, G <: AbstractGrid{T, DS}}
    names = dimnames(G)

    if (names === ()) || !all(k in names for k in keys(kwargs))
        error("Trying to subset on dimension names that are not part of type $T")
    end

    selectors = ntuple(d -> get(kwargs, names[d], :), Val(DS))
    subset(g, selectors...)
end

"""
    subset(gf::AbstractGriddedFunction, selectors...)
    subset(gf::AbstractGriddedFunction; name = selector, ...)

Create a [`SubGriddedFunction`](@ref) of `gf` from the given [`Selector`](@ref)s.

Accepts the same selectors as `subset` on a grid. See [`subset(g::AbstractGrid, ...)`](@ref)
for details on selector syntax. The returned view shares memory with `gf`.
"""
function subset(gf::GF, args...; kwargs...) where {TY, DS, GF <: AbstractGriddedFunction{TY, DS}}
    sg = subset(grid(gf), args...; kwargs...)
    inds = ntuple(ds -> indices(subaxes(sg, ds)), Val(DS))
    SubGriddedFunction(gf, inds...)
end
