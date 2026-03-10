"""
    SubGrid{T, D, DS, GS, DC} <: AbstractGrid{T, D}

A view of a `DS`-dimensional source grid that restricts and/or fixes some of
its axes, yielding a `D`-dimensional view (`D ≤ DS`).

- **Free axes** are kept in full (`:`) or restricted to a subrange / value
  subset — they contribute one dimension each to the view.
- **Fixed axes** are collapsed to a single value, reducing the dimensionality
  from `DS` to `D`. `getindex` on the view still returns a complete `T`
  (all `DS` components) with the fixed-axis values embedded at their original
  positions.

`DC` is the number of continuous dimensions in the view.

# Constructors

    SubGrid(g::Grid{T, DS}, selectors...)

Pass one selector per axis of `g`:

- `:` — keep the full axis unchanged (free)
- `(lo, hi)` — restrict a [`ContinuousAxis`](@ref) to elements in `[lo, hi]` (free)
- scalar — fix this axis to the given value (removes one dimension from the view)

    SubGrid(g::Grid{T, DS}; name = selector, ...)

Construct a view using named keyword selectors. Any axis not mentioned defaults
to `:` (unrestricted). The axis names are resolved via [`dimnames`](@ref) on
the point type `T`, so the keywords may be given in any order. Equivalent to
calling the positional constructor with the selectors placed in the order
defined by `dimnames(T)`.

# Example

```julia
g = Grid(LinearAxis(range(0.0, 1.0; length=11)),
         LinearAxis(range(0.0, 2.0; length=6)),
         DiscreteAxis([10, 20, 30]))

sg = SubGrid(g, :, :, 20)          # fix discrete axis → 2-D view
sg[3, 2]                            # returns (0.2, 0.4, 20)

sg2 = SubGrid(g, (0.0, 0.5), :, :) # restrict first axis, keep the rest

# with named axes (T must have dimnames defined, e.g. a NamedTuple):
g_named = Grid(x = LinearAxis(range(0.0, 1.0; length=11)),
               y = LinearAxis(range(0.0, 2.0; length=6)),
               z = DiscreteAxis([10, 20, 30]))
sg3 = SubGrid(g_named; z = 20)     # fix :z, keep :x and :y free

```
"""
struct SubGrid{T, D, DS, GS <: AbstractGrid{T, DS}, DC} <: AbstractGrid{T, D}
    source::GS
    slices::NTuple{DS, UnitRange{Int}}
    fixed::NTuple{DS, Bool}
    axismap::NTuple{DS, Int}

    function SubGrid(g::GS, slices::Vararg{AbstractUnitRange{Int}, DS}) where {T, DS, GS <: Grid{T, DS}}

        if !all(maximum(slices[ds]) <= size(g, ds) for ds in 1:DS)
            error("Trying to create out of bounds view")
        end

        fixed = ntuple(ds -> length(slices[ds]) == 1, Val(DS))
        fixedbefore = cumsum(fixed)
        axismap = ntuple(ds -> fixed[ds] ? 0 : ds - fixedbefore[ds], Val(DS))
        DC = ncontinuousdims(GS) - fixedbefore[ncontinuousdims(GS)]
        D = DS - fixedbefore[DS]
        new{T, D, DS, GS, DC}(g, slices, fixed, axismap)
    end
end

function SubGrid(g::AbstractGrid{T, DS}, selectors::Vararg{Any, DS}) where {T, DS}
    slices = _parseselectors(g, selectors...)
    SubGrid(g, slices...)
end

function SubGrid(g::AbstractGrid{T, DS}; kwargs...) where {T, DS}
    names = dimnames(T)

    if (names === ()) || !all(k in names for k in keys(kwargs))
        error("Trying to subset on dimension names that are not part of type $T")
    end

    selectors = ntuple(d -> get(kwargs, names[d], :), Val(DS))
    SubGrid(g, selectors...)
end

"""
    gfview(g::Grid, selectors...)
    gfview(g::Grid; name = selector, ...)

Return a [`SubGrid`](@ref) view of `g`.

Pass one selector per axis of `g`:

- `:` — keep the full axis unchanged (free)
- `(lo, hi)` — restrict a [`ContinuousAxis`](@ref) to elements in `[lo, hi]` (free)
- scalar — fix this axis to the given value, dropping it from the view

The keyword form accepts any subset of the named axes of `g` (requires
[`dimnames`](@ref) to be defined for the element type of `g`). Any axis not
mentioned defaults to `:` (unrestricted).

# Examples

```julia
g = Grid(x = LinearAxis(range(0.0, 1.0; length=11)),
         y = LinearAxis(range(0.0, 2.0; length=6)),
         z = DiscreteAxis([10, 20, 30]))

gfview(g, :, :, 20)          # fix :z → 2-D SubGrid
gfview(g, (0.0, 0.5), :, :)  # restrict :x
gfview(g; z = 20)             # fix :z by name
gfview(g; z = 20, x = (0.0, 0.5))  # mixed, any order
```
"""
gfview(g::Grid, args...) = SubGrid(g, args...)
gfview(g::Grid; kwargs...) = SubGrid(g; kwargs...)

"Return tuple of index slices as unit ranges"
function _parseselectors(g::AbstractGrid{T, D}, selectors::Vararg{Any, D}) where {T, D}
    ntuple(d -> _parseselector(gridaxes(g, d), selectors[d]), Val(D))
end

"Get the source grid underlying sub-grid `g`"
source(g::SubGrid) = g.source

"Get the index slices corresponding to source axis `ds`"
slices(g::SubGrid) = g.slices
slices(g::SubGrid, ds) = slices(g)[ds]

"True if source dimension `ds` is fixed in sub-grid `g`."
isfixed(g::SubGrid, ds) = g.fixed[ds]

"Get the source index at which source dimension `ds` is fixed in sub-grid `g`."
fixedat(g::SubGrid, ds) = only(slices(g, ds))

"Get the value to which source dimension `ds` is fixed in sub-grid `g`."
fixedto(g::SubGrid, ds) = gridaxes(source(g), ds)[fixedat(g, ds)]

"Get the dimension corresponding to source dimension `ds` in sub-grid `g`."
viewdim(g::SubGrid, ds) = g.axismap[ds]

"Return the source dimension corresponding to sub-grid dimension `d`."
function sourcedim(g::SubGrid{T, D, DS}, d) where {T, D, DS}
    for ds in 1:DS
        (viewdim(g, ds) == d) && (return ds)
    end
    error("SubGrid $g does not have free dimension $d")
end

"Return source grid index given an index on the sub-grid"
function sourceindex(g::SubGrid{T, D, DS}, I::CartesianIndex{D}) where {T, D, DS}
    CartesianIndex(
        ntuple(
            ds -> isfixed(g, ds) ? fixedat(g, ds) : (d = viewdim(g, ds); I[d] + minimum(slices(g, ds)) - 1), Val(DS)
        )
    )
end

"Return sub-grid index given an index on the source grid"
function viewindex(g::SubGrid{T, D, DS}, I::CartesianIndex{DS}; strict = false)::NTuple{D, Int} where {T, D, DS}
    !strict || all(I[ds] == fixedat(g, ds) for ds in 1:DS if isfixed(g, ds))
    CartesianIndex(
        Tuple(
            I[ds] - minimum(slices(g, ds)) + 1 for ds in 1:DS if !isfixed(g, ds)
        )
    )
end

# implement AbstractGrid interface

function gridaxes(g::SubGrid{T, D}) where {T, D}
    ntuple(
        d -> (ds = sourcedim(g, d); gridaxes(source(g), ds)[slices(g, ds)]),
        Val(D)
    )
end

function gridaxes(g::SubGrid, d::Int)
    ds = sourcedim(g, d)
    gridaxes(source(g), ds)[slices(g, sourcedim(g, d))]
end

function topoint(t::NTuple{D, Any}, g::SubGrid{T, D, DS}) where {T, D, DS}
    convert(T, ntuple(ds -> isfixed(g, ds) ? fixedto(g, ds) : t[viewdim(g, ds)], Val(DS)))
end

function totuple(x::T, g::SubGrid{T, D, DS}) where {T, D, DS}
    Tuple(x[ds] for ds in 1:DS if !isfixed(g, ds))
end

ncontinuousdims(::Type{SubGrid{T, D, DS, GS, DC}}) where {T, D, DS, GS, DC} = DC

# --- SubGriddedFunction -------------------------------------------

"""
    SubGriddedFunction{TY, D, GF, VV, GV} <: AbstractGriddedFunction{TY, D}

A view of a [`GriddedFunction`](@ref) (or another `AbstractGriddedFunction`)
that restricts and/or fixes some of its axes, analogous to [`SubGrid`](@ref)
for grids. The underlying value array is a non-copying `SubArray`.

Fixed axes (singleton slices) drop their dimension from both `values` and the
grid, so the view appears as a lower-dimensional function.

# Constructors

    SubGriddedFunction(source, selectors...)

Accepts the same selectors as [`SubGrid`](@ref). The view shares memory with
`source`; mutating `values(view)` also mutates `values(source)`.

    SubGriddedFunction(source; name = selector, ...)

Keyword-selector variant. Any axis not mentioned defaults to `:` (unrestricted).
Axis names are resolved via [`dimnames`](@ref) on the point type of the
underlying grid (see [`SubGrid`](@ref) keyword constructor).
"""
struct SubGriddedFunction{TY, D, GF, VV, GV} <: AbstractGriddedFunction{TY, D}
    source::GF
    values::VV
    subgrid::GV

    function SubGriddedFunction(
            source::GF, slices::Vararg{AbstractUnitRange{Int}}
        ) where {GF <: AbstractGriddedFunction}

        slices_fixed = ntuple(
            d -> length(slices[d]) == 1 ? only(slices[d]) : slices[d],
            length(slices)
        )
        vals = view(fvalues(source), slices_fixed...)
        sg = SubGrid(grid(source), slices...)
        new{eltype(GF), ndims(sg), GF, typeof(vals), typeof(sg)}(source, vals, sg)
    end
end

function SubGriddedFunction(source::AbstractGriddedFunction, selectors::Vararg)
    sg = SubGrid(grid(source), selectors...)
    index_slices = slices(sg)

    SubGriddedFunction(source, index_slices...)
end

function SubGriddedFunction(source::AbstractGriddedFunction; kwargs...)
    sg = SubGrid(grid(source); kwargs...)
    SubGriddedFunction(source, slices(sg)...)
end

grid(sgf::SubGriddedFunction) = sgf.subgrid
fvalues(sgf::SubGriddedFunction) = sgf.values
gridtype(::Type{SubGriddedFunction{TY, D, GF, VV, GV}}) where {TY, D, GF, VV, GV} = GV

"""
    gfview(gf::GriddedFunction, selectors...)
    gfview(gf::GriddedFunction; name = selector, ...)

Return a [`SubGriddedFunction`](@ref) view of `gf`. The view shares memory
with `gf`; mutating its values also mutates `gf`.

Pass one selector per axis of `gf`:

- `:` — keep the full axis unchanged (free)
- `(lo, hi)` — restrict a [`ContinuousAxis`](@ref) to elements in `[lo, hi]` (free)
- scalar — fix this axis to the given value, dropping it from the view

The keyword form accepts any subset of the named axes (requires
[`dimnames`](@ref) to be defined for the element type of the underlying grid).
Any axis not mentioned defaults to `:` (unrestricted).

# Examples

```julia
g  = Grid(x = LinearAxis(range(0.0, 1.0; length=11)),
          y = LinearAxis(range(0.0, 2.0; length=6)),
          z = DiscreteAxis([10, 20, 30]))
gf = GriddedFunction(Float64, g, (x, y, z) -> x + y + z)

gfview(gf, :, :, 20)          # fix :z → 2-D SubGriddedFunction
gfview(gf, (0.0, 0.5), :, :)  # restrict :x
gfview(gf; z = 20)             # fix :z by name
gfview(gf; z = 20, x = (0.0, 0.5))  # mixed, any order
```
"""
gfview(gf::GriddedFunction, args...) = SubGriddedFunction(gf, args...)
gfview(gf::GriddedFunction; kwargs...) = SubGriddedFunction(gf; kwargs...)

"""
    continuousview(source::AbstractGriddedFunction, I_disc::CartesianIndex)

Return a [`SubGriddedFunction`](@ref) of `source` that fixes all discrete
axes at the discrete index combination `I_disc` and keeps all continuous axes
free. Used internally by [`GFInterpolation`](@ref) to slice out the
continuous subgrid for each discrete point.
"""
function continuousview(source::GF, I_disc::CartesianIndex) where {GF <: AbstractGriddedFunction}
    # tuple of colons for continuousdims and I_disc individual values
    indices = tuple(
        ntuple(dc -> (1:size(source, dc)), Val(ncontinuousdims(GF)))...,
        ntuple(dd -> (I_disc[dd]:I_disc[dd]), Val(ndiscretedims(GF)))...
    )

    SubGriddedFunction(source, indices...)
end
