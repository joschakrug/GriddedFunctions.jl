"""
    GridView{T, D, DS, GS, DC} <: AbstractGrid{T, D}

A view of a `DS`-dimensional source grid that restricts and/or fixes some of
its axes, yielding a `D`-dimensional view (`D ≤ DS`).

- **Free axes** are kept in full (`:`) or restricted to a subrange / value
  subset — they contribute one dimension each to the view.
- **Fixed axes** are collapsed to a single value, reducing the dimensionality
  from `DS` to `D`. `getindex` on the view still returns a complete `T`
  (all `DS` components) with the fixed-axis values embedded at their original
  positions.

`DC` is the number of continuous dimensions in the view.

# Constructor

    GridView(g::Grid{T, DS}, selectors...)

Pass one selector per axis of `g`:

- `:` — keep the full axis unchanged (free)
- `(lo, hi)` — restrict a [`ContinuousAxis`](@ref) to elements in `[lo, hi]` (free)
- scalar — fix this axis to the given value (removes one dimension from the view)

# Example

```julia
g = Grid(LinearAxis(range(0.0, 1.0; length=11)),
         LinearAxis(range(0.0, 2.0; length=6)),
         DiscreteAxis([10, 20, 30]))

gv = GridView(g, :, :, 20)          # fix discrete axis → 2-D view
gv[3, 2]                             # returns (0.2, 0.4, 20)

gv2 = GridView(g, (0.0, 0.5), :, :) # restrict first axis, keep the rest
```
"""
struct GridView{T, D, DS, GS <: AbstractGrid{T, DS}, DC} <: AbstractGrid{T, D}
    source::GS
    slices::NTuple{DS, UnitRange{Int}}
    fixed::NTuple{DS, Bool}
    axismap::NTuple{DS, Int}

    function GridView(g::GS, slices::Vararg{AbstractUnitRange{Int}, DS}) where {T, DS, GS <: Grid{T, DS}}

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

function GridView(g::AbstractGrid{T, DS}, selectors::Vararg{Any, DS}) where {T, DS}
    slices = _parseselectors(g, selectors...)
    GridView(g, slices...)
end

"Return tuple of index slices as unit ranges"
function _parseselectors(g::AbstractGrid{T, D}, selectors::Vararg{Any, D}) where {T, D}
    ntuple(d -> _parseselector(gridaxes(g, d), selectors[d]), Val(D))
end

"Get the source grid underlying grid view `g`"
source(g::GridView) = g.source

"Get the index slices corresponding to source axis `ds`"
slices(g::GridView) = g.slices
slices(g::GridView, ds) = slices(g)[ds]

"True if source dimension `ds` is fixed in grid view `g`."
isfixed(g::GridView, ds) = g.fixed[ds]

"Get the source index at which source dimension `ds` is fixed in grid view `g`."
fixedat(g::GridView, ds) = only(slices(g, ds))

"Get the value to which source dimension `ds` is fixed in grid view `g`."
fixedto(g::GridView, ds) = gridaxes(source(g), ds)[fixedat(g, ds)]

"Get the dimension corresponding to source dimension `ds` in grid view `g`."
viewdim(g::GridView, ds) = g.axismap[ds]

"Return the dimension of grid view dimension `d` in the source grid."
function sourcedim(g::GridView{T, D, DS}, d) where {T, D, DS}
    for ds in 1:DS
        (viewdim(g, ds) == d) && (return ds)
    end
    error("Grid view $g does not have free dimension $d")
end

"Return source grid index given an index on the grid view"
function sourceindex(g::GridView{T, D, DS}, I::CartesianIndex{D}) where {T, D, DS}
    CartesianIndex(
        ntuple(
            ds -> isfixed(g, ds) ? fixedat(g, ds) : (d = viewdim(g, ds); I[d] + minimum(slices(g, ds)) - 1), Val(DS)
        )
    )
end

"Return grid view index given an index on the source grid"
function viewindex(g::GridView{T, D, DS}, I::CartesianIndex{DS}; strict = false)::NTuple{D, Int} where {T, D, DS}
    !strict || all(I[ds] == fixedat(g, ds) for ds in 1:DS if isfixed(g, ds))
    CartesianIndex(
        Tuple(
            I[ds] - minimum(slices(g, ds)) + 1 for ds in 1:DS if !isfixed(g, ds)
        )
    )
end

# implement AbstractGrid interface

function gridaxes(g::GridView{T, D}) where {T, D}
    ntuple(
        d -> (ds = sourcedim(g, d); gridaxes(source(g), ds)[slices(g, ds)]),
        Val(D)
    )
end

function gridaxes(g::GridView, d)
    ds = sourcedim(g, d)
    gridaxes(source(g), ds)[slices(g, sourcedim(g, d))]
end

function topoint(t::NTuple{D, Any}, g::GridView{T, D, DS}) where {T, D, DS}
    convert(T, ntuple(ds -> isfixed(g, ds) ? fixedto(g, ds) : t[viewdim(g, ds)], Val(DS)))
end

function totuple(x::T, g::GridView{T, D, DS}) where {T, D, DS}
    Tuple(x[ds] for ds in 1:DS if !isfixed(g, ds))
end

ncontinuousdims(::Type{GridView{T, D, DS, GS, DC}}) where {T, D, DS, GS, DC} = DC

# --- GriddedFunctionView -------------------------------------------

"""
    GriddedFunctionView{TY, D, GF, VV, GV} <: AbstractGriddedFunction{TY, D}

A view of a [`GriddedFunction`](@ref) (or another `AbstractGriddedFunction`)
that restricts and/or fixes some of its axes, analogous to [`GridView`](@ref)
for grids. The underlying value array is a non-copying `SubArray`.

Fixed axes (singleton slices) drop their dimension from both `values` and the
grid, so the view appears as a lower-dimensional function.

# Constructor

    GriddedFunctionView(source, selectors...)

Accepts the same selectors as [`GridView`](@ref). The view shares memory with
`source`; mutating `values(view)` also mutates `values(source)`.
"""
struct GriddedFunctionView{TY, D, GF, VV, GV} <: AbstractGriddedFunction{TY, D}
    source::GF
    values::VV
    gridview::GV

    function GriddedFunctionView(
            source::GF, slices::Vararg{AbstractUnitRange{Int}}
        ) where {GF <: AbstractGriddedFunction}

        slices_fixed = ntuple(
            d -> length(slices[d]) == 1 ? only(slices[d]) : slices[d],
            length(slices)
        )
        vals = view(values(source), slices_fixed...)
        gridview = GridView(grid(source), slices...)
        new{eltype(GF), ndims(gridview), GF, typeof(vals), typeof(gridview)}(source, vals, gridview)
    end
end

function GriddedFunctionView(source::AbstractGriddedFunction, selectors::Vararg)
    gv = GridView(grid(source), selectors...)
    index_slices = slices(gv)
    
    GriddedFunctionView(source, index_slices...)
end

grid(gfv::GriddedFunctionView) = gfv.gridview
values(gfv::GriddedFunctionView) = gfv.values
gridtype(::Type{GriddedFunctionView{TY, D, GF, VV, GV}}) where {TY, D, GF, VV, GV} = GV 

"""
    continuousview(source::AbstractGriddedFunction, I_disc::CartesianIndex)

Return a [`GriddedFunctionView`](@ref) of `source` that fixes all discrete
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

    GriddedFunctionView(source, indices...)
end
