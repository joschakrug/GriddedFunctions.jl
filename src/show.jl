# Custom showarg methods for AbstractArray subtypes.
#
# Julia calls Base.showarg(io, x, toplevel) to render the type label in the
# array-summary header (e.g. "5-element LinearAxis{Float64}:").  By overriding
# it here we truncate the implementation-detail type parameters while keeping
# all other array printing behaviour (dimensions, element layout) unchanged.

# --- Axes -------------------------------------------------------------------

Base.showarg(io::IO, ::LinearAxis{T}, ::Bool) where T =
    print(io, "LinearAxis{", T, "}")

Base.showarg(io::IO, ::DiscreteAxis{T}, ::Bool) where T =
    print(io, "DiscreteAxis{", T, "}")

Base.showarg(io::IO, ::SubAxis{T, <:Any, A}, ::Bool) where {T, A} =
    print(io, "SubAxis{", T, "}")

# --- Grids ------------------------------------------------------------------

Base.showarg(io::IO, ::Grid{T, D}, ::Bool) where {T, D} =
    print(io, "Grid{", T, "}")

Base.showarg(io::IO, ::SubGrid{T, D, DS, GS}, ::Bool) where {T, D, DS, GS} =
    print(io, "SubGrid{", T, "}")

# --- Gridded functions ------------------------------------------------------

Base.showarg(io::IO, ::GriddedFunction{TY, D}, ::Bool) where {TY, D} =
    print(io, "GriddedFunction{$TY}")

Base.showarg(io::IO, ::SubGriddedFunction{TY, D}, ::Bool) where {TY, D} =
    print(io, "SubGriddedFunction{$TY}")

