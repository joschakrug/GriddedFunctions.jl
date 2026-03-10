using Revise
using Test
using GriddedFunctions
import Interpolations

@testset "GriddedFunctions" begin

    struct SimpleType
        x::Float64
    end

    Base.getindex(m::SimpleType, i) = i == 1 ? m.x : nothing
    Base.convert(::Type{SimpleType}, t::T) where {T <: Tuple} = SimpleType(t...)
    GriddedFunctions.dimnames(::Type{SimpleType}) = fieldnames(SimpleType)

    struct DoubleType
        x::Float64
        y::Int64
    end

    Base.getindex(m::DoubleType, i) = getfield(m, i)
    Base.convert(::Type{DoubleType}, t::T) where {T <: Tuple} = DoubleType(t...)
    GriddedFunctions.dimnames(::Type{DoubleType}) = fieldnames(DoubleType)

    @testset "LinearAxis" begin
        lax = LinearAxis(range(0.0, 10.0, length = 100))

        @test length(lax) == 100
        @test minimum(lax) ≈ 0.0
        @test maximum(lax) ≈ 10.0
        @test lax[1] ≈ 0.0
        @test lax[end] ≈ 10.0

        # membership
        @test 0.0 in lax
        @test -1.0 ∉ lax
        @test 11.0 ∉ lax

        # iteration matches the underlying range
        @test collect(lax) ≈ collect(range(0.0, 10.0, length = 100))
    end

    @testset "DiscreteAxis" begin
        dax = DiscreteAxis([0, 1])

        @test length(dax) == 2
        @test dax[1] == 0
        @test dax[2] == 1

        # membership
        @test 0 in dax
        @test 1 in dax
        @test 2 ∉ dax

        # constructor requires sorted points
        @test_throws Exception DiscreteAxis([2, 1])
    end

    @testset "Grid construction" begin
        linax1 = LinearAxis(range(0.0, 10.0, length = 500))
        linax2 = LinearAxis(range(5.0, 20.0, length = 500))
        disax  = DiscreteAxis([0, 1])

        # Grid factory returns the right subtypes
        mg = Grid(linax1, linax2, disax)
        @test size(mg) == (500, 500, 2)
        @test GriddedFunctions.ncontinuousdims(typeof(mg)) == 2
        @test GriddedFunctions.ndiscretedims(mg) == 1

        # continuous axes must come before discrete axes
        @test_throws Exception Grid(disax, linax1)

        # indexing a grid returns the coordinate tuple at that index
        @test mg[1, 1, 1] == (0.0, 5.0, 0)
        @test mg[end, end, end] == (10.0, 20.0, 1)
    end

   @testset "Grid construction — named kwargs" begin
        linax = LinearAxis(range(0.0, 10.0, length = 11))
        disax = DiscreteAxis([2, 3, 4])

        # Grid(T; kwargs...) — axes reordered to match dimnames(T)
        g = Grid(DoubleType; x = linax, y = disax)
        @test g isa Grid{DoubleType}
        @test size(g) == (11, 3)
        @test g[1, 1] == DoubleType(0.0, 2)
        @test g[end, end] == DoubleType(10.0, 4)

        # order of kwargs should not matter
        g_rev = Grid(DoubleType; y = disax, x = linax)
        @test g_rev[1, 1] == DoubleType(0.0, 2)
        @test g_rev[end, end] == DoubleType(10.0, 4)

        # single-axis type
        g_simple = Grid(SimpleType; x = linax)
        @test g_simple isa Grid{SimpleType}
        @test g_simple[1] == SimpleType(0.0)
        @test g_simple[end] == SimpleType(10.0)

        # Grid(; kwargs...) — defaults to NamedTuple element type
        g_nt = Grid(x = linax, y = disax)
        @test eltype(g_nt) == NamedTuple{(:x, :y), Tuple{Float64, Int64}}
        @test size(g_nt) == (11, 3)
        @test g_nt[1, 1]     == (x = 0.0,  y = 2)
        @test g_nt[end, end] == (x = 10.0, y = 4)
        @test GriddedFunctions.ncontinuousdims(typeof(g_nt)) == 1
    end

    @testset "Grid construction with custom types" begin
        g = Grid(DoubleType, LinearAxis(range(0., 10., length = 50)), DiscreteAxis([2, 3, 4]))

        @test g[1, 1] == DoubleType(0., 2)
        @test GriddedFunctions.decompose(DoubleType(0., 4), g) == (tuple(0.), tuple(4))
        @test GriddedFunctions.finddiscrete((3,), g) == CartesianIndex(2)
        @test GriddedFunctions.find(DoubleType(0., 4), g) == CartesianIndex(1, 3)
        

        g = Grid(SimpleType, LinearAxis(range(0., 10., length = 50)))
        @test g[1] == SimpleType(0.)
        @test GriddedFunctions.discreteaxes(g) == ()
    end

    @testset "GriddedFunction — construction" begin
        grid = Grid(
            LinearAxis(range(0.0, 10.0, length = 500)),
            LinearAxis(range(5.0, 20.0, length = 500)),
            DiscreteAxis([0, 1])
        )

        gf = GriddedFunction(Float64, grid, (x, y, z) -> (x * y) * exp(z))

        @test gf isa GriddedFunction
        @test size(gf) == (500, 500, 2)
        @test eltype(gf) == Float64
        @test eltype(typeof(gf)) == Float64

        # values at corners
        @test gf[1, 1, 1] ≈ 0.0 * 5.0 * exp(0)      # (0, 5, 0)  → 0
        @test gf[end, end, 1] ≈ 10.0 * 20.0 * exp(0)  # (10, 20, 0) → 200
        @test gf[end, end, 2] ≈ 10.0 * 20.0 * exp(1)  # (10, 20, 1) → 200e
    end

    @testset "GriddedFunction — arithmetic" begin
        grid = Grid(
            LinearAxis(range(0.0, 10.0, length = 50)),
            LinearAxis(range(5.0, 20.0, length = 50)),
            DiscreteAxis([0, 1])
        )
        gf1 = GriddedFunction(Float64, grid, (x, y, z) -> x * y * exp(z))
        gf2 = GriddedFunction(Float64, grid, (x, y, z) -> x + y + z)

        gfsum  = gf1 + gf2
        gfdiff = gf1 - gf2
        gfprod = gf1 * gf2
        gfdiv  = gf1 / gf2

        @test gfsum[2, 3, 1]  ≈ gf1[2, 3, 1] + gf2[2, 3, 1]
        @test gfdiff[2, 3, 1] ≈ gf1[2, 3, 1] - gf2[2, 3, 1]
        @test gfprod[2, 3, 1] ≈ gf1[2, 3, 1] * gf2[2, 3, 1]
        @test gfdiv[2, 3, 1]  ≈ gf1[2, 3, 1] / gf2[2, 3, 1]
    end

    @testset "GriddedFunction — points" begin
        # 1-D: keys are scalars (matching 1-D grid iteration), values are function values
        g1  = Grid(LinearAxis(range(0.0, 1.0; length=5)))
        gf1 = GriddedFunction(Float64, g1, x -> x^2)

        ps1 = collect(points(gf1))
        @test length(ps1) == 5
        @test all(p isa Pair for p in ps1)

        # keys and values match the grid and function in iteration order
        for (p, x, fx) in zip(ps1, g1, fvalues(gf1))
            @test p.first  == x
            @test p.second ≈ fx
        end
        # values satisfy the defining relation (p.first is a 1-element Tuple)
        @test all(p.second ≈ only(p.first)^2 for p in ps1)

        # 2-D: keys are Tuples, values are function values
        g2  = Grid(LinearAxis(range(0.0, 1.0; length=3)),
                   LinearAxis(range(0.0, 2.0; length=4)))
        gf2 = GriddedFunction(Float64, g2, (x, y) -> x + y)

        ps2 = collect(points(gf2))
        @test length(ps2) == 3 * 4
        @test all(p isa Pair for p in ps2)

        # keys and values match the grid and function in iteration order
        for (p, x, fx) in zip(ps2, g2, fvalues(gf2))
            @test p.first  == x
            @test p.second ≈ fx
        end
        # values satisfy the defining relation
        @test all(p.second ≈ p.first[1] + p.first[2] for p in ps2)
    end

    @testset "GriddedFunction — findmax" begin
        grid = Grid(
            LinearAxis(range(0.0, 10.0, length = 500)),
            LinearAxis(range(5.0, 20.0, length = 500)),
            DiscreteAxis([0, 1])
        )

        # f = (x * y) * exp(z) is maximised at x=10, y=20, z=1
        gf = GriddedFunction(Float64, grid, (x, y, z) -> (x * y) * exp(z))

        maxval, maxI = findmax(gf)

        @test maxval ≈ 10.0 * 20.0 * exp(1)
        @test maxI == CartesianIndex(500, 500, 2)
    end

    @testset "GriddedFunction — maxpoint" begin
        grid = Grid(
            LinearAxis(range(0.0, 10.0, length = 500)),
            LinearAxis(range(5.0, 20.0, length = 500)),
            DiscreteAxis([0, 1])
        )

        # f = (x * y) * exp(z) is maximised at x=10, y=20, z=1
        gf = GriddedFunction(Float64, grid, (x, y, z) -> (x * y) * exp(z))

        maxpt, maxval = maxpoint(gf)

        @test maxval ≈ 10.0 * 20.0 * exp(1)
        # maxpoint returns the grid point directly (via pairs)
        @test maxpt == (10.0, 20.0, 1)
    end

    @testset "SubGrid" begin
        linax1 = LinearAxis(range(0.0, 10.0, length = 11))  # 0, 1, …, 10
        linax2 = LinearAxis(range(0.0,  4.0, length =  5))  # 0, 1, 2, 3, 4
        disax  = DiscreteAxis([10, 20, 30])
        g = Grid(linax1, linax2, disax)

        # all-free view preserves dimensionality and size
        sg_free = SubGrid(g, :, :, :)
        @test ndims(sg_free) == 3
        @test size(sg_free)  == (11, 5, 3)

        # fix discrete axis → 2-D view
        sg = SubGrid(g, :, :, 10)
        @test ndims(sg) == 2
        @test size(sg)  == (11, 5)
        @test GriddedFunctions.ncontinuousdims(typeof(sg)) == 2
        # indexing reconstructs the full point with the fixed discrete value embedded
        @test sg[1, 1]     == (0.0,  0.0, 10)
        @test sg[end, end] == (10.0, 4.0, 10)
        @test sg[3, 2]     == (2.0,  1.0, 10)

        # fix second discrete value
        sg_d2 = SubGrid(g, :, :, 30)
        @test size(sg_d2) == (11, 5)
        @test sg_d2[1, 1] == (0.0, 0.0, 30)

        # fix one continuous axis → 2-D view (free axes: linax1 + disax)
        sg2 = SubGrid(g, :, 2.0, :)
        @test ndims(sg2) == 2
        @test size(sg2)  == (11, 3)
        @test sg2[1, 1]     == (0.0,  2.0, 10)
        @test sg2[end, end] == (10.0, 2.0, 30)

        # restrict first continuous axis to a subrange → still 3-D, but smaller
        sg3 = SubGrid(g, (2.0, 6.0), :, :)
        @test ndims(sg3) == 3
        @test size(sg3)  == (5, 5, 3)   # 2,3,4,5,6 → 5 points
        @test sg3[1, 1, 1]       == (2.0,  0.0, 10)
        @test sg3[end, end, end] == (6.0,  4.0, 30)

        # out-of-bounds slice should throw
        @test_throws Exception SubGrid(g, 1:20, :, :)

        # 1-D grid with SimpleType: restrict the single continuous axis
        g_simple = Grid(SimpleType, LinearAxis(range(0.0, 5.0; length = 6)))
        sg_s = SubGrid(g_simple, (1.0, 4.0))
        @test ndims(sg_s) == 1
        @test size(sg_s)  == (4,)
        @test sg_s[1]   == SimpleType(1.0)
        @test sg_s[end] == SimpleType(4.0)

        # DoubleType: 1 continuous + 1 discrete, fix discrete → 1-D view
        g_double = Grid(DoubleType, LinearAxis(range(0.0, 10.0; length = 11)), DiscreteAxis([2, 3, 4]))
        sg_d = SubGrid(g_double, :, 3)
        @test ndims(sg_d) == 1
        @test size(sg_d)  == (11,)
        @test sg_d[1]   == DoubleType(0.0,  3)
        @test sg_d[end] == DoubleType(10.0, 3)

        # gfview passes through to SubGrid
        @test gfview(g, :, :, 10) == SubGrid(g, :, :, 10)
    end

    @testset "SubGrid — named kwargs" begin
        linax1 = LinearAxis(range(0.0, 10.0, length = 11))
        linax2 = LinearAxis(range(0.0,  4.0, length =  5))
        disax  = DiscreteAxis([10, 20, 30])

        # NamedTuple grid via Grid(; kwargs...)
        g = Grid(x = linax1, y = linax2, z = disax)

        # fix :z — should give a 2-D view
        sg = SubGrid(g; z = 10)
        @test ndims(sg) == 2
        @test size(sg)  == (11, 5)
        @test sg[1, 1]     == (x = 0.0,  y = 0.0, z = 10)
        @test sg[end, end] == (x = 10.0, y = 4.0, z = 10)

        # restrict :x, leave :y and :z free
        sg2 = SubGrid(g; x = (2.0, 6.0))
        @test ndims(sg2) == 3
        @test size(sg2)  == (5, 5, 3)
        @test sg2[1, 1, 1] == (x = 2.0, y = 0.0, z = 10)

        # kwargs in arbitrary order — same result
        sg3 = SubGrid(g; z = 20, x = (0.0, 5.0))
        @test ndims(sg3) == 2
        @test sg3[1, 1] == (x = 0.0, y = 0.0, z = 20)

        # DoubleType grid
        g_d = Grid(DoubleType; x = linax1, y = disax)
        sg_d = SubGrid(g_d; y = 20)
        @test ndims(sg_d) == 1
        @test size(sg_d)  == (11,)
        @test sg_d[1]   == DoubleType(0.0,  20)
        @test sg_d[end] == DoubleType(10.0, 20)

        # gfview with kwargs passes through to SubGrid
        @test gfview(g; z = 10) == SubGrid(g; z = 10)
    end

    @testset "SubGriddedFunction" begin
        linax1 = LinearAxis(range(0.0, 10.0, length = 11))
        linax2 = LinearAxis(range(0.0,  4.0, length =  5))
        disax  = DiscreteAxis([10, 20])
        g  = Grid(linax1, linax2, disax)
        gf = GriddedFunction(Float64, g, (x, y, z) -> x + y + z)

        # fix discrete dim: values should be 2-D (singleton dim dropped)
        sgf = SubGriddedFunction(gf, :, :, 10)
        @test ndims(fvalues(sgf)) == 2
        @test size(fvalues(sgf))  == (11, 5)
        @test fvalues(sgf)[1, 1]     ≈  0.0 + 0.0 + 10
        @test fvalues(sgf)[end, end] ≈ 10.0 + 4.0 + 10
        # the grid of the view has matching reduced dimensionality
        @test ndims(grid(sgf)) == 2
        @test size(grid(sgf))  == (11, 5)

        # second discrete slice
        sgf2 = SubGriddedFunction(gf, :, :, 20)
        @test ndims(fvalues(sgf2)) == 2
        @test fvalues(sgf2)[1, 1] ≈ 0.0 + 0.0 + 20

        # restrict (non-singleton) → values remain 3-D
        sgf3 = SubGriddedFunction(gf, (2.0, 6.0), :, :)
        @test ndims(fvalues(sgf3)) == 3
        @test size(fvalues(sgf3))  == (5, 5, 2)
        @test fvalues(sgf3)[1, 1, 1] ≈ 2.0 + 0.0 + 10

        # 1-D SimpleType grid: restrict the single axis
        g_simple  = Grid(SimpleType, LinearAxis(range(0.0, 5.0; length = 6)))
        gf_simple = GriddedFunction(Float64, g_simple, x -> x^2)
        sgf_s = SubGriddedFunction(gf_simple, (1.0, 4.0))
        @test ndims(fvalues(sgf_s)) == 1
        @test size(fvalues(sgf_s))  == (4,)
        @test fvalues(sgf_s)[1]   ≈  1.0
        @test fvalues(sgf_s)[end] ≈ 16.0

        # DoubleType: fix discrete → 1-D values
        g_double  = Grid(DoubleType, LinearAxis(range(0.0, 10.0; length = 11)), DiscreteAxis([2, 3, 4]))
        gf_double = GriddedFunction(Float64, g_double, (x, y) -> x * y)
        sgf_d = SubGriddedFunction(gf_double, :, 3)
        @test ndims(fvalues(sgf_d)) == 1
        @test size(fvalues(sgf_d))  == (11,)
        @test fvalues(sgf_d)[1]   ≈  0.0 * 3
        @test fvalues(sgf_d)[end] ≈ 10.0 * 3

        # gfview passes through to SubGriddedFunction
        @test fvalues(gfview(gf, :, :, 10)) == fvalues(SubGriddedFunction(gf, :, :, 10))
    end

    @testset "SubGriddedFunction — named kwargs" begin
        linax1 = LinearAxis(range(0.0, 10.0, length = 11))
        linax2 = LinearAxis(range(0.0,  4.0, length =  5))
        disax  = DiscreteAxis([10, 20])

        g  = Grid(x = linax1, y = linax2, z = disax)
        gf = GriddedFunction(Float64, g, (x, y, z) -> x + y + z)

        # fix :z — 2-D values
        sgf = SubGriddedFunction(gf; z = 10)
        @test ndims(fvalues(sgf)) == 2
        @test size(fvalues(sgf))  == (11, 5)
        @test fvalues(sgf)[1, 1]     ≈  0.0 + 0.0 + 10
        @test fvalues(sgf)[end, end] ≈ 10.0 + 4.0 + 10

        # restrict :x only — values remain 3-D
        sgf2 = SubGriddedFunction(gf; x = (2.0, 6.0))
        @test ndims(fvalues(sgf2)) == 3
        @test size(fvalues(sgf2))  == (5, 5, 2)
        @test fvalues(sgf2)[1, 1, 1] ≈ 2.0 + 0.0 + 10

        # kwargs in arbitrary order
        sgf3 = SubGriddedFunction(gf; z = 20, x = (0.0, 5.0))
        @test ndims(fvalues(sgf3)) == 2
        @test fvalues(sgf3)[1, 1] ≈ 0.0 + 0.0 + 20

        # gfview with kwargs passes through to SubGriddedFunction
        @test fvalues(gfview(gf; z = 10)) == fvalues(SubGriddedFunction(gf; z = 10))
    end

    @testset "Interpolation" begin
        grid = Grid(
            LinearAxis(range(0.0, 10.0, length = 500)),
            LinearAxis(range(5.0, 20.0, length = 500)),
            DiscreteAxis([0, 1])
        )

        gf  = GriddedFunction(Float64, grid, (x, y, z) -> (x * y) * exp(z))
        gfi = interpolate(gf)

        @test gfi isa GriddedFunctions.GFInterpolation

        # at grid-coincident points the interpolated value must match exactly
        @test gfi(0.0, 5.0, 0)  ≈ 0.0
        @test gfi(10.0, 20.0, 0) ≈ 200.0
        @test gfi(10.0, 20.0, 1) ≈ 200.0 * exp(1)

        # f(x, y) = x * y is bilinear, so bilinear interpolation is exact
        @test gfi(2.031, 11.007, 0) ≈ 2.031 * 11.007 * exp(0) atol=1e-6
        @test gfi(2.031, 11.007, 1) ≈ 2.031 * 11.007 * exp(1) atol=1e-6

        # from the README: valid calls on both discrete values
        @test gfi(2.031, 11.007, 0) isa Float64
        @test gfi(2.031, 11.007, 1) isa Float64

        # from the README: a discrete value not on the axis must throw
        @test_throws Exception gfi(2.031, 11.007, 2)
    end

    @testset "Interpolation — no discrete axes" begin
        grid = Grid(
            LinearAxis(range(0.0, 10.0, length = 200)),
            LinearAxis(range(0.0, 10.0, length = 200))
        )

        gf  = GriddedFunction(Float64, grid, (x, y) -> x * y)
        gfi = interpolate(gf)

        @test gfi isa GriddedFunctions.GFInterpolation
        @test gfi(3.7, 6.2) ≈ 3.7 * 6.2 atol=1e-6
        @test gfi(0.0, 0.0) ≈ 0.0
        @test gfi(10.0, 10.0) ≈ 100.0
    end

    @testset "Interpolation - custom type" begin
        g = Grid(
            DoubleType,
            LinearAxis(range(0.0, 5.0; length = 100)),
            DiscreteAxis([0, 4, 5])
        )

        gf  = GriddedFunction(Float64, g, (x, y) -> x + y)

        # Interpolation construction
        gfi = interpolate(gf)
        @test gfi isa GriddedFunctions.GFInterpolation

        # f(x, y) = x + y is linear, so BSpline(Linear()) is exact everywhere
        @test gfi(0.0, 0) ≈ 0.0
        @test gfi(5.0, 5) ≈ 10.0
        @test gfi(0.5, 4) ≈ 4.5 atol=1e-10

        # evaluate with a MyPoint directly (exercises the TX dispatch path)
        @test GriddedFunctions.evaluate(gfi, DoubleType(0.5, 4)) ≈ 4.5 atol=1e-10
        @test gfi(DoubleType(0.5, 4)) ≈ 4.5 atol=1e-10
    end

    @testset "Custom singleton type" begin

        g = Grid(
            SimpleType,
            LinearAxis(range(0.0, 5.0; length = 100))
        )

        # grid type and indexing
        @test g isa GriddedFunctions.Grid{SimpleType}
        @test g[1]      == SimpleType(0.0)
        @test g[end]    == SimpleType(5.0)

        # membership (exercises p[d])
        @test  g[1]     in g      # corner points are always on the grid
        @test  g[end]   in g
        @test  SimpleType(-1.0) ∉ g   # x < 0 is outside the grid

        # GriddedFunction construction and indexing
        gf = GriddedFunction(Float64, g, x -> x^2)
        @test gf isa GriddedFunction
        @test gf[1]     ≈ 0.0
        @test gf[end] ≈ 25.0

        # Interpolation construction
        gfi = interpolate(gf)
        @test gfi isa GriddedFunctions.GFInterpolation

        # f(x, y) = x + y is linear, so BSpline(Linear()) is exact everywhere
        @test gfi(0.0) ≈ 0.0
        @test gfi(5.0) ≈ 25.0
        @test gfi(3.0) ≈ 9.0 atol=1e-2

        @test GriddedFunctions.finddiscrete((), g) == CartesianIndex()
        @test first(GriddedFunctions.decompose(SimpleType(5.0), g)) == (5.0,)
        @test gfi.interpolations[CartesianIndex()](5.0) ≈ 25 atol=1e-2

        # evaluate with a SimpleType directly (exercises the TX dispatch path)
        @test GriddedFunctions.evaluate(gfi, SimpleType(2.0)) ≈ 4.0 atol=1e-2
        @test gfi(SimpleType(2.0)) ≈ 4.0 atol=1e-2
        @test gfi(2.0) ≈ 4.0 atol=1e-2
    end

    @testset "argmap / argmap!" begin
        g  = Grid(LinearAxis(range(0.0, 1.0; length = 5)),
                  DiscreteAxis([0, 1]))
        f  = (x, z) -> x^2 + z

        # argmap! (serial) — mutates in place and returns the same object
        gf = GriddedFunction(Float64, g, undef)
        result = argmap!(t -> t[1]^2 + t[2], gf)
        @test result === gf
        @test fvalues(gf)[1, 1] ≈ 0.0^2 + 0
        @test fvalues(gf)[end, 1] ≈ 1.0^2 + 0
        @test fvalues(gf)[1, 2] ≈ 0.0^2 + 1
        @test fvalues(gf)[end, 2] ≈ 1.0^2 + 1

        # argmap! (parallel) — same result as serial
        gf_par = GriddedFunction(Float64, g, undef)
        argmap!(t -> t[1]^2 + t[2], gf_par; parallel = true)
        @test fvalues(gf_par) ≈ fvalues(gf)

        # argmap (serial) — returns a new object, source unchanged
        gf_src = GriddedFunction(Float64, g, (x, z) -> 0.0)
        gf_new = argmap(t -> t[1]^2 + t[2], gf_src)
        @test gf_new !== gf_src
        @test fvalues(gf_new)[1, 1]   ≈ 0.0^2 + 0
        @test fvalues(gf_new)[end, 1] ≈ 1.0^2 + 0
        @test fvalues(gf_new)[1, 2]   ≈ 0.0^2 + 1
        @test fvalues(gf_new)[end, 2] ≈ 1.0^2 + 1
        @test all(fvalues(gf_src) .== 0.0)   # source not mutated

        # argmap (parallel) — same result as serial
        gf_par2 = argmap(t -> t[1]^2 + t[2], gf_src; parallel = true)
        @test fvalues(gf_par2) ≈ fvalues(gf_new)
    end

    @testset "inbounds" begin
        g = Grid(
            LinearAxis(range(0.0, 10.0; length = 200)),
            LinearAxis(range(5.0, 20.0; length = 200)),
            DiscreteAxis([0, 1])
        )

        # both components valid
        @test  inbounds((3.7, 12.0, 0), g)
        @test  inbounds((3.7, 12.0, 1), g)
        # boundary continuous, valid discrete
        @test  inbounds((0.0, 5.0, 0), g)
        @test  inbounds((10.0, 20.0, 1), g)
        # continuous out of range
        @test !inbounds((-0.1, 12.0, 0), g)
        @test !inbounds((10.1, 12.0, 1), g)
        @test !inbounds((5.0, 4.9, 0),   g)
        # discrete value not on axis
        @test !inbounds((3.7, 12.0, 2), g)
        # non-grid-coincident continuous is still in-bounds
        @test  inbounds((0.001, 5.001, 0), g)
        @test  (0.001, 5.001, 0) ∉ g
    end

end
