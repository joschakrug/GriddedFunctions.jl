using Test
using GriddedFunctions
import Interpolations

@testset "GriddedFunctions" begin

    struct MyType
        x::Float64
        y::Int64
    end

    MyType(iter) = MyType(iter...)
    Base.getindex(m::MyType, i) = getfield(m, i)

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
        @test mg isa GriddedFunctions.MixedGrid
        @test size(mg) == (500, 500, 2)

        cg = Grid(linax1, linax2)
        @test cg isa GriddedFunctions.ContinuousGrid
        @test size(cg) == (500, 500)

        dg = Grid(disax)
        @test dg isa GriddedFunctions.DiscreteGrid
        @test size(dg) == (2,)

        # continuous axes must come before discrete axes
        @test_throws Exception Grid(disax, linax1)

        # indexing a grid returns the coordinate tuple at that index
        @test mg[1, 1, 1] == (0.0, 5.0, 0)
        @test mg[end, end, end] == (10.0, 20.0, 1)
    end

    @testset "Grid construction with custom type" begin
        g = Grid(MyType, LinearAxis(range(0., 10., length = 50)), DiscreteAxis([2, 3, 4]))

        @test g[1, 1] == MyType(0., 2)
        @test GriddedFunctions.finddiscrete(g, MyType(0.2, 3)) == 2
        @test GriddedFunctions.find(g, MyType(0., 4)) == CartesianIndex(1, 3)
        @test GriddedFunctions.decompose(g, MyType(0., 4)) == (tuple(0.), tuple(4))
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
        for (p, x, fx) in zip(ps1, g1, values(gf1))
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
        for (p, x, fx) in zip(ps2, g2, values(gf2))
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

    @testset "Interpolation — ContinuousGrid" begin
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

    @testset "Custom point type" begin
        # Minimal custom point type replacing the default Tuple{Float64, Float64}.
        # Must satisfy the Grid{T} interface:
        #   - T(a, b)       positional constructor  (used by getindex)
        #   - T(t::Tuple)   from-Tuple constructor  (used by the GFInterpolation callable)
        #   - p[i]          component access        (used by Base.in, decompose, find)
        #   - iterate/length                        (used for splatting in evaluate)
        struct MyPoint
            x::Float64
            y::Float64
        end

        MyPoint(iter)                 = MyPoint(iter...)
        Base.getindex(p::MyPoint, i)  = i == 1 ? p.x : p.y
        Base.iterate(p::MyPoint, s=1) = s > 2 ? nothing : (p[s], s + 1)

        g = Grid(
            MyPoint,
            LinearAxis(range(0.0, 5.0; length = 100)),
            DiscreteAxis([0.0, 4.5, 5.0])
        )

        # grid type and indexing
        @test g isa GriddedFunctions.MixedGrid{MyPoint}
        @test g[1, 1]     == MyPoint(0.0, 0.0)
        @test g[end, end] == MyPoint(5.0, 5.0)

        # membership (exercises p[d])
        @test  g[1, 1]     in g      # corner points are always on the grid
        @test  g[end, end] in g
        @test  MyPoint(-1.0, 2.0) ∉ g   # x < 0 is outside the grid

        # GriddedFunction construction and indexing
        gf = GriddedFunction(Float64, g, (x, y) -> x + y)
        @test gf isa GriddedFunction
        @test gf[1, 1]     ≈ 0.0
        @test gf[end, end] ≈ 10.0

        # Interpolation construction
        gfi = interpolate(gf)
        @test gfi isa GriddedFunctions.GFInterpolation

        # f(x, y) = x + y is linear, so BSpline(Linear()) is exact everywhere
        @test gfi(0.0, 0.0) ≈ 0.0
        @test gfi(5.0, 5.0) ≈ 10.0
        @test gfi(0.5, 4.5) ≈ 5.0 atol=1e-10

        # evaluate with a MyPoint directly (exercises the TX dispatch path)
        @test GriddedFunctions.evaluate(gfi, MyPoint(0.5, 4.5)) ≈ 5.0 atol=1e-10
        @test gfi(MyPoint(0.5, 4.5)) ≈ 5.0 atol=1e-10
    end

end
