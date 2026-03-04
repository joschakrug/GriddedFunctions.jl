using Revise, GriddedFunctions, Plots

g1c = Grid(LinearAxis(range(0.0, 2π; length=100)))
gf1c = GriddedFunction(Float64, g1c, x -> sin(x))
plot(gf1c)  # should show a sine curve

g1d = Grid(DiscreteAxis([1, 2, 3, 4]))
gf1d = GriddedFunction(Float64, g1d, x -> x^2)
plot(gf1d)  # should show 4 scatter points

g2c = Grid(LinearAxis(range(0.0, 1.0; length=50)),
           LinearAxis(range(0.0, 1.0; length=50)))
gf2c = GriddedFunction(Float64, g2c, (x, y) -> x^2 + y^2)
plot(gf2c)               # 3D surface
plot(gf2c, seriestype=:heatmap)  # 2D heatmap

struct Point; x::Float64; y::Float64; end
Point(iter) = Point(iter...)
Base.getindex(p::Point, i) = getfield(p, i)
Base.iterate(p::Point, s = 1) = s > 2 ? nothing : (p[s], s + 1)
g = Grid(Point, LinearAxis(range(0.0,1.0;length=50)), LinearAxis(range(0.0,1.0;length=50)))
gf = GriddedFunction(Float64, g, (x, y) -> x + y)
plot(gf)  # x-label "x", y-label "y"
