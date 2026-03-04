using Revise
using Interpolations
using GriddedFunctions

linax = LinearAxis(range(0., 10., length = 100))
linax2 = LinearAxis(range(4., 17., length = 150))
disax = DiscreteAxis([2, 3])
disax2 = DiscreteAxis([1, 3, 5])

grid = Grid(linax, linax2, disax)

gf = GriddedFunction(Float64, grid, (x1, x2, x3) -> (x1 + x2 + x3))

gfi = interpolate(gf)

gfi(0.5, 6.7, 3)

gf2 = GriddedFunction(Float64, grid, (x1, x2, x3) -> (x1 / x2 - x3))

gf + gf2

gf3 = GriddedFunction(Int64, grid, (x1, x2, x3) -> x2 != 0 ? ceil(x1 / x2 - x3) : 0)

gf / gf3

grid

GriddedFunctions.evaluate(gfi, (0.5, 4.5, 2))

view(gf, 1:3)

grid4 = Grid(linax, linax2, disax, disax)
gf4 = GriddedFunction(Float64, grid4, (x1, x2, x3, x4) -> (x1 + x2 + x3 + x4))

for I in eachindex(grid)
    println(grid[I])
end
