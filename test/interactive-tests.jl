using Revise
using Interpolations
using GriddedFunctions

linax = LinearAxis(range(0., 10., length = 100))
linax2 = LinearAxis(range(4., 17., length = 150))
disax = DiscreteAxis([2, 3])
disax2 = DiscreteAxis([1, 3, 5])

grid = Grid(linax, linax2, disax)

gf = GriddedFunction(grid, (x1, x2, x3) -> (x1 + x2 + x3), Float64)

gfi = interpolate(gf)

gfi(0.5, 6.7, 3)

gf2 = GriddedFunction(grid, (x1, x2, x3) -> (x1 / x2 - x3), Float64)

gf + gf2

gf3 = GriddedFunction(grid, (x1, x2, x3) -> x2 != 0 ? ceil(x1 / x2 - x3) : 0, Int64)

gf / gf3

grid

GriddedFunctions.evaluate(gf, (0.5, 4.5, 2))

view(gf, 1:3)

grid4 = Grid(linax, linax2, disax, disax)
gf4 = GriddedFunction(grid4, (x1, x2, x3, x4) -> (x1 + x2 + x3 + x4), Float64)

GriddedFunctions.evaluate(gf4, (0.5, 4.750573, 2, 2))

A = [2, 3, 4] * [1 2]

CartesianIndices(A)

for I in eachindex(grid)
    println(grid[I])
end