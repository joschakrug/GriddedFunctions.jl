# Approximate functions over a grid

This package provides tools to comfortably work with functions that cannot be defined analytically but only approximated over a grid. It builds on the [Interpolations.jl](https://juliamath.github.io/Interpolations.jl/latest/) package for actually computing function values but offers tools to define, store, modify and access approximated functions in a convenient and efficient way.

This is useful in particular when trying to solve dynamic problems via value function iteration.

## A simple example

Imagine you want to approximate a function of three variables, two of which live in a continuous range and one of which can only take a discrete set of values. Using the `GriddedFunctions.jl` package, you can easily set up such a function as follows:

```julia
using GriddedFunctions

grid = Grid(
    LinearAxis(range(0, 10, length = 500)),
    LinearAxis(range(5, 20, length = 500)),
    DiscreteAxis([0, 1])
)

gf = GriddedFunction(
    Float64,
    grid,
    ((x, y, z),) -> (x * y) * exp(z)
)
```

If you want to perform a simple grid search optimisation of `gf`, this is as easy as using Julia's built-in `findmax` function:

```julia
maxval, maxidx = findmax(gf)

# return the actual (x, y, z) values that maximise gf on the given grid
grid[maxidx]
```

If you want to get an interpolated continuous version of `gfi`, the package provides an implementation of the `interpolate` method tailored to gridded functions. It will interpolate between values on continuous axes but only accept exact values on discrete axes:

```julia
gfi = interpolate(gf)

# valid
gfi(2.031, 11.007, 0)
gfi(2.031, 11.007, 1)

# throws an error
gfi(2.031, 11.007, 2)
```

## Installation

To install the latest stable version, add [JuliaRegistryJKG](https://github.com/joschakrug/JuliaRegistryJKG) to your Julia registries by running

```julia
Pkg.Registry.add(RegistrySpec(url="git@github.com:joschakrug/JuliaRegistryJKG.git"))
```

in your REPL. With this registry added, you can simply `] add` and `] updated` the `GriddedFunctions` package using your package manager.

To install the latest development version, clone this git repository to a local folder and add that folder to your main project as a development dependency running `] dev local/repo/path`.

## For developers

### Testing

Testing is as simple as running `] test GriddedFunctions` with the `test` environment activated. Manual tests in the `test/manual` folder require the `test` environment to be activated as well.

### Pushing updated versions

To register an updated package version in `JuliaRegistryJKG`, bump the version number in the `Project.toml`, push a tagged commit with the same version number to GitHub and then run

```julia
julia> using LocalRegistry
julia> register("path/to/local/copy/of/project", registry = "JuliaRegistryJKG", push = true)
```
