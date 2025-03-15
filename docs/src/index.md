```@meta
CurrentModule = NeuralLyapunov
```

# NeuralLyapunov.jl

[NeuralLyapunov.jl](https://github.com/SciML/NeuralLyapunov.jl) is a library for searching for neural Lyapunov functions in Julia.

This package provides an API for setting up the search for a neural Lyapunov function. Such a search can be formulated as a partial differential inequality, and this library generates a [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) PDESystem to be solved using [NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl). Since the Lyapunov conditions can be formulated in several different ways and a neural Lyapunov function can be set up in many different forms, this library presents an extensible interface for users to choose how they wish to set up the search, with useful pre-built options for common setups.

## Getting Started

If this is your first time using the library, start by familiarizing yourself with the [components of a neural Lyapunov problem](man.md) in NeuralLyapunov.jl.
Then, you can dive in with any of the following demonstrations (the [damped simple harmonic oscillator](demos/damped_SHO.md) is recommended to begin):

```@contents
Pages = Main.DEMONSTRATION_PAGES
Depth = 1
```

When you begin to write your own neural Lyapunov code, especially if you hope to define your own neural Lyapunov formulation, you may find any of the following manual pages useful:

```@contents
Pages = Main.MANUAL_PAGES
```

Finally, when you wish to test your neural Lyapunov code, you may wish to use one of the built-in example systems:

```@contents
Pages = Main.NEURALLYAPUNOVPROBLEMLIBRARY_PAGES
```
