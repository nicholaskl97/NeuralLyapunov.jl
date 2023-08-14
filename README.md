# NeuralLyapunov.jl
A library for searching for neural Lyapunov functions in Julia.

This package provides an API for setting up the search for a neural Lyapunov function.
Such a search can be formulated as a partial differential inequality, and this library generates a ModelingToolkit PDESystem to be solved using NeuralPDE.jl.
Since the Lyapunov conditions can be formulated in several different ways and a neural Lyapunov function can be set up in many different forms, this library presents an extensible interface for users to choose how they wish to set up the search, with useful pre-built options for common setups.
