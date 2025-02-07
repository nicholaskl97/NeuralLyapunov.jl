# Lyapunov Minimization Condition

The condition that the Lyapunov function ``V(x)`` must be minimized uniquely at the fixed point ``x_0`` is often represented as a requirement for ``V(x)`` to be positive away from the fixed point and zero at the fixed point.
Put mathematically, it is sufficient to require ``V(x) > 0 \, \forall x \ne x_0`` and ``V(x_0) = 0``.
We call such functions positive definite (around the fixed point ``x_0``).
The weaker condition that ``V(x) \ge 0 \, \forall x \ne x_0`` and ``V(x_0) = 0`` is positive *semi-*definiteness.

NeuralLyapunov.jl provides the [`LyapunovMinimizationCondition`](@ref) struct for users to specify the form of the minimization condition to enforce through training.

```@docs
LyapunovMinimizationCondition     
```

## Pre-defined minimization conditions

```@docs
StrictlyPositiveDefinite
PositiveSemiDefinite
DontCheckNonnegativity
```

## Defining your own minimization condition

```@meta
CurrentModule = NeuralLyapunov
```

If a user wishes to define their own version of the minimization condition in a form other than
``V(x) ≥ \texttt{strength}(x, x_0)``,
they must define their own subtype of [`AbstractLyapunovMinimizationCondition`](@ref).

```@docs
AbstractLyapunovMinimizationCondition
```

When constructing the PDESystem, [`NeuralLyapunovPDESystem`](@ref) uses [`check_nonnegativity`](@ref) to determine if it should include an equation equating the result of [`get_minimization_condition`](@ref) to zero.
It additionally uses [`check_zero_fixed_point`](@ref) to determine if it should include the equation ``V(x_0) = 0`` and [`check_minimal_fixed_point`](@ref) to determine if it should include the equation ``\nabla V(x_0) = 0``.

```@docs
check_nonnegativity
check_zero_fixed_point
check_minimal_fixed_point
get_minimization_condition
```
