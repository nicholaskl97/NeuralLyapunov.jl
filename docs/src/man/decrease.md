# Lyapunov Decrease Condition

To represent the condition that the Lyapunov function ``V(x)`` must decrease along system trajectories, we typically introduce a new function ``\dot{V}(x) = \nabla V(x) \cdot f(x)``.
This function represents the rate of change of ``V`` along system trajectories.
That is to say, if ``x(t)`` is a trajectory defined by ``\frac{dx}{dt} = f(x)``, then ``\dot{V}(x(t)) = \frac{d}{dt} [ V(x(t)) ]``.
It is then sufficient to show that ``\dot{V}(x)`` is negative away from the fixed point and zero at the fixed point, since a negative derivative means a decreasing function.

Put mathematically, it is sufficient to require ``\dot{V}(x) < 0 \, \forall x \ne x_0`` and ``\dot{V}(x_0) = 0``.
We call such functions negative definite (around the fixed point ``x_0``).
The weaker condition that ``\dot{V}(x) \le 0 \, \forall x \ne x_0`` and ``\dot{V}(x_0) = 0`` is negative *semi-*definiteness.

The condition that ``\dot{V}(x_0) = 0`` is satisfied by the definition of ``\dot{V}`` and the fact that ``x_0`` is a fixed point, so we do not need to train for it.
In cases where the dynamics have some dependence on the neural network (e.g., in [policy search](policy_search.md)), we should rather train directly for ``f(x_0) = 0``, since the minimization condition will also guarantee ``[\nabla V](x_0) = 0``, so ``\dot{V}(x_0) = 0``.

NeuralLyapunov.jl provides the [`LyapunovDecreaseCondition`](@ref) struct for users to specify the form of the decrease condition to enforce through training.

```@docs
LyapunovDecreaseCondition
```

## Pre-defined decrease conditions

```@docs
AsymptoticStability
ExponentialStability
StabilityISL
DontCheckDecrease
```

## Defining your own decrease condition

```@meta
CurrentModule = NeuralLyapunov
```

If a user wishes to define their own version of the decrease condition in a form other than
``\texttt{rate\_metric}(V(x), \dot{V}(x)) \le - \texttt{strength}(x, x_0)``,
they must define their own subtype of [`AbstractLyapunovDecreaseCondition`](@ref).

```@docs
AbstractLyapunovDecreaseCondition
```

When constructing the PDESystem, [`NeuralLyapunovPDESystem`](@ref) uses [`check_decrease`](@ref) to determine if it should include an equation equating the result of [`get_decrease_condition`](@ref) to zero.
It additionally uses [`check_maximal_fixed_point`](@ref) to determine if it should include the equation ``\nabla \dot{V}(x_0) = 0``, which helps enforce the local maximality of the fixed point.

```@docs
check_decrease
get_decrease_condition
check_maximal_fixed_point
```
