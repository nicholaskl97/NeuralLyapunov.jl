"""
    RoAAwareDecreaseCondition(check_decrease, rate_metric, strength, rectifier, ρ, out_of_RoA_penalty, check_fixed_point_gradient)

Specifies the form of the Lyapunov decrease condition to be used, training for a region of
attraction estimate of ``\\{ x : V(x) ≤ ρ \\}``.

# Fields
  - `check_decrease::Bool`: whether or not to train for negativity/nonpositivity of
    ``V̇(x)``.
  - `rate_metric::Function`: should increase with ``V̇(x)``; used when
    `check_decrease == true`.
  - `strength::Function`: specifies the level of strictness for negativity training; should
    be zero when the two inputs are equal and nonnegative otherwise; used when
    `check_decrease == true`.
  - `rectifier::Function`: positive when the input is positive and (approximately) zero when
    the input is negative.
  - `sigmoid::Function`: approximately one when the input is positive and approximately zero
    when the input is negative.
  - `ρ`: the level of the sublevel set forming the estimate of the region of attraction.
  - `out_of_RoA_penalty::Function`: a loss function to be applied penalizing points outside
    the sublevel set forming the region of attraction estimate.

# Training conditions

If `check_decrease == true`, training will attempt to enforce

``\\texttt{rate\\_metric}(V(x), V̇(x)) ≤ - \\texttt{strength}(x, x_0)``

whenever ``V(x) ≤ ρ``, and will instead apply a loss of

``\\lvert \\texttt{out\\_of\\_RoA\\_penalty}(V(x), V̇(x), x, x_0, ρ) \\rvert^2``

when ``V(x) > ρ``.

The inequality will be approximated by the equation

``\\texttt{rectifier}(\\texttt{rate\\_metric}(V(x), V̇(x)) + \\texttt{strength}(x, x_0)) = 0``.

Note that the approximate equation and inequality are identical when
``\\texttt{rectifier}(t) = \\max(0, t)``.

The `sigmoid` function allows for a smooth transition between the ``V(x) ≤ ρ`` case and the
``V(x) > ρ`` case, by combining the above equations into one:

``\\texttt{sigmoid}(ρ - V(x)) (\\text{in-RoA expression}) + \\texttt{sigmoid}(V(x) - ρ) (\\text{out-of-RoA expression}) = 0``.

Note that a hard transition, which only enforces the in-RoA equation when ``V(x) ≤ ρ`` and
the out-of-RoA equation when ``V(x) > ρ`` can be provided by a `sigmoid` which is exactly
one when the input is nonnegative and exactly zero when the input is negative.

If the dynamics truly have a fixed point at ``x_0`` and ``V̇(x)`` is truly the rate of
decrease of ``V(x)`` along the dynamics, then ``V̇(x_0)`` will be ``0`` and there is no need
to train for ``V̇(x_0) = 0``. So, if `check_fixed_point_gradient` is `true`, then training
will also attempt to enforce the local maximality of the fixed point via ``∇V̇(x_0) = 0``.

# Examples:

Asymptotic decrease can be enforced by requiring
    ``V̇(x) ≤ -C \\lVert x - x_0 \\rVert^2``,
for some positive ``C``, which corresponds to

    rate_metric = (V, dVdt) -> dVdt
    strength = (x, x0) -> C * (x - x0) ⋅ (x - x0)

Exponential decrease of rate ``k`` is proven by
    ``V̇(x) ≤ - k V(x)``,
which corresponds to

    rate_metric = (V, dVdt) -> dVdt + k * V
    strength = (x, x0) -> 0.0

Enforcing either condition only in the region of attraction and not penalizing any points
outside that region would correspond to

    out_of_RoA_penalty = (V, dVdt, state, fixed_point, ρ) -> 0.0

whereas an example of a penalty that decays farther in state space from the fixed point is

    out_of_RoA_penalty = (V, dVdt, x, x0, ρ) -> 1.0 / ((x - x0) ⋅ (x - x0))

Note that this penalty could also depend on values of ``V`` and ``V̇`` at various points, as
well as ``ρ``.

In any of these cases, the rectified linear unit `rectifier = (t) -> max(zero(t), t)`
exactly represents the inequality, but differentiable approximations of this function may be
employed.

See also: [`LyapunovDecreaseCondition`](@ref)
"""
struct RoAAwareDecreaseCondition <: AbstractLyapunovDecreaseCondition
    check_decrease::Bool
    rate_metric::Function
    strength::Function
    rectifier::Function
    sigmoid::Function
    ρ::Real
    out_of_RoA_penalty::Function
    check_fixed_point_gradient::Bool
end

function check_decrease(cond::RoAAwareDecreaseCondition)::Bool
    cond.check_decrease
end

function check_maximal_fixed_point(cond::RoAAwareDecreaseCondition)::Bool
    cond.check_fixed_point_gradient
end

function get_decrease_condition(cond::RoAAwareDecreaseCondition)
    if cond.check_decrease
        return function (V, dVdt, x, fixed_point)
            _V = V(x)
            _V = _V isa AbstractVector ? _V[] : _V
            _V̇ = dVdt(x)
            _V̇ = _V̇ isa AbstractVector ? _V̇[] : _V̇
            return cond.sigmoid(cond.ρ - _V) * cond.rectifier(
                cond.rate_metric(_V, _V̇) + cond.strength(x, fixed_point)
            ) +
                   cond.sigmoid(_V - cond.ρ) *
                   cond.out_of_RoA_penalty(_V, _V̇, x, fixed_point, cond.ρ)
        end
    else
        return nothing
    end
end

"""
    make_RoA_aware(cond; ρ, out_of_RoA_penalty, sigmoid)

Add awareness of the region of attraction (RoA) estimation task to the supplied
[`LyapunovDecreaseCondition`](@ref).

When estimating the region of attraction using a Lyapunov function, the decrease condition
only needs to be met within a bounded sublevel set ``\\{ x : V(x) ≤ ρ \\}``.
The returned [`RoAAwareDecreaseCondition`](@ref) enforces the decrease condition represented
by `cond` only in that sublevel set.

# Arguments
  - `cond::LyapunovDecreaseCondition`: specifies the loss to be applied when ``V(x) ≤ ρ``.

# Keyword Arguments
  - `ρ`: the target level such that the RoA will be ``\\{ x : V(x) ≤ ρ \\}``, defaults to 1.
  - `out_of_RoA_penalty::Function`: specifies the loss to be applied when ``V(x) > ρ``,
    defaults to no loss.
  - `sigmoid::Function`: approximately one when the input is positive and approximately zero
    when the input is negative, defaults to unit step function.

The loss applied to samples ``x`` such that ``V(x) > ρ`` is
``\\lvert \\texttt{out\\_of\\_RoA\\_penalty}(V(x), V̇(x), x, x_0, ρ) \\rvert^2``.

The `sigmoid` function allows for a smooth transition between the ``V(x) ≤ ρ`` case and the
``V(x) > ρ`` case, by combining the above equations into one:

``\\texttt{sigmoid}(ρ - V(x)) (\\text{in-RoA expression}) + \\texttt{sigmoid}(V(x) - ρ) (\\text{out-of-RoA expression}) = 0``.

Note that a hard transition, which only enforces the in-RoA equation when ``V(x) ≤ ρ`` and
the out-of-RoA equation when ``V(x) > ρ`` can be provided by a `sigmoid` which is exactly
one when the input is nonnegative and exactly zero when the input is negative.
As such, the default value is `sigmoid(t) = t ≥ zero(t)`.

See also: [`RoAAwareDecreaseCondition`](@ref)
"""
function make_RoA_aware(
        cond::LyapunovDecreaseCondition;
        ρ = 1.0,
        out_of_RoA_penalty = (V, dVdt, state, fixed_point, _ρ) -> 0.0,
        sigmoid = (x) -> x .≥ zero.(x)
)::RoAAwareDecreaseCondition
    RoAAwareDecreaseCondition(
        cond.check_decrease,
        cond.rate_metric,
        cond.strength,
        cond.rectifier,
        sigmoid,
        ρ,
        out_of_RoA_penalty,
        cond.check_fixed_point_gradient
    )
end
