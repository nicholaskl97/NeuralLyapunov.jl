"""
    LyapunovMinimizationCondition(check_nonnegativity, strength, rectifier, check_fixed_point)

Specifies the form of the Lyapunov minimization condition to be used.

# Fields
  - `check_nonnegativity::Bool`: whether or not to train for positivity/nonnegativity of
    ``V(x)``
  - `strength::Function`: specifies the level of strictness for positivity training; should
    be zero when the two inputs are equal and nonnegative otherwise; used when
    `check_nonnegativity == true`
  - `rectifier::Function`: positive when the input is positive and (approximately) zero when
    the input is negative
  - `check_fixed_point`: whether or not to train for ``V(x_0) = 0``.
  - `check_fixed_point_gradient`: whether or not to train for ``∇V(x_0) = 0``.

# Training conditions

If `check_nonnegativity` is `true`, training will attempt to enforce:

``V(x) ≥ \\texttt{strength}(x, x_0).``

The inequality will be approximated by the equation:

``\\texttt{rectifier}(\\texttt{strength}(x, x_0) - V(x_0)) = 0.``

Note that the approximate equation and inequality are identical when
``\\texttt{rectifier}(t) = \\max(0, t)``.

If `check_fixed_point` is `true`, then training will also attempt to enforce
``V(x_0) = 0``.

If `check_fixed_point_gradient` is `true`, then training will also attempt to enforce
``∇V(x_0) = 0``.

# Examples

When training for a strictly positive definite ``V``, an example of an appropriate `strength`
is ``\\texttt{strength}(x, x_0) = \\lVert x - x_0 \\rVert^2``.
This form is used in [`StrictlyPositiveDefinite`](@ref).

If ``V`` were structured such that it is always nonnegative, then ``V(x_0) = 0`` is all
that must be enforced in training for the Lyapunov function to be uniquely minimized at
``x_0``. In that case, we would use
    `check_nonnegativity = false;  check_fixed_point = true`.
This can also be accomplished with [`DontCheckNonnegativity(true)`](@ref).

In either case, the rectified linear unit `rectifier(t) = max(zero(t), t)` exactly
represents the inequality, but differentiable approximations of this function may be
employed.
"""
struct LyapunovMinimizationCondition <: AbstractLyapunovMinimizationCondition
    check_nonnegativity::Bool
    strength::Function
    rectifier::Function
    check_fixed_point::Bool
    check_fixed_point_gradient::Bool
end

function check_nonnegativity(cond::LyapunovMinimizationCondition)::Bool
    cond.check_nonnegativity
end

function check_zero_fixed_point(cond::LyapunovMinimizationCondition)::Bool
    cond.check_fixed_point
end

function check_minimal_fixed_point(cond::LyapunovMinimizationCondition)::Bool
    cond.check_fixed_point_gradient
end

function get_minimization_condition(cond::LyapunovMinimizationCondition)
    if cond.check_nonnegativity
        return function (V, x, fixed_point)
            _V = V(x)
            _V = _V isa AbstractVector ? _V[] : _V
            return cond.rectifier(cond.strength(x, fixed_point) - _V)
        end
    else
        return nothing
    end
end

"""
    StrictlyPositiveDefinite(; C, check_fixed_point, check_fixed_point_gradient, rectifier)

Construct a [`LyapunovMinimizationCondition`](@ref) representing
    ``V(x) ≥ C \\lVert x - x_0 \\rVert^2``.
If `check_fixed_point == true` (as is the default), then training will also attempt to
enforce ``V(x_0) = 0``. Similarly, if `check_fixed_point_gradient == true`, (as is the
default) then training will also attempt to enforce ``∇V(x_0) = 0``.

The inequality is approximated by
    ``\\texttt{rectifier}(C \\lVert x - x_0 \\rVert^2 - V(x)) = 0``,
and the default `rectifier` is the rectified linear unit `(t) -> max(zero(t), t)`, which
exactly represents ``V(x) ≥ C \\lVert x - x_0 \\rVert^2``. ``C`` defaults to `1e-6`.
"""
function StrictlyPositiveDefinite(;
        check_fixed_point = true,
        check_fixed_point_gradient = true,
        C::Real = 1e-6,
        rectifier = (t) -> max(zero(t), t)
)::LyapunovMinimizationCondition
    LyapunovMinimizationCondition(
        true,
        (state, fixed_point) -> C * (state - fixed_point) ⋅ (state - fixed_point),
        rectifier,
        check_fixed_point,
        check_fixed_point_gradient
    )
end

"""
    PositiveSemiDefinite(; check_fixed_point, check_fixed_point_gradient, rectifier)

Construct a [`LyapunovMinimizationCondition`](@ref) representing ``V(x) ≥ 0``.
If `check_fixed_point == true` (as is the default), then training will also attempt to
enforce ``V(x_0) = 0``. Similarly, if `check_fixed_point_gradient == true` (as is the
default), then training will also attempt to enforce ``∇V(x_0) = 0``.

The inequality is approximated by ``\\texttt{rectifier}( -V(x) ) = 0`` and the default
`rectifier` is the rectified linear unit `(t) -> max(zero(t), t)`, which exactly represents
``V(x) ≥ 0``.
"""
function PositiveSemiDefinite(;
        check_fixed_point = true,
        check_fixed_point_gradient = true,
        rectifier = (t) -> max(zero(t), t)
)::LyapunovMinimizationCondition
    LyapunovMinimizationCondition(
        true,
        (state, fixed_point) -> 0.0,
        rectifier,
        check_fixed_point,
        check_fixed_point_gradient
    )
end

"""
    DontCheckNonnegativity(; check_fixed_point, check_fixed_point_gradient)

Construct a [`LyapunovMinimizationCondition`](@ref) which represents not checking for
nonnegativity of the Lyapunov function. This is appropriate in cases where this condition
has been structurally enforced.

Even in this case, it is still possible to check for ``V(x_0) = 0`` and ``∇V(x_0) = 0``, for
example if `V` is structured to be positive for ``x ≠ x_0`` but does not guarantee
``V(x_0) = 0`` (such as [`NonnegativeNeuralLyapunov`](@ref)). Both `check_fixed_point` and
`check_fixed_point_gradient` default to `true`, since in cases where ``V(x_0) = 0`` and
``∇V(x_0) = 0`` is enforced structurally, the equation will reduce to `0.0 ~ 0.0`, which
gets automatically removed in most cases.
"""
function DontCheckNonnegativity(;
    check_fixed_point = true,
    check_fixed_point_gradient = true
)::LyapunovMinimizationCondition
    LyapunovMinimizationCondition(
        false,
        (state, fixed_point) -> 0.0,
        (t) -> zero(t),
        check_fixed_point,
        check_fixed_point_gradient
    )
end
