"""
    NeuralLyapunovStructure

Specifies the structure of the neural Lyapunov function and its derivative.

Allows the user to define the Lyapunov in terms of the neural network to 
structurally enforce Lyapunov conditions. 
network_dim is the dimension of the output of the neural network.
V(phi::Function, state, fixed_point) takes in the neural network, the state,
and the fixed_point, and outputs the value of the Lyapunov function at state
V̇(phi::Function, J_phi::Function, f::Function, state, fixed_point) takes in the
neural network, the jacobian of the neural network, the dynamics (as a function
of the state alone), the state, and the fixed_point, and outputs the time 
derivative of the Lyapunov function at state.
"""
struct NeuralLyapunovStructure
    V::Function
    ∇V::Function
    V̇::Function
    network_dim::Integer
end

"""
    UnstructuredNeuralLyapunov()

Creates a NeuralLyapunovStructure where the Lyapunov function is the neural
network evaluated at state. This does not structurally enforce any Lyapunov 
conditions.
"""
function UnstructuredNeuralLyapunov()::NeuralLyapunovStructure
    NeuralLyapunovStructure(
        (net, state, fixed_point) -> net(state), 
        (net, grad_net, state, fixed_point) -> grad_net(state),
        (net, grad_net, f, state, fixed_point) -> grad_net(state) ⋅ f(state),
        1
        )
end

"""
    NonnegativeNeuralLyapunov(network_dim, δ, pos_def; grad_pos_def, grad)

Creates a NeuralLyapunovStructure where the Lyapunov function is the L2 norm of
the neural network output plus a constant δ times a function pos_def.

The condition that the Lyapunov function must be minimized uniquely at the
fixed point can be represented as V(fixed_point) = 0, V(state) > 0 when 
state != fixed_point. This structure ensures V(state) ≥ 0. Further, if δ > 0
and pos_def(fixed_point, fixed_point) = 0, but pos_def(state, fixed_point) > 0 
when state != fixed_point, this ensures that V(state) > 0 when 
state != fixed_point. This does not enforce V(fixed_point) = 0, so that 
condition must included in the neural Lyapunov loss function.

grad_pos_def(state, fixed_point) should be the gradient of pos_def with respect
to state at state. If grad_pos_def is not defined, it is evaluated using grad,
which defaults to ForwardDiff.gradient.

The neural network output has dimension network_dim.
"""
function NonnegativeNeuralLyapunov(
    network_dim::Integer;
    δ::Real = 0.0, 
    pos_def::Function = (state, fixed_point) -> log(1.0 + (state - fixed_point) ⋅ (state - fixed_point)),
    grad_pos_def = nothing,
    grad = ForwardDiff.gradient,
    )::NeuralLyapunovStructure
    if δ == 0.0
        NeuralLyapunovStructure(
            (net, state, fixed_point) -> net(state) ⋅ net(state), 
            (net, J_net, state, fixed_point) -> 2 * transpose(net(state)) * J_net(state),
            (net, J_net, f, state, fixed_point) -> 2 * dot(net(state), J_net(state), f(state)),
            network_dim
            )
    else
        grad_pos_def = if isnothing(grad_pos_def)
            (state, fixed_point) -> grad((x) -> pos_def(x, fixed_point), state)
        else
            grad_pos_def
        end
        NeuralLyapunovStructure(
            (net, state, fixed_point) -> net(state) ⋅ net(state) + δ * pos_def(state, fixed_point), 
            (net, J_net, state, fixed_point) -> 2 * transpose(net(state)) * J_net(state) + 
                δ * grad_pos_def(state, fixed_point),
            (net, J_net, f, state, fixed_point) -> 2 * dot(
                net(state), 
                J_net(state), 
                f(state)
                ) + δ * grad_pos_def(state, fixed_point) ⋅ f(state),
            network_dim
            )
    end
end

"""
    PositiveSemiDefiniteStructure(network_dim; pos_def, non_neg, grad_pos_def, grad_non_neg, grad)

Creates a NeuralLyapunovStructure where the Lyapunov function is the product of
a positive (semi-)definite function pos_def which does not depend on the 
network and a nonnegative function non_neg which does depend the network.

The condition that the Lyapunov function must be minimized uniquely at the
fixed point can be represented as V(fixed_point) = 0, V(state) > 0 when 
state != fixed_point. This structure ensures V(state) ≥ 0. Further, if pos_def
is 0 only at fixed_point (and positive elsewhere) and if non_neg is strictly 
positive away from fixed_point (as is the case for the default values of pos_def
and non_neg), then this structure ensures V(fixed_point) = 0 and V(state) > 0 
when state != fixed_point.

grad_pos_def(state, fixed_point) should be the gradient of pos_def with respect
to state at state. Similarly, grad_non_neg(net, J_net, state, fixed_point) 
should be the gradient of non_neg(net, state, fixed_point) with respect to 
state at state. If grad_pos_def or grad_non_neg is not defined, it is evaluated 
using grad, which defaults to ForwardDiff.gradient. 

The neural network output has dimension network_dim.
"""
function PositiveSemiDefiniteStructure(
        network_dim::Integer;
        pos_def::Function = (state, fixed_point) -> log(1.0 + (state - fixed_point) ⋅ (state - fixed_point)),
        non_neg::Function = (net, state, fixed_point) -> 1 + net(state) ⋅ net(state),
        grad_pos_def = nothing,
        grad_non_neg = nothing,
        grad = ForwardDiff.gradient,
    )
    _grad(f::Function, x::AbstractArray{T}) where T<:Num = Symbolics.gradient(f(x), x)
    _grad(f::Function, x) = grad(f, x)
    grad_pos_def = if isnothing(grad_pos_def)
        (state, fixed_point) -> _grad((x) -> pos_def(x, fixed_point), state)
    else
        grad_pos_def
    end
    grad_non_neg = if isnothing(grad_non_neg)
        (net, J_net, state, fixed_point) -> _grad((x) -> non_neg(net, x, fixed_point), state)
    else
        grad_non_neg
    end
    NeuralLyapunovStructure(
        (net, state, fixed_point) -> pos_def(state, fixed_point) * non_neg(net, state, fixed_point),
        (net, J_net, state, fixed_point) -> 
            grad_pos_def(state, fixed_point) * non_neg(net, state, fixed_point) + 
            pos_def(state, fixed_point) * grad_non_neg(net, J_net, state, fixed_point),
        (net, J_net, f, state, fixed_point) -> 
            (f(state) ⋅ grad_pos_def(state, fixed_point)) * non_neg(net, state, fixed_point) + 
            pos_def(state, fixed_point) * (f(state) ⋅ grad_non_neg(net, J_net, state, fixed_point)),
        network_dim
    )
end

abstract type AbstractLyapunovMinimizationCondition end

"""
check_nonnegativity(cond::AbstractLyapunovMinimizationCondition)

True if cond specifies training to meet the Lyapunov minimization condition, 
false if cond specifies no training to meet this condition.
"""
function check_nonnegativity(cond::AbstractLyapunovMinimizationCondition)::Bool
    error("check_nonnegativity not implemented for AbstractLyapunovMinimizationCondition of type $(typeof(cond))")
end

"""
    check_fixed_point(cond::AbstractLyapunovMinimizationCondition)

True if cond specifies training for the Lyapunov function to equal zero at the
fixed point, false if cond specifies no training to meet this condition.
"""
function check_fixed_point(cond::AbstractLyapunovMinimizationCondition)::Bool
    error("check_fixed_point not implemented for AbstractLyapunovMinimizationCondition of type $(typeof(cond))")
end

"""
    get_minimization_condition(cond::AbstractLyapunovMinimizationCondition)

Returns a function of V, state, and fixed_point that equals zero when the 
Lyapunov minimization condition is met and greater than zero when it's violated
"""
function get_minimization_condition(cond::AbstractLyapunovMinimizationCondition)
    error("get_condition not implemented for AbstractLyapunovMinimizationCondition of type $(typeof(cond))")
end

"""
    LyapunovMinimizationCondition

Specifies the form of the Lyapunov conditions to be used.

If check_nonnegativity is true, training will attempt to enforce
    V(state) ≥ strength(state, fixed_point)
The inequality will be approximated by the equation
    relu(strength(state, fixed_point) - V(state)) = 0.0
If check_fixed_point is true, then training will attempt to enforce 
    V(fixed_point) = 0

# Examples

The condition that the Lyapunov function must be minimized uniquely at the
fixed point can be represented as V(fixed_point) = 0, V(state) > 0 when 
state != fixed_point. This could be enfored by V ≥ ||state - fixed_point||^2,
which would be represented, with check_nonnegativity = true, by
    strength(state, fixed_point) = ||state - fixed_point||^2,
paired with V(fixed_point) = 0, which can be enforced with 
    check_fixed_point = true

If V were structured such that it is always nonnegative, then V(fixed_point) = 0
is all that must be enforced in training for the Lyapunov function to be 
uniquely minimized at fixed_point. So, in that case, we would use 
    check_nonnegativity = false;  check_fixed_point = true

In either case, relu = (t) -> max(0.0, t) exactly represents the inequality, 
but approximations of this function are allowed.
"""
struct LyapunovMinimizationCondition <: AbstractLyapunovMinimizationCondition
    check_nonnegativity::Bool
    strength::Function
    relu::Function
    check_fixed_point::Bool
end

function check_nonnegativity(cond::LyapunovMinimizationCondition)::Bool
    cond.check_nonnegativity
end

function check_fixed_point(cond::LyapunovMinimizationCondition)::Bool
    cond.check_fixed_point
end

function get_minimization_condition(cond::LyapunovMinimizationCondition)
    if cond.check_nonnegativity
        return (V, x, fixed_point) -> cond.relu(cond.strength(x, fixed_point) - V(x))
    else
        return nothing
    end
end

"""
    StrictlyPositiveDefinite(C; check_fixed_point, relu)

Constructs a LyapunovMinimizationCondition representing 
    V(state) ≥ C * ||state - fixed_point||^2
If check_fixed_point is true, then training will also attempt to enforce 
    V(fixed_point) = 0

The inequality is represented by a ≥ b <==> relu(b-a) = 0.0
"""
function StrictlyPositiveDefinite(; 
    check_fixed_point = true, 
    C::Real = 1e-6, 
    relu = (t) -> max(0.0, t)
    )::LyapunovMinimizationCondition
    LyapunovMinimizationCondition(
        true,
        (state, fixed_point) -> C * (state - fixed_point) ⋅ (state - fixed_point),
        relu,
        check_fixed_point
    )
end

"""
    PositiveSemiDefinite(check_fixed_point)

Constructs a LyapunovMinimizationCondition representing 
    V(state) ≥ 0
If check_fixed_point is true, then training will also attempt to enforce 
    V(fixed_point) = 0

The inequality is represented by a ≥ b <==> relu(b-a) = 0.0
"""
function PositiveSemiDefinite(; 
    check_fixed_point = true, 
    relu = (t) -> max(0.0, t)
    )::LyapunovMinimizationCondition
    LyapunovMinimizationCondition(
        true,
        (state, fixed_point) -> 0.0,
        relu,
        check_fixed_point
    )
end

"""
    DontCheckNonnegativity(check_fixed_point)

Constructs a LyapunovMinimizationCondition which represents not checking for 
nonnegativity of the Lyapunov function. This is appropriate in cases where this
condition has been structurally enforced.

It is still possible to check for V(fixed_point) = 0, even in this case, for
example if V is structured to be positive for state != fixed_point, but it is
not guaranteed structurally that V(fixed_point) = 0.
"""
function DontCheckNonnegativity(;check_fixed_point = false)::LyapunovMinimizationCondition
    LyapunovMinimizationCondition(
        false,
        (state, fixed_point) -> 0.0,
        (t) -> 0.0,
        check_fixed_point
    )    
end

abstract type AbstractLyapunovDecreaseCondition end

"""
    check_decrease(cond::AbstractLyapunovDecreaseCondition)

True if cond specifies training to meet the Lyapunov decrease condition, false
if cond specifies no training to meet this condition.
"""
function check_decrease(cond::AbstractLyapunovDecreaseCondition)::Bool
    error("check_decrease not implemented for AbstractLyapunovDecreaseCondition of type $(typeof(cond))")
end

"""
    check_stationary_fixed_point(cond::AbstractLyapunovDecreaseCondition)

True if cond specifies training for the Lyapunov function not to change at the 
fixed point, false if cond specifies no training to meet this condition.
"""
function check_stationary_fixed_point(cond::AbstractLyapunovDecreaseCondition)::Bool
    error("check_fixed_point not implemented for AbstractLyapunovDecreaseCondition of type $(typeof(cond))")
end

"""
    get_decrease_condition(cond::AbstractLyapunovDecreaseCondition)

Returns a function of V, dVdt, state, and fixed_point that is equal to zero
when the Lyapunov decrease condition is met and greater than zero when it is 
violated
"""
function get_decrease_condition(cond::AbstractLyapunovDecreaseCondition)
    error("get_condition not implemented for AbstractLyapunovDecreaseCondition of type $(typeof(cond))")
end

"""
LyapunovDecreaseCondition(decrease, strength, check_fixed_point)

Specifies the form of the Lyapunov conditions to be used; training will enforce
    decrease(V, dVdt) ≤ strength(state, fixed_point)
The inequality will be approximated by the equation
    relu(decrease(V, dVdt) - strength(state, fixed_point)) = 0.0
If check_fixed_point is false, then training assumes dVdt(fixed_point) = 0, but
if check_fixed_point is true, then training will enforce dVdt(fixed_point) = 0.

If the dynamics truly have a fixed point at fixed_point and dVdt has been 
defined properly in terms of the dynamics, then dVdt(fixed_point) will be 0 and
there is no need to enforce dVdt(fixed_point) = 0, so check_fixed_point defaults
to false.

# Examples:

Asymptotic decrease can be enforced by requiring
    dVdt ≤ -C |state - fixed_point|^2,
which corresponds to
    decrease = (V, dVdt) -> dVdt
    strength = (x, x0) -> -C * (x - x0) ⋅ (x - x0)

Exponential decrease of rate k is proven by dVdt ≤ - k * V, so corresponds to
    decrease = (V, dVdt) -> dVdt + k * V
    strength = (x, x0) -> 0.0
"""
struct LyapunovDecreaseCondition <: AbstractLyapunovDecreaseCondition
    check_decrease::Bool
    decrease::Function
    strength::Function
    relu::Function
    check_fixed_point::Bool
end

function check_decrease(cond::LyapunovDecreaseCondition)::Bool
    cond.check_decrease
end

function check_stationary_fixed_point(cond::LyapunovDecreaseCondition)::Bool
    cond.check_fixed_point
end

function get_decrease_condition(cond::LyapunovDecreaseCondition)
    if cond.check_decrease
        return (V, dVdt, x, fixed_point) -> cond.relu(
            cond.decrease(V(x), dVdt(x)) - cond.strength(x, fixed_point)
            )
    else
        return nothing
    end
end

"""
    AsymptoticDecrease(strict; check_fixed_point, C)

Constructs a LyapunovDecreaseCondition corresponding to asymptotic decrease.

If strict is false, the condition is dV/dt ≤ 0, and if strict is true, the 
condition is dV/dt ≤ - C | state - fixed_point |^2

The inequality is represented by a ≥ b <==> relu(b-a) = 0.0
"""
function AsymptoticDecrease(;
    strict::Bool = false, 
    check_fixed_point::Bool = false, 
    C::Real = 1e-6, 
    relu = (t) -> max(0.0, t)
    )::LyapunovDecreaseCondition
    if strict
        return LyapunovDecreaseCondition(
            true,
            (V, dVdt) -> dVdt, 
            (x, x0) -> -C * (x - x0) ⋅ (x - x0),
            relu,
            check_fixed_point
            )
    else
        return LyapunovDecreaseCondition(
            true,
            (V, dVdt) -> dVdt, 
            (x, x0) -> 0.0,
            relu,
            check_fixed_point
            )
    end
end

"""
    ExponentialDecrease(k, strict; check_fixed_point, C)

Constructs a LyapunovDecreaseCondition corresponding to exponential decrease of rate k.

If strict is false, the condition is dV/dt ≤ -k * V, and if strict is true, the 
condition is dV/dt ≤ -k * V - C * ||state - fixed_point||^2

The inequality is represented by a ≥ b <==> relu(b-a) = 0.0
"""
function ExponentialDecrease(
    k::Real; 
    strict::Bool = false, 
    check_fixed_point::Bool = false, 
    C::Real = 1e-6, 
    relu = (t) -> max(0.0, t)
    )::LyapunovDecreaseCondition
    if strict
        return LyapunovDecreaseCondition(
            true,
            (V, dVdt) -> dVdt + k * V, 
            (x, x0) -> -C * (x - x0) ⋅ (x - x0),
            relu,
            check_fixed_point
            )
    else
        return LyapunovDecreaseCondition(
            true,
            (V, dVdt) -> dVdt + k * V, 
            (x, x0) -> 0.0,
            relu,
            check_fixed_point
            )
    end
end

"""
    DontCheckDecrease(check_fixed_point = false)

Constructs a LyapunovDecreaseCondition which represents not checking for 
decrease of the Lyapunov function along system trajectories. This is appropriate
in cases when the Lyapunov decrease condition has been structurally enforced.

It is still possible to check for dV/dt = 0 at fixed_point, even in this case.
"""
function DontCheckDecrease(check_fixed_point::Bool = false)::LyapunovDecreaseCondition
    return LyapunovDecreaseCondition(
        false,
        (V, dVdt) -> 0.0,
        (state, fixed_point) -> 0.0,
        (t) -> 0.0,
        check_fixed_point
    )
end

struct NeuralLyapunovSpecification
    structure::NeuralLyapunovStructure
    minimzation_condition::AbstractLyapunovMinimizationCondition
    decrease_condition::AbstractLyapunovDecreaseCondition
end