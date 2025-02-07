"""
    NeuralLyapunovStructure(V, V̇, f_call, network_dim)

Specifies the structure of the neural Lyapunov function and its derivative.

Allows the user to define the Lyapunov in terms of the neural network, potentially
structurally enforcing some Lyapunov conditions.

# Fields
  - `V(phi::Function, state, fixed_point)`: outputs the value of the Lyapunov function at
    `state`.
  - `V̇(phi::Function, J_phi::Function, dynamics::Function, state, params, t, fixed_point)`:
    outputs the time derivative of the Lyapunov function at `state`.
  - `f_call(dynamics::Function, phi::Function, state, params, t)`: outputs the derivative of
    the state; this is useful for making closed-loop dynamics which depend on the neural
    network, such as in the policy search case.
  - `network_dim`: the dimension of the output of the neural network.

`phi` and `J_phi` above are both functions of `state` alone.
"""
struct NeuralLyapunovStructure
    V::Function
    V̇::Function
    f_call::Function
    network_dim::Integer
end

"""
    AbstractLyapunovMinimizationCondition

Represents the minimization condition in a neural Lyapunov problem

All concrete `AbstractLyapunovMinimizationCondition` subtypes should define the
`check_nonnegativity`, `check_fixed_point`, and `get_minimization_condition` functions.
"""
abstract type AbstractLyapunovMinimizationCondition end

"""
    AbstractLyapunovDecreaseCondition

Represents the decrease condition in a neural Lyapunov problem

All concrete `AbstractLyapunovDecreaseCondition` subtypes should define the
`check_decrease` and `get_decrease_condition` functions.
"""
abstract type AbstractLyapunovDecreaseCondition end

"""
    NeuralLyapunovSpecification(structure, minimzation_condition, decrease_condition)

Specifies a neural Lyapunov problem.

# Fields
  - `structure`: a [`NeuralLyapunovStructure`](@ref) specifying the relationship between the
    neural network and the candidate Lyapunov function.
  - `minimzation_condition`: an [`AbstractLyapunovMinimizationCondition`](@ref) specifying
    how the minimization condition will be enforced.
  - `decrease_condition`: an [`AbstractLyapunovDecreaseCondition`](@ref) specifying how the
    decrease condition will be enforced.
"""
struct NeuralLyapunovSpecification
    structure::NeuralLyapunovStructure
    minimization_condition::AbstractLyapunovMinimizationCondition
    decrease_condition::AbstractLyapunovDecreaseCondition
end

"""
    check_nonnegativity(cond::AbstractLyapunovMinimizationCondition)

Return `true` if `cond` specifies training to meet the Lyapunov minimization condition, and
`false` if `cond` specifies no training to meet this condition.
"""
function check_nonnegativity(cond::AbstractLyapunovMinimizationCondition)::Bool
    error("check_nonnegativity not implemented for AbstractLyapunovMinimizationCondition " *
          "of type $(typeof(cond))")
end

"""
    check_zero_fixed_point(cond::AbstractLyapunovMinimizationCondition)

Return `true` if `cond` specifies training for the Lyapunov function to equal zero at the
fixed point, and `false` if `cond` specifies no training to meet this condition.
"""
function check_zero_fixed_point(cond::AbstractLyapunovMinimizationCondition)::Bool
    error("check_zero_fixed_point not implemented for " *
          "AbstractLyapunovMinimizationCondition of type $(typeof(cond))")
end

"""
    check_minimal_fixed_point(cond::AbstractLyapunovMinimizationCondition)

Return `true` if `cond` specifies training for the Lyapunov function to have a local minimum
at the fixed point, and `false` if `cond` specifies no training to meet this condition.
"""
function check_minimal_fixed_point(cond::AbstractLyapunovMinimizationCondition)::Bool
    error("check_minimal_fixed_point not implemented for " *
          "AbstractLyapunovMinimizationCondition of type $(typeof(cond))")
end

"""
    get_minimization_condition(cond::AbstractLyapunovMinimizationCondition)

Return a function of ``V``, ``x``, and ``x_0`` that equals zero when the Lyapunov
minimization condition is met for the Lyapunov candidate function ``V`` at the point ``x``,
and is greater than zero if it's violated.

Note that the first input, ``V``, is a function, so the minimization condition can depend on
the value of the candidate Lyapunov function at multiple points.
"""
function get_minimization_condition(cond::AbstractLyapunovMinimizationCondition)
    error("get_minimization_condition not implemented for " *
          "AbstractLyapunovMinimizationCondition of type $(typeof(cond))")
end

"""
    check_decrease(cond::AbstractLyapunovDecreaseCondition)

Return `true` if `cond` specifies training to meet the Lyapunov decrease condition, and
`false` if `cond` specifies no training to meet this condition.
"""
function check_decrease(cond::AbstractLyapunovDecreaseCondition)::Bool
    error("check_decrease not implemented for AbstractLyapunovDecreaseCondition of type " *
          string(typeof(cond)))
end

"""
    check_maximal_fixed_point(cond::AbstractLyapunovMinimizationCondition)

Return `true` if `cond` specifies training for the Lyapunov decrease function to have a
local maximum at the fixed point, and `false` if `cond` specifies no training to meet this
condition.
"""
function check_maximal_fixed_point(cond::AbstractLyapunovDecreaseCondition)::Bool
    error("check_maximal_fixed_point not implemented for " *
          "AbstractLyapunovDecreaseCondition of type $(typeof(cond))")
end

"""
    get_decrease_condition(cond::AbstractLyapunovDecreaseCondition)

Return a function of ``V``, ``V̇``, ``x``, and ``x_0`` that returns zero when the Lyapunov
decrease condition is met and a value greater than zero when it is violated.

Note that the first two inputs, ``V`` and ``V̇``, are functions, so the decrease condition
can depend on the value of these functions at multiple points.
"""
function get_decrease_condition(cond::AbstractLyapunovDecreaseCondition)
    error("get_decrease_condition not implemented for AbstractLyapunovDecreaseCondition " *
          "of type $(typeof(cond))")
end
