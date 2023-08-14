"""
    NeuralLyapunovPDESystem(dynamics, lb, ub, spec; fixed_point, ps)

Constructs a ModelingToolkit PDESystem to train a neural Lyapunov function

Returns the PDESystem and a function representing the neural network, which
operates columnwise.

The neural Lyapunov function will only be trained for { x : lb .≤ x .≤ ub }.
The Lyapunov function will be for the dynamical system represented by dynamics
If dynamics is an ODEProblem or ODEFunction, then the corresponding ODE; if 
dynamics is a function, then the ODE is ẋ = dynamics(x, p, t). This ODE should 
not depend on t (time t=0.0 alone will be used) and should have a fixed point 
at x = fixed_point. The particular Lyapunov conditions to be used and structure
of the neural Lyapunov function are specified through spec, which is a 
NeuralLyapunovSpecification.

The returned neural network function takes three inputs: the neural network 
structure phi, the trained network parameters, and a matrix of inputs to 
operate on columnwise.

If dynamics requires parameters, their values can be supplied through the 
Vector p, or through dynamics.p if dynamics isa ODEProblem (in which case, let
the other be SciMLBase.NullParameters()). If dynamics is an ODEFunction and 
dynamics.paramsyms is defined, then p should have the same order.
"""
function NeuralLyapunovPDESystem(
    dynamics::ODEFunction,
    lb,
    ub,
    spec::NeuralLyapunovSpecification;
    fixed_point = zeros(length(lb)),
    p = SciMLBase.NullParameters(),
)::Tuple{PDESystem,Function}
    if dynamics.mass_matrix !== I
        throw(ErrorException("DAEs are not supported at this time"))
    end

    ########################## Unpack specifications ##########################
    structure = spec.structure
    minimzation_condition = spec.minimzation_condition
    decrease_condition = spec.decrease_condition

    ########################## Define state symbols ###########################
    state_dim = length(lb)

    # Define state symbols, if not already defined
    state_syms = if isnothing(dynamics.syms)
        [Symbol(:state, i) for i = 1:state_dim]  
    else
        dynamics.syms
    end
    state = [first(@parameters $s) for s in state_syms]

    # Define domains
    domains = [state[i] ∈ (lb[i], ub[i]) for i = 1:state_dim]

    ######################## Define parameter symbols #########################
    # Define parameter symbols, if not already defined
    param_syms = if p == SciMLBase.NullParameters()
        []
    else
        if isnothing(dynamics.paramsyms)
            [Symbol(:param, i) for i = 1:length(p)]
        else
            dynamics.paramsyms
        end
    end
    
    params = [first(@parameters $s) for s in param_syms]
    defaults = if p == SciMLBase.NullParameters()
        Dict()
    else
        Dict([param => param_val for (param, param_val) in zip(params, p)])
    end

    ################## Define Lyapunov function & derivative ##################
    output_dim = structure.network_dim
    net_syms = [Symbol(:u, i) for i = 1:output_dim]
    net = [first(@variables $s(..)) for s in net_syms]

    # u(x) is the symbolic form of neural network output
    u(x) = Num.([ui(x...) for ui in net])

    # V_sym(x) is the symobolic form of the Lyapunov function
    V_sym(x) = structure.V(u, x, fixed_point)

    # V̇_sym(x) is the symbolic time derivative of the Lyapunov function
    V̇_sym(x) = structure.V̇(
        u, 
        y -> Symbolics.jacobian(u(y), y), 
        y -> dynamics(y, params, 0.0), 
        x, 
        fixed_point
        )

    ################ Define equations and boundary conditions #################
    eqs = []

    if check_nonnegativity(minimzation_condition)
        cond = get_minimization_condition(minimzation_condition)
        push!(eqs, cond(V_sym, state, fixed_point) ~ 0.0)
    end

    if check_decrease(decrease_condition)
        cond = get_decrease_condition(decrease_condition)
        push!(eqs, cond(V_sym, V̇_sym, state, fixed_point) ~ 0.0)
    end

    bcs = [] 
    
    if check_fixed_point(minimzation_condition)
        push!(bcs, V_sym(fixed_point) ~ 0.0)
    end
    if check_stationary_fixed_point(decrease_condition)
        push!(bcs, V̇_sym(fixed_point) ~ 0.0)
    end

    if isempty(eqs) && isempty(bcs)
        error("No training conditions specified.")
    end
    
    # NeuralPDE requires an equation and a boundary condition, even if they are
    # trivial like 0.0 == 0.0
    if isempty(eqs)
        push!(eqs, 0.0 ~ 0.0)
    end
    if isempty(bcs)
        push!(bcs, 0.0 ~ 0.0)
    end

    ########################### Construct PDESystem ###########################
    @named lyapunov_pde_system = PDESystem(
        eqs, 
        bcs, 
        domains, 
        state, 
        u(state),
        params; 
        defaults = defaults
        )

    ################### Return PDESystem and neural network ###################
    # u_func is the numerical form of neural network output
    u_func(phi, θ, x) = reduce(
        vcat, 
        Array(phi[i](x, θ.depvar[net_syms[i]])) for i = 1:output_dim
        )

    return lyapunov_pde_system, u_func
end

function NeuralLyapunovPDESystem(
    dynamics::Function,
    lb,
    ub,
    spec::NeuralLyapunovSpecification;
    fixed_point = zeros(length(lb)),
    p = SciMLBase.NullParameters(),
)::Tuple{PDESystem,Function}
    return NeuralLyapunovPDESystem(
            ODEFunction(dynamics),
            lb,
            ub,
            spec;
            fixed_point = fixed_point,
            p = p
        )      
end

function NeuralLyapunovPDESystem(
    dynamics::ODEProblem,
    lb,
    ub,
    spec::NeuralLyapunovSpecification;
    fixed_point = zeros(length(lb)),
    p = SciMLBase.NullParameters(),
)::Tuple{PDESystem,Function}
    f = dynamics.f
    
    p = if dynamics.p == SciMLBase.NullParameters()
        p
    elseif p == SciMLBase.NullParameters()
        dynamics.p
    elseif dynamics.p == p
        p
    else
        throw(ErrorException("Conflicting parameter definitions. Please define parameters only through p or dynamics.p; the other should be SciMLBase.NullParameters()"))
    end

    return NeuralLyapunovPDESystem(
            f, 
            lb, 
            ub, 
            spec; 
            fixed_point = fixed_point, 
            p = p
        )
end

function NeuralLyapunovPDESystem(
    dynamics::ODESystem,
    lb,
    ub,
    spec::NeuralLyapunovSpecification;
    fixed_point = zeros(length(lb)),
    p = SciMLBase.NullParameters(),
)::Tuple{PDESystem,Function}
    p = if p == SciMLBase.NullParameters() && !isempty(dynamics.ps)
        [dynamics.defaults[param] for param in dynamics.ps]
    else
        p
    end
    
    NeuralLyapunovPDESystem(
        ODEFunction(dynamics),
        lb,
        ub,
        spec;
        fixed_point = fixed_point,
        p = p
    )
end

"""
    NumericalNeuralLyapunovFunctions(phi, θ, network_func, structure, dynamics, fixed_point; jac, J_net)

Returns the Lyapunov function, its time derivative, and its gradient: V(state), 
V̇(state), and ∇V(state)

These functions can operate on a state vector or columnwise on a matrix of state
vectors. phi is the neural network with parameters θ. network_func(phi, θ, state) 
is an output of NeuralLyapunovPDESystem, which evaluates the neural network 
represented phi with parameters θ at state.

The Lyapunov function structure is specified in structure, which is a 
NeuralLyapunovStructure. The Jacobian of the network is either specified via
J_net(_phi, _θ, state) or calculated using jac, which defaults to 
ForwardDiff.jacobian
"""
function NumericalNeuralLyapunovFunctions(
    phi,
    θ, 
    network_func::Function,
    structure::NeuralLyapunovStructure,
    dynamics::Function,
    fixed_point;
    p,
    jac = ForwardDiff.jacobian,
    J_net = (_phi, _θ, x) -> jac((y) -> network_func(_phi, _θ, y), x)
)::Tuple{Function, Function, Function}
    # Make Network function
    _net_func = (x) -> network_func(phi, θ, x)
    _J_net = (x) -> J_net(phi, θ, x)

    # Numerical form of Lyapunov function
    V_func(state::AbstractVector) = structure.V(_net_func, state, fixed_point)
    V_func(state::AbstractMatrix) = mapslices(V_func, state, dims = [1])

    # Numerical gradient of Lyapunov function
    ∇V_func(state::AbstractVector) = structure.∇V(
        _net_func, 
        _J_net, 
        state, 
        fixed_point
        )
    ∇V_func(state::AbstractMatrix) = mapslices(∇V_func, state, dims = [1])

    # Numerical time derivative of Lyapunov function
    V̇_func(state::AbstractVector) = structure.V̇(
        _net_func, 
        _J_net, 
        y -> dynamics(y, p, 0.0), 
        state, 
        fixed_point
        )
    V̇_func(state::AbstractMatrix) = mapslices(V̇_func, state, dims = [1])

    return V_func, V̇_func, ∇V_func
end

"""
    NumericalNeuralLyapunovFunctions(phi, θ, network_func, V_structure, dynamics, fixed_point, grad)

Returns the Lyapunov function, its time derivative, and its gradient: V(state), 
V̇(state), and ∇V(state)

These functions can operate on a state vector or columnwise on a matrix of state
vectors. phi is the neural network with parameters θ. network_func is an output
of NeuralLyapunovPDESystem.

The Lyapunov function structure is defined by 
    V_structure(_network_func, state, fixed_point)
Its gradient is calculated using grad, which defaults to ForwardDiff.gradient. 
"""
function NumericalNeuralLyapunovFunctions(
    phi,
    θ,
    network_func::Function,
    V_structure::Function,
    dynamics::Function,
    fixed_point;
    p = SciMLBase.NullParameters,
    grad = ForwardDiff.gradient,
)::Tuple{Function, Function, Function}
    # Make network function
    _net_func = (x) -> network_func(phi, θ, x)

    # Numerical form of Lyapunov function
    V_func(state::AbstractVector) = V_structure(_net_func, state, fixed_point)
    V_func(state::AbstractMatrix) = mapslices(V_func, state, dims = [1])

    # Numerical gradient of Lyapunov function
    ∇V_func(state::AbstractVector) = grad(V_func, state)
    ∇V_func(state::AbstractMatrix) = mapslices(∇V_func, state, dims = [1])

    # Numerical time derivative of Lyapunov function
    V̇_func(state::AbstractVector) = dynamics(state, p, 0.0) ⋅ ∇V_func(state)
    V̇_func(state::AbstractMatrix) = mapslices(V̇_func, state, dims = [1])

    return V_func, V̇_func, ∇V_func
end
