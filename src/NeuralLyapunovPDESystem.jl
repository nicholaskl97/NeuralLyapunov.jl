"""
    NeuralLyapunovPDESystem(dynamics::ODESystem, bounds, spec; <keyword_arguments>)
    NeuralLyapunovPDESystem(dynamics::Function, lb, ub, spec; <keyword_arguments>)

Construct a `ModelingToolkit.PDESystem` representing the specified neural Lyapunov problem.

# Positional Arguments
  - `dynamics`: the dynamical system being analyzed, represented as an `ODESystem` or the
    function `f` such that `ẋ = f(x[, u], p, t)`; either way, the ODE should not depend on
    time and only `t = 0.0` will be used. (For an example of when `f` would have a `u`
    argument, see [`add_policy_search`](@ref).)
  - `bounds`: an array of domains, defining the training domain by bounding the states (and
    derivatives, when applicable) of `dynamics`; only used when `dynamics isa
    ODESystem`, otherwise use `lb` and `ub`.
  - `lb` and `ub`: the training domain will be ``[lb_1, ub_1]×[lb_2, ub_2]×...``; not used
    when `dynamics isa ODESystem`, then use `bounds`.
  - `spec`: a [`NeuralLyapunovSpecification`](@ref) defining the Lyapunov function
    structure, as well as the minimization and decrease conditions.

# Keyword Arguments
  - `fixed_point`: the equilibrium being analyzed; defaults to the origin.
  - `p`: the values of the parameters of the dynamical system being analyzed; defaults to
    `SciMLBase.NullParameters()`; not used when `dynamics isa ODESystem`, then use the
    default parameter values of `dynamics`.
  - `state_syms`: an array of the `Symbol` representing each state; not used when `dynamics
    isa ODESystem`, then the symbols from `dynamics` are used; if `dynamics isa ODEFunction`,
    symbols stored there are used, unless overridden here; if not provided here and cannot
    be inferred, `[:state1, :state2, ...]` will be used.
  - `parameter_syms`: an array of the `Symbol` representing each parameter; not used when
    `dynamics isa ODESystem`, then the symbols from `dynamics` are used; if `dynamics isa
    ODEFunction`, symbols stored there are used, unless overridden here; if not provided
    here and cannot be inferred, `[:param1, :param2, ...]` will be used.
  - `policy_search::Bool`: whether or not to include a loss term enforcing `fixed_point` to
    actually be a fixed point; defaults to `false`; only used when `dynamics isa Function &&
    !(dynamics isa ODEFunction)`; when `dynamics isa ODEFunction`, `policy_search` should
    not be supplied (as it must be false); when `dynamics isa ODESystem`, value inferred by
    the presence of unbound inputs.
  - `name`: the name of the constructed `PDESystem`
"""
function NeuralLyapunovPDESystem(
        dynamics::Function,
        lb,
        ub,
        spec::NeuralLyapunovSpecification;
        fixed_point = zeros(length(lb)),
        p = SciMLBase.NullParameters(),
        state_syms = [],
        parameter_syms = [],
        policy_search::Bool = false,
        name
)::PDESystem
    ########################## Define state symbols ###########################
    state_dim = length(lb)

    # Define state symbols, if not already defined
    state_syms = if isempty(state_syms)
        [Symbol(:state, i) for i in 1:state_dim]
    else
        state_syms
    end
    state = [first(@parameters $s) for s in state_syms]

    ######################## Define parameter symbols #########################
    # Define parameter symbols, if not already defined
    param_syms = if p == SciMLBase.NullParameters()
        []
    else
        if isempty(parameter_syms)
            [Symbol(:param, i) for i in 1:length(p)]
        else
            parameter_syms
        end
    end

    params = [first(@parameters $s) for s in param_syms]

    ##################### Define default parameter values #####################
    defaults = if p == SciMLBase.NullParameters()
        Dict()
    else
        Dict([param => param_val for (param, param_val) in zip(params, p)])
    end

    ############################# Define domains ##############################
    domains = [state[i] ∈ (lb[i], ub[i]) for i in 1:state_dim]

    ########################### Construct PDESystem ###########################
    return _NeuralLyapunovPDESystem(
        dynamics,
        domains,
        spec,
        fixed_point,
        state,
        params,
        defaults,
        policy_search,
        name
    )
end

function NeuralLyapunovPDESystem(
        dynamics::ODEFunction,
        lb,
        ub,
        spec::NeuralLyapunovSpecification;
        fixed_point = zeros(length(lb)),
        p = SciMLBase.NullParameters(),
        state_syms = [],
        parameter_syms = [],
        policy_search::Bool = false,
        name
)::PDESystem
    if dynamics.mass_matrix !== I
        throw(ErrorException("DAEs are not supported at this time. Please supply dynamics" *
                             " without a mass matrix."))
    end
    if policy_search
        throw(ErrorException("Got policy_search == true when dynamics were supplied as an" *
                             " ODEFunction f(x,p,t), so no input can be supplied."))
    end

    # Extract state and parameter symbols from ODEFunction
    s_syms, p_syms = if dynamics.sys isa ODESystem
        s_syms = Symbol.(operation.(unknowns(dynamics.sys)))
        p_syms = Symbol.(parameters(dynamics.sys))
        (s_syms, p_syms)
    elseif dynamics.sys isa SciMLBase.SymbolCache
        s_syms = SciMLBase.variable_symbols(dynamics.sys)
        p_syms = if isnothing(dynamics.sys.parameters)
            []
        else
            keys(dynamics.sys.parameters)
        end
        (s_syms, p_syms)
    else
        ([], [])
    end

    # Override ODEFunction state and parameter symbols when supplied
    s_syms = if state_syms == []
        s_syms
    else
        state_syms
    end
    p_syms = if parameter_syms == []
        p_syms
    else
        parameter_syms
    end

    return NeuralLyapunovPDESystem(
        dynamics.f,
        lb,
        ub,
        spec;
        fixed_point = fixed_point,
        p = p,
        state_syms = s_syms,
        parameter_syms = p_syms,
        policy_search = false,
        name
    )
end

function NeuralLyapunovPDESystem(
        dynamics::ODESystem,
        bounds,
        spec::NeuralLyapunovSpecification;
        fixed_point = zeros(length(bounds)),
        name
)::PDESystem
    ######################### Check for policy search #########################
    f, x, params, policy_search = if isempty(ModelingToolkit.unbound_inputs(dynamics))
        (ODEFunction(dynamics), unknowns(dynamics), parameters(dynamics), false)
    else
        (f, _), x, params = ModelingToolkit.generate_control_function(
            dynamics;
            simplify = true,
            split = false
        )
        (f, x, params, true)
    end

    ########################## Define state symbols ###########################
    # States should all be functions of time, but we just want the symbol
    # e.g., if the state is ω(t), we just want ω
    _state = operation.(x)
    state_syms = Symbol.(_state)
    state = [first(@parameters $s) for s in state_syms]

    ###################### Remove derivatives in domains ######################
    domains = map(
        d -> Num(operation(Symbolics.diff2term(Symbolics.value(d.variables)))) ∈ d.domain,
        bounds
    )
    domain_vars = map(d -> d.variables, domains)
    if Set(_state) != Set(domain_vars)
        error("Domain variables from `domains` do not match those extracted from " *
              "`dynamics`. Got $_state from `dynamics` and $domain_vars from `domains`.")
    end

    ########################### Construct PDESystem ###########################
    return _NeuralLyapunovPDESystem(
        f,
        domains,
        spec,
        fixed_point,
        state,
        params,
        ModelingToolkit.get_defaults(dynamics),
        policy_search,
        name
    )
end

function _NeuralLyapunovPDESystem(
        dynamics::Function,
        domains,
        spec::NeuralLyapunovSpecification,
        fixed_point,
        state,
        params,
        defaults,
        policy_search::Bool,
        name
)::PDESystem
    ########################## Unpack specifications ##########################
    structure = spec.structure
    minimization_condition = spec.minimization_condition
    decrease_condition = spec.decrease_condition
    f_call = structure.f_call
    state_dim = length(domains)

    ################## Define Lyapunov function & derivative ##################
    output_dim = structure.network_dim
    net_syms = [Symbol(:φ, i) for i in 1:output_dim]
    net = [first(@variables $s(..)) for s in net_syms]

    # φ(x) is the symbolic form of neural network output
    φ(x) = Num.([φi(x...) for φi in net])

    # V(x) is the symobolic form of the Lyapunov function
    V(x) = structure.V(φ, x, fixed_point)

    # V̇(x) is the symbolic time derivative of the Lyapunov function
    function V̇(x)
        structure.V̇(
            φ,
            y -> Symbolics.jacobian(φ(y), y),
            dynamics,
            x,
            params,
            0.0,
            fixed_point
        )
    end

    ################ Define equations and boundary conditions #################
    eqs = Equation[]

    if check_nonnegativity(minimization_condition)
        cond = get_minimization_condition(minimization_condition)
        push!(eqs, cond(V, state, fixed_point) ~ 0.0)
    end

    if check_decrease(decrease_condition)
        cond = get_decrease_condition(decrease_condition)
        push!(eqs, cond(V, V̇, state, fixed_point) ~ 0.0)
    end

    bcs = Equation[]

    if check_zero_fixed_point(minimization_condition)
        _V = V(fixed_point)
        _V = _V isa AbstractVector ? _V[] : _V
        push!(bcs, _V ~ 0.0)
    end

    if check_minimal_fixed_point(minimization_condition)
        _V = V(fixed_point)
        _V = _V isa AbstractVector ? _V[] : _V
        append!(bcs, Symbolics.gradient(_V, state) .~ 0.0)
    end

    if check_maximal_fixed_point(decrease_condition)
        _V̇ = V̇(fixed_point)
        _V̇ = _V̇ isa AbstractVector ? _V̇[] : _V̇
        append!(bcs, Symbolics.gradient(_V̇, state) .~ 0.0)
    end

    if policy_search
        append!(bcs, f_call(dynamics, φ, fixed_point, params, 0.0) .~ zeros(state_dim))
    end

    if isempty(eqs) && isempty(bcs)
        error("No training conditions specified.")
    end

    # NeuralPDE requires an equation and a boundary condition, even if they are
    # trivial like φ(0.0) == φ(0.0), so we remove those trivial equations if they showed up
    # naturally alongside other equations and add them in if we have no other equations
    eqs = filter(eq -> eq != (0.0 ~ 0.0), eqs)
    bcs = filter(eq -> eq != (0.0 ~ 0.0), bcs)

    if isempty(eqs)
        push!(eqs, φ(fixed_point)[1] ~ φ(fixed_point)[1])
    end
    if isempty(bcs)
        push!(bcs, φ(fixed_point)[1] ~ φ(fixed_point)[1])
    end

    ########################### Construct PDESystem ###########################
    return PDESystem(
        eqs,
        bcs,
        domains,
        state,
        φ(state),
        params;
        defaults = defaults,
        name = name
    )
end
