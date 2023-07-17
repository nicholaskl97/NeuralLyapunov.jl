function NeuralLyapunovProblem(
    dynamics::ODEFunction,
    lb,
    ub,
    spec::NeuralLyapunovSpecification,
    discretization::PhysicsInformedNN;
    fixed_point = zeros(length(lb)),
    p = SciMLBase.NullParameters(),
)::Tuple{OptimizationProblem,Function}
    if dynamics.mass_matrix !== I
        throw(ErrorException("DAEs are not supported at this time"))
    end

    ########################## Unpack specifications ##########################
    structure = spec.structure
    minimzation_condition = spec.minimzation_condition
    decrease_condition = spec.decrease_condition

    ########################## Unpack discretization ##########################
    chain = discretization.chain

    param_estim = discretization.param_estim
    additional_loss = discretization.additional_loss
    adaloss = discretization.adaptive_loss

    multioutput = discretization.multioutput
    init_params = discretization.init_params
    phi = discretization.phi

    derivative = discretization.derivative
    strategy = discretization.strategy

    logger = discretization.logger
    log_frequency = discretization.log_options.log_frequency
    iteration = discretization.iteration
    self_increment = discretization.self_increment

    ################### Initialize neural network parameters ##################
    init_params = initialize_network_parameters(init_params, chain)

    ################## Define Lyapunov function & derivative ##################
    f(x) = dynamics(x, p, 0.0)
    V(x) = structure.V(??phi??, x, fixed_point)
    V̇(x) = structure.V̇(??phi??, ??J_phi??, f, x, fixed_point)

    ################### Construct data-free loss functions ####################
    data_free_loss_functions = []

    if check_nonnegativity(minimzation_condition)
        cond = get_minimization_condition(minimzation_condition)
        push!(data_free_loss_functions, (state) -> cond(V, state, fixed_point))
    end

    if check_decrease(decrease_condition)
        cond = get_decrease_condition(decrease_condition)
        push!(data_free_loss_functions, (state) -> cond(V, V̇, state, fixed_point))
    end
    
    if check_fixed_point(minimzation_condition)
        push!(data_free_loss_functions, V(fixed_point))
    end
    if check_stationary_fixed_point(decrease_condition)
        push!(data_free_loss_functions, V̇(fixed_point))
    end

    if isempty(data_free_loss_functions)
        error("No training conditions specified.")
    end

    #################### Construct decrease condition loss ####################

end

function initialize_network_parameters(init_params, chain)
    # This code taken (almost) directly from NeuralPDE.jl
    if init_params === nothing
        # Use the initialization of the neural network framework
        # But for Lux, default to Float64
        # For Flux, default to the types matching the values in the neural network
        # This is done because Float64 is almost always better for these applications
        # But with Flux there's already a chosen type from the user

        if chain isa AbstractArray
            if chain[1] isa Flux.Chain
                init_params = map(x -> Flux.destructure(x)[1], chain)
            else 
                # chain is from Lux
                y = map(chain) do x
                    _x = ComponentArrays.ComponentArray(
                            Lux.initialparameters(Random.default_rng(), x)
                        )
                    Float64.(_x) # No ComponentArray GPU support
                end
                ###################?????????????????? TODO TODO Fix line belowl ??????????????????????????????#####################
                names = ntuple(i -> depvars[i], length(chain))
                init_params = ComponentArrays.ComponentArray(
                        NamedTuple{names}(i for i in y)
                    )
            end
        else
            if chain isa Flux.Chain
                init_params = Flux.destructure(chain)[1]
                init_params = init_params isa Array ? Float64.(init_params) :
                                init_params
            else
                init_params = Float64.(
                        ComponentArrays.ComponentArray(
                            Lux.initialparameters(Random.default_rng(), chain)
                        )
                    )
            end
        end
    else
        init_params = init_params
    end
    return init_params
end