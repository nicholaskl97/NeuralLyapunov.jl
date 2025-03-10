"""
    DoublePendulum(; actuation=:fully_actuated, name, defaults)

Create an `ODESystem` representing an undamped double pendulum.

The posture of the double pendulum is determined by `θ1` and `θ2`, the angle of the
first and second pendula, respectively.
`θ1` is measured counter-clockwise relative to the downward equilibrium and `θ2` is
measured counter-clockwise relative to `θ1` (i.e., when `θ2` is fixed at 0, the double
pendulum appears as a single pendulum).

The ODESystem uses the explicit manipulator form of the equations:
```math
q̈ = M^{-1}(q) (-C(q,q̇)q̇ + τ_g(q) + Bu).
```

The name of the `ODESystem` is `name`.

# Actuation modes

The four actuation modes are described in the table below and selected via `actuation`.

| Actuation mode (`actuation`) | Torque around `θ1` | Torque around `θ2` |
| ---------------------------- | ------------------ | ------------------ |
| `:fully_actuated` (default)  | `τ1`               | `τ2`               |
| `:acrobot`                   | Not actuated       | `τ`                |
| `:pendubot`                  | `τ`                | Not actuated       |
| `:undriven`                  | Not actuated       | Not actuated       |

# ODESystem Parameters
  - `I1`: moment of inertia of the first pendulum around its pivot (not its center of
    mass).
  - `I2`:  moment of inertia of the second pendulum around its pivot (not its center of
    mass).
  - `l1`: length of the first pendulum.
  - `l2`: length of the second pendulum.
  - `lc1`: distance from pivot to the center of mass of the first pendulum.
  - `lc2`: distance from the link to the center of mass of the second pendulum.
  - `m1`: mass of the first pendulum.
  - `m2`: mass of the second pendulum.
  - `g`: gravitational acceleration (defaults to 9.81).

Users may optionally provide default values of the parameters through `defaults`: a vector
of the default values for `[I1, I2, l1, l2, lc1, lc2, m1, m2, g]`.
"""
function DoublePendulum(; actuation=:fully_actuated, name, defaults=NullParameters())
    @independent_variables t
    Dt = Differential(t); DDt = Dt^2

    @variables θ1(t) θ2(t)
    @parameters I1 I2 l1 l2 lc1 lc2 m1 m2 g=9.81

    M = [
        I1 + I2 + m2 * l1^2 + 2 * m2 * l1 * lc2 * cos(θ2)   I2 + m2 * l1 * lc2 * cos(θ2);
        I2 + m2 * l1 * lc2 * cos(θ2)                        I2
    ]
    C = [
        -2 * m2 * l1 * lc2 * sin(θ2) * Dt(θ2)   -m2 * l1 * lc2 * sin(θ2) * Dt(θ2);
        m2 * l1 * lc2 * sin(θ2) * Dt(θ1)        0
    ]
    G = [
        -m1 * g * lc1 * sin(θ1) - m2 * g * (l1 * sin(θ1) + lc2 * sin(θ1 + θ2));
        -m2 * g * lc2 * sin(θ1 + θ2)
    ]
    q = [θ1, θ2]
    params = [I1, I2, l1, l2, lc1, lc2, m1, m2, g]

    kwargs = if defaults == NullParameters()
        (; name = name)
    else
        (; name = name, defaults = Dict(params .=> defaults))
    end

    if actuation == :fully_actuated
        ########################## Fully-actuated double pendulum ##########################
        @variables τ1(t) [input = true] τ2(t) [input = true]
        u = [τ1, τ2]

        eqs = DDt.(q) .~ M \ (-C * Dt.(q) + G + u)
        return ODESystem(eqs, t, vcat(q, u), params; kwargs...)
    elseif actuation == :undriven
        ############################# Undriven double pendulum #############################
        eqs = DDt.(q) .~ M \ (-C * Dt.(q) + G)
        return ODESystem(eqs, t, q, params; kwargs...)
    else
        ########################## Underactuated double pendulum ###########################
        @variables τ(t) [input = true]

        if actuation == :acrobot
            #################################### Acrobot ###################################
            B = [0, 1]
            eqs = DDt.(q) .~ M \ (-C * Dt.(q) + G + B * τ)

            return ODESystem(eqs, t, vcat(q, τ), params; kwargs...)
        elseif actuation == :pendubot
            ################################### Pendubot ###################################
            B = [1, 0]
            eqs = DDt.(q) .~ M \ (-C * Dt.(q) + G + B * τ)

            return ODESystem(eqs, t, vcat(q, τ), params; kwargs...)
        else
            error(
                "Invalid actuation for DoublePendulum. Received actuation = :",
                string(name)
            )
        end
    end
end

"""
    Acrobot(; name, defaults)

Alias for [`DoublePendulum(; actuation = :acrobot, name, defaults)`](@ref).
"""
function Acrobot(; name, defaults=NullParameters())
    return DoublePendulum(; actuation = :acrobot, name, defaults)
end

"""
    Pendubot(; name, defaults)

Alias for [`DoublePendulum(; actuation = :pendubot, name, defaults)`](@ref).
"""
function Pendubot(; name, defaults=NullParameters())
    DoublePendulum(; actuation = :pendubot, name, defaults)
end

"""
    plot_double_pendulum(θ1, θ2, p, t; title)
    plot_double_pendulum(sol, p; title, N, angle_symbol)

Plot the pendulum's trajectory.

# Arguments
  - `θ1`: The angle of the first pendulum link at each time step.
  - `θ2`: The angle of the second pendulum link at each time step.
  - `t`: The time steps.
  - `sol`: The solution to the ODE problem.
  - `p`: The parameters of the double pendulum.

# Keyword arguments
  - `title`: The title of the plot; defaults to no title (i.e., `title=""`).
  - `N`: The number of points to plot; when using `θ` and `t`, uses `length(t)`; defaults to
    500 when using `sol`.
  - `angle1_symbol`: The symbol of the angle of the first link in `sol`; defaults to `:θ1`.
  - `angle2_symbol`: The symbol of the angle of the second link in `sol`; defaults to `:θ2`.
"""
function plot_double_pendulum end
