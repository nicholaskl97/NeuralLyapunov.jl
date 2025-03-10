"""
    Pendulum(; driven = true, name, defaults)

Create an `ODESystem` representing a damped, driven or undriven pendulum, depending on the
value of driven (defaults to `true`, i.e., driven pendulum).

The equation used in this model is
``\\ddot{θ} + 2 ζ ω_0 \\dot{θ} + ω_0^2 \\sin(θ) = τ,``
where ``θ`` is the angle counter-clockwise from the downward equilibrium, ``ζ`` is the
damping parameter, ``ω_0`` is the resonant angular frequency, and ``τ`` is the input torque
divided by the moment of inertia of the pendulum around the pivot (`driven = false` sets
``τ ≡ 0``).

The name of the `ODESystem` is `name`.

Users may optionally provide default values of the parameters through `defaults`: a vector
of the default values for `[ζ, ω_0]`.

# Example

```jldoctest; output = false
@named pendulum = Pendulum(driven = false)
pendulum = structural_simplify(pendulum)

x0 = π * rand(2)
p = rand(2)
τ = 1 / prod(p)

prob = ODEProblem(pendulum, x0, 15τ, p)
sol = solve(prob, Tsit5())

# Check that the undriven pendulum fell to the downward equilibrium
θ_end, ω_end = sol.u[end]
x_end, y_end = sin(θ_end), -cos(θ_end)

sqrt(sum(abs2, [x_end, y_end, ω_end] .- [0, -1, 0])) < 1e-4
# output
true
```
"""
function Pendulum(; driven = true, name, defaults=NullParameters())
    @independent_variables t
    Dt = Differential(t); DDt = Dt^2

    @variables θ(t) τ(t) [input = true]
    @parameters ζ ω_0

    params = [ζ, ω_0]
    kwargs = if defaults == NullParameters()
        (; name = name)
    else
        (; name = name, defaults = Dict(params .=> defaults))
    end

    torque = if driven; τ else 0 end
    variables = if driven; [θ, τ] else [θ] end

    eqs = [DDt(θ) + 2ζ * ω_0 * Dt(θ) + ω_0^2 * sin(θ) ~ torque]

    return ODESystem(
        eqs,
        t,
        variables,
        params;
        kwargs...
    )
end

"""
    plot_pendulum(θ, t; title)
    plot_pendulum(sol; title, N, angle_symbol)

Plot the pendulum's trajectory.

# Arguments
  - `θ`: The angle of the pendulum at each time step.
  - `t`: The time steps.
  - `sol`: The solution to the ODE problem.

# Keyword arguments
  - `title`: The title of the plot; defaults to no title (i.e., `title=""`).
  - `N`: The number of points to plot; when using `θ` and `t`, uses `length(t)`; defaults to
    500 when using `sol`.
  - `angle_symbol`: The symbol of the angle in `sol`; defaults to `:θ`.
"""
function plot_pendulum end
