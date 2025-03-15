##################################### Planar quadrotor #####################################
"""
    QuadrotorPlanar(; name, defaults)

Create an `ODESystem` representing a planar approximation of the quadrotor (technically a
birotor).

This birotor is a rigid body with two rotors in line with the center of mass.
The location of the center of mass is determined by `x` and `y`.
Its orientation is determined by `θ`, measured counter-clockwise from the ``x``-axis.
The thrust from the right rotor (on the positive ``x``-axis when ``θ = 0``) is the input
`u1`.
The thrust from the other rotor is `u2`.
Note that these thrusts should be nonnegative and if a negative input is provided, the model
replaces it with 0.

The equations governing the planar quadrotor are:
```math
\\begin{align}
    mẍ &= -(u_1 + u_2)\\sin(θ), \\\\
    mÿ &= (u_1 + u_2)\\cos(θ) - mg, \\\\
    I_{quad} \\ddot{θ} &= r (u_1 - u_2).
\\end{align}
```

The name of the `ODESystem` is `name`.

# ODESystem Parameters
  - `m`: mass of the quadrotor.
  - `I_quad`: moment of inertia of the quadrotor around its center of mass.
  - `g`: gravitational acceleration in the direction of the negative ``y``-axis (defaults to
    9.81).
  - `r`: distance from center of mass to each rotor.

Users may optionally provide default values of the parameters through `defaults`: a vector
of the default values for `[m, I_quad, g, r]`.
"""
function QuadrotorPlanar(; name, defaults=NullParameters())
    @independent_variables t
    Dt = Differential(t); DDt = Dt^2
    @variables x(t) y(t) θ(t)
    @variables u1(t) [input=true] u2(t) [input=true]
    @parameters m I_quad g r

    # Thrusts must be nonnegative
    ũ1 = max(0, u1)
    ũ2 = max(0, u2)

    eqs = [
        m * DDt(x) ~ -(ũ1 + ũ2) * sin(θ);
        m * DDt(y) ~ (ũ1 + ũ2) * cos(θ) - m * g;
        I_quad * DDt(θ) ~ r * (ũ1 - ũ2)
    ]

    params = [m, I_quad, g, r]
    kwargs = if defaults == NullParameters()
        (; name = name)
    else
        (; name = name, defaults = Dict(params .=> defaults))
    end

    return ODESystem(eqs, t, [x, y, θ, u1, u2], params; kwargs...)
end

####################################### 3D quadrotor #######################################

"""
    Quadrotor3D(; name, defaults)

Create an `ODESystem` representing a quadrotor in 3D space.

The quadrotor is a rigid body in an X-shape (90°-angles between the rotors).
The equations governing the quadrotor can be found in [quadrotor](@cite).

# ODESystem State Variables
  - `x`: ``x``-position (world frame).
  - `y`: ``y``-position (world frame).
  - `z`: ``z``-position (world frame).
  - `φ`: roll around body ``x``-axis (Z-X-Y Euler angles).
  - `θ`: pitch around body ``y``-axis (Z-X-Y Euler angles).
  - `ψ`: yaw around body ``z``-axis (Z-X-Y Euler angles).
  - `vx`: ``x``-velocity (world frame).
  - `vy`: ``y``-velocity (world frame).
  - `vz`: ``z``-velocity (world frame).
  - `ωφ`: roll angular velocity (world frame).
  - `ωθ`: pitch angular velocity (world frame).
  - `ωψ`: yaw angular velocity (world frame).

# ODESystem Input Variables
  - `T`: thrust (should be nonnegative).
  - `τφ`: roll torque.
  - `τθ`: pitch torque.
  - `τψ`: yaw torque.

Not only should the aggregate thrust be nonnegative, but the torques should have been
generated from nonnegative individual rotor thrusts.
The model calculates individual rotor thrusts and replaces any negative values with 0.

# ODESystem Parameters
  - `m`: mass of the quadrotor.
  - `g`: gravitational acceleration in the direction of the negative ``z``-axis (defaults to
    9.81).
  - `Ixx`,`Ixy`,`Ixz`,`Iyy`,`Iyz`,`Izz`: components of the moment of inertia matrix of the
    quadrotor around its center of mass:
    ```math
    I = \\begin{pmatrix}
            I_{xx} & I_{xy} & I_{xz} \\\\
            I_{xy} & I_{yy} & I_{yz} \\\\
            I_{xz} & I_{yz} & I_{zz}
        \\end{pmatrix}.
    ```

Users may optionally provide default values of the parameters through `defaults`: a vector
of the default values for `[m, g, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]`.
"""
function Quadrotor3D(; name, defaults=NullParameters())
    # Model from "Minimum Snap Trajectory Generation and Control for Quadrotors"
    # https://doi.org/10.1109/ICRA.2011.5980409
    @independent_variables t
    Dt = Differential(t)

    # Position (world frame)
    @variables x(t) y(t) z(t)
    position_world = [x, y, z]

    # Velocity (world frame)
    @variables vx(t) vy(t) vz(t)
    velocity_world = [vx, vy, vz]

    # Attitude
    # φ-roll (around body x-axis), θ-pitch (around body y-axis), ψ-yaw (around body z-axis)
    @variables φ(t) θ(t) ψ(t)
    attitude = [φ, θ, ψ]
    R = RotZXY(roll=φ, pitch=θ, yaw=ψ)

    # Angular velocity (world frame)
    @variables ωφ(t), ωθ(t), ωψ(t)
    angular_velocity_world = [ωφ, ωθ, ωψ]

    # Inputs
    # T-thrust, τφ-roll torque, τθ-pitch torque, τψ-yaw torque
    @variables T(t) [input=true]
    @variables τφ(t) [input=true] τθ(t) [input=true] τψ(t) [input=true]

    # Individual rotor thrusts must be nonnegative
    f = ([1 0 -2 -1; 1 2 0 1; 1 0 2 -1; 1 -2 0 1] * [T; τφ; τθ; τψ]) ./ 4
    f̃ = max.(0, f)
    T̃ = [1 1 1 1; 0 1 0 -1; -1 0 1 0; -1 1 -1 1] * f̃

    F = T̃[1] .* R[3, :]
    τ = T̃[2:4]

    # Parameters
    # m-mass, g-gravitational accelerationz
    @parameters m g=9.81 Ixx Ixy Ixz Iyy Iyz Izz
    params = [m, g, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
    g_vec = [0, 0, -g]
    inertia_matrix = [Ixx Ixy Ixz; Ixy Iyy Iyz; Ixz Ixy Izz]

    eqs = vcat(
        Dt.(position_world) .~ velocity_world,
        Dt.(velocity_world) .~ F ./ m .+ g_vec,
        Dt.(attitude) .~ inv(R) * angular_velocity_world,
        Dt.(angular_velocity_world) .~ inertia_matrix \
                (τ - angular_velocity_world × (inertia_matrix * angular_velocity_world))
    )

    kwargs = if defaults == NullParameters()
        (; name = name)
    else
        (; name = name, defaults = Dict(params .=> defaults))
    end

    return ODESystem(
        eqs,
        t,
        vcat(position_world, attitude, velocity_world, angular_velocity_world, T, τφ, τθ, τψ),
        params;
        kwargs...
    )
end

"""
    plot_quadrotor_planar(x, y, θ, [u1, u2,] p, t; title)
    plot_quadrotor_planar(sol, p; title, N, x_symbol, y_symbol, θ_symbol)

Plot the planar quadrotor's trajectory.

When thrusts are supplied, the arrows scale with thrust, otherwise the arrows are of
constant length.

# Arguments
  - `x`: The x-coordinate of the quadrotor at each time step.
  - `y`: The y-coordinate of the quadrotor at each time step.
  - `θ`: The angle of the quadrotor at each time step.
  - `u1`: The thrust of the first rotor at each time step.
  - `u2`: The thrust of the second rotor at each time step.
  - `t`: The time steps.
  - `sol`: The solution to the ODE problem.
  - `p`: The parameters of the quadrotor.

# Keyword arguments
  - `title`: The title of the plot; defaults to no title (i.e., `title=""`).
  - `N`: The number of points to plot; when using `x`, `y`, `θ`, and `t`, uses `length(t)`; defaults to
    500 when using `sol`.
  - `x_symbol`: The symbol of the x-coordinate in `sol`; defaults to `:x`.
  - `y_symbol`: The symbol of the y-coordinate in `sol`; defaults to `:y`.
  - `θ_symbol`: The symbol of the angle in `sol`; defaults to `:θ`.
  - `u1_symbol`: The symbol of the thrust of the first rotor in `sol`; defaults to `:u1`.
  - `u2_symbol`: The symbol of the thrust of the second rotor in `sol`; defaults to `:u2`.
"""
function plot_quadrotor_planar end

"""
    plot_quadrotor_3d(x, y, z, φ, θ, ψ, [T, τφ, τθ, τψ,] p, t; title)
    plot_quadrotor_3d(sol, p; title, N, x_symbol, y_symbol, z_symbol, φ_symbol, θ_symbol, ψ_symbol, T_symbol, τφ_symbol, τθ_symbol, τψ_symbol)

Plot the 3D quadrotor's trajectory.

When thrusts are supplied, the arrows scale with thrust, otherwise the arrows are of
constant length.

# Arguments
  - `x`: The x-coordinate of the quadrotor at each time step.
  - `y`: The y-coordinate of the quadrotor at each time step.
  - `z`: The z-coordinate of the quadrotor at each time step.
  - `φ`: The roll of the quadrotor at each time step.
  - `θ`: The pitch of the quadrotor at each time step.
  - `ψ`: The yaw of the quadrotor at each time step.
  - `T`: The thrust of the quadrotor at each time step.
  - `τφ`: The roll torque of the quadrotor at each time step.
  - `τθ`: The pitch torque of the quadrotor at each time step.
  - `τψ`: The yaw torque of the quadrotor at each time step.
  - `t`: The time steps.
  - `sol`: The solution to the ODE problem.
  - `p`: The parameters of the quadrotor.

# Keyword arguments
  - `title`: The title of the plot; defaults to no title (i.e., `title=""`).
  - `N`: The number of points to plot; when using `x`, `y`, `z`, etc., uses `length(t)`;
    defaults to 500 when using `sol`.
  - `x_symbol`: The symbol of the x-coordinate in `sol`; defaults to `:x`.
  - `y_symbol`: The symbol of the y-coordinate in `sol`; defaults to `:y`.
  - `z_symbol`: The symbol of the z-coordinate in `sol`; defaults to `:z`.
  - `φ_symbol`: The symbol of the roll in `sol`; defaults to `:φ`.
  - `θ_symbol`: The symbol of the pitch in `sol`; defaults to `:θ`.
  - `ψ_symbol`: The symbol of the yaw in `sol`; defaults to `:ψ`.
  - `T_symbol`: The symbol of the thrust in `sol`; defaults to `:T`.
  - `τφ_symbol`: The symbol of the roll torque in `sol`; defaults to `:τφ`.
  - `τθ_symbol`: The symbol of the pitch torque in `sol`; defaults to `:τθ`.
  - `τψ_symbol`: The symbol of the yaw torque in `sol`; defaults to `:τψ`.
"""
function plot_quadrotor_3d end
