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
function NeuralLyapunovProblemLibrary.plot_quadrotor_planar(
    sol,
    p;
    title="",
    N = 500,
    x_symbol=:x,
    y_symbol=:y,
    θ_symbol=:θ,
    u1_symbol=:u1,
    u2_symbol=:u2
)
    t = LinRange(sol.t[1], sol.t[end], N)
    x = sol(t)[x_symbol]
    y = sol(t)[y_symbol]
    θ = sol(t)[θ_symbol]
    u1 = sol(t)[u1_symbol]
    u2 = sol(t)[u2_symbol]
    return plot_quadrotor_planar(x, y, θ, u1, u2, p, t; title=title)
end

function NeuralLyapunovProblemLibrary.plot_quadrotor_planar(x, y, θ, p, t; title="")
    m, _, g, _ = p
    T = m * g / 2
    u = fill(T, length(t))
    return plot_quadrotor_planar(x, y, θ, u, u, p, t; title=title)
end

function NeuralLyapunovProblemLibrary.plot_quadrotor_planar(x, y, θ, u1, u2, p, t; title="")
    function quadrotor_body(x, y, θ, r; aspect_ratio=10)
        pos = [x, y]
        r_vec = [r * cos(θ), r * sin(θ)]
        r_perp = [r_vec[2], -r_vec[1]] / aspect_ratio
        C1 = pos + r_vec + r_perp
        C2 = pos + r_vec - r_perp
        C3 = pos - r_vec - r_perp
        C4 = pos - r_vec + r_perp
        corners = [C1, C2, C3, C4]
        return Shape(first.(corners), last.(corners))
    end
    function rotors(x, y, θ, u1, u2, r)
        pos = [x, y]
        r_vec = [r * cos(θ), r * sin(θ)] * 0.75
        r_perp = [-r_vec[2], r_vec[1]]
        arrow1_begin = pos + r_vec
        arrow1_end = pos + r_vec + r_perp * u1
        arrow2_begin = pos - r_vec
        arrow2_end = pos - r_vec + r_perp * u2
        arrow1 = [arrow1_begin, arrow1_end]
        arrow2 = [arrow2_begin, arrow2_end]
        return (
            (first.(arrow1), last.(arrow1)),
            (first.(arrow2), last.(arrow2))
        )
    end

    m, _, g, r = p
    u1 = max.(0.0, u1 ./ (m * g / 2))
    u2 = max.(0.0, u2 ./ (m * g / 2))
    return @animate for i in eachindex(t)
        # Quadcopter body
        plot(quadrotor_body(x[i], y[i], θ[i], r), aspect_ratio=1, legend=false)

        # Rotors
        top_rotor, bottom_rotor = rotors(x[i], y[i], θ[i], u1[i], u2[i], r)
        plot!(top_rotor..., arrow=true, linewdith=2, color=:red)
        plot!(bottom_rotor..., arrow=true, linewdith=2, color=:red)

        # Trajectory so far
        traj_x = x[1:i]
        traj_y = y[1:i]
        plot!(traj_x, traj_y, color=:orange)
        scatter!(
            traj_x,
            traj_y,
            color = :orange,
            markersize = 2,
            markerstrokewidth = 0,
            markerstrokecolor = :orange,
        )

        # Plot settings and timestamp
        title!(title)
        #annotate!(-0.5L, 0.75L, "time= $(rpad(round(t[i]; digits=1),4,"0"))")
    end
end

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
function NeuralLyapunovProblemLibrary.plot_quadrotor_3d(
    sol,
    p;
    title="",
    N = 500,
    x_symbol=:x,
    y_symbol=:y,
    z_symbol=:z,
    φ_symbol=:φ,
    θ_symbol=:θ,
    ψ_symbol=:ψ,
    T_symbol=:T,
    τφ_symbol=:τφ,
    τθ_symbol=:τθ,
    τψ_symbol=:τψ
)
    t = LinRange(sol.t[1], sol.t[end], N)
    x = sol(t)[x_symbol]
    y = sol(t)[y_symbol]
    z = sol(t)[z_symbol]
    φ = sol(t)[φ_symbol]
    θ = sol(t)[θ_symbol]
    ψ = sol(t)[ψ_symbol]
    T = sol(t)[T_symbol]
    τφ = sol(t)[τφ_symbol]
    τθ = sol(t)[τθ_symbol]
    τψ = sol(t)[τψ_symbol]
    return plot_quadrotor_3d(x, y, z, φ, θ, ψ, T, τφ, τθ, τψ, p, t; title=title)
end

function NeuralLyapunovProblemLibrary.plot_quadrotor_3d(
    x, y, z, φ, θ, ψ, p, t; title=""
)
    m, g = p[1:2]
    T = fill(m * g / 4, length(t))
    τφ = zeros(length(t))
    τθ = zeros(length(t))
    τψ = zeros(length(t))
    return plot_quadrotor_3d(x, y, z, φ, θ, ψ, T, τφ, τθ, τψ, p, t; title=title)
end

function NeuralLyapunovProblemLibrary.plot_quadrotor_3d(
    x, y, z, φ, θ, ψ, T, τφ, τθ, τψ, p, t; title=""
)
    m, g = p[1:2]
    L = 1.0
    k = 1.0

    # Calculate thrusts (rescaled as lengths)
    τ = transpose(hcat(T, τφ ./ L, τθ ./ L, τψ ./ k))
    F = max.(0.0, [1 0 -2 -1; 1 2 0 1; 1 0 2 -1; 1 -2 0 1] * τ ./ (m * g) .* L ./ 3)

    return @animate for i in eachindex(t)
        # CoM position (world coordinates)
        pos = [x[i], y[i], z[i]]

        # Rotor positions (body coordinates)
        rotor1 = [ L,  0, 0]
        rotor2 = [ 0,  L, 0]
        rotor3 = [-L,  0, 0]
        rotor4 = [ 0, -L, 0]

        # Rotate legs to world coordinates
        R = RotZXY(roll=φ[i], pitch=θ[i], yaw=ψ[i])
        rotor1 = pos + R * rotor1
        rotor2 = pos + R * rotor2
        rotor3 = pos + R * rotor3
        rotor4 = pos + R * rotor4

        # Rotor thrusts in world coordinates
        dir = R[:, 3]
        f1, f2, f3, f4 = F[:, i]
        F1 = f1 * dir
        F2 = f2 * dir
        F3 = f3 * dir
        F4 = f4 * dir

        # Plot quadrotor body
        # Conntect rotors 1 and 3
        plot(
            [rotor1[1], rotor3[1]],
            [rotor1[2], rotor3[2]],
            [rotor1[3], rotor3[3]],
            legend=false,
            lw=3,
            color=:black
        )
        # Connect rotors 2 and 4
        plot!(
            [rotor2[1], rotor4[1]],
            [rotor2[2], rotor4[2]],
            [rotor2[3], rotor4[3]],
            lw=3,
            color=:black
        )

        # Mark center
        scatter!([pos[1]], [pos[2]], [pos[3]], markersize=5, color=:black)

        # Plot thrusts
        quiver!(
            [rotor1[1], rotor2[1], rotor3[1], rotor4[1]],
            [rotor1[2], rotor2[2], rotor3[2], rotor4[2]],
            [rotor1[3], rotor2[3], rotor3[3], rotor4[3]],
            quiver=(
                [F1[1], F2[1], F3[1], F4[1]],
                [F1[2], F2[2], F3[2], F4[2]],
                [F1[3], F2[3], F3[3], F4[3]]
            ),
            markersize=3,
            markershape=:utriangle,
            lw=2,
            color=:red
        )

        # Mark rotors
        scatter!(
            [rotor1[1], rotor2[1], rotor3[1], rotor4[1]],
            [rotor1[2], rotor2[2], rotor3[2], rotor4[2]],
            [rotor1[3], rotor2[3], rotor3[3], rotor4[3]],
            markersize=5,
            color=:black
        )

        # Plot trajectory so far
        traj_x = x[1:i]
        traj_y = y[1:i]
        traj_z = z[1:i]
        plot!(traj_x, traj_y, traj_z, color=:orange)
        scatter!(
            traj_x,
            traj_y,
            traj_z,
            color=:orange,
            markersize=2,
            markerstrokewidth=0,
            markerstrokecolor=:orange
        )

        # Plot settings and timestamp
        title!(title)
        xl, yl, zl = xlims(), ylims(), zlims()
        xrange = xl[2] - xl[1]
        yrange = yl[2] - yl[1]
        zrange = zl[2] - zl[1]
        range = maximum([xrange, yrange, zrange])
        x̄ = (xl[1] + xl[2]) / 2
        ȳ = (yl[1] + yl[2]) / 2
        z̄ = (zl[1] + zl[2]) / 2
        xlims!(x̄ - range / 2, x̄ + range / 2)
        ylims!(ȳ - range / 2, ȳ + range / 2)
        zlims!(z̄ - range / 2, z̄ + range / 2)
    end
end
