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
function NeuralLyapunovProblemLibrary.plot_double_pendulum(
    sol,
    p;
    title="",
    N = 500,
    angle1_symbol=:θ1,
    angle2_symbol=:θ2,
)
    t = LinRange(sol.t[1], sol.t[end], N)
    θ1 = sol(t)[angle1_symbol]
    θ2 = sol(t)[angle2_symbol]
    return plot_double_pendulum(θ1, θ2, p, t; title=title)
end

function NeuralLyapunovProblemLibrary.plot_double_pendulum(θ1, θ2, p, t; title="")
    l1, l2 = p[3:4]
    L = l1 + l2

    x1 = +l1 * sin.(θ1)
    y1 = -l1 * cos.(θ1)

    x2 = x1 + l2 * sin.(θ2)
    y2 = y1 - l2 * cos.(θ2)

    return @animate for i in eachindex(t)
        # Pendulum bars
        pend_x = [0, x1[i], x2[i]]
        pend_y = [0, y1[i], y2[i]]
        plot(pend_x, pend_y, legend=false, lw=3)
        scatter!(pend_x, pend_y)

        # Trajectory so far
        traj_x = x2[1:i]
        traj_y = y2[1:i]
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
        xlims!(-L, L)
        ylims!(-L, L)
        title!(title)
        annotate!(-0.5L, 0.75L, "time= $(rpad(round(t[i]; digits=1),4,"0"))")
    end
end
