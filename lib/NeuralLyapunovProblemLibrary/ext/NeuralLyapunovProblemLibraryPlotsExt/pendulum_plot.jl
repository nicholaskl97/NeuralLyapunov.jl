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
function NeuralLyapunovProblemLibrary.plot_pendulum(sol; title="", N = 500, angle_symbol=:θ)
    t = LinRange(sol.t[1], sol.t[end], N)
    θ = sol(t)[angle_symbol]
    return plot_pendulum(θ, t; title=title)
end

function NeuralLyapunovProblemLibrary.plot_pendulum(θ, t; title="")
    x = sin.(θ)
    y = -cos.(θ)

    return @animate for i in eachindex(t)
        # Pendulum bar
        pend_x = [0, x[i]]
        pend_y = [0, y[i]]
        plot(pend_x, pend_y, legend=false, lw=3)
        scatter!(pend_x, pend_y)

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
        xlims!(-1.5, 1.5)
        ylims!(-1.5, 1.5)
        title!(title)
        annotate!(-0.75, 1.25, "time= $(rpad(round(t[i]; digits=1),4,"0"))")
    end
end
