module NeuralLyapunovProblemLibraryPlotsExt

using NeuralLyapunovProblemLibrary
using Plots
using Rotations: RotZXY

include("pendulum_plot.jl")
include("double_pendulum_plot.jl")
include("quadrotor_plot.jl")

end
