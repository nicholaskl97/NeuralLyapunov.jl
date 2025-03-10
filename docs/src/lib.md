# NeuralLyapunovProblemLibrary.jl

For testing and benchmarking purposes, the NeuralLyapunovProblemLibrary.jl package is provided in the same [repository](https://github.com/SciML/NeuralLyapunov.jl) (in the `lib` folder).
This package contains ModelingToolkit models for the simple pendulum ([`Pendulum`](@ref)), the double pendulum ([`DoublePendulum`](@ref)), a planar approximation of the quadrotor ([`QuadrotorPlanar`](@ref)), and a full 3D model of the quadrotor ([`Quadrotor3D`](@ref)).
Additionally, when used with the Plots.jl package, methods are provided for generating animations of the trajectory of each model (shown below).

```@contents
Pages = Main.NEURALLYAPUNOVPROBLEMLIBRARY_PAGES[2:end]
Depth = 3
```

|                                                          |                                                        |
| -------------------------------------------------------- | ------------------------------------------------------ |
| ![3D quadrotor animation](imgs/quadrotor_3d.gif)         | ![Double pendulum animation](imgs/double_pendulum.gif) |
| ![Planar quadrotor animation](imgs/quadrotor_planar.gif) | ![Pendulum animation](imgs/pendulum.gif)               |