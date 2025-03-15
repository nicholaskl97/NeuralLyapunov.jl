using ModelingToolkit
import ModelingToolkit: inputs, generate_control_function
using NeuralLyapunovProblemLibrary
using OrdinaryDiffEq
using Plots
using LinearAlgebra
using ControlSystemsBase: lqr, Continuous
using Test, StableRNGs

rng = StableRNG(0)

#################################### Hovering quadrotor ####################################
println("3D quadrotor vertical only test")

function π_vertical_only(x, p; z_goal=0.0, k_p=1.0, k_d=1.0)
    z = x[3]
    ż = x[9]
    m, g = p[1:2]
    Izz = p[end]

    # Calculate a characteristic length scale for scaling the PD controller
    r = sqrt(Izz / m)
    k_p = k_p * m * g / r
    k_d = k_d * m * sqrt(g / r)

    # PD controller for vertical motion, limited to nonnegative thrust
    T0 = m * g
    T = T0 - k_p * (z - z_goal) - k_d * ż
    return [T, 0, 0, 0]
end

@named quadrotor_3d = Quadrotor3D()

_, _, p, quadrotor_3d_simplified = generate_control_function(
    quadrotor_3d;
    simplify=true,
    split=false
)

t, = independent_variables(quadrotor_3d)
Dt = Differential(t)
x = setdiff(unknowns(quadrotor_3d), inputs(quadrotor_3d))

params = map(Base.Fix1(getproperty, quadrotor_3d), toexpr.(p))
u = map(
        Base.Fix1(getproperty, quadrotor_3d),
        toexpr.(getproperty.(inputs(quadrotor_3d_simplified), :f))
)
x = map(
        Base.Fix1(getproperty, quadrotor_3d),
        toexpr.(getproperty.(x, :f))
)
q, q̇ = x[1:6], x[7:12]

@named vertical_only_controller = ODESystem(
    u .~ π_vertical_only(x, params),
    t,
    vcat(x, u),
    params
)

@named quadrotor_3d_vertical_only = compose(vertical_only_controller, quadrotor_3d)
quadrotor_3d_vertical_only = structural_simplify(quadrotor_3d_vertical_only)

# Hovering
# Assume rotors are negligible mass when calculating the moment of inertia
m, L = ones(2)
g = 1.0
Ixx = Iyy = m * L^2 / 6
Izz = m * L^2 / 3
Ixy = Ixz = Iyz = 0.0
p = Dict(params .=> [m, g, Ixx, Ixy, Ixz, Iyy, Iyz, Izz])

τ = sqrt(L / g)

x0 = Dict(x .=> zeros(12))
x0[q[3]] = rand(rng)
x0[q̇[3]] = rand(rng)
x0[q̇[6]] = rand(rng)

prob = ODEProblem(quadrotor_3d_vertical_only, x0, 15τ, p)
sol = solve(prob, Tsit5())

x_end, y_end, z_end, φ_end, θ_end, ψ_end = sol[q][end]
vx_end, vy_end, vz_end, ωφ_end, ωθ_end, ωψ_end = sol[q̇][end]
@test x_end ≈ 0.0 atol=1e-4
@test y_end ≈ 0.0 atol=1e-4
@test z_end ≈ 0.0 atol=1e-4
@test φ_end ≈ 0.0 atol=1e-4
@test θ_end ≈ 0.0 atol=1e-4
@test vx_end ≈ 0.0 atol=1e-4
@test vy_end ≈ 0.0 atol=1e-4
@test vz_end ≈ 0.0 atol=1e-4
@test ωφ_end ≈ 0.0 atol=1e-4
@test ωθ_end ≈ 0.0 atol=1e-4
@test ωψ_end ≈ x0[q̇[6]] atol=1e-4

anim = plot_quadrotor_3d(
    sol,
    [m, g, Ixx, Ixy, Ixz, Iyy, Iyz, Izz];
    x_symbol=q[1],
    y_symbol=q[2],
    z_symbol=q[3],
    φ_symbol=q[4],
    θ_symbol=q[5],
    ψ_symbol=q[6],
    T_symbol=u[1],
    τφ_symbol=u[2],
    τθ_symbol=u[3],
    τψ_symbol=u[4]
)
@test anim isa Plots.Animation
# gif(anim, fps = 50)

############################## LQR planar quadrotor controller #############################
println("3D quadrotor LQR test")

function quadrotor_3d_lqr_matrix(
    p;
    x_eq = zeros(12),
    u_eq = [p[1]*p[2], 0, 0, 0],
    Q = I(12),
    R = I(4)
)
    u = inputs(quadrotor_3d)
    x = setdiff(unknowns(quadrotor_3d), u)
    params = parameters(quadrotor_3d)

    op = Dict(vcat(x .=> x_eq, u .=> u_eq, params .=> p))

    mats, sys = linearize(quadrotor_3d, u, x; op)

    # Create permutation matrices Px : x_new = Px * x and Pu : u_new = Pu * u
    x_new = unknowns(sys)
    u_new = inputs(sys)

    Px = (x_new .- x') .=== 0
    Pu = (u_new .- u') .=== 0

    A_lin = Px' * mats[:A] * Px
    B_lin = Px' * mats[:B] * Pu

    return lqr(Continuous, A_lin, B_lin, Q, R)
end

function π_lqr(p; x_eq = zeros(12), u_eq = [p[1]*p[2], 0, 0, 0], Q = I(12), R = I(4))
    L = quadrotor_3d_lqr_matrix(p; Q, R, x_eq, u_eq)
    return (x) -> -L * (x - x_eq) + u_eq
end

@named quadrotor_3d = Quadrotor3D()

_, _, p, quadrotor_3d_simplified = generate_control_function(
    quadrotor_3d;
    simplify=true,
    split=false
)

t, = independent_variables(quadrotor_3d)
Dt = Differential(t)
x = setdiff(unknowns(quadrotor_3d), inputs(quadrotor_3d))

params = map(Base.Fix1(getproperty, quadrotor_3d), toexpr.(p))
u = map(
        Base.Fix1(getproperty, quadrotor_3d),
        toexpr.(getproperty.(inputs(quadrotor_3d_simplified), :f))
)
x = map(
        Base.Fix1(getproperty, quadrotor_3d),
        toexpr.(getproperty.(x, :f))
)
q, q̇ = x[1:6], x[7:12]

# Assume rotors are negligible mass when calculating the moment of inertia
m, L = ones(2)
g = 1.0
Ixx = Iyy = m * L^2 / 6
Izz = m * L^2 / 3
Ixy = Ixz = Iyz = 0.0
p = [m, g, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]

@named lqr_controller = ODESystem(
    u .~ π_lqr(p)(x),
    t,
    vcat(x, u),
    params
)

@named quadrotor_3d_lqr = compose(lqr_controller, quadrotor_3d)
quadrotor_3d_lqr = structural_simplify(quadrotor_3d_lqr)

# Fly to origin
p = Dict(params .=> p)
δ = 0.5
x0 = Dict(x .=> δ .* (2 .* rand(rng, 12) .- 1))
τ = sqrt(L / g)

prob = ODEProblem(quadrotor_3d_lqr, x0, 15τ, p)
sol = solve(prob, Tsit5())

x_end, y_end, z_end, φ_end, θ_end, ψ_end = sol[q][end]
vx_end, vy_end, vz_end, ωφ_end, ωθ_end, ωψ_end = sol[q̇][end]
@test x_end ≈ 0.0 atol=1e-4
@test y_end ≈ 0.0 atol=1e-4
@test z_end ≈ 0.0 atol=1e-4
@test φ_end ≈ 0.0 atol=1e-4
@test θ_end ≈ 0.0 atol=1e-4
@test ψ_end ≈ 0.0 atol=1e-4
@test vx_end ≈ 0.0 atol=1e-4
@test vy_end ≈ 0.0 atol=1e-4
@test vz_end ≈ 0.0 atol=1e-4
@test ωφ_end ≈ 0.0 atol=1e-4
@test ωθ_end ≈ 0.0 atol=1e-4
@test ωψ_end ≈ 0.0 atol=1e-4

anim = plot_quadrotor_3d(
    sol,
    [m, g, Ixx, Ixy, Ixz, Iyy, Iyz, Izz];
    x_symbol=q[1],
    y_symbol=q[2],
    z_symbol=q[3],
    φ_symbol=q[4],
    θ_symbol=q[5],
    ψ_symbol=q[6],
    T_symbol=u[1],
    τφ_symbol=u[2],
    τθ_symbol=u[3],
    τψ_symbol=u[4]
)
@test anim isa Plots.Animation
# gif(anim, fps = 50)
