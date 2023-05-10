using LinearAlgebra
using DifferentialEquations
using NeuralPDE, Lux
using Optimization, OptimizationOptimisers, OptimizationOptimJL, NLopt
using Plots
using NLsolve
using NeuralLyapunov

p_vsm = [
    # Outer Control VSM
    2.0, # Ta
    400.0, # kd
    20.0, # kω
    # Outer Control Reactive Droop
    0.2, #kq
    1000.0, #ωf
    # Inner Control
    0.59, #kpv
    736.0, #kiv
    0.0, #kffv
    0.0, #rv
    0.2, #lv
    1.27, #kpc
    14.3, #kic
    0.0, #kffi
    50.0, #ωad
    0.2, #kad
    # PLL
    500.0, #ω_lp
    0.084, #kp_pll
    4.69, #ki_pll
    # Filter
    0.08, #lf
    0.003, #rf
    0.074, #cf
    0.2, #lg
    0.01, #rg
    # References
    1.0, #ω_ref
    1.0229153603373207, #V_ref
    0.5025002107429647, #P_ref
    0.04541354000106637, # Q_ref
]

init_guess_x0 = [
    0.1979, #θ_oc
    1.0, #ω_oc
    0.0454, #q_oc
    0.0007, #ξd_ic
    -0.0001, #ξq_ic
    0.0703, #γd_ic
    -0.0069, #γq_ic
    1.0043, #ϕd_ic
    -0.0982, #ϕq_ic
    1.0091, #vd_pll
    0.0, #vq_pll
    0.0, #ϵ_pll
    0.1003, #θ_pll
    0.4925, #ir_cnv
    0.0794, #ii_cnv
    1.004, #vr_filter
    0.1011, #vi_filter
    0.5, #ir_filter
    0.0051, #ii_filter
]

function inverter_vsm!(dx, x, p, t)
    # Unwrap states
    θ_oc, ω_oc, q_oc, ξd_ic, ξq_ic = @view x[1:5]
    γd_ic, γq_ic, ϕd_ic, ϕq_ic, vd_pll = @view x[6:10]
    vq_pll, ϵ_pll, θ_pll, ir_cnv, ii_cnv = @view x[11:15]
    vr_filter, vi_filter, ir_filter, ii_filter = @view x[16:19]

    # Unwrap params
    # Outer Control
    Ta, kd, kω, kq, ωf = @view p[1:5]
    # Inner Control
    kpv, kiv, kffv, rv, lv = @view p[6:10]
    kpc, kic, kffi, ωad, kad = @view p[11:15]
    # PLL
    ω_lp, kp_pll, ki_pll = @view p[16:18]
    # Filter
    lf, rf, cf, lg, rg = @view p[19:23]
    # References
    ω_ref, V_ref, P_ref, Q_ref = @view p[24:27]

    # Additional params
    vr = 1.0
    vi = 0.0
    ω_sys = 1.0
    Ω_base = 2 * pi * 60

    # PLL calculations
    δω_pll = kp_pll * atan(vq_pll / vd_pll) + ki_pll * ϵ_pll
    ω_pll = δω_pll + ω_sys
    vd_filt_pll = sin(θ_pll + pi / 2) * vr_filter - cos(θ_pll + pi / 2) * vi_filter
    vq_filt_pll = cos(θ_pll + pi / 2) * vr_filter + sin(θ_pll + pi / 2) * vi_filter

    # Outer Loop Control
    pe = vr_filter * ir_filter + vi_filter * ii_filter
    qe = vi_filter * ir_filter + vr_filter * ii_filter
    v_ref_olc = V_ref + kq * (Q_ref - q_oc)

    dx[1] = Ω_base * (ω_oc - ω_sys) # θ_oc_dot
    dx[2] = (P_ref - pe - kd * (ω_oc - ω_pll) - kω * (ω_oc - ω_ref)) / Ta
    dx[3] = ωf * (qe - q_oc)

    # Inner Loop
    vd_filt_olc = sin(θ_oc + pi / 2) * vr_filter - cos(θ_oc + pi / 2) * vi_filter
    vq_filt_olc = cos(θ_oc + pi / 2) * vr_filter + sin(θ_oc + pi / 2) * vi_filter
    id_filt_olc = sin(θ_oc + pi / 2) * ir_filter - cos(θ_oc + pi / 2) * ii_filter
    iq_filt_olc = cos(θ_oc + pi / 2) * ir_filter + sin(θ_oc + pi / 2) * ii_filter
    id_cnv_olc = sin(θ_oc + pi / 2) * ir_cnv - cos(θ_oc + pi / 2) * ii_cnv
    iq_cnv_olc = cos(θ_oc + pi / 2) * ir_cnv + sin(θ_oc + pi / 2) * ii_cnv

    #Voltage control equations
    Vd_filter_ref = v_ref_olc - rv * id_filt_olc + ω_oc * lv * iq_filt_olc
    Vq_filter_ref = -rv * iq_filt_olc - ω_oc * lv * id_filt_olc
    dx[4] = Vd_filter_ref - vd_filt_olc # ξd_ic_dot
    dx[5] = Vq_filter_ref - vq_filt_olc # ξq_ic_dot

    #current control equations
    Id_cnv_ref = (
        kpv * (Vd_filter_ref - vd_filt_olc) + kiv * ξd_ic - cf * ω_oc * vq_filt_olc +
        kffi * id_filt_olc
    )
    Iq_cnv_ref = (
        kpv * (Vq_filter_ref - vq_filt_olc) +
        kiv * ξq_ic +
        cf * ω_oc * vd_filt_olc +
        kffi * iq_filt_olc
    )
    dx[6] = Id_cnv_ref - id_cnv_olc # γd_ic_dot
    dx[7] = Iq_cnv_ref - iq_cnv_olc # γq_ic_dot

    # Active damping equations
    Vd_cnv_ref = (
        kpc * (Id_cnv_ref - id_cnv_olc) + kic * γd_ic - lf * ω_oc * iq_cnv_olc +
        kffv * vd_filt_olc - kad * (vd_filt_olc - ϕd_ic)
    )
    Vq_cnv_ref = (
        kpc * (Iq_cnv_ref - iq_cnv_olc) +
        kic * γq_ic +
        lf * ω_oc * id_cnv_olc +
        kffv * vq_filt_olc - kad * (vq_filt_olc - ϕq_ic)
    )
    dx[8] = ωad * (vd_filt_olc - ϕd_ic) # ϕd_ic_dot
    dx[9] = ωad * (vq_filt_olc - ϕq_ic) # ϕq_ic_dot

    # PLL Differential Equations
    dx[10] = ω_lp * (vd_filt_pll - vd_pll) # vd_pll_dot
    dx[11] = ω_lp * (vq_filt_pll - vq_pll) # vq_pll_dot
    dx[12] = atan(vq_pll / vd_pll) # ϵ_pll_dot
    dx[13] = Ω_base * δω_pll # θ_pll_dot

    # Filter equations
    Vr_cnv = sin(θ_oc + pi / 2) * Vd_cnv_ref + cos(θ_oc + pi / 2) * Vq_cnv_ref
    Vi_cnv = -cos(θ_oc + pi / 2) * Vd_cnv_ref + sin(θ_oc + pi / 2) * Vq_cnv_ref

    dx[14] = (Ω_base / lf) * (Vr_cnv - vr_filter - rf * ir_cnv + ω_sys * lf * ii_cnv) #ir_cnv_dot
    dx[15] = (Ω_base / lf) * (Vi_cnv - vi_filter - rf * ii_cnv - ω_sys * lf * ir_cnv) #ii_cnv_dot
    dx[16] = (Ω_base / cf) * (ir_cnv - ir_filter + ω_sys * cf * vi_filter) #vr_filter_dot
    dx[17] = (Ω_base / cf) * (ii_cnv - ii_filter - ω_sys * cf * vr_filter) #vi_filter_dot
    dx[18] = (Ω_base / lg) * (vr_filter - vr - rg * ir_filter + ω_sys * lg * ii_filter) # ir_filter_dot
    dx[19] = (Ω_base / lg) * (vi_filter - vi - rg * ii_filter - ω_sys * lg * ir_filter)
    return
end

function inverter_vsm(x, p, t)
    # Unwrap states
    θ_oc, ω_oc, q_oc, ξd_ic, ξq_ic = @view x[1:5]
    γd_ic, γq_ic, ϕd_ic, ϕq_ic, vd_pll = @view x[6:10]
    vq_pll, ϵ_pll, θ_pll, ir_cnv, ii_cnv = @view x[11:15]
    vr_filter, vi_filter, ir_filter, ii_filter = @view x[16:19]

    # Unwrap params
    # Outer Control
    Ta, kd, kω, kq, ωf = @view p[1:5]
    # Inner Control
    kpv, kiv, kffv, rv, lv = @view p[6:10]
    kpc, kic, kffi, ωad, kad = @view p[11:15]
    # PLL
    ω_lp, kp_pll, ki_pll = @view p[16:18]
    # Filter
    lf, rf, cf, lg, rg = @view p[19:23]
    # References
    ω_ref, V_ref, P_ref, Q_ref = @view p[24:27]

    # Additional params
    vr = 1.0
    vi = 0.0
    ω_sys = 1.0
    Ω_base = 2 * pi * 60

    # PLL calculations
    δω_pll = kp_pll * atan(vq_pll / vd_pll) + ki_pll * ϵ_pll
    ω_pll = δω_pll + ω_sys
    vd_filt_pll = sin(θ_pll + pi / 2) * vr_filter - cos(θ_pll + pi / 2) * vi_filter
    vq_filt_pll = cos(θ_pll + pi / 2) * vr_filter + sin(θ_pll + pi / 2) * vi_filter

    # Outer Loop Control
    pe = vr_filter * ir_filter + vi_filter * ii_filter
    qe = vi_filter * ir_filter + vr_filter * ii_filter
    v_ref_olc = V_ref + kq * (Q_ref - q_oc)

    dx1 = Ω_base * (ω_oc - ω_sys) # θ_oc_dot
    dx2 = (P_ref - pe - kd * (ω_oc - ω_pll) - kω * (ω_oc - ω_ref)) / Ta
    dx3 = ωf * (qe - q_oc)

    # Inner Loop
    vd_filt_olc = sin(θ_oc + pi / 2) * vr_filter - cos(θ_oc + pi / 2) * vi_filter
    vq_filt_olc = cos(θ_oc + pi / 2) * vr_filter + sin(θ_oc + pi / 2) * vi_filter
    id_filt_olc = sin(θ_oc + pi / 2) * ir_filter - cos(θ_oc + pi / 2) * ii_filter
    iq_filt_olc = cos(θ_oc + pi / 2) * ir_filter + sin(θ_oc + pi / 2) * ii_filter
    id_cnv_olc = sin(θ_oc + pi / 2) * ir_cnv - cos(θ_oc + pi / 2) * ii_cnv
    iq_cnv_olc = cos(θ_oc + pi / 2) * ir_cnv + sin(θ_oc + pi / 2) * ii_cnv

    #Voltage control equations
    Vd_filter_ref = v_ref_olc - rv * id_filt_olc + ω_oc * lv * iq_filt_olc
    Vq_filter_ref = -rv * iq_filt_olc - ω_oc * lv * id_filt_olc
    dx4 = Vd_filter_ref - vd_filt_olc # ξd_ic_dot
    dx5 = Vq_filter_ref - vq_filt_olc # ξq_ic_dot

    #current control equations
    Id_cnv_ref = (
        kpv * (Vd_filter_ref - vd_filt_olc) + kiv * ξd_ic - cf * ω_oc * vq_filt_olc +
        kffi * id_filt_olc
    )
    Iq_cnv_ref = (
        kpv * (Vq_filter_ref - vq_filt_olc) +
        kiv * ξq_ic +
        cf * ω_oc * vd_filt_olc +
        kffi * iq_filt_olc
    )
    dx6 = Id_cnv_ref - id_cnv_olc # γd_ic_dot
    dx7 = Iq_cnv_ref - iq_cnv_olc # γq_ic_dot

    # Active damping equations
    Vd_cnv_ref = (
        kpc * (Id_cnv_ref - id_cnv_olc) + kic * γd_ic - lf * ω_oc * iq_cnv_olc +
        kffv * vd_filt_olc - kad * (vd_filt_olc - ϕd_ic)
    )
    Vq_cnv_ref = (
        kpc * (Iq_cnv_ref - iq_cnv_olc) +
        kic * γq_ic +
        lf * ω_oc * id_cnv_olc +
        kffv * vq_filt_olc - kad * (vq_filt_olc - ϕq_ic)
    )
    dx8 = ωad * (vd_filt_olc - ϕd_ic) # ϕd_ic_dot
    dx9 = ωad * (vq_filt_olc - ϕq_ic) # ϕq_ic_dot

    # PLL Differential Equations
    dx10 = ω_lp * (vd_filt_pll - vd_pll) # vd_pll_dot
    dx11 = ω_lp * (vq_filt_pll - vq_pll) # vq_pll_dot
    dx12 = atan(vq_pll / vd_pll) # ϵ_pll_dot
    dx13 = Ω_base * δω_pll # θ_pll_dot

    # Filter equations
    Vr_cnv = sin(θ_oc + pi / 2) * Vd_cnv_ref + cos(θ_oc + pi / 2) * Vq_cnv_ref
    Vi_cnv = -cos(θ_oc + pi / 2) * Vd_cnv_ref + sin(θ_oc + pi / 2) * Vq_cnv_ref

    dx14 = (Ω_base / lf) * (Vr_cnv - vr_filter - rf * ir_cnv + ω_sys * lf * ii_cnv) #ir_cnv_dot
    dx15 = (Ω_base / lf) * (Vi_cnv - vi_filter - rf * ii_cnv - ω_sys * lf * ir_cnv) #ii_cnv_dot
    dx16 = (Ω_base / cf) * (ir_cnv - ir_filter + ω_sys * cf * vi_filter) #vr_filter_dot
    dx17 = (Ω_base / cf) * (ii_cnv - ii_filter - ω_sys * cf * vr_filter) #vi_filter_dot
    dx18 = (Ω_base / lg) * (vr_filter - vr - rg * ir_filter + ω_sys * lg * ii_filter) # ir_filter_dot
    dx19 = (Ω_base / lg) * (vi_filter - vi - rg * ii_filter - ω_sys * lg * ir_filter)
    return [
        dx1,
        dx2,
        dx3,
        dx4,
        dx5,
        dx6,
        dx7,
        dx8,
        dx9,
        dx10,
        dx11,
        dx12,
        dx13,
        dx14,
        dx15,
        dx16,
        dx17,
        dx18,
        dx19,
    ]
end

# Initialize
init_model = (dx, x) -> inverter_vsm!(dx, x, p_vsm, 0.0)
sys_solve = nlsolve(init_model, init_guess_x0)
fixed_point = sys_solve.zero

# Construct Problem
tspan = (0.0, 2.0)
ODEprob = ODEProblem(inverter_vsm, fixed_point, tspan, p_vsm)

lb = -0.5 * ones(size(fixed_point));
ub = 1.5 * ones(size(fixed_point));

# Make log version
dim_output = 1
κ = 20.0
pde_system_log, lyapunov_func = NeuralLyapunovPDESystem(
    ODEprob,
    lb,
    ub,
    dim_output,
    relu = (t) -> log(1.0 + exp(κ * t)) / κ,
    fixed_point = fixed_point,
);

# Set up neural net 
dim_state = length(lb)
dim_hidden = 3 * dim_state
chain = [
    Lux.Chain(
        Dense(dim_state, dim_hidden, tanh),
        Dense(dim_hidden, dim_hidden, tanh),
        Dense(dim_hidden, 1, use_bias = false),
    ) for _ = 1:dim_output
]

# Define neural network discretization
strategy = QuasiRandomTraining(100)
discretization = PhysicsInformedNN(chain, strategy)

# Build optimization problem
prob_log = discretize(pde_system_log, discretization)
sym_prob_log = symbolic_discretize(pde_system_log, discretization)

callback = function (p, l)
    println("loss: ", l)
    return false
end

# Optimize log version
res = Optimization.solve(prob_log, Adam(); callback = callback, maxiters = 300)

# Optimize ReLU verion
pde_system_relu, _ = NeuralLyapunovPDESystem(ODEprob, lb, ub, dim_output);
prob_relu = discretize(pde_system_relu, discretization);
sym_prob_relu = symbolic_discretize(pde_system_relu, discretization);
prob_relu = Optimization.remake(prob_relu, u0 = res.u);
println("Switching from log(1 + κ exp(V̇))/κ to max(0,V̇)");
res = Optimization.solve(prob_relu, Adam(); callback = callback, maxiters = 300)
prob_relu = Optimization.remake(prob_relu, u0 = res.u);
println("Switching from Adam to BFGS");
res = Optimization.solve(prob_relu, BFGS(); callback = callback, maxiters = 300)

# Get numerical numerical functions
V_func, V̇_func =
    NumericalNeuralLyapunovFunctions(discretization.phi, res, lyapunov_func, ODEprob)

# TODO: Check Lyapunov functions
