using DifferentialEquations
using Plots
using NLsolve

r_line = 0.01 # line+gen resistance pu
x_line = 0.1 # line+gen reactance pu
y_line = 1.0 / (r_line + x_line * 1im) # line+gen admittance

g_line = real(y_line) # line conductance pu
b_line = imag(y_line) # line+gen susceptance pu

Pm = 0.5 # mechanical power from gen in pu

E = 1.01 # Generator voltage
E_∞ = 1.0 # Infinite bus voltage
H = 8.0 # Generator Inertia [seconds]
Ω_b = 2 * pi * 60 # Base Frequency
T = 0.1 # Damping Coefficient

p_omib = [
    Pm - E^2 * g_line, # P: Total Power with losses
    -E * E_∞ * b_line, # C: sine coefficient term
    -E * E_∞ * g_line, # D: cosine coefficient term
    T, # damping coefficient
    H, # inertia term
    Ω_b, # nominal frequency
]

init_guess_x0 = [0.05, 0.0]

"""
This system describes a classic machine model against an infinite bus, described as:
δ_dot = ω
ω_dot = (Ω_b / H) * (P - C * sin(δ) - D * cos(δ) - T * ω)
where δ is the rotor angle and ω is the frequency deviation from the synchronous frequency.

This model incorporates line losses and has a general Lyapunov function in the usual sense as:
V(δ, ω) = (H / Ω_b) * (ω^2 / 2) - P * δ - C * cos(δ) + D * sin(δ) + α
where α is an arbitrary constant. It can be shown that the derivative of V along the orbits satisfies:
V_dot = - T * ω^2 ≤ 0
which is a negative semi-definite function. V also satisfies the requirements of LaSalle's invariance principle
and hence V can be used to study the stability of this system in the usual way.
"""
function omib_diffeq!(dx, x, p, t)
    δ, ω = @view x[1:2]
    P, C, D, T, H, Ω_b = @view p[1:6]

    dx[1] = ω
    dx[2] = (Ω_b / H) * (P - C * sin(δ) - D * cos(δ) - T * ω)
end


init_model = (dx, x) -> omib_diffeq!(dx, x, p_omib, 0.0)
sys_solve = nlsolve(init_model, init_guess_x0)
init_x0 = sys_solve.zero

tspan = (0.0, 5.0)
prob = ODEProblem(omib_diffeq!, init_x0, tspan, p_omib)

# Solve Problem
sol = solve(prob, TRBDF2())
# Should stay still since is a stable operating point
plot(sol)

# Add some dynamics
x0_perturbed = [0.06, 0.0]
prob2 = ODEProblem(omib_diffeq!, x0_perturbed, tspan, p_omib)

# Solve Problem
sol2 = solve(prob2, TRBDF2())
# Should stay still since is a stable operating point
plot(sol2)
