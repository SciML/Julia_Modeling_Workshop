### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 07dae75d-c939-4393-9452-67817f44ed7a
using LinearAlgebra

# ╔═╡ a950257a-9773-11f0-2da5-ffdda63d000c
using DifferentialEquations

# ╔═╡ 62a5c4e9-4c09-445b-b1b0-258be58e3a2b
using Plots

# ╔═╡ a95040fa-9773-11f0-0b7e-e92b841a56e1
using NonlinearSolve

# ╔═╡ a95046a4-9773-11f0-1018-0fefed30aefa
using Optimization, OptimizationOptimJL

# ╔═╡ a95054be-9773-11f0-0a1e-e94b285d51f6
# With automatic differentiation for gradients
using OptimizationOptimisers, ForwardDiff

# ╔═╡ a95055ae-9773-11f0-043d-15e66f32784a
# Box constraints
using OptimizationMOI, Ipopt

# ╔═╡ a950640e-9773-11f0-29e7-f3fff8c09e6c
using Symbolics

# ╔═╡ a9506c18-9773-11f0-0e54-431798773d2a
using ModelingToolkit

# ╔═╡ e6b1f3bb-671b-4d3b-a9a0-eab2f62aac51
using ModelingToolkit: t_nounits as t, D_nounits as D

# ╔═╡ a9509ab4-9773-11f0-0e4d-95d8808c524e
using BenchmarkTools

# ╔═╡ d96df0b2-bebb-472b-a077-874ebed3c16f
md"""
# Julia Scientific Modeling Quick Reference

## Table of Contents
- Basic Syntax
- Arrays and Linear Algebra
- Solving ODEs
- Solving Nonlinear Systems
- Optimization
- Symbolic Computing
- Plotting
- Performance Tips
"""

# ╔═╡ ac30d390-e879-4333-8c45-6498378d501d
md"""
## Basic Syntax

### Variables and Types
"""

# ╔═╡ 474e5f09-0de1-4dc6-81e0-8e759acebcdf
begin
    x_var = 3.14                 # Float64
    n_var = 42                   # Int64
    s_var = "Hello"              # String
    c_var = 'A'                  # Char
    b_var = true                 # Bool
end

# ╔═╡ d2abc7fb-b95a-451d-a5da-109c0101aba1
# Type conversions
# Int(3.14)                # Error! Use round/floor/ceil
round(Int, 3.14)         # 3

# ╔═╡ cde593cf-3ae7-47f6-8883-7393af3d94dd
parse(Int, "42")         # 42

# ╔═╡ 1724b3a2-ed4f-44f7-90b3-90cd70664d9d
string(42)               # "42"

# ╔═╡ 43e45b51-3cb1-4c85-8b84-5fe681cb8a42
md"""
### Functions
"""

# ╔═╡ 771c5eb5-6e94-4ccf-a0d7-b629b1452fce
# One-liner
f1(x) = x^2 + 2x - 1

# ╔═╡ e753679d-34ed-4d30-9274-b9476bb0acd5
# Multi-line
function g1(x, y)
    z = x + y
    return z^2
end

# ╔═╡ a9c726e7-2357-4110-9ea8-99b374e88639
# Anonymous functions
map(x -> x^2, [1,2,3])

# ╔═╡ f7330414-a677-4069-a7cb-3239bac81570
# Multiple dispatch
h1(x::Float64) = "Got a float"

# ╔═╡ 9d6d56a4-c982-47c0-9407-2d8a722e3e7f
h1(x::Int) = "Got an integer"

# ╔═╡ 2215dbbc-1c35-4c2a-9749-f52c58b09f4d
# Optional & keyword arguments
function myplot(x, y; color="red", width=2)
    return "Plotting with color=$color and width=$width"
end

# ╔═╡ 1cd34230-1299-4af9-8093-e369e47f05ed
md"""
### Control Flow
"""

# ╔═╡ 3ccae72a-565f-4f99-b6d9-2fb2bc44d9ab
x_test = 5

# ╔═╡ e4bc599b-b03d-4300-a6b1-55ea8974d29c
# If-else
if x_test > 0
    println("positive")
elseif x_test < 0
    println("negative")
else
    println("zero")
end

# ╔═╡ 85d34909-dba6-4368-9e7d-137d71273f97
# Ternary operator
y_test = x_test > 0 ? 1 : -1

# ╔═╡ 03c30a6d-90f7-4d43-8cd2-4d2904437785
# For loops
for i in 1:5
    println(i)
end

# ╔═╡ 4047f4b2-fdb6-460f-b789-5674bb5cc649
x_while = 8.0

# ╔═╡ 69359e85-cbbc-4145-9d4b-56cd49a88f78
# While loops
while x_while > 1
    println(x_while)
	break
end

# ╔═╡ f005406f-38e7-485c-92ba-1829b1534058
# Comprehensions
[x^2 for x in 1:10]

# ╔═╡ 448451e1-24b2-4d19-9e91-26cfd3f16493
[x*y for x in 1:3, y in 1:3]

# ╔═╡ f56c31ee-1528-4913-999f-40aa76026b23
md"""
## Arrays and Linear Algebra

### Creating Arrays
"""

# ╔═╡ fd4b5a08-e1bd-4d20-a39d-df66a3d9fd75
# Vectors
v1 = [1, 2, 3]                    # Column vector

# ╔═╡ 27f0e376-3828-4b0a-93b3-531e268b1248
v2 = [1; 2; 3]                    # Same

# ╔═╡ ca4b7b33-608a-4d0f-9b40-4084d6c1e756
v3 = [1 2 3]'                     # Transpose row to column

# ╔═╡ ba54aa80-a8cd-4972-b407-c9394e7556bc
# Matrices
A1 = [1 2 3; 4 5 6]              # 2×3 matrix

# ╔═╡ 8286672c-85e6-4ba5-9540-3d2ec0abd527
A2 = [1 2 3
      4 5 6]                      # Same

# ╔═╡ e9fd0565-f9e1-463f-863d-62b66f863b9c
# Special matrices
zeros(3, 4)                      # 3×4 zero matrix

# ╔═╡ cc9f470f-051d-4cd8-8415-008019f30fa1
ones(2, 3)                       # 2×3 ones matrix

# ╔═╡ 26d12a25-9151-4c20-8a24-eb7723b7fb32
I(3)                            # 3×3 identity (from LinearAlgebra)

# ╔═╡ 09d5e5cc-99c8-41a5-ac0d-be1e6d07d1b7
diagm([1, 2, 3])                # Diagonal matrix

# ╔═╡ a94ff584-9773-11f0-169a-69b540829e8a
rand(3, 3)                      # Random uniform [0,1]

# ╔═╡ a94ff5c8-9773-11f0-3586-17e283ccb322
randn(3, 3)                     # Random normal

# ╔═╡ a94ff5fa-9773-11f0-0af6-e582f16ebebe
# Ranges and linspace
1:10                            # 1, 2, ..., 10

# ╔═╡ a94ff64a-9773-11f0-3bed-bfd3e7f3a7a7
1:2:10                          # 1, 3, 5, 7, 9

# ╔═╡ a94ff67c-9773-11f0-09dc-2ba7b0957b86
range(0, 1, length=11)          # 0.0, 0.1, ..., 1.0

# ╔═╡ a94ff6ba-9773-11f0-375f-cfc5e027869b
md"""
### Array Operations
"""

# ╔═╡ a94ff6f4-9773-11f0-0d0d-2d16f63c3383
# Indexing (1-based!)
A_test = rand(3, 3)

# ╔═╡ 7182b0f4-c3fd-4bd4-b41a-91a66a9c485b
A_test[1, 2]                         # Element at row 1, col 2

# ╔═╡ a94ff758-9773-11f0-16d2-05eaa6cd8cd0
A_test[1, :]                         # First row

# ╔═╡ a94ff794-9773-11f0-0969-755e398c46a1
A_test[:, 2]                         # Second column

# ╔═╡ a94ff7d0-9773-11f0-3ae0-1919a24ec6c1
A_test[1:2, 2:3]                     # Submatrix

# ╔═╡ a94ff802-9773-11f0-32e3-dd8a527a3f92
# Element-wise operations (broadcasting)
B_test = rand(3, 3)

# ╔═╡ 0ff4929c-d835-4168-986f-3d5ad3e33b7d
A_test .+ B_test                          # Element-wise addition

# ╔═╡ a94ff882-9773-11f0-0349-7599be769c50
A_test .* B_test                          # Element-wise multiplication

# ╔═╡ a94ff8d4-9773-11f0-1d6a-518b9e35a092
sin.(A_test)                         # Apply sin to each element

# ╔═╡ a94ff91a-9773-11f0-3d5e-b56619ad0feb
A_test .^ 2                          # Square each element

# ╔═╡ a94ff958-9773-11f0-0f6c-6b1eabaef6ac
# Matrix operations
A_test * B_test                           # Matrix multiplication

# ╔═╡ a94ff9a6-9773-11f0-0634-07a9553513b1
A_test' # or transpose(A_test)              # Transpose

# ╔═╡ a94ff9e2-9773-11f0-22fd-0d005fafc010
inv(A_test)                          # Inverse

# ╔═╡ a94ffa1c-9773-11f0-0960-adb43c187185
det(A_test)                          # Determinant

# ╔═╡ a94ffa5a-9773-11f0-3006-c555024c3d7b
tr(A_test)                           # Trace

# ╔═╡ a94ffa8c-9773-11f0-3e11-fb6b6a0ef529
eigvals(A_test)                      # Eigenvalues

# ╔═╡ a95004aa-9773-11f0-3ef9-c7b4255fb618
eigvecs(A_test)                      # Eigenvectors

# ╔═╡ a9500806-9773-11f0-3761-3d9d108fe1fa
norm(v1)                         # Vector/matrix norm

# ╔═╡ 1cb46d3d-0a35-470c-91bf-c766621d561f
# Solving linear systems
b_vec = rand(3)

# ╔═╡ 55e65992-a43f-4a2f-a78c-7435542cc559
A_test \ b_vec                           # Solve Ax = b

# ╔═╡ a9500d36-9773-11f0-28af-db3304977705
md"""
### Useful Array Functions
"""

# ╔═╡ a9500fb8-9773-11f0-33f2-7113956b5dec
size(A_test)                         # Tuple of dimensions

# ╔═╡ a9502296-9773-11f0-1211-67ce99ea71c4
size(A_test, 1)                      # Number of rows

# ╔═╡ a950235e-9773-11f0-3653-4b3abaf73301
length(v1)                       # Total number of elements

# ╔═╡ a95023a4-9773-11f0-2787-958f5163342b
ndims(A_test)                        # Number of dimensions

# ╔═╡ a95023e0-9773-11f0-3295-911e8cea76e9
reshape(A_test, 1, 9)                # Reshape array

# ╔═╡ a950241c-9773-11f0-26d0-c79549537d16
vec(A_test)                          # Flatten to vector

# ╔═╡ a9502476-9773-11f0-3c61-3deb69a5acc8
cat(A_test, B_test, dims=1)              # Concatenate (vcat for vertical)

# ╔═╡ a95024b2-9773-11f0-09c4-b58fd0feb74a
hcat(A_test, B_test)                      # Horizontal concatenation

# ╔═╡ a95024e4-9773-11f0-3f92-af591ba245d4
vcat(A_test, B_test)                      # Vertical concatenation

# ╔═╡ a9502516-9773-11f0-2386-272f2fed3bf5
md"""
## Solving ODEs

### Basic ODE Solving with DifferentialEquations.jl
"""

# ╔═╡ a95025be-9773-11f0-049c-9b8fa5d6fbe6
# Define ODE: du/dt = f(u, p, t)
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

# ╔═╡ a9502642-9773-11f0-37ed-b92da7ad835c
# Initial condition
u0_lorenz = [1.0, 0.0, 0.0]

# ╔═╡ a9502688-9773-11f0-375d-95481f3c34f0
# Parameters
p_lorenz = [10.0, 28.0, 8/3]

# ╔═╡ a95026ce-9773-11f0-0997-f7c1b6d2ecdc
# Time span
tspan_lorenz = (0.0, 100.0)

# ╔═╡ a950270a-9773-11f0-04fb-e5cbd99f623c
# Define problem
prob_lorenz = ODEProblem(lorenz!, u0_lorenz, tspan_lorenz, p_lorenz)

# ╔═╡ a9502758-9773-11f0-3260-e195cfdc1c9b
# Solve
sol_lorenz = solve(prob_lorenz, Tsit5())

# ╔═╡ a95027a0-9773-11f0-1280-5129046d073f
# Plot solution
plot(sol_lorenz, vars=(1,2,3))

# ╔═╡ a95027dc-9773-11f0-26c2-5dcc62bc7fe7
md"""
### Common ODE Solvers
"""

# ╔═╡ a9502822-9773-11f0-2fca-a1091e080746
# Explicit methods (non-stiff)
sol_tsit5 = solve(prob_lorenz, Tsit5())      # 5th order Runge-Kutta

# ╔═╡ a9502872-9773-11f0-37af-adb9cfdcd57e
sol_vern7 = solve(prob_lorenz, Vern7())      # 7th order accurate

# ╔═╡ a95028ae-9773-11f0-1202-5d63738a1835
# Implicit methods (stiff)
sol_ros = solve(prob_lorenz, Rosenbrock23())

# ╔═╡ a9502d86-9773-11f0-2f1c-fba9f1352c9c
sol_rodas = solve(prob_lorenz, Rodas5P())

# ╔═╡ a950327c-9773-11f0-29d3-eff1b9d6883d
sol_radau = solve(prob_lorenz, FBDF())

# ╔═╡ a95035f6-9773-11f0-0e38-f7cccd719bfd
# Adaptive timestep control
sol_tol = solve(prob_lorenz, abstol=1e-8, reltol=1e-8)

# ╔═╡ a950390a-9773-11f0-2b1f-b36846c46101
# Save at specific times
sol_save = solve(prob_lorenz, saveat=0.1)

# ╔═╡ a9503bbe-9773-11f0-2864-2106756926c4
sol_times = solve(prob_lorenz, saveat=[1.0, 2.0, 5.0])

# ╔═╡ a9503e78-9773-11f0-30ba-cb2c532d4e73
md"""
### Parameterized Functions and Callbacks
"""

# ╔═╡ a9503ed4-9773-11f0-1f0d-2db7bd6de83b
# Parameterized ODE
function pendulum!(du, u, p, t)
    g, L = p
    θ, ω = u
    du[1] = ω
    du[2] = -(g/L) * sin(θ)
end

# ╔═╡ a9503f38-9773-11f0-113b-51dbf80ae71b
# With callbacks (events)
condition(u, t, integrator) = u[1]  # Stop when u[1] = 0

# ╔═╡ a9503f92-9773-11f0-0bd9-6d85dd75c400
affect!(integrator) = terminate!(integrator)

# ╔═╡ a9503fc4-9773-11f0-0f8f-cfc89571781d
cb = ContinuousCallback(condition, affect!)

# ╔═╡ 5e2ca688-a485-4fa2-859b-8470f24deefd
u0_pend = [1.0, 0.0]

# ╔═╡ 01612662-ca6f-4e2f-8922-3022fb6adb84
p_pend = [9.81, 1.0]

# ╔═╡ dcaf27d3-83ad-4574-9477-7cbc2ce05e2f
tspan_pend = (0.0, 10.0)

# ╔═╡ 6cad5ecc-3b63-4611-a8a2-5b9ccc620e69
prob_pend = ODEProblem(pendulum!, u0_pend, tspan_pend, p_pend)

# ╔═╡ a9504078-9773-11f0-072e-73fff6f8021b
sol_cb = solve(prob_pend, Tsit5(), callback=cb)

# ╔═╡ a95040aa-9773-11f0-194a-dd335a206bc5
md"""
## Solving Nonlinear Systems

### Using NonlinearSolve.jl
"""

# ╔═╡ a950412c-9773-11f0-3db2-7df6300679fd
# Define system: f(u, p) = 0
function nl_system!(du, u, p)
    du[1] = u[1]^2 + u[2]^2 - 4
    du[2] = u[1] - u[2]
end

# ╔═╡ a9504190-9773-11f0-2b32-cdd5293c5a17
# Initial guess
u0_nl = [1.0, 1.0]

# ╔═╡ a95041cc-9773-11f0-2c72-dbf74fb1944f
# Define problem
prob_nl = NonlinearProblem(nl_system!, u0_nl)

# ╔═╡ a9504212-9773-11f0-2581-7bd0c046d9ac
# Solve with different algorithms
sol_newton = solve(prob_nl, NewtonRaphson())

# ╔═╡ a9504258-9773-11f0-26a5-d3cfe8b2533e
sol_trust = solve(prob_nl, TrustRegion())

# ╔═╡ a950428a-9773-11f0-1fc1-55cd36659bdf
sol_lm = solve(prob_nl, LevenbergMarquardt())

# ╔═╡ a95042bc-9773-11f0-07a3-756736127a69
# Access solution
u_sol = sol_newton.u

# ╔═╡ a95042f8-9773-11f0-37fc-519e71b86624
# For scalar equations
f_scalar(u, p) = u^3 - 2u - 5

# ╔═╡ a950433e-9773-11f0-12ae-a77bc5063503
u0_scalar = 2.0

# ╔═╡ a9504370-9773-11f0-2b30-b5d08ae8f733
prob_scalar = NonlinearProblem(f_scalar, u0_scalar)

# ╔═╡ a95043a2-9773-11f0-2185-3b48fc34a363
sol_scalar = solve(prob_scalar)

# ╔═╡ a95043d4-9773-11f0-2b70-3157562eac2b
md"""
## Optimization

### Using Optimization.jl
"""

# ╔═╡ a950494c-9773-11f0-0c31-4121b08a6307
# Define objective function
rosenbrock(x, p) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

# ╔═╡ a9504c08-9773-11f0-16fb-7f24591a7cd7
# Initial guess
x0_opt = [0.0, 0.0]

# ╔═╡ 662a3264-536d-4001-94be-6533c235dde0
optf = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())

# ╔═╡ a9504ece-9773-11f0-095f-293d02c85b7c
# Create optimization problem
optprob1 = OptimizationProblem(optf, x0_opt)

# ╔═╡ a9505192-9773-11f0-3f74-f1f69349557e
# Solve with different algorithms
sol_bfgs = solve(optprob1, BFGS())

# ╔═╡ a950543c-9773-11f0-3871-dfdb831d101b
sol_newton_opt = solve(optprob1, Newton())

# ╔═╡ a9505478-9773-11f0-0241-671420b679c8
sol_nelder = solve(optprob1, NelderMead())

# ╔═╡ a950550e-9773-11f0-3ee1-3f355a7a63e3


# ╔═╡ a9505540-9773-11f0-0fa0-09f98c7bdffc
optprob2 = OptimizationProblem(optf, x0_opt)

# ╔═╡ a950557c-9773-11f0-254c-55abc2f49428
sol_adam = solve(optprob2, ADAM(), maxiters=1000)

# ╔═╡ a95055f4-9773-11f0-2051-ed5cbb17cd58
f_constrained(x, p) = (x[1] - 2)^2 + (x[2] - 3)^2

# ╔═╡ 3bcc1f05-eb24-4cad-8d41-55d08d0038de
optf2 = OptimizationFunction(f_constrained, Optimization.AutoForwardDiff())

# ╔═╡ a9505626-9773-11f0-3a85-2f4ef025bbfb
optprob3 = OptimizationProblem(optf2, x0_opt, lb=[0.0, 0.0], ub=[10.0, 10.0])

# ╔═╡ a9505662-9773-11f0-045e-3de748f29d5c
sol_ipopt = solve(optprob3, Ipopt.Optimizer())

# ╔═╡ a9505694-9773-11f0-1ba2-69efa4d12e68
# Access results
x_opt = sol_bfgs.u

# ╔═╡ a95056da-9773-11f0-2098-af3b528a1535
f_min = sol_bfgs.objective

# ╔═╡ a950570c-9773-11f0-23b0-f5a328397f3f
md"""
### Using JuMP.jl for Constrained Optimization
"""

# ╔═╡ a950575c-9773-11f0-1ffd-0171fc8be532
import JuMP
# Ipopt already loaded

# ╔═╡ a95057a2-9773-11f0-3d2f-e9ba06c45451
# Create model
model = JuMP.Model(Ipopt.Optimizer)

# ╔═╡ a95057e8-9773-11f0-0c67-116a6d9b6142
# Variables
JuMP.@variable(model, x_jump >= 0)

# ╔═╡ a9505824-9773-11f0-2b42-b1d8386aa0a9
JuMP.@variable(model, y_jump >= 0)

# ╔═╡ a9505856-9773-11f0-0bbf-67056a82bc97
# Objective
JuMP.@objective(model, Min, x_jump^2 + y_jump^2)

# ╔═╡ a95058c4-9773-11f0-12eb-a9740f1716ea
# Constraints
JuMP.@constraint(model, x_jump + y_jump == 1)

# ╔═╡ a9505900-9773-11f0-028e-65f34de357e7
JuMP.@constraint(model, x_jump <= 0.8)

# ╔═╡ a9505932-9773-11f0-2ba7-37b335a1d4a2
# Solve
JuMP.optimize!(model)

# ╔═╡ a9505bbc-9773-11f0-0a63-e9b55afb8d71
# Get results
JuMP.value(x_jump), JuMP.value(y_jump)

# ╔═╡ a9505ec8-9773-11f0-1ac6-97047a2fb56d
JuMP.objective_value(model)

# ╔═╡ a9506166-9773-11f0-32dd-df7461826e15
md"""
## Symbolic Computing

### Using Symbolics.jl
"""

# ╔═╡ a95066c0-9773-11f0-30f5-b927b25710d5
# Define symbolic variables
@variables x_sym y_sym z_sym t_sym

# ╔═╡ a9506986-9773-11f0-2cee-59dcf3ac9fba
# Build expressions
expr1 = x_sym^2 + 2x_sym*y_sym + y_sym^2

# ╔═╡ a95069cc-9773-11f0-2f88-af68a4958443
expr2 = sin(x_sym) * exp(-t_sym)

# ╔═╡ a9506a08-9773-11f0-28da-4700cbb01464
# Derivatives
Dx = Differential(x_sym)

# ╔═╡ a9506a44-9773-11f0-2050-272f876a7fa2
Dx(x_sym^3 + 2x_sym)                    # 3x^2 + 2

# ╔═╡ a9506a7e-9773-11f0-220f-912ac0e463b1
# Gradients and Jacobians
@variables x_vec[1:3]

# ╔═╡ a9506ab0-9773-11f0-1c59-81f4ae3a85d1
f_sym = x_vec[1]^2 + x_vec[2]*x_vec[3]

# ╔═╡ a9506aee-9773-11f0-0039-f5a63067f405
Symbolics.gradient(f_sym, x_vec)

# ╔═╡ a9506b34-9773-11f0-1e68-5d240efc214b
simplify(sin(x_sym)^2 + cos(x_sym)^2)

# ╔═╡ a9506b8e-9773-11f0-0fb8-51a6e806b698
substitute(x_sym^2 + y_sym, Dict(x_sym => 2, y_sym => 3))

# ╔═╡ a9506bde-9773-11f0-02d3-21d30930f60c
md"""
### Using ModelingToolkit.jl
"""

# ╔═╡ a9506c56-9773-11f0-1c6c-2bfa4605f3c9
@variables x(t) y(t) z(t)

# ╔═╡ a9506c8a-9773-11f0-0a32-57cbf7b4b38b
@parameters σ ρ β

# ╔═╡ a9506cd8-9773-11f0-237a-37b7bcfcc1bd
# Define equations
eqs = [D(x) ~ σ*(y-x),
       D(y) ~ x*(ρ-z)-y,
       D(z) ~ x*y - β*z]

# ╔═╡ a9506d5a-9773-11f0-1fef-375b4ba650ad
# Create system
@named sys = System(eqs, t)

# ╔═╡ a9506d8c-9773-11f0-3e6c-b37f56ee6391
simpsys = mtkcompile(sys)

# ╔═╡ a9506dbe-9773-11f0-2b96-f92eb60ed404
# Convert to ODE problem
u0_mtk = [x => 1.0, y => 0.0, z => 0.0, σ => 10.0, ρ => 28.0, β => 8/3]

# ╔═╡ a9506e2c-9773-11f0-2d1e-c393fd8b0bd0
tspan_mtk = (0.0, 10.0)

# ╔═╡ a9506e56-9773-11f0-056e-8fb014d2ce1b
prob_mtk = ODEProblem(simpsys, u0_mtk, tspan_mtk)

# ╔═╡ a95070d4-9773-11f0-3a44-53e15619932e
sol_mtk = solve(prob_mtk)

# ╔═╡ 48f2cd73-4e3d-4664-9aef-6d3adf86b6f6
plot(sol_mtk)

# ╔═╡ e94f7342-f6c8-496d-9a2b-7ca038d59ccd
plot(sol_mtk, idxs = (t,x))

# ╔═╡ 4749377b-daf2-464d-adf6-86ef1c0d1a48
plot(sol_mtk, idxs = (x,y))

# ╔═╡ 34479ce9-c7d3-4f86-ae27-c53dc40f4e26
plot(sol_mtk, idxs = (x,y,z))

# ╔═╡ a9507360-9773-11f0-1efb-abe8edbdb4e9
md"""
## Plotting

### Using Plots.jl
"""

# ╔═╡ a9507900-9773-11f0-3607-53cc14351f48
# Plots already loaded
# Basic plotting
x_plot = 0:0.1:2π

# ╔═╡ a9507bd0-9773-11f0-0613-e720372a6a11
y_plot = sin.(x_plot)

# ╔═╡ a9507f2a-9773-11f0-0b6f-f7b153f14811
plot(x_plot, y_plot)                      # Line plot

# ╔═╡ a9508182-9773-11f0-02e3-63373863f26e
scatter(x_plot, y_plot)                   # Scatter plot

# ╔═╡ ecc2eae4-c56f-4ac1-b6f5-5cef680e63d4
plot(x_plot, y_plot)

# ╔═╡ 2224ebd3-cadd-4f1a-987a-9a03e76483ee
plot!(x_plot, cos.(x_plot))              # Add to existing plot

# ╔═╡ a9508222-9773-11f0-27dc-1fb53022a1d5
# Customization
plot(x_plot, y_plot,
    label="sin(x)",
    xlabel="x",
    ylabel="y",
    title="Trig Functions",
    lw=2,                       # Line width
    color=:red,
    linestyle=:dash,
    marker=:circle,
    markersize=3,
    legend=:topright,
    grid=true
)

# ╔═╡ a95082c2-9773-11f0-29f4-15c9d9711373
# Multiple series
plot(x_plot, [sin.(x_plot) cos.(x_plot) tan.(x_plot)],
    label=["sin" "cos" "tan"],
    lw=[2 1 1])

# ╔═╡ a950831c-9773-11f0-1e18-4b5c1b666461
# Subplots
p1 = plot(x_plot, sin.(x_plot))

# ╔═╡ a950834e-9773-11f0-0738-292ea6617918
p2 = plot(x_plot, cos.(x_plot))

# ╔═╡ a9508380-9773-11f0-2061-97f6be7926d5
plot(p1, p2, layout=(2,1))

# ╔═╡ a95083b2-9773-11f0-0f88-9fe002f360db
plot(p1, p2, layout=(1,2))

# ╔═╡ a95083dc-9773-11f0-06ed-153f74cb2f3c
# Heatmaps and contours
z_plot = rand(10, 10)

# ╔═╡ a9508416-9773-11f0-00c1-4b96177c44d7
heatmap(z_plot)

# ╔═╡ a9508440-9773-11f0-0468-456ab3321954
contour(z_plot)

# ╔═╡ a95084ac-9773-11f0-1638-4561db13624c
contourf(z_plot)                    # Filled contour

# ╔═╡ a95084de-9773-11f0-17ca-ebefd6103bb8
# 3D plots
surface(x_plot, x_plot, (x,y)->sin(x)*cos(y))

# ╔═╡ a950851a-9773-11f0-1437-697984f663df
plot3d(sin.(x_plot), cos.(x_plot), x_plot)

# ╔═╡ a950854c-9773-11f0-364d-95517de5c4ec
# Save figure
# savefig("myplot.png")
# savefig("myplot.pdf")

# ╔═╡ a950859c-9773-11f0-22b4-b716257facb1
md"""
### Using Makie.jl (for publication-quality plots)
"""

# ╔═╡ a95085e2-9773-11f0-3b48-bd54b886bb7b
# using GLMakie  # or CairoMakie for static plots
# Uncomment to use:
# begin
#     fig = Figure()
#     ax = Axis(fig[1, 1], xlabel="x", ylabel="y")
#     lines!(ax, x_plot, sin.(x_plot))
#     scatter!(ax, x_plot[1:5:end], sin.(x_plot[1:5:end]))
#     fig
# end

# ╔═╡ a9508664-9773-11f0-0d7f-33e43397186f
# Save
# save("figure.pdf", fig)

# ╔═╡ a9508696-9773-11f0-36a0-b744c388b21f
md"""
## Performance Tips

### Type Stability
"""

# ╔═╡ a95086d2-9773-11f0-045a-df6220764b85
# Bad: type-unstable
function foo_bad(x)
    if x > 0
        return 1.0
    else
        return 1  # Returns Int
    end
end

# ╔═╡ a950872c-9773-11f0-0bbd-e13ebd0d5a86
# Good: type-stable
function foo_good(x)
    if x > 0
        return 1.0
    else
        return 1.0
    end
end

# ╔═╡ a95089d4-9773-11f0-1918-871f58139161
# Check type stability
# @code_warntype foo_good(1.0)

# ╔═╡ a9508c36-9773-11f0-2cf3-931774b45b28
md"""
### Preallocate Arrays
"""

# ╔═╡ a9508eca-9773-11f0-0622-537f01b955b2
# Bad
function bad_sum(n)
    arr = []
    for i in 1:n
        push!(arr, i^2)  # Grows array
    end
    return sum(arr)
end

# ╔═╡ a95091d6-9773-11f0-3fca-3fe549980f60
# Good
function good_sum(n)
    arr = zeros(n)
    for i in 1:n
        arr[i] = i^2
    end
    return sum(arr)
end

# ╔═╡ a95094ea-9773-11f0-1b8a-a1258ed85b74
md"""
### Use Views Instead of Copies
"""

# ╔═╡ a9509802-9773-11f0-25ac-b93ddf8a399f
A_big = rand(1000, 1000)

# ╔═╡ a950983e-9773-11f0-1d74-cfbfaa288302
# Creates a copy
B_copy = A_big[1:100, 1:100]

# ╔═╡ a950987a-9773-11f0-29b8-c53303609f8c
# Creates a view (no copy)
B_view = @view A_big[1:100, 1:100]

# ╔═╡ a95098b6-9773-11f0-1d1c-fde1591eb09d
B_view2 = view(A_big, 1:100, 1:100)

# ╔═╡ a95098de-9773-11f0-27af-0363afd8dd27
md"""
### Avoid Global Variables
"""

# ╔═╡ decfafe4-3526-438a-994a-2f59470622d1
# Bad
global_x = 10

# ╔═╡ 60b194b5-43c8-4db8-94c1-46f8cc6ac86d
function bad_func(y)
    return global_x + y  # x is global
end

# ╔═╡ a9509974-9773-11f0-16d7-057e1415c17e
# Good
function good_func(x, y)
    return x + y
end

# ╔═╡ a95099b0-9773-11f0-17d9-7f1bd6657f5a
# Or use const for globals
const const_x = 10

# ╔═╡ a95099e2-9773-11f0-031f-85acb445143d
md"""
### Use @inbounds for Trusted Code
"""

# ╔═╡ a9509a1e-9773-11f0-24a9-0b5cd3034f27
function sum_array(A)
    s = 0.0
    @inbounds for i in eachindex(A)
        s += A[i]  # Skip bounds checking
    end
    return s
end

# ╔═╡ a9509a82-9773-11f0-13f1-2725db277304
md"""
### Benchmarking
"""

# ╔═╡ a9509ae6-9773-11f0-2080-d954c041242f
# Time a function
@time sum(rand(1000))

# ╔═╡ a9509b36-9773-11f0-396d-9da3f01171a4
# More accurate benchmarking
@benchmark sum(rand(1000))

# ╔═╡ a9509b72-9773-11f0-01c2-8f04fb479a7b
@btime sum($(rand(1000)))

# ╔═╡ a9509b9a-9773-11f0-2c9b-fffec5097f25
# Profile code
# using Profile
# @profile my_slow_function()
# Profile.print()

# ╔═╡ a9509be0-9773-11f0-1ab8-d50e8abd48b6
md"""
## Quick Reference Card

### REPL Commands
```julia
?function          # Help for function
]                  # Package mode
]add Package       # Install package
]up               # Update packages
]st               # Show installed packages
;                 # Shell mode
using Package     # Load package
include("file.jl") # Run script
```

### Useful Macros
```julia
@time expr        # Time execution
@elapsed expr     # Return elapsed time
@assert cond      # Assert condition
@show var         # Print var = value
@. expr           # Vectorize all operations
@views expr       # Use views for slicing
@inbounds expr    # Skip bounds checking
@threads for ...  # Parallel for loop
```

### String Interpolation
```julia
x = 10
s = "x = $x"                    # "x = 10"
s = "x² = $(x^2)"              # "x² = 100"
```

### Broadcasting
```julia
# Add . to apply element-wise
A .+ B            # Element-wise addition
sin.(X)           # Apply to each element
f.(X, Y)          # Element-wise f(x,y)
```

### Type Assertions
```julia
x::Float64        # Assert x is Float64
f(x::Int)         # Function expecting Int
Vector{Float64}   # Vector of Float64s
Matrix{Int}       # Matrix of Ints
```
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
Ipopt = "b6b21f68-93f8-5de0-b562-5493be1d77c9"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
ModelingToolkit = "961ee093-0014-501f-94e3-6117800e7a78"
NonlinearSolve = "8913a72c-1f9b-4ce2-8d82-65094dcecaec"
Optimization = "7f7a1694-90dd-40f0-9382-eb1efda571ba"
OptimizationMOI = "fd9f6733-72f4-499f-8506-86b2bdd0dea1"
OptimizationOptimJL = "36348300-93cb-4f02-beb5-3c3902f8871e"
OptimizationOptimisers = "42dfb2eb-d2b4-4451-abcd-913932933ac1"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"

[compat]
BenchmarkTools = "~1.8.0"
DifferentialEquations = "~8.1.1"
ForwardDiff = "~1.4.6"
Ipopt = "~1.16.0"
JuMP = "~1.31.2"
ModelingToolkit = "~11.43.1"
NonlinearSolve = "~4.32.0"
Optimization = "~5.9.1"
OptimizationMOI = "~1.4.1"
OptimizationOptimJL = "~0.4.21"
OptimizationOptimisers = "~0.3.24"
Plots = "~1.41.7"
Symbolics = "~7.39.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.7"
manifest_format = "2.0"
project_hash = "a719e5e010b483d14068c7a4e341d0bd528926d9"

[[deps.ADTypes]]
deps = ["PrecompileTools"]
git-tree-sha1 = "629de23e1c16911b439dabd2303c08af9575b226"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.24.0"
weakdeps = ["ChainRulesCore", "ConstructionBase", "EnzymeCore"]

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

[[deps.AMD]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse_jll"]
git-tree-sha1 = "344f3d221a6bee75925b5c97a30cfd870f12a138"
uuid = "14f7f29c-3bd6-536c-9a0b-7339e30b5a3e"
version = "0.5.4"

[[deps.ASL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "21a22cd1a6d45034933824665cfb874ec425a14b"
uuid = "ae81ac8f-d209-56e5-92de-9978fef736f9"
version = "0.1.5+0"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "e71ee7b4aa06b045259a7d6101e1cb45ad140bce"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.1"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "7063ad1083578215c7c4bf410368150abe8d5524"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.45"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "7c2c19b5a26e601634bf718490b89d59685f122e"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.7.1"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "1aec1ff0dcaa83a484e7f72097562fa3102c6722"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.30.2"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceAMDGPUExt = "AMDGPU"
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = ["CUDSS", "CUDA"]
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceFillArraysExt = "FillArrays"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceGPUArraysCoreTrackerExt = ["GPUArraysCore", "Tracker"]
    ArrayInterfaceMetalExt = "Metal"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "e0b47732a192dd59b9d079a06d04235e2f833963"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.12.2"
weakdeps = ["SparseArrays"]

    [deps.ArrayLayouts.extensions]
    ArrayLayoutsSparseArraysExt = "SparseArrays"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.BandedMatrices]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "PrecompileTools"]
git-tree-sha1 = "545a7b37b3ccde0a6a2948f7bb884024609ae368"
uuid = "aae01518-5342-5314-be14-df237901396f"
version = "1.12.0"

    [deps.BandedMatrices.extensions]
    BandedMatricesSparseArraysExt = "SparseArrays"
    CliqueTreesExt = "CliqueTrees"

    [deps.BandedMatrices.weakdeps]
    CliqueTrees = "60701a23-6482-424a-84db-faee86b9b1f8"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "PrecompileTools", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "9670d3febc2b6da60a0ae57846ba74670290653f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.8.0"

[[deps.Bijections]]
git-tree-sha1 = "a2d308fcd4c2fb90e943cf9cd2fbfa9c32b69733"
uuid = "e2ed5e7c-b2de-5872-ae92-c73ca462fb04"
version = "0.2.2"

[[deps.BinaryHeaps]]
deps = ["PrecompileTools"]
git-tree-sha1 = "0adb6d8d0a4bae93a07d79c8d59eaf617f4c59da"
uuid = "b2a6c25c-c996-4615-94ab-6e519ffb8690"
version = "1.1.0"

[[deps.BipartiteGraphs]]
deps = ["DataStructures", "DocStringExtensions", "Graphs", "PrecompileTools"]
git-tree-sha1 = "ba317c3a2b08853880832d5a034690c2fba539bf"
uuid = "caf10ac8-0290-4205-88aa-f15908547e8d"
version = "0.1.14"
weakdeps = ["SparseArrays"]

    [deps.BipartiteGraphs.extensions]
    BipartiteGraphsSparseArraysExt = "SparseArrays"

[[deps.BlockArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra"]
git-tree-sha1 = "75c9c4d41f387b58ac7ecac17a02062f4cf8e92a"
uuid = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
version = "1.10.0"
weakdeps = ["Adapt", "BandedMatrices"]

    [deps.BlockArrays.extensions]
    BlockArraysAdaptExt = "Adapt"
    BlockArraysBandedMatricesExt = "BandedMatrices"

[[deps.BracketingNonlinearSolve]]
deps = ["CommonSolve", "ConcreteStructs", "NonlinearSolveBase", "PrecompileTools", "Reexport", "SciMLBase", "SciMLLogging"]
git-tree-sha1 = "027dcb51dcf326023e015902b3d8e88ca59efb8a"
uuid = "70df07ce-3d50-431d-a3e7-ca6ddb60ac1e"
version = "1.12.7"
weakdeps = ["ChainRulesCore", "ForwardDiff"]

    [deps.BracketingNonlinearSolve.extensions]
    BracketingNonlinearSolveChainRulesCoreExt = ["ChainRulesCore", "ForwardDiff"]
    BracketingNonlinearSolveForwardDiffExt = "ForwardDiff"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1fa950ebc3e37eccd51c6a8fe1f92f7d86263522"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.7+0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "12177ad6b3cad7fd50c8b3825ce24a99ad61c18f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "TranscodingStreams"]
git-tree-sha1 = "84990fa864b7f2b4901901ca12736e45ee79068c"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.5"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "970758a3d591a2a5c2a907c53f2e2f8c1b1d3537"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.9"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
deps = ["PrecompileTools"]
git-tree-sha1 = "6c389fa857f6ca5a95474b52a52023fd77f24cb7"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.14"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.CommonWorldInvalidations]]
deps = ["PrecompileTools"]
git-tree-sha1 = "7c4ea0adfb7238bf4678aa36325863fab6163f6f"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.2.2"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.1+2"

[[deps.CompositeTypes]]
git-tree-sha1 = "bce26c3dab336582805503bed209faab1c279768"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.4"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcreteStructs]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a72c3b5ce6d2a477f55b5c9b8756e91e695c67f6"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.8"

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "54b76cbb40d9a0f5368c880725b2f141da77c94f"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.2.0"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "b0bc6d2cad1fed8b7fd59a1551a991cb3d2809e6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.6"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffEqBase]]
deps = ["ArrayInterface", "BracketingNonlinearSolve", "ConcreteStructs", "DocStringExtensions", "FastBroadcast", "FastClosures", "FastPower", "FunctionWrappers", "FunctionWrappersWrappers", "LinearAlgebra", "Logging", "Markdown", "MuladdMacro", "PrecompileTools", "Printf", "RecursiveArrayTools", "Reexport", "RespecializeParams", "SciMLBase", "SciMLLogging", "SciMLOperators", "SciMLStructures", "Setfield", "StaticArraysCore", "SymbolicIndexingInterface", "TruncatedStacktraces"]
git-tree-sha1 = "e2a0c69c358ed6a8a6ac5ae8f1c7e36150a4dbd9"
uuid = "2b5f629d-d688-5b77-993f-72d75c75574e"
version = "7.21.1"

    [deps.DiffEqBase.extensions]
    DiffEqBaseCUDAExt = "CUDA"
    DiffEqBaseChainRulesCoreExt = "ChainRulesCore"
    DiffEqBaseDynamicQuantitiesExt = "DynamicQuantities"
    DiffEqBaseEnzymeExt = ["ChainRulesCore", "Enzyme"]
    DiffEqBaseFlexUnitsExt = "FlexUnits"
    DiffEqBaseForwardDiffExt = ["ForwardDiff"]
    DiffEqBaseGTPSAExt = "GTPSA"
    DiffEqBaseGeneralizedGeneratedExt = "GeneralizedGenerated"
    DiffEqBaseMPIExt = "MPI"
    DiffEqBaseMeasurementsExt = "Measurements"
    DiffEqBaseMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    DiffEqBaseMooncakeExt = "Mooncake"
    DiffEqBaseReverseDiffExt = "ReverseDiff"
    DiffEqBaseSparseArraysExt = "SparseArrays"
    DiffEqBaseTrackerExt = "Tracker"
    DiffEqBaseUnitfulExt = "Unitful"

    [deps.DiffEqBase.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    DynamicQuantities = "06fc5a27-2a28-4c7c-a15d-362465fb6821"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FlexUnits = "76e01b6b-c995-4ce6-8559-91e72a3d4e95"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    GeneralizedGenerated = "6b9d7cbe-bcb9-11e9-073f-15a7a543e2eb"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.DiffEqCallbacks]]
deps = ["ConcreteStructs", "DataStructures", "DiffEqBase", "DifferentiationInterface", "LinearAlgebra", "Markdown", "PrecompileTools", "RecipesBase", "RecursiveArrayTools", "SciMLBase", "StaticArraysCore"]
git-tree-sha1 = "74d18419405b8c196f8443aa9be9c6527bd3e6e2"
uuid = "459566f4-90b8-5000-8ac3-15dfb0a30def"
version = "4.19.4"
weakdeps = ["Functors"]

    [deps.DiffEqCallbacks.extensions]
    DiffEqCallbacksFunctorsExt = "Functors"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "79a2aca180a85c690c58a020d47b426954b590f8"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.16.0"

[[deps.DifferentialEquations]]
deps = ["OrdinaryDiffEq", "PrecompileTools", "Reexport", "SciMLBase"]
git-tree-sha1 = "d167917674ca92bbf51b56a3e00d019792695110"
uuid = "0c46a032-eb83-5123-abaf-570d42b7fbaa"
version = "8.1.1"

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "LinearAlgebra"]
git-tree-sha1 = "0693d8b0a4608ff289d228ab4c598df5894845cd"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.7.21"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = ["EnzymeCore", "Enzyme"]
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = ["ForwardDiff", "DiffResults"]
    DifferentiationInterfaceGPUArraysCoreExt = ["GPUArraysCore", "Adapt"]
    DifferentiationInterfaceGTPSAExt = "GTPSA"
    DifferentiationInterfaceHyperHessiansExt = "HyperHessians"
    DifferentiationInterfaceMooncakeExt = "Mooncake"
    DifferentiationInterfacePolyesterForwardDiffExt = ["PolyesterForwardDiff", "ForwardDiff", "DiffResults"]
    DifferentiationInterfaceReverseDiffExt = ["ReverseDiff", "DiffResults"]
    DifferentiationInterfaceSparseArraysExt = "SparseArrays"
    DifferentiationInterfaceSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DifferentiationInterfaceSparseMatrixColoringsExt = "SparseMatrixColorings"
    DifferentiationInterfaceStaticArraysExt = "StaticArrays"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    HyperHessians = "06b494a0-c8e0-40cc-ad32-d99506a00a6c"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.DomainSets]]
deps = ["CompositeTypes", "FunctionMaps", "IntervalSets", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "c0f576ae49bd2d1bc904b9946f4783db8f0ef530"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.8.1"

    [deps.DomainSets.extensions]
    DomainSetsMakieExt = "Makie"
    DomainSetsRandomExt = "Random"

    [deps.DomainSets.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.DynamicPolynomials]]
deps = ["LinearAlgebra", "MultivariatePolynomials", "MutableArithmetics", "Reexport", "StarAlgebras"]
git-tree-sha1 = "b95c4325301d86a6a1b59790b71f7269a8702dd5"
uuid = "7c1d4256-1411-5781-91ec-d7bc3513ac07"
version = "0.6.8"

[[deps.EnumX]]
git-tree-sha1 = "c49898e8438c828577f04b92fc9368c388ac783c"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.7"

[[deps.EnzymeCore]]
git-tree-sha1 = "971d7831cc85f43bc9f51d615a3f7f21270c2f1d"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.21"
weakdeps = ["Adapt", "ChainRulesCore"]

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"
    EnzymeCoreChainRulesCoreExt = "ChainRulesCore"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2bfb1e047e2ad0a5ca94365340bde8005d637568"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.4+0"

[[deps.ExprTools]]
git-tree-sha1 = "d2e49e7efd29719d6f28b891b0e0e159daa9d2b4"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.11"

[[deps.ExproniconLite]]
git-tree-sha1 = "c13f0b150373771b0fdc1713c97860f8df12e6c2"
uuid = "55351af7-c7e9-48d6-89ff-24e801d99491"
version = "0.10.14"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "7a58e45171b63ed4782f2d36fdee8713a469e6e0"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.1.2+0"

[[deps.FastBroadcast]]
deps = ["ArrayInterface", "LinearAlgebra", "PrecompileTools"]
git-tree-sha1 = "c8f8eefadaa330d982bbf5e395c34e53a13ab668"
uuid = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
version = "1.4.0"

    [deps.FastBroadcast.extensions]
    FastBroadcastPolyesterExt = "Polyester"
    FastBroadcastStaticExt = "Static"

    [deps.FastBroadcast.weakdeps]
    Polyester = "f517fe37-dbe3-4b94-8317-1923a5111588"
    Static = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"

[[deps.FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[deps.FastPower]]
deps = ["PrecompileTools"]
git-tree-sha1 = "3c7269c236978d434a16ebe99f9743b9d706270a"
uuid = "a4df4552-cc26-4903-aec0-212e50a0e84b"
version = "1.5.0"

    [deps.FastPower.extensions]
    FastPowerEnzymeExt = "Enzyme"
    FastPowerForwardDiffExt = "ForwardDiff"
    FastPowerMeasurementsExt = "Measurements"
    FastPowerMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    FastPowerMooncakeExt = "Mooncake"
    FastPowerReverseDiffExt = "ReverseDiff"
    FastPowerTrackerExt = "Tracker"

    [deps.FastPower.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "5bad39456d9f0166184fce2248783dd9862645c1"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.17.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FindFirstFunctions]]
deps = ["PrecompileTools"]
git-tree-sha1 = "bba475ce782fc77777f539cf0b7c029207798db1"
uuid = "64ca27bc-2ba2-4a57-88aa-44e436879224"
version = "3.2.1"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "5031f23e040bf17082e5b52422d77b5e844eefb1"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.33.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffSparseArraysExt = "SparseArrays"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "3b0f72e2ffef1a139ac450725c9ecd01c0a3c050"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.4.6"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.FunctionMaps]]
deps = ["CompositeTypes", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "31bd99a57edf98990d1c21486032963955450e8d"
uuid = "a85aefff-f8ca-4649-a888-c8e5398bc76c"
version = "0.1.2"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "2bcce3ad6f6977d617928d7707fdc86ac83cce03"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "1.13.0"

    [deps.FunctionWrappersWrappers.extensions]
    FunctionWrappersWrappersEnzymeExt = ["Enzyme", "EnzymeCore"]
    FunctionWrappersWrappersMooncakeExt = "Mooncake"

    [deps.FunctionWrappersWrappers.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"

[[deps.Functors]]
deps = ["Compat", "ConstructionBase", "LinearAlgebra", "Random"]
git-tree-sha1 = "1ac2813982db52b974c9343124ca61adbf297316"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.5.3"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "64bbbb7d1499297751b536dd39c58b20750ab1db"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.5.1+0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "4d777f73c46b46b8b5276206059cf8a195499314"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.27"

    [deps.GR.extensions]
    IJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f8eb8f7ba13ea75083531647fc8faeda8d541f07"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.27+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "090526e65de8f69648ac156daae153de8b56df62"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.88.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Inflate", "LinearAlgebra", "Random", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "dbb2b976f605b220dfe139d6db9f3859680da09e"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.15.0"

    [deps.Graphs.extensions]
    GraphsSharedArraysExt = "SharedArrays"

    [deps.Graphs.weakdeps]
    Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"
    SharedArrays = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "9d9531a9cb63a9edc33836414e82a07e81710de2"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "100.14004.0+0"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XML2_jll", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "c35847ca5b4997fc8418836354a56c459bcf48d8"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.14.0+0"

[[deps.ImplicitDiscreteSolve]]
deps = ["CommonSolve", "ConcreteStructs", "DiffEqBase", "NonlinearSolveBase", "NonlinearSolveFirstOrder", "OrdinaryDiffEqCore", "Reexport", "SciMLBase", "SymbolicIndexingInterface"]
git-tree-sha1 = "a796d5bbd868177f97a840742e69061d89993a99"
uuid = "3263718b-31ed-49cf-8a0f-35a466e8af96"
version = "2.3.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "c72458f1962faeb003bf23cbdb75164fe6280906"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.4"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IntervalSets]]
git-tree-sha1 = "79d6bd28c8d9bccc2229784f1bd637689b256377"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.14"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.Ipopt]]
deps = ["Ipopt_jll", "LinearAlgebra", "OpenBLAS32_jll", "PrecompileTools"]
git-tree-sha1 = "d5a4fb77d030c56258a33a7046195506055ce5cc"
uuid = "b6b21f68-93f8-5de0-b562-5493be1d77c9"
version = "1.16.0"
weakdeps = ["MathOptInterface"]

    [deps.Ipopt.extensions]
    IpoptMathOptInterfaceExt = "MathOptInterface"

[[deps.Ipopt_jll]]
deps = ["ASL_jll", "Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "MUMPS_seq_jll", "SPRAL_jll", "libblastrampoline_jll"]
git-tree-sha1 = "d58bb7f6393b8ec1f6fb39eb6c1a7a83b8f8b521"
uuid = "9cc047cb-c261-5740-88fc-0cf96f7bdcc7"
version = "300.1400.1902+0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "88352712893ec50bee3680605891eaf0e9ed6368"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.8.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.Jieko]]
deps = ["ExproniconLite"]
git-tree-sha1 = "2f05ed29618da60c06a87e9c033982d4f71d0b6c"
uuid = "ae98c720-c025-4a4a-838c-29b094483192"
version = "0.2.1"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "037babc10853eeb8e585418922246cb97b8e5b74"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.2.0+1"

[[deps.JuMP]]
deps = ["LinearAlgebra", "MacroTools", "MathOptInterface", "MutableArithmetics", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays"]
git-tree-sha1 = "4f27b21df3b47e8c08a83ead049afb621b2f5b3c"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.31.2"

    [deps.JuMP.extensions]
    JuMPDimensionalDataExt = "DimensionalData"

    [deps.JuMP.weakdeps]
    DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.JumpProcesses]]
deps = ["ADTypes", "ArrayInterface", "DataStructures", "DiffEqBase", "DiffEqCallbacks", "DocStringExtensions", "FunctionWrappers", "Graphs", "LinearAlgebra", "PoissonRandom", "PrecompileTools", "Random", "RecursiveArrayTools", "SciMLBase", "SimpleNonlinearSolve", "StaticArrays", "SymbolicIndexingInterface"]
git-tree-sha1 = "fb405676df22fcfd4c8912b9fef77c634d744386"
uuid = "ccbc3e58-028d-4f4c-8cd5-9ae44345cda5"
version = "9.32.4"

    [deps.JumpProcesses.extensions]
    JumpProcessesForwardDiffExt = "ForwardDiff"
    JumpProcessesKernelAbstractionsExt = ["Adapt", "KernelAbstractions"]
    JumpProcessesOrdinaryDiffEqCoreExt = "OrdinaryDiffEqCore"

    [deps.JumpProcesses.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    OrdinaryDiffEqCore = "bbf590c4-e513-4bbe-9b18-05decba2e5d8"

[[deps.Krylov]]
deps = ["LinearAlgebra", "Printf", "SparseArrays"]
git-tree-sha1 = "909890a0d1486c392128cf4bd6e038ee0c6a7af4"
uuid = "ba0b0d4f-ebba-5204-a429-3ac8c609bfb7"
version = "0.10.10"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "39bca05343661c347aae0bca57a5994a0bf4f08d"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.2.0+0"

[[deps.LHLFactorization]]
deps = ["LinearAlgebra", "PrecompileTools"]
git-tree-sha1 = "51d99d9892d0f95b4c9d4d7310785971f2d48298"
uuid = "2faa5264-e118-4071-8864-e10559f68c7c"
version = "2.2.2"

    [deps.LHLFactorization.extensions]
    LHLFactorizationPolyesterExt = "Polyester"
    LHLFactorizationSparseExt = ["PureKLU", "SparseArrays"]

    [deps.LHLFactorization.weakdeps]
    Polyester = "f517fe37-dbe3-4b94-8317-1923a5111588"
    PureKLU = "0c0d3e7f-3a8b-4f7e-b6f1-9a4d2e7c1f01"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e5b100780d4d30d63b4618d7930d48af409c1772"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "23.1.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f88f3ccef05a6a72a0cf0ed417c8fd68530f4ab2"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.1"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "df7566479bd64f20bd16b09960145e70160ffb3b"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.12"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "d4816abce26971e3237b46a47b99991f306e4832"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.3.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "aebd334d06cee9f24cea70bd19a39749daf73881"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.3+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d620582b1f0cbe2c72dd1d5bd195a9ce73370ab1"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.0+0"

[[deps.LineSearch]]
deps = ["ADTypes", "CommonSolve", "ConcreteStructs", "FastClosures", "LinearAlgebra", "MaybeInplace", "PrecompileTools", "SciMLBase", "SciMLJacobianOperators", "StaticArraysCore"]
git-tree-sha1 = "e8fb8395e72f73cb8432572dbafcd941514fb94d"
uuid = "87fe0de2-c867-4266-b59a-2f0a94fc965b"
version = "0.1.18"
weakdeps = ["LineSearches"]

    [deps.LineSearch.extensions]
    LineSearchLineSearchesExt = "LineSearches"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Printf"]
git-tree-sha1 = "a09a85e94e681def2446752cddaf9401a4d82b6d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.8.2"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LinearSolve]]
deps = ["AMD", "ArrayInterface", "ConcreteStructs", "DocStringExtensions", "EnumX", "GPUArraysCore", "InteractiveUtils", "Krylov", "LHLFactorization", "Libdl", "LinearAlgebra", "MKL_jll", "Markdown", "OpenBLAS_jll", "PrecompileTools", "Preferences", "PureKLU", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLLogging", "SciMLOperators", "SciMLStructures", "Setfield", "SparseArrays", "SparseColumnPivotedQR", "StaticArraysCore"]
git-tree-sha1 = "b57acc4a96a542efd0a8c9c7262f9168b8822788"
uuid = "7ed4a6bd-45f5-4d41-b270-4a48e9bafcae"
version = "5.17.5"

    [deps.LinearSolve.extensions]
    LinearSolveAMDGPUExt = "AMDGPU"
    LinearSolveAMGXExt = ["AMGX", "CUDA"]
    LinearSolveAlgebraicMultigridExt = "AlgebraicMultigrid"
    LinearSolveArnoldiMethodExt = "ArnoldiMethod"
    LinearSolveArpackExt = "Arpack"
    LinearSolveBLISExt = ["blis_jll", "LAPACK_jll"]
    LinearSolveBandedMatricesExt = "BandedMatrices"
    LinearSolveBlockDiagonalsExt = "BlockDiagonals"
    LinearSolveCKTSOExt = "CKTSO"
    LinearSolveCUDAExt = ["cuSOLVER"]
    LinearSolveCUDSSExt = "CUDSS"
    LinearSolveCUSOLVERRFExt = "CUSOLVERRF"
    LinearSolveChainRulesCoreExt = "ChainRulesCore"
    LinearSolveCliqueTreesExt = "CliqueTrees"
    LinearSolveConjugateGradientsExt = "ConjugateGradients"
    LinearSolveElementalExt = "Elemental"
    LinearSolveEnzymeExt = "EnzymeCore"
    LinearSolveFastAlmostBandedMatricesExt = "FastAlmostBandedMatrices"
    LinearSolveFastLapackInterfaceExt = "FastLapackInterface"
    LinearSolveForwardDiffExt = "ForwardDiff"
    LinearSolveGinkgoExt = "Ginkgo"
    LinearSolveHSLExt = "HSL"
    LinearSolveHYPREExt = "HYPRE"
    LinearSolveIterativeSolversExt = "IterativeSolvers"
    LinearSolveJacobiDavidsonExt = "JacobiDavidson"
    LinearSolveKernelAbstractionsExt = "KernelAbstractions"
    LinearSolveKrylovKitExt = "KrylovKit"
    LinearSolveMUMPSExt = "MUMPS"
    LinearSolveMetalExt = "Metal"
    LinearSolveMooncakeExt = "Mooncake"
    LinearSolvePETScExt = ["PETSc", "SparseMatricesCSR"]
    LinearSolvePETScMPIExt = ["PETSc", "PartitionedArrays", "SparseMatricesCSR"]
    LinearSolveParUExt = "ParU_jll"
    LinearSolvePardisoExt = "Pardiso"
    LinearSolvePartitionedSolversExt = ["PartitionedArrays", "PartitionedSolvers"]
    LinearSolvePureUMFPACKExt = "PureUMFPACK"
    LinearSolveReactantExt = "Reactant"
    LinearSolveRecursiveFactorizationExt = ["RecursiveFactorization", "TriangularSolve"]
    LinearSolveSLATEExt = "SLATE_jll"
    LinearSolveSTRUMPACKExt = "STRUMPACK_jll"
    LinearSolveSparspakExt = "Sparspak"
    LinearSolveSpecializingFactorizationsExt = "SpecializingFactorizations"
    LinearSolveSuperLUDISTExt = "SuperLUDIST"

    [deps.LinearSolve.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    AMGX = "c963dde9-0319-47f5-bf0c-b07d3c80ffa6"
    AlgebraicMultigrid = "2169fc97-5a83-5252-b627-83903c6c433c"
    ArnoldiMethod = "ec485272-7323-5ecc-a04f-4719b315124d"
    Arpack = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockDiagonals = "0a1fb500-61f7-11e9-3c65-f5ef3456f9f0"
    CKTSO = "8e543fc2-2ef1-4e2d-a99e-39beaa046845"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    CUSOLVERRF = "a8cc9031-bad2-4722-94f5-40deabb4245c"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    CliqueTrees = "60701a23-6482-424a-84db-faee86b9b1f8"
    ConjugateGradients = "f59de78d-195d-4e7b-a078-2e47da4c3ad6"
    Elemental = "902c3f28-d1ec-5e7e-8399-a24c3845ee38"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastAlmostBandedMatrices = "9d29842c-ecb8-4973-b1e9-a27b1157504e"
    FastLapackInterface = "29a986be-02c6-4525-aec4-84b980013641"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Ginkgo = "4c8bd3c9-ead9-4b5e-a625-08f1338ba0ec"
    HSL = "34c5aeac-e683-54a6-a0e9-6e0fdc586c50"
    HYPRE = "b5ffcf37-a2bd-41ab-a3da-4bd9bc8ad771"
    IterativeSolvers = "42fd0dbc-a981-5370-80f2-aaf504508153"
    JLArrays = "27aeb0d3-9eb9-45fb-866b-73c2ecf80fcb"
    JacobiDavidson = "11c68b98-9c9b-11e8-267b-bbb95576cead"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    KrylovKit = "0b1a1467-8014-51b9-945f-bf0ae24f4b77"
    LAPACK_jll = "51474c39-65e3-53ba-86ba-03b1b862ec14"
    MUMPS = "55d2b088-9f4e-11e9-26c0-150b02ea6a46"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PETSc = "ace2c81b-2b5f-4b1e-a30d-d662738edfe0"
    ParU_jll = "9e0b026c-e8ce-559c-a2c4-6a3d5c955bc9"
    Pardiso = "46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2"
    PartitionedArrays = "5a9dfac6-5c52-46f7-8278-5e2210713be9"
    PartitionedSolvers = "11b65f7f-80ac-401b-9ef2-3db765482d62"
    PureUMFPACK = "b7e1f0a2-3c4d-4e5f-9a0b-1c2d3e4f5a6b"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    RecursiveFactorization = "f2c3362d-daeb-58d1-803e-2bc74f2840b4"
    SLATE_jll = "cb01ff19-587c-516c-bcf9-532d49da298d"
    STRUMPACK_jll = "86fbd0b9-476f-557c-b766-62c724b42d8c"
    SparseMatricesCSR = "a0a7dd2c-ebf4-11e9-1f05-cf50bc540ca1"
    Sparspak = "e56a9233-b9d6-4f03-8d0f-1825330902ac"
    SpecializingFactorizations = "fa08b7a1-13d3-4faf-875d-5cbc1520e3f3"
    SuperLUDIST = "4cd002a6-0da4-410d-a012-232df062f478"
    TriangularSolve = "d5829a12-d9aa-46ab-831f-fb7c9ab06edf"
    blis_jll = "6136c539-28a5-5bf0-87cc-b183200dce32"
    cuSOLVER = "887afef0-6a32-4de5-add4-7827692ba8fc"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "bba2d9aa057d8f126415de240573e86a8f39d2a1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.1"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.METIS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5ecd92439c60a50c49bd6e96192e9d486b24b050"
uuid = "d00139f3-1899-568f-a2f0-47f597d42d70"
version = "5.1.4+0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MUMPS_seq_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "METIS_jll", "libblastrampoline_jll"]
git-tree-sha1 = "fc1daa0ab5a0d7a0a34d5168a711cbf02b2cb0cc"
uuid = "d7ed1dd3-d0ae-5e8e-bfb4-87a502085b8d"
version = "500.900.100+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathOptInterface]]
deps = ["CodecBzip2", "CodecZlib", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test"]
git-tree-sha1 = "d10ba577e0b5a0481fab01dfd31fb20af3326954"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.53.0"

    [deps.MathOptInterface.extensions]
    MathOptInterfaceBenchmarkToolsExt = "BenchmarkTools"
    MathOptInterfaceCliqueTreesExt = "CliqueTrees"

    [deps.MathOptInterface.weakdeps]
    BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
    CliqueTrees = "60701a23-6482-424a-84db-faee86b9b1f8"

[[deps.MaybeInplace]]
deps = ["ArrayInterface", "LinearAlgebra", "MacroTools", "PrecompileTools"]
git-tree-sha1 = "f2cde0ae772162f20287b803e623966d0d94dea9"
uuid = "bb5d69b7-63fc-4a16-80bd-7e42200c7bdb"
version = "0.1.8"
weakdeps = ["SparseArrays"]

    [deps.MaybeInplace.extensions]
    MaybeInplaceSparseArraysExt = "SparseArrays"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.ModelingToolkit]]
deps = ["ADTypes", "BipartiteGraphs", "BlockArrays", "Combinatorics", "CommonSolve", "ConstructionBase", "DataStructures", "DiffEqBase", "DifferentiationInterface", "DocStringExtensions", "FillArrays", "ForwardDiff", "Graphs", "InteractiveUtils", "Libdl", "LinearAlgebra", "ModelingToolkitBase", "ModelingToolkitTearing", "Moshi", "OffsetArrays", "OrderedCollections", "PreallocationTools", "PrecompileTools", "REPL", "Reexport", "RuntimeGeneratedFunctions", "SCCNonlinearSolve", "SciMLBase", "SciMLPublic", "SciMLStructures", "Serialization", "Setfield", "SimpleNonlinearSolve", "SparseArrays", "StateSelection", "StaticArrays", "SymbolicIndexingInterface", "SymbolicUtils", "Symbolics", "TermInterface", "UnPack"]
git-tree-sha1 = "a37c6714e14ce0ceea37de362649113c6e1dba78"
uuid = "961ee093-0014-501f-94e3-6117800e7a78"
version = "11.43.1"

    [deps.ModelingToolkit.extensions]
    MTKFMIExt = "FMIImport"
    MTKOrdinaryDiffEqBDFExt = "OrdinaryDiffEqBDF"
    MTKOrdinaryDiffEqDefaultExt = "OrdinaryDiffEqDefault"
    MTKOrdinaryDiffEqRosenbrockExt = "OrdinaryDiffEqRosenbrock"
    MTKOrdinaryDiffEqRosenbrockNonlinearSolveExt = ["OrdinaryDiffEqRosenbrock", "OrdinaryDiffEqNonlinearSolve"]
    MTKOrdinaryDiffEqTsit5Ext = "OrdinaryDiffEqTsit5"

    [deps.ModelingToolkit.weakdeps]
    FMIImport = "9fcbc62e-52a0-44e9-a616-1359a0008194"
    OrdinaryDiffEqBDF = "6ad6398a-0878-4a85-9266-38940aa047c8"
    OrdinaryDiffEqDefault = "50262376-6c5a-4cf5-baba-aaf4f84d72d7"
    OrdinaryDiffEqNonlinearSolve = "127b3ac7-2247-4354-8eb6-78cf4e7c58e8"
    OrdinaryDiffEqRosenbrock = "43230ef6-c299-4910-a778-202eb28ce4ce"
    OrdinaryDiffEqTsit5 = "b1df2697-797e-41e3-8120-5422d3b24e4a"

[[deps.ModelingToolkitBase]]
deps = ["ADTypes", "AbstractTrees", "ArrayInterface", "BandedMatrices", "BipartiteGraphs", "BlockArrays", "Combinatorics", "CommonSolve", "Compat", "ConstructionBase", "DataStructures", "DiffEqBase", "DiffEqCallbacks", "DiffRules", "DifferentiationInterface", "DocStringExtensions", "DomainSets", "EnumX", "EnzymeCore", "ExprTools", "FillArrays", "ForwardDiff", "FunctionWrappers", "FunctionWrappersWrappers", "Graphs", "ImplicitDiscreteSolve", "InteractiveUtils", "IntervalSets", "JumpProcesses", "Libdl", "LinearAlgebra", "Moshi", "NaNMath", "NonlinearSolveBase", "OffsetArrays", "OrderedCollections", "PreallocationTools", "PrecompileTools", "Printf", "REPL", "Random", "ReadOnlyDicts", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SCCNonlinearSolve", "SciMLBase", "SciMLPublic", "SciMLStructures", "Serialization", "Setfield", "SimpleNonlinearSolve", "SparseArrays", "SpecialFunctions", "StaticArrays", "StaticArraysCore", "SymbolicIndexingInterface", "SymbolicUtils", "Symbolics", "TaskLocalValues", "TermInterface", "UnPack"]
git-tree-sha1 = "eabb97d470bd4c281a52e9e08e2cc21c1e268d36"
uuid = "7771a370-6774-4173-bd38-47e70ca0b839"
version = "1.73.0"

    [deps.ModelingToolkitBase.extensions]
    MTKBifurcationKitExt = "BifurcationKit"
    MTKCasADiDynamicOptExt = "CasADi"
    MTKChainRulesCoreExt = "ChainRulesCore"
    MTKDiffEqNoiseProcessExt = "DiffEqNoiseProcess"
    MTKDynamicQuantitiesExt = "DynamicQuantities"
    MTKInfiniteOptExt = "InfiniteOpt"
    MTKJuliaFormatterExt = "JuliaFormatter"
    MTKLabelledArraysExt = "LabelledArrays"
    MTKLatexifyExt = "Latexify"
    MTKMooncakeExt = "Mooncake"
    MTKPyomoDynamicOptExt = ["Pyomo", "PythonCall"]
    MTKTrackerExt = "Tracker"

    [deps.ModelingToolkitBase.weakdeps]
    BifurcationKit = "0f109fa4-8a5d-4b75-95aa-f515264e7665"
    CasADi = "c49709b8-5c63-11e9-2fb2-69db5844192f"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffEqNoiseProcess = "77a26b50-5914-5dd7-bc55-306e6241c503"
    DynamicQuantities = "06fc5a27-2a28-4c7c-a15d-362465fb6821"
    InfiniteOpt = "20393b10-9daf-11e9-18c9-8db751c92c57"
    JuliaFormatter = "98e50ef6-434e-11e9-1051-2b60c6c9e899"
    LabelledArrays = "2ee39098-c373-598a-b85f-a56591580800"
    Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    Pyomo = "0e8e1daf-01b5-4eba-a626-3897743a3816"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ModelingToolkitTearing]]
deps = ["BipartiteGraphs", "CommonSolve", "DocStringExtensions", "Graphs", "LinearAlgebra", "ModelingToolkitBase", "Moshi", "OffsetArrays", "OrderedCollections", "SciMLBase", "Setfield", "SparseArrays", "StateSelection", "SymbolicIndexingInterface", "SymbolicUtils", "Symbolics", "UUIDs"]
git-tree-sha1 = "543a9abbe1dbf6aa27a5cae349df946485399e8d"
uuid = "6bb917b9-1269-42b9-9f7c-b0dca72083ab"
version = "1.20.6"

[[deps.Moshi]]
deps = ["ExproniconLite", "Jieko"]
git-tree-sha1 = "3f37e471438f418e14b85e02376234c037218f07"
uuid = "2e0e35c7-a2e4-4343-998d-7ef72827ed2d"
version = "0.3.9"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.MuladdMacro]]
deps = ["PrecompileTools"]
git-tree-sha1 = "283bf85d4a767481dd924dff0eee1735e95f449e"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.7"

[[deps.MultivariatePolynomials]]
deps = ["DataStructures", "LinearAlgebra", "MutableArithmetics", "StarAlgebras"]
git-tree-sha1 = "4838893d9b035c2f6967c0d533350e1755b58a70"
uuid = "102ac46a-7ee4-5c85-9060-abc95bfdeaa3"
version = "0.5.19"
weakdeps = ["ChainRulesCore"]

    [deps.MultivariatePolynomials.extensions]
    MultivariatePolynomialsChainRulesCoreExt = "ChainRulesCore"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "dc5b2c4c111c46bc79ac4405eeb563523b39c004"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.8.0"

[[deps.NLSolversBase]]
deps = ["ADTypes", "DifferentiationInterface", "FiniteDiff", "LinearAlgebra"]
git-tree-sha1 = "f96d38936d92d610318dec4d3b5ef37b0373c20f"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "8.0.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "dbd2e8cd2c1c27f0b584f6661b4309609c5a685e"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.4"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.NonlinearSolve]]
deps = ["ADTypes", "ArrayInterface", "BracketingNonlinearSolve", "CommonSolve", "ConcreteStructs", "DifferentiationInterface", "FastClosures", "FiniteDiff", "ForwardDiff", "LineSearch", "LinearAlgebra", "LinearSolve", "NonlinearSolveBase", "NonlinearSolveFirstOrder", "NonlinearSolveQuasiNewton", "NonlinearSolveSpectralMethods", "PrecompileTools", "Preferences", "Reexport", "SciMLBase", "Setfield", "SimpleNonlinearSolve", "StaticArraysCore", "SymbolicIndexingInterface"]
git-tree-sha1 = "90afae5b2b33dcd1098e4c1a7b6066f18741bf6f"
uuid = "8913a72c-1f9b-4ce2-8d82-65094dcecaec"
version = "4.32.0"

    [deps.NonlinearSolve.extensions]
    NonlinearSolveFastLevenbergMarquardtExt = "FastLevenbergMarquardt"
    NonlinearSolveFixedPointAccelerationExt = "FixedPointAcceleration"
    NonlinearSolveLeastSquaresOptimExt = "LeastSquaresOptim"
    NonlinearSolveMINPACKExt = "MINPACK"
    NonlinearSolveNLSolversExt = "NLSolvers"
    NonlinearSolveNLsolveExt = ["NLsolve", "LineSearches"]
    NonlinearSolvePETScExt = ["PETSc", "MPI", "SparseArrays"]
    NonlinearSolveSIAMFANLEquationsExt = "SIAMFANLEquations"
    NonlinearSolveSpeedMappingExt = "SpeedMapping"
    NonlinearSolveSundialsExt = "Sundials"

    [deps.NonlinearSolve.weakdeps]
    FastLevenbergMarquardt = "7a0df574-e128-4d35-8cbd-3d84502bf7ce"
    FixedPointAcceleration = "817d07cb-a79a-5c30-9a31-890123675176"
    LeastSquaresOptim = "0fc2ff8b-aaa3-5acd-a817-1944a5e08891"
    LineSearches = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
    MINPACK = "4854310b-de5a-5eb6-a2a5-c1dee2bd17f9"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    NLSolvers = "337daf1e-9722-11e9-073e-8b9effe078ba"
    NLsolve = "2774e3e8-f4cf-5e23-947b-6d7e65073b56"
    PETSc = "ace2c81b-2b5f-4b1e-a30d-d662738edfe0"
    SIAMFANLEquations = "084e46ad-d928-497d-ad5e-07fa361a48c4"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SpeedMapping = "f1835b91-879b-4a3f-a438-e4baacf14412"
    Sundials = "c3572dad-4567-51f8-b174-8c6c989267f4"

[[deps.NonlinearSolveBase]]
deps = ["ADTypes", "Adapt", "ArrayInterface", "CommonSolve", "Compat", "ConcreteStructs", "DifferentiationInterface", "EnumX", "EnzymeCore", "FastClosures", "FunctionWrappers", "FunctionWrappersWrappers", "LinearAlgebra", "LogExpFunctions", "Markdown", "MaybeInplace", "PreallocationTools", "PrecompileTools", "Preferences", "Printf", "RecursiveArrayTools", "RespecializeParams", "SciMLBase", "SciMLJacobianOperators", "SciMLLogging", "SciMLOperators", "SciMLStructures", "Setfield", "StaticArraysCore", "SymbolicIndexingInterface", "TimerOutputs"]
git-tree-sha1 = "a166907a5f5d6722aa09f74172a51332e43267f0"
uuid = "be0214bd-f91f-a760-ac4e-3421ce2b2da0"
version = "2.51.1"

    [deps.NonlinearSolveBase.extensions]
    NonlinearSolveBaseBandedMatricesExt = "BandedMatrices"
    NonlinearSolveBaseChainRulesCoreExt = "ChainRulesCore"
    NonlinearSolveBaseEnzymeExt = ["ChainRulesCore", "Enzyme"]
    NonlinearSolveBaseForwardDiffExt = "ForwardDiff"
    NonlinearSolveBaseLineSearchExt = "LineSearch"
    NonlinearSolveBaseLinearSolveExt = "LinearSolve"
    NonlinearSolveBaseMooncakeExt = ["ChainRulesCore", "Mooncake"]
    NonlinearSolveBaseReverseDiffExt = "ReverseDiff"
    NonlinearSolveBaseSparseArraysExt = "SparseArrays"
    NonlinearSolveBaseSparseMatrixColoringsExt = "SparseMatrixColorings"
    NonlinearSolveBaseTrackerExt = "Tracker"

    [deps.NonlinearSolveBase.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LineSearch = "87fe0de2-c867-4266-b59a-2f0a94fc965b"
    LinearSolve = "7ed4a6bd-45f5-4d41-b270-4a48e9bafcae"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.NonlinearSolveFirstOrder]]
deps = ["ADTypes", "ArrayInterface", "CommonSolve", "ConcreteStructs", "FiniteDiff", "ForwardDiff", "LineSearch", "LinearAlgebra", "LinearSolve", "MaybeInplace", "NonlinearSolveBase", "PrecompileTools", "Reexport", "SciMLBase", "SciMLJacobianOperators", "SciMLLogging", "SciMLOperators", "Setfield", "Sobol", "StaticArraysCore"]
git-tree-sha1 = "d46d0e313b095ab48b02394e851a17ccbeff70e6"
uuid = "5959db7a-ea39-4486-b5fe-2dd0bf03d60d"
version = "2.8.0"

[[deps.NonlinearSolveQuasiNewton]]
deps = ["ArrayInterface", "CommonSolve", "ConcreteStructs", "LinearAlgebra", "LinearSolve", "MaybeInplace", "NonlinearSolveBase", "PrecompileTools", "Reexport", "SciMLBase", "SciMLLogging", "SciMLOperators", "StaticArraysCore"]
git-tree-sha1 = "db0bca65e9c849333023a0d5827813947d003f09"
uuid = "9a2c21bd-3a47-402d-9113-8faf9a0ee114"
version = "1.15.3"
weakdeps = ["ForwardDiff"]

    [deps.NonlinearSolveQuasiNewton.extensions]
    NonlinearSolveQuasiNewtonForwardDiffExt = "ForwardDiff"

[[deps.NonlinearSolveSpectralMethods]]
deps = ["CommonSolve", "ConcreteStructs", "LineSearch", "MaybeInplace", "NonlinearSolveBase", "PrecompileTools", "Reexport", "SciMLBase", "SciMLLogging"]
git-tree-sha1 = "c3e5a641a867b7f37155a78086cf3f22134c088f"
uuid = "26075421-4e9a-44e1-8bd1-420ed7ad02b2"
version = "1.8.3"
weakdeps = ["ForwardDiff"]

    [deps.NonlinearSolveSpectralMethods.extensions]
    NonlinearSolveSpectralMethodsForwardDiffExt = "ForwardDiff"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS32_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "libblastrampoline_jll"]
git-tree-sha1 = "30870d0f2dc0b2dba76b10df1c58c7f018413e56"
uuid = "656ef2d0-ae68-5445-9ca0-591084a874a2"
version = "0.3.34+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.6+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optim]]
deps = ["ADTypes", "EnumX", "FillArrays", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "PositiveFactorizations", "Printf", "SparseArrays", "Statistics"]
git-tree-sha1 = "81f338ad984b25bc82471db4d235e7b8bf21b8d6"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "2.3.2"
weakdeps = ["MathOptInterface"]

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Compat", "ConstructionBase", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "b6a586b581eccc60a181145ffd4382099e32e2df"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.4.9"

    [deps.Optimisers.extensions]
    OptimisersAdaptExt = ["Adapt"]
    OptimisersEnzymeCoreExt = "EnzymeCore"
    OptimisersReactantExt = "Reactant"
    OptimisersReactantMLDataDevicesExt = ["Reactant", "MLDataDevices"]

    [deps.Optimisers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    MLDataDevices = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"

[[deps.Optimization]]
deps = ["ADTypes", "ArrayInterface", "ConsoleProgressMonitor", "DocStringExtensions", "LinearAlgebra", "Logging", "LoggingExtras", "OptimizationBase", "Printf", "Reexport", "SciMLBase", "SparseArrays", "TerminalLoggers"]
git-tree-sha1 = "b4d47691ca17e5f8e26cc7e453f6ab92d480285f"
uuid = "7f7a1694-90dd-40f0-9382-eb1efda571ba"
version = "5.9.1"

[[deps.OptimizationBase]]
deps = ["ADTypes", "ArrayInterface", "DifferentiationInterface", "DocStringExtensions", "FastClosures", "LinearAlgebra", "PrecompileTools", "SciMLBase", "SciMLLogging", "SparseArrays", "SparseConnectivityTracer", "SparseMatrixColorings", "SymbolicIndexingInterface"]
git-tree-sha1 = "c390a9e18ec082cee22136a42f50b208b8312d0b"
uuid = "bca83a33-5cc9-4baa-983d-23429ab6bcbb"
version = "5.6.1"

    [deps.OptimizationBase.extensions]
    OptimizationChainRulesCoreExt = "ChainRulesCore"
    OptimizationEnzymeExt = ["ChainRulesCore", "Enzyme"]
    OptimizationFiniteDiffExt = "FiniteDiff"
    OptimizationForwardDiffExt = "ForwardDiff"
    OptimizationMLDataDevicesExt = "MLDataDevices"
    OptimizationMLUtilsExt = "MLUtils"
    OptimizationMooncakeExt = "Mooncake"
    OptimizationReverseDiffExt = "ReverseDiff"
    OptimizationSymbolicAnalysisExt = "SymbolicAnalysis"
    OptimizationZygoteExt = "Zygote"

    [deps.OptimizationBase.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    MLDataDevices = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
    MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SymbolicAnalysis = "4297ee4d-0239-47d8-ba5d-195ecdf594fe"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.OptimizationMOI]]
deps = ["ADTypes", "LinearAlgebra", "MathOptInterface", "ModelingToolkitBase", "OptimizationBase", "Reexport", "SciMLBase", "SciMLLogging", "SciMLStructures", "SparseArrays", "SymbolicIndexingInterface", "SymbolicUtils", "Symbolics"]
git-tree-sha1 = "39798d0de6807b244d50423577ddee50263cf282"
uuid = "fd9f6733-72f4-499f-8506-86b2bdd0dea1"
version = "1.4.1"
weakdeps = ["ModelingToolkit"]

[[deps.OptimizationOptimJL]]
deps = ["Optim", "OptimizationBase", "SciMLBase", "SparseArrays"]
git-tree-sha1 = "6bc626d531967889260563d7cd42294047fd91e8"
uuid = "36348300-93cb-4f02-beb5-3c3902f8871e"
version = "0.4.21"

[[deps.OptimizationOptimisers]]
deps = ["Logging", "Optimisers", "OptimizationBase", "SciMLBase", "SciMLLogging"]
git-tree-sha1 = "304bef9d4beab854782b722255d97cd061383636"
uuid = "42dfb2eb-d2b4-4451-abcd-913932933ac1"
version = "0.3.24"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05f45c2e0de6259db764adbfd2f1dc6d3f8de13c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "2.0.1"

[[deps.OrdinaryDiffEq]]
deps = ["ADTypes", "CommonSolve", "DiffEqBase", "DocStringExtensions", "OrdinaryDiffEqBDF", "OrdinaryDiffEqCore", "OrdinaryDiffEqDefault", "OrdinaryDiffEqRosenbrock", "OrdinaryDiffEqTsit5", "OrdinaryDiffEqVerner", "SciMLBase", "SciMLLogging"]
git-tree-sha1 = "bae9b8e5f1d00e51c554648e5094e215164a19f8"
uuid = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
version = "7.8.1"

[[deps.OrdinaryDiffEqBDF]]
deps = ["ADTypes", "ArrayInterface", "DiffEqBase", "FastBroadcast", "LinearAlgebra", "MacroTools", "MuladdMacro", "OrdinaryDiffEqCore", "OrdinaryDiffEqDifferentiation", "OrdinaryDiffEqNonlinearSolve", "OrdinaryDiffEqSDIRK", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "SciMLBase", "TruncatedStacktraces"]
git-tree-sha1 = "042c06109378fec9d4a204d81942f9eb45b157e7"
uuid = "6ad6398a-0878-4a85-9266-38940aa047c8"
version = "2.4.10"

[[deps.OrdinaryDiffEqCore]]
deps = ["ADTypes", "Accessors", "Adapt", "ArrayInterface", "BinaryHeaps", "CommonSolve", "ConstructionBase", "DiffEqBase", "DocStringExtensions", "EnumX", "EnzymeCore", "FastBroadcast", "FastClosures", "FastPower", "FindFirstFunctions", "FunctionWrappers", "FunctionWrappersWrappers", "InteractiveUtils", "LinearAlgebra", "Logging", "MacroTools", "MuladdMacro", "PrecompileTools", "Preferences", "Printf", "Random", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLLogging", "SciMLOperators", "SciMLStructures", "SymbolicIndexingInterface", "TruncatedStacktraces"]
git-tree-sha1 = "b4351cee919a8312d0fa6186b9e90d464d62a397"
uuid = "bbf590c4-e513-4bbe-9b18-05decba2e5d8"
version = "4.17.4"

    [deps.OrdinaryDiffEqCore.extensions]
    OrdinaryDiffEqCoreMooncakeExt = "Mooncake"
    OrdinaryDiffEqCorePolyesterExt = "Polyester"
    OrdinaryDiffEqCoreSparseArraysExt = "SparseArrays"

    [deps.OrdinaryDiffEqCore.weakdeps]
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    Polyester = "f517fe37-dbe3-4b94-8317-1923a5111588"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.OrdinaryDiffEqDefault]]
deps = ["ADTypes", "DiffEqBase", "EnumX", "LinearAlgebra", "LinearSolve", "OrdinaryDiffEqBDF", "OrdinaryDiffEqCore", "OrdinaryDiffEqRosenbrock", "OrdinaryDiffEqTsit5", "OrdinaryDiffEqVerner", "PrecompileTools", "Preferences", "Reexport", "SciMLBase"]
git-tree-sha1 = "1ce6ba1f52c94059089f835f173fddda90233959"
uuid = "50262376-6c5a-4cf5-baba-aaf4f84d72d7"
version = "2.6.2"

[[deps.OrdinaryDiffEqDifferentiation]]
deps = ["ADTypes", "ArrayInterface", "ConcreteStructs", "ConstructionBase", "DiffEqBase", "DifferentiationInterface", "FastBroadcast", "FiniteDiff", "ForwardDiff", "FunctionWrappersWrappers", "LinearAlgebra", "LinearSolve", "OrdinaryDiffEqCore", "SciMLBase", "SciMLOperators", "SparseMatrixColorings", "StaticArraysCore"]
git-tree-sha1 = "d24ac464cc0e8b3d23f957186b31920ed0055a66"
uuid = "4302a76b-040a-498a-8c04-15b101fed76b"
version = "3.12.2"
weakdeps = ["SparseArrays"]

    [deps.OrdinaryDiffEqDifferentiation.extensions]
    OrdinaryDiffEqDifferentiationSparseArraysExt = "SparseArrays"

[[deps.OrdinaryDiffEqNonlinearSolve]]
deps = ["ADTypes", "ArrayInterface", "CommonSolve", "ConstructionBase", "DiffEqBase", "FastBroadcast", "FastClosures", "ForwardDiff", "LinearAlgebra", "LinearSolve", "MuladdMacro", "NonlinearSolve", "NonlinearSolveBase", "OrdinaryDiffEqCore", "OrdinaryDiffEqDifferentiation", "PreallocationTools", "RecursiveArrayTools", "SciMLBase", "SciMLOperators", "SciMLPublic", "SimpleNonlinearSolve", "SparseArrays", "StaticArraysCore"]
git-tree-sha1 = "d37ed37d7f2c6d29770f8235b54dbb614572b228"
uuid = "127b3ac7-2247-4354-8eb6-78cf4e7c58e8"
version = "2.9.8"

[[deps.OrdinaryDiffEqRosenbrock]]
deps = ["ADTypes", "ArrayInterface", "DiffEqBase", "DifferentiationInterface", "FastBroadcast", "FiniteDiff", "ForwardDiff", "LinearAlgebra", "LinearSolve", "MacroTools", "MuladdMacro", "OrdinaryDiffEqCore", "OrdinaryDiffEqDifferentiation", "OrdinaryDiffEqRosenbrockTableaus", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "SciMLBase"]
git-tree-sha1 = "b2f5f9afb665548c598c287b9d62954bb96d8418"
uuid = "43230ef6-c299-4910-a778-202eb28ce4ce"
version = "2.7.4"

[[deps.OrdinaryDiffEqRosenbrockTableaus]]
git-tree-sha1 = "bab0d4ccee6fcb89a786b89207774314523e2e5d"
uuid = "b4bd8bb3-f80f-41d2-9b21-73a655b304b9"
version = "2.4.2"

[[deps.OrdinaryDiffEqSDIRK]]
deps = ["ADTypes", "CommonSolve", "ConstructionBase", "DiffEqBase", "FastBroadcast", "LinearAlgebra", "MacroTools", "MuladdMacro", "OrdinaryDiffEqCore", "OrdinaryDiffEqDifferentiation", "OrdinaryDiffEqNonlinearSolve", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "SciMLBase", "TruncatedStacktraces"]
git-tree-sha1 = "81e3394743eaf8ff282260104f97b7ae5155effb"
uuid = "2d112036-d095-4a1e-ab9a-08536f3ecdbf"
version = "2.9.6"

[[deps.OrdinaryDiffEqTsit5]]
deps = ["CommonSolve", "DiffEqBase", "FastBroadcast", "LinearAlgebra", "MuladdMacro", "OrdinaryDiffEqCore", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "SciMLBase", "TruncatedStacktraces"]
git-tree-sha1 = "2fdf22d85a4c35e913d8b746480167e0206cf0c3"
uuid = "b1df2697-797e-41e3-8120-5422d3b24e4a"
version = "2.1.4"

[[deps.OrdinaryDiffEqVerner]]
deps = ["DiffEqBase", "FastBroadcast", "LinearAlgebra", "MuladdMacro", "OrdinaryDiffEqCore", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "SciMLBase", "TruncatedStacktraces"]
git-tree-sha1 = "50756dd4eb1a6ae0b3b5f33e5fdfaf1a83e6a8e9"
uuid = "79d7bb75-1356-48c1-b8c0-6832512096c2"
version = "2.4.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1912a9f1b9ca55005b03ba075f8e19993583e237"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.58.2+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools"]
git-tree-sha1 = "663e8b48b789916221e0765393b289ca6c88f24e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "3.0.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "83bd514e8ff16b5858ac54c53fa0bcf6002a3b00"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.7"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PoissonRandom]]
deps = ["LogExpFunctions", "PrecompileTools", "Random"]
git-tree-sha1 = "faa278c0dc901606775554f6171350ad962d0e18"
uuid = "e409e4f3-bfea-5376-8464-e040bb5c01ab"
version = "0.4.13"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "cd00e28071a75c98664663f87c2ae447d25c0503"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "1.7.1"

    [deps.PreallocationTools.extensions]
    PreallocationToolsEnzymeCoreExt = "EnzymeCore"
    PreallocationToolsForwardDiffExt = "ForwardDiff"
    PreallocationToolsReverseDiffExt = "ReverseDiff"

    [deps.PreallocationTools.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "5005266de4bfe50e53ff44a5cb5c540b6e47a254"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.6.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1b8aa19f229b1cea7fc93874a52e49db6a854450"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.4.8"

    [deps.PrettyTables.extensions]
    PrettyTablesExcelExt = "XLSX"
    PrettyTablesTypstryExt = "Typstry"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"
    XLSX = "fdbf4ff8-1666-58a4-91e7-1b58723a45e0"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "25cdd1d20cd005b52fc12cb6be3f75faaf59bb9b"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
deps = ["StyledStrings"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "f0803bc1171e455a04124affa9c21bba5ac4db32"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.6"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "fbb92c6c56b34e1a2c4c36058f68f332bec840e7"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.PureKLU]]
deps = ["LinearAlgebra", "PrecompileTools", "SparseArrays"]
git-tree-sha1 = "6719681d517f9b02fc3e3c3a0befbde5e4a2320e"
uuid = "0c0d3e7f-3a8b-4f7e-b6f1-9a4d2e7c1f01"
version = "1.5.0"
weakdeps = ["ForwardDiff"]

    [deps.PureKLU.extensions]
    PureKLUForwardDiffExt = "ForwardDiff"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "144895f6166994730ee7ff8113b981fc360638f1"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.10.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll", "Qt6Svg_jll"]
git-tree-sha1 = "159d253ab126d5b29230cf53521899bea4ef4648"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.10.2+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "4d85eedf69d875982c46643f6b4f66919d7e157b"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.10.2+1"

[[deps.Qt6Svg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "81587ff5ff25a4e1115ce191e36285ede0334c9d"
uuid = "6de9746b-f93d-5813-b365-ba18ad4a9cf3"
version = "6.10.2+0"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "672c938b4b4e3e0169a07a5f227029d4905456f2"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.10.2+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.ReadOnlyArrays]]
git-tree-sha1 = "e6f7ddf48cf141cb312b078ca21cb2d29d0dc11d"
uuid = "988b38a3-91fc-5605-94a2-ee2116b3bd83"
version = "0.2.0"

[[deps.ReadOnlyDicts]]
deps = ["DocStringExtensions"]
git-tree-sha1 = "711acef70140078d808be9cd33040f510af57f5e"
uuid = "795d4caa-f5a7-4580-b5d8-c01d53451803"
version = "1.0.1"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "GPUArraysCore", "LinearAlgebra", "PrecompileTools", "RecipesBase", "SciMLPublic", "SciMLStructures", "StaticArraysCore", "SymbolicIndexingInterface"]
git-tree-sha1 = "775b6023c34c401e35b9a159568d2f6d051280d5"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "4.5.1"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsCUDAExt = "CUDA"
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsFastBroadcastPolyesterExt = ["FastBroadcast", "Polyester"]
    RecursiveArrayToolsForwardDiffExt = "ForwardDiff"
    RecursiveArrayToolsKernelAbstractionsExt = "KernelAbstractions"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsMooncakeExt = "Mooncake"
    RecursiveArrayToolsReverseDiffExt = ["ReverseDiff", "Zygote"]
    RecursiveArrayToolsSparseArraysExt = ["SparseArrays"]
    RecursiveArrayToolsStatisticsExt = "Statistics"
    RecursiveArrayToolsStructArraysExt = "StructArrays"
    RecursiveArrayToolsTablesExt = ["Tables"]
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    Polyester = "f517fe37-dbe3-4b94-8317-1923a5111588"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.RespecializeParams]]
deps = ["FunctionWrappersWrappers", "PrecompileTools"]
git-tree-sha1 = "140c074bb63518a0e1ec2c8aba2a8a5719bcf21c"
uuid = "9fe22ead-9e00-4db2-8b46-706a60d40f5e"
version = "1.3.0"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "PrecompileTools", "SHA", "Serialization"]
git-tree-sha1 = "1719b26eee2c5e40d4b41c647b91dbaac1dcbf05"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.26"

[[deps.SCCNonlinearSolve]]
deps = ["CommonSolve", "NonlinearSolveBase", "PrecompileTools", "SciMLBase", "SymbolicIndexingInterface"]
git-tree-sha1 = "3244efe86d4fbd5817251bb5c3c93ff111e0e9f1"
uuid = "9dfe8606-65a1-4bb3-9748-cb89d1561431"
version = "1.15.3"
weakdeps = ["ChainRulesCore"]

    [deps.SCCNonlinearSolve.extensions]
    SCCNonlinearSolveChainRulesCoreExt = "ChainRulesCore"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SPRAL_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "Libdl", "METIS_jll", "libblastrampoline_jll"]
git-tree-sha1 = "b78d602475715c9c6ea21c9650fa5dba58e9945b"
uuid = "319450e9-13b8-58e8-aa9f-8fd1420848ab"
version = "2025.9.18+1"

[[deps.SciMLBase]]
deps = ["ADTypes", "Accessors", "Adapt", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FindFirstFunctions", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "LoggingExtras", "Markdown", "PreallocationTools", "PrecompileTools", "Preferences", "Printf", "Random", "RecipesBase", "RecursiveArrayTools", "RuntimeGeneratedFunctions", "SciMLOperators", "SciMLPublic", "SciMLStructures", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface"]
git-tree-sha1 = "8a14bac10be51ecf8cb167e483e39c068ca24bfd"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "3.55.0"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBaseDifferentiationInterfaceExt = "DifferentiationInterface"
    SciMLBaseDistributionsExt = "Distributions"
    SciMLBaseEnzymeExt = "Enzyme"
    SciMLBaseForwardDiffExt = "ForwardDiff"
    SciMLBaseFunctionPropertiesExt = "FunctionProperties"
    SciMLBaseMakieExt = ["Makie", "GeometryBasics"]
    SciMLBaseMeasurementsExt = "Measurements"
    SciMLBaseMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    SciMLBaseMooncakeExt = "Mooncake"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseReverseDiffExt = ["ReverseDiff", "ForwardDiff"]
    SciMLBaseStaticArraysExt = "StaticArrays"
    SciMLBaseTrackerExt = "Tracker"
    SciMLBaseZygoteExt = ["Zygote", "ChainRulesCore", "ZygoteRules", "FillArrays"]

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DifferentiationInterface = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    FunctionProperties = "f62d2435-5019-4c03-9749-2d4c77af0cbc"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
    ZygoteRules = "700de1a5-db45-46bc-99cf-38207098b444"

[[deps.SciMLJacobianOperators]]
deps = ["ADTypes", "ArrayInterface", "ConcreteStructs", "ConstructionBase", "DifferentiationInterface", "FastClosures", "LinearAlgebra", "SciMLBase", "SciMLOperators"]
git-tree-sha1 = "e58036956d27635a57276b05bdbbe8b22376a805"
uuid = "19f34311-ddf3-4b8b-af20-060888a46c0e"
version = "0.1.19"

[[deps.SciMLLogging]]
deps = ["Logging", "LoggingExtras", "PrecompileTools", "Preferences"]
git-tree-sha1 = "3aafee7edc2ec7e71f50dcc6816cb031fac390a0"
uuid = "a6db7da4-7206-11f0-1eab-35f2a5dbe1d1"
version = "2.1.0"

    [deps.SciMLLogging.extensions]
    SciMLLoggingTracyExt = "Tracy"

    [deps.SciMLLogging.weakdeps]
    Tracy = "e689c965-62c8-4b79-b2c5-8359227902fd"

[[deps.SciMLOperators]]
deps = ["Accessors", "Adapt", "ArrayInterface", "DocStringExtensions", "LinearAlgebra", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "9eba48f738160e04268c0770e01dae2451199bc2"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "1.30.1"

    [deps.SciMLOperators.extensions]
    SciMLOperatorsLoopVectorizationExt = "LoopVectorization"
    SciMLOperatorsSparseArraysExt = "SparseArrays"

    [deps.SciMLOperators.weakdeps]
    LoopVectorization = "bdcacae8-1622-11e9-2a5c-532679323890"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SciMLPublic]]
deps = ["PrecompileTools"]
git-tree-sha1 = "74685afb51732a464fbce79a72708f7c4203ceb7"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.3.0"

[[deps.SciMLStructures]]
deps = ["ArrayInterface", "PrecompileTools"]
git-tree-sha1 = "5c2f9dbf6f07eea6bc9e93f117b00b7939a79f9e"
uuid = "53ae85a6-f571-4167-b2af-e1d143709226"
version = "1.10.5"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.Showoff]]
deps = ["Dates"]
git-tree-sha1 = "8238217340ad0aaabe11afe39c1098b5bc9f4c8e"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.1.1"

[[deps.SimpleNonlinearSolve]]
deps = ["ADTypes", "ArrayInterface", "BracketingNonlinearSolve", "CommonSolve", "ConcreteStructs", "DifferentiationInterface", "FastClosures", "FiniteDiff", "ForwardDiff", "LineSearch", "LinearAlgebra", "MaybeInplace", "NonlinearSolveBase", "PrecompileTools", "Reexport", "SciMLBase", "SciMLLogging", "Setfield", "StaticArraysCore"]
git-tree-sha1 = "ff7b8621512721ba9bd86b43a1a2f6466b7caa72"
uuid = "727e6d20-b764-4bd8-a329-72de5adea6c7"
version = "2.14.5"

    [deps.SimpleNonlinearSolve.extensions]
    SimpleNonlinearSolveChainRulesCoreExt = "ChainRulesCore"
    SimpleNonlinearSolveReverseDiffExt = "ReverseDiff"
    SimpleNonlinearSolveTrackerExt = "Tracker"

    [deps.SimpleNonlinearSolve.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "7ddb0b49c109481b046972c0e4ab02b2127d6a75"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.6"

[[deps.Sobol]]
deps = ["DelimitedFiles", "Random"]
git-tree-sha1 = "5a74ac22a9daef23705f010f72c81d6925b19df8"
uuid = "ed01d8cd-4d21-5b2a-85b4-cc3bdc58bad4"
version = "1.5.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SparseColumnPivotedQR]]
deps = ["LinearAlgebra", "PrecompileTools", "SparseArrays"]
git-tree-sha1 = "8cf1a59c78dd6f900feb366e5a574f99ed278ef1"
uuid = "a57abbd0-fea5-4d57-96be-5e525945e8e4"
version = "2.1.8"
weakdeps = ["AMD"]

    [deps.SparseColumnPivotedQR.extensions]
    SparseColumnPivotedQRAMDExt = "AMD"

[[deps.SparseConnectivityTracer]]
deps = ["ADTypes", "DocStringExtensions", "FillArrays", "LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "5c127b6b5e36e1ba38556b578c6346b2375ec0b4"
uuid = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
version = "1.2.3"

    [deps.SparseConnectivityTracer.extensions]
    SparseConnectivityTracerChainRulesCoreExt = "ChainRulesCore"
    SparseConnectivityTracerLogExpFunctionsExt = "LogExpFunctions"
    SparseConnectivityTracerNNlibExt = "NNlib"
    SparseConnectivityTracerNaNMathExt = "NaNMath"
    SparseConnectivityTracerSpecialFunctionsExt = "SpecialFunctions"

    [deps.SparseConnectivityTracer.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
    NNlib = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
    NaNMath = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.SparseMatrixColorings]]
deps = ["ADTypes", "DocStringExtensions", "LinearAlgebra", "PrecompileTools", "Random", "SparseArrays"]
git-tree-sha1 = "63e1776f40bbd5e7394edce08e62f852b1384baf"
uuid = "0a514795-09f3-496d-8182-132a7b665d35"
version = "0.4.28"

    [deps.SparseMatrixColorings.extensions]
    SparseMatrixColoringsCUDAExt = ["CUDA", "cuSPARSE"]
    SparseMatrixColoringsCliqueTreesExt = "CliqueTrees"
    SparseMatrixColoringsColorsExt = "Colors"
    SparseMatrixColoringsGPUArraysExt = "GPUArrays"
    SparseMatrixColoringsJuMPExt = ["JuMP", "MathOptInterface"]

    [deps.SparseMatrixColorings.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CliqueTrees = "60701a23-6482-424a-84db-faee86b9b1f8"
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
    cuSPARSE = "b26da814-b3bc-49ef-b0ee-c816305aa060"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "429071b23f4c9a13fb6582f807cc2ef454082408"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.9.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.StarAlgebras]]
deps = ["LinearAlgebra", "MutableArithmetics", "SparseArrays"]
git-tree-sha1 = "235b1f9d287bbf34083b3d0829343a7942c0ad1c"
uuid = "0c0c59c1-dc5f-42e9-9a8b-b5dc384a6cd1"
version = "0.3.0"

[[deps.StateSelection]]
deps = ["BipartiteGraphs", "DocStringExtensions", "FindFirstFunctions", "Graphs", "LinearAlgebra", "OrderedCollections", "Setfield", "SparseArrays"]
git-tree-sha1 = "10914498005246949b0474ad26a650570e08948b"
uuid = "64909d44-ed92-46a8-bbd9-f047dfbdc84b"
version = "1.11.1"

    [deps.StateSelection.extensions]
    StateSelectionDeepDiffsExt = "DeepDiffs"

    [deps.StateSelection.weakdeps]
    DeepDiffs = "ab62b9b5-e342-54a8-a765-a90f495de1a6"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "e206cf4850fd7ac4255ffd2b98922f563e18ac53"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.20"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e2b53ce13a53367e96601081e33d34746b571bad"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.5"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "adb9da019510162e67a4493fc235c23203d8b09e"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.13"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "773065c6e0e903924a9d838259be74338422aef2"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.5.0"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "2d0fc55c61321ba245c47be599570d11bac50303"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.5"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.SymbolicIndexingInterface]]
deps = ["Accessors", "ArrayInterface", "PrecompileTools", "RuntimeGeneratedFunctions", "StaticArraysCore"]
git-tree-sha1 = "7eb6da9656581ac9fbffaef3ef7950a090a5002a"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.55"
weakdeps = ["PrettyTables"]

    [deps.SymbolicIndexingInterface.extensions]
    SymbolicIndexingInterfacePrettyTablesExt = "PrettyTables"

[[deps.SymbolicLimits]]
deps = ["PrecompileTools", "SymbolicUtils", "TermInterface"]
git-tree-sha1 = "3780b4d8aaa1db8996a67146ac7d4ac20606f56e"
uuid = "19f23fe9-fdab-4a78-91af-e7b7767979c3"
version = "1.2.1"

[[deps.SymbolicUtils]]
deps = ["AbstractTrees", "ArrayInterface", "Combinatorics", "ConstructionBase", "DataStructures", "DocStringExtensions", "DynamicPolynomials", "EnumX", "ExproniconLite", "Graphs", "LinearAlgebra", "MacroTools", "Moshi", "MultivariatePolynomials", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "ReadOnlyArrays", "SciMLPublic", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArraysCore", "SymbolicIndexingInterface", "TaskLocalValues", "TermInterface", "WeakCacheSets"]
git-tree-sha1 = "fbe177f8b6e50734335a32beab6e055ce151eea7"
uuid = "d1185830-fcd6-423d-90d6-eec64667417b"
version = "4.46.6"

    [deps.SymbolicUtils.extensions]
    SymbolicUtilsChainRulesCoreExt = "ChainRulesCore"
    SymbolicUtilsDistributionsExt = "Distributions"
    SymbolicUtilsLabelledArraysExt = "LabelledArrays"
    SymbolicUtilsReverseDiffExt = "ReverseDiff"

    [deps.SymbolicUtils.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    LabelledArrays = "2ee39098-c373-598a-b85f-a56591580800"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

[[deps.Symbolics]]
deps = ["ADTypes", "AbstractPlutoDingetjes", "ArrayInterface", "Bijections", "CommonWorldInvalidations", "ConstructionBase", "DataStructures", "DiffRules", "DocStringExtensions", "DomainSets", "DynamicPolynomials", "IntervalSets", "Libdl", "LinearAlgebra", "LogExpFunctions", "MacroTools", "Markdown", "Moshi", "MultivariatePolynomials", "MutableArithmetics", "NaNMath", "PrecompileTools", "Preferences", "Primes", "RecipesBase", "RuntimeGeneratedFunctions", "SciMLPublic", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArraysCore", "SymbolicIndexingInterface", "SymbolicLimits", "SymbolicUtils", "TermInterface"]
git-tree-sha1 = "5541e29d374404512fa677dcbad9792228a54f25"
uuid = "0c5d862f-8b57-4792-8d23-62f2024744c7"
version = "7.39.2"

    [deps.Symbolics.extensions]
    SymbolicsD3TreesExt = "D3Trees"
    SymbolicsDistributionsExt = "Distributions"
    SymbolicsForwardDiffExt = "ForwardDiff"
    SymbolicsGroebnerExt = "Groebner"
    SymbolicsHypergeometricFunctionsExt = "HypergeometricFunctions"
    SymbolicsLatexifyExt = ["Latexify", "LaTeXStrings"]
    SymbolicsNemoExt = "Nemo"
    SymbolicsStaticArraysExt = "StaticArrays"
    SymbolicsSymPyExt = "SymPy"
    SymbolicsSymPyPythonCallExt = "SymPyPythonCall"

    [deps.Symbolics.weakdeps]
    D3Trees = "e3df1716-f71e-5df9-9e2d-98e193103c45"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Groebner = "0b43b601-686d-58a3-8a1c-6623616c7cd4"
    HypergeometricFunctions = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
    LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
    Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
    Nemo = "2edaba10-b0f1-5616-af89-8c11ac63239a"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "a94d9bdda1b7bed0046cea645639ab3f62196fac"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.14.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TaskLocalValues]]
git-tree-sha1 = "67e469338d9ce74fc578f7db1736a74d93a49eb8"
uuid = "ed4db957-447d-4319-bfb6-7fa9ae7ecf34"
version = "0.1.3"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TermInterface]]
git-tree-sha1 = "d673e0aca9e46a2f63720201f55cc7b3e7169b16"
uuid = "8ea1fca8-c5ef-4a55-8b96-4e9afe9c9a3c"
version = "2.0.0"

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "81c9b4137edfe56a56efcdcb35d721b2ce3e2416"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.8"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TimerOutputs]]
deps = ["ExprTools", "PrettyTables", "Printf", "Tables"]
git-tree-sha1 = "03271b02dd1c8f87281271575448b9cf6326a961"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "1.2.1"

    [deps.TimerOutputs.extensions]
    FlameGraphsExt = "FlameGraphs"

    [deps.TimerOutputs.weakdeps]
    FlameGraphs = "08572546-2f56-4bcf-ba4e-bab62c3a3f89"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.TruncatedStacktraces]]
deps = ["InteractiveUtils", "MacroTools", "Preferences"]
git-tree-sha1 = "ea3e54c2bdde39062abf5a9758a23735558705e1"
uuid = "781d530d-4396-4725-bb49-402e4bee1e77"
version = "1.4.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.WeakCacheSets]]
git-tree-sha1 = "386050ae4353310d8ff9c228f83b1affca2f7f38"
uuid = "d30d5f5c-d141-4870-aa07-aabb0f5fe7d5"
version = "0.1.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "80d3930c6347cfce7ccf96bd3bafdf079d9c0390"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.9+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e52eca002a11c30a858185efdfb15311e1c7a6bf"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.4+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "dcb316b3ce0941f195537dda56bea4517fcd3ff5"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.4+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "2e59214e017a55cb87474a00fa76035c82ac0e17"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.47.0+2"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ef17c47d22224aaecc76e597ab21a072e025cf7b"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.14.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "cb007192783c56d8249db4cf0e3495001edfe414"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.5+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "28e57478e8a160d346a19c28b3fffb9273bcc9c2"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.134+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e51150d5ab85cee6fc36726850f0e627ad2e4aba"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.58+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "da8c1f6eee04831f14edcfa5dae611d309807e57"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.3.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"
"""

# ╔═╡ Cell order:
# ╟─d96df0b2-bebb-472b-a077-874ebed3c16f
# ╟─ac30d390-e879-4333-8c45-6498378d501d
# ╠═474e5f09-0de1-4dc6-81e0-8e759acebcdf
# ╠═d2abc7fb-b95a-451d-a5da-109c0101aba1
# ╠═cde593cf-3ae7-47f6-8883-7393af3d94dd
# ╠═1724b3a2-ed4f-44f7-90b3-90cd70664d9d
# ╟─43e45b51-3cb1-4c85-8b84-5fe681cb8a42
# ╠═771c5eb5-6e94-4ccf-a0d7-b629b1452fce
# ╠═e753679d-34ed-4d30-9274-b9476bb0acd5
# ╠═a9c726e7-2357-4110-9ea8-99b374e88639
# ╠═f7330414-a677-4069-a7cb-3239bac81570
# ╠═9d6d56a4-c982-47c0-9407-2d8a722e3e7f
# ╠═2215dbbc-1c35-4c2a-9749-f52c58b09f4d
# ╟─1cd34230-1299-4af9-8093-e369e47f05ed
# ╠═3ccae72a-565f-4f99-b6d9-2fb2bc44d9ab
# ╠═e4bc599b-b03d-4300-a6b1-55ea8974d29c
# ╠═85d34909-dba6-4368-9e7d-137d71273f97
# ╠═03c30a6d-90f7-4d43-8cd2-4d2904437785
# ╠═4047f4b2-fdb6-460f-b789-5674bb5cc649
# ╠═69359e85-cbbc-4145-9d4b-56cd49a88f78
# ╠═f005406f-38e7-485c-92ba-1829b1534058
# ╠═448451e1-24b2-4d19-9e91-26cfd3f16493
# ╟─f56c31ee-1528-4913-999f-40aa76026b23
# ╠═07dae75d-c939-4393-9452-67817f44ed7a
# ╠═fd4b5a08-e1bd-4d20-a39d-df66a3d9fd75
# ╠═27f0e376-3828-4b0a-93b3-531e268b1248
# ╠═ca4b7b33-608a-4d0f-9b40-4084d6c1e756
# ╠═ba54aa80-a8cd-4972-b407-c9394e7556bc
# ╠═8286672c-85e6-4ba5-9540-3d2ec0abd527
# ╠═e9fd0565-f9e1-463f-863d-62b66f863b9c
# ╠═cc9f470f-051d-4cd8-8415-008019f30fa1
# ╠═26d12a25-9151-4c20-8a24-eb7723b7fb32
# ╠═09d5e5cc-99c8-41a5-ac0d-be1e6d07d1b7
# ╠═a94ff584-9773-11f0-169a-69b540829e8a
# ╠═a94ff5c8-9773-11f0-3586-17e283ccb322
# ╠═a94ff5fa-9773-11f0-0af6-e582f16ebebe
# ╠═a94ff64a-9773-11f0-3bed-bfd3e7f3a7a7
# ╠═a94ff67c-9773-11f0-09dc-2ba7b0957b86
# ╟─a94ff6ba-9773-11f0-375f-cfc5e027869b
# ╠═a94ff6f4-9773-11f0-0d0d-2d16f63c3383
# ╠═7182b0f4-c3fd-4bd4-b41a-91a66a9c485b
# ╠═a94ff758-9773-11f0-16d2-05eaa6cd8cd0
# ╠═a94ff794-9773-11f0-0969-755e398c46a1
# ╠═a94ff7d0-9773-11f0-3ae0-1919a24ec6c1
# ╠═a94ff802-9773-11f0-32e3-dd8a527a3f92
# ╠═0ff4929c-d835-4168-986f-3d5ad3e33b7d
# ╠═a94ff882-9773-11f0-0349-7599be769c50
# ╠═a94ff8d4-9773-11f0-1d6a-518b9e35a092
# ╠═a94ff91a-9773-11f0-3d5e-b56619ad0feb
# ╠═a94ff958-9773-11f0-0f6c-6b1eabaef6ac
# ╠═a94ff9a6-9773-11f0-0634-07a9553513b1
# ╠═a94ff9e2-9773-11f0-22fd-0d005fafc010
# ╠═a94ffa1c-9773-11f0-0960-adb43c187185
# ╠═a94ffa5a-9773-11f0-3006-c555024c3d7b
# ╠═a94ffa8c-9773-11f0-3e11-fb6b6a0ef529
# ╠═a95004aa-9773-11f0-3ef9-c7b4255fb618
# ╠═a9500806-9773-11f0-3761-3d9d108fe1fa
# ╠═1cb46d3d-0a35-470c-91bf-c766621d561f
# ╠═55e65992-a43f-4a2f-a78c-7435542cc559
# ╟─a9500d36-9773-11f0-28af-db3304977705
# ╠═a9500fb8-9773-11f0-33f2-7113956b5dec
# ╠═a9502296-9773-11f0-1211-67ce99ea71c4
# ╠═a950235e-9773-11f0-3653-4b3abaf73301
# ╠═a95023a4-9773-11f0-2787-958f5163342b
# ╠═a95023e0-9773-11f0-3295-911e8cea76e9
# ╠═a950241c-9773-11f0-26d0-c79549537d16
# ╠═a9502476-9773-11f0-3c61-3deb69a5acc8
# ╠═a95024b2-9773-11f0-09c4-b58fd0feb74a
# ╠═a95024e4-9773-11f0-3f92-af591ba245d4
# ╟─a9502516-9773-11f0-2386-272f2fed3bf5
# ╠═a950257a-9773-11f0-2da5-ffdda63d000c
# ╠═62a5c4e9-4c09-445b-b1b0-258be58e3a2b
# ╠═a95025be-9773-11f0-049c-9b8fa5d6fbe6
# ╠═a9502642-9773-11f0-37ed-b92da7ad835c
# ╠═a9502688-9773-11f0-375d-95481f3c34f0
# ╠═a95026ce-9773-11f0-0997-f7c1b6d2ecdc
# ╠═a950270a-9773-11f0-04fb-e5cbd99f623c
# ╠═a9502758-9773-11f0-3260-e195cfdc1c9b
# ╠═a95027a0-9773-11f0-1280-5129046d073f
# ╟─a95027dc-9773-11f0-26c2-5dcc62bc7fe7
# ╠═a9502822-9773-11f0-2fca-a1091e080746
# ╠═a9502872-9773-11f0-37af-adb9cfdcd57e
# ╠═a95028ae-9773-11f0-1202-5d63738a1835
# ╠═a9502d86-9773-11f0-2f1c-fba9f1352c9c
# ╠═a950327c-9773-11f0-29d3-eff1b9d6883d
# ╠═a95035f6-9773-11f0-0e38-f7cccd719bfd
# ╠═a950390a-9773-11f0-2b1f-b36846c46101
# ╠═a9503bbe-9773-11f0-2864-2106756926c4
# ╟─a9503e78-9773-11f0-30ba-cb2c532d4e73
# ╠═a9503ed4-9773-11f0-1f0d-2db7bd6de83b
# ╠═a9503f38-9773-11f0-113b-51dbf80ae71b
# ╠═a9503f92-9773-11f0-0bd9-6d85dd75c400
# ╠═a9503fc4-9773-11f0-0f8f-cfc89571781d
# ╠═5e2ca688-a485-4fa2-859b-8470f24deefd
# ╠═01612662-ca6f-4e2f-8922-3022fb6adb84
# ╠═dcaf27d3-83ad-4574-9477-7cbc2ce05e2f
# ╠═6cad5ecc-3b63-4611-a8a2-5b9ccc620e69
# ╠═a9504078-9773-11f0-072e-73fff6f8021b
# ╠═a95040aa-9773-11f0-194a-dd335a206bc5
# ╠═a95040fa-9773-11f0-0b7e-e92b841a56e1
# ╠═a950412c-9773-11f0-3db2-7df6300679fd
# ╠═a9504190-9773-11f0-2b32-cdd5293c5a17
# ╠═a95041cc-9773-11f0-2c72-dbf74fb1944f
# ╠═a9504212-9773-11f0-2581-7bd0c046d9ac
# ╠═a9504258-9773-11f0-26a5-d3cfe8b2533e
# ╠═a950428a-9773-11f0-1fc1-55cd36659bdf
# ╠═a95042bc-9773-11f0-07a3-756736127a69
# ╠═a95042f8-9773-11f0-37fc-519e71b86624
# ╠═a950433e-9773-11f0-12ae-a77bc5063503
# ╠═a9504370-9773-11f0-2b30-b5d08ae8f733
# ╠═a95043a2-9773-11f0-2185-3b48fc34a363
# ╟─a95043d4-9773-11f0-2b70-3157562eac2b
# ╠═a95046a4-9773-11f0-1018-0fefed30aefa
# ╠═a950494c-9773-11f0-0c31-4121b08a6307
# ╠═a9504c08-9773-11f0-16fb-7f24591a7cd7
# ╠═662a3264-536d-4001-94be-6533c235dde0
# ╠═a9504ece-9773-11f0-095f-293d02c85b7c
# ╠═a9505192-9773-11f0-3f74-f1f69349557e
# ╠═a950543c-9773-11f0-3871-dfdb831d101b
# ╠═a9505478-9773-11f0-0241-671420b679c8
# ╠═a95054be-9773-11f0-0a1e-e94b285d51f6
# ╠═a950550e-9773-11f0-3ee1-3f355a7a63e3
# ╠═a9505540-9773-11f0-0fa0-09f98c7bdffc
# ╠═a950557c-9773-11f0-254c-55abc2f49428
# ╠═a95055ae-9773-11f0-043d-15e66f32784a
# ╠═a95055f4-9773-11f0-2051-ed5cbb17cd58
# ╠═3bcc1f05-eb24-4cad-8d41-55d08d0038de
# ╠═a9505626-9773-11f0-3a85-2f4ef025bbfb
# ╠═a9505662-9773-11f0-045e-3de748f29d5c
# ╠═a9505694-9773-11f0-1ba2-69efa4d12e68
# ╠═a95056da-9773-11f0-2098-af3b528a1535
# ╟─a950570c-9773-11f0-23b0-f5a328397f3f
# ╠═a950575c-9773-11f0-1ffd-0171fc8be532
# ╠═a95057a2-9773-11f0-3d2f-e9ba06c45451
# ╠═a95057e8-9773-11f0-0c67-116a6d9b6142
# ╠═a9505824-9773-11f0-2b42-b1d8386aa0a9
# ╠═a9505856-9773-11f0-0bbf-67056a82bc97
# ╠═a95058c4-9773-11f0-12eb-a9740f1716ea
# ╠═a9505900-9773-11f0-028e-65f34de357e7
# ╠═a9505932-9773-11f0-2ba7-37b335a1d4a2
# ╠═a9505bbc-9773-11f0-0a63-e9b55afb8d71
# ╠═a9505ec8-9773-11f0-1ac6-97047a2fb56d
# ╟─a9506166-9773-11f0-32dd-df7461826e15
# ╠═a950640e-9773-11f0-29e7-f3fff8c09e6c
# ╠═a95066c0-9773-11f0-30f5-b927b25710d5
# ╠═a9506986-9773-11f0-2cee-59dcf3ac9fba
# ╠═a95069cc-9773-11f0-2f88-af68a4958443
# ╠═a9506a08-9773-11f0-28da-4700cbb01464
# ╠═a9506a44-9773-11f0-2050-272f876a7fa2
# ╠═a9506a7e-9773-11f0-220f-912ac0e463b1
# ╠═a9506ab0-9773-11f0-1c59-81f4ae3a85d1
# ╠═a9506aee-9773-11f0-0039-f5a63067f405
# ╠═a9506b34-9773-11f0-1e68-5d240efc214b
# ╠═a9506b8e-9773-11f0-0fb8-51a6e806b698
# ╟─a9506bde-9773-11f0-02d3-21d30930f60c
# ╠═a9506c18-9773-11f0-0e54-431798773d2a
# ╠═e6b1f3bb-671b-4d3b-a9a0-eab2f62aac51
# ╠═a9506c56-9773-11f0-1c6c-2bfa4605f3c9
# ╠═a9506c8a-9773-11f0-0a32-57cbf7b4b38b
# ╠═a9506cd8-9773-11f0-237a-37b7bcfcc1bd
# ╠═a9506d5a-9773-11f0-1fef-375b4ba650ad
# ╠═a9506d8c-9773-11f0-3e6c-b37f56ee6391
# ╠═a9506dbe-9773-11f0-2b96-f92eb60ed404
# ╠═a9506e2c-9773-11f0-2d1e-c393fd8b0bd0
# ╠═a9506e56-9773-11f0-056e-8fb014d2ce1b
# ╠═a95070d4-9773-11f0-3a44-53e15619932e
# ╠═48f2cd73-4e3d-4664-9aef-6d3adf86b6f6
# ╠═e94f7342-f6c8-496d-9a2b-7ca038d59ccd
# ╠═4749377b-daf2-464d-adf6-86ef1c0d1a48
# ╠═34479ce9-c7d3-4f86-ae27-c53dc40f4e26
# ╟─a9507360-9773-11f0-1efb-abe8edbdb4e9
# ╠═a9507900-9773-11f0-3607-53cc14351f48
# ╠═a9507bd0-9773-11f0-0613-e720372a6a11
# ╠═a9507f2a-9773-11f0-0b6f-f7b153f14811
# ╠═a9508182-9773-11f0-02e3-63373863f26e
# ╠═ecc2eae4-c56f-4ac1-b6f5-5cef680e63d4
# ╠═2224ebd3-cadd-4f1a-987a-9a03e76483ee
# ╠═a9508222-9773-11f0-27dc-1fb53022a1d5
# ╠═a95082c2-9773-11f0-29f4-15c9d9711373
# ╠═a950831c-9773-11f0-1e18-4b5c1b666461
# ╠═a950834e-9773-11f0-0738-292ea6617918
# ╠═a9508380-9773-11f0-2061-97f6be7926d5
# ╠═a95083b2-9773-11f0-0f88-9fe002f360db
# ╠═a95083dc-9773-11f0-06ed-153f74cb2f3c
# ╠═a9508416-9773-11f0-00c1-4b96177c44d7
# ╠═a9508440-9773-11f0-0468-456ab3321954
# ╠═a95084ac-9773-11f0-1638-4561db13624c
# ╠═a95084de-9773-11f0-17ca-ebefd6103bb8
# ╠═a950851a-9773-11f0-1437-697984f663df
# ╠═a950854c-9773-11f0-364d-95517de5c4ec
# ╟─a950859c-9773-11f0-22b4-b716257facb1
# ╠═a95085e2-9773-11f0-3b48-bd54b886bb7b
# ╠═a9508664-9773-11f0-0d7f-33e43397186f
# ╟─a9508696-9773-11f0-36a0-b744c388b21f
# ╠═a95086d2-9773-11f0-045a-df6220764b85
# ╠═a950872c-9773-11f0-0bbd-e13ebd0d5a86
# ╠═a95089d4-9773-11f0-1918-871f58139161
# ╠═a9508c36-9773-11f0-2cf3-931774b45b28
# ╠═a9508eca-9773-11f0-0622-537f01b955b2
# ╠═a95091d6-9773-11f0-3fca-3fe549980f60
# ╠═a95094ea-9773-11f0-1b8a-a1258ed85b74
# ╠═a9509802-9773-11f0-25ac-b93ddf8a399f
# ╠═a950983e-9773-11f0-1d74-cfbfaa288302
# ╠═a950987a-9773-11f0-29b8-c53303609f8c
# ╠═a95098b6-9773-11f0-1d1c-fde1591eb09d
# ╠═a95098de-9773-11f0-27af-0363afd8dd27
# ╠═decfafe4-3526-438a-994a-2f59470622d1
# ╠═60b194b5-43c8-4db8-94c1-46f8cc6ac86d
# ╠═a9509974-9773-11f0-16d7-057e1415c17e
# ╠═a95099b0-9773-11f0-17d9-7f1bd6657f5a
# ╠═a95099e2-9773-11f0-031f-85acb445143d
# ╠═a9509a1e-9773-11f0-24a9-0b5cd3034f27
# ╠═a9509a82-9773-11f0-13f1-2725db277304
# ╠═a9509ab4-9773-11f0-0e4d-95d8808c524e
# ╠═a9509ae6-9773-11f0-2080-d954c041242f
# ╠═a9509b36-9773-11f0-396d-9da3f01171a4
# ╠═a9509b72-9773-11f0-01c2-8f04fb479a7b
# ╠═a9509b9a-9773-11f0-2c9b-fffec5097f25
# ╟─a9509be0-9773-11f0-1ab8-d50e8abd48b6
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
