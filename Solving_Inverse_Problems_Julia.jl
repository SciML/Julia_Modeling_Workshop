### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 885f5685-5239-42aa-942c-2b9bdf7df220
# For most simplicity, you can `using OrdinaryDiffEq`
# Or `using DifferentialEquations`
# This is done to cut the install size.
using OrdinaryDiffEqTsit5

# ╔═╡ f5c9ea83-6dba-4848-8340-1a8c772bd941
using Plots

# ╔═╡ c7193526-d32f-4f2a-95aa-09e98b88b30b
using Optimization, OptimizationOptimJL

# ╔═╡ 36985121-9fe5-4961-9ec2-3f5ee57cfaa1
using Zygote

# ╔═╡ ba06d75c-9f29-48c0-9db3-b3822568bf39
using StableRNGs # Make the RNG consistent across Julia versions

# ╔═╡ 6f04db2c-3c38-4244-8d70-22a15a7dd72e
using LinearAlgebra # norm is defined in the LinearAlgebra Base library

# ╔═╡ 8a3108d7-0bec-40d1-b5a9-e4deec2f88c0
using SciMLSensitivity

# ╔═╡ f860ea33-eb7c-43f6-a1b5-859b0b33d367
using OptimizationBBO

# ╔═╡ 5550a2d3-bfc7-433e-a3d8-4c4791ecd7fc
using NonlinearSolveFirstOrder

# ╔═╡ 6402bee8-18dd-4c1b-8616-d57030310503
using Turing, MCMCChains

# ╔═╡ b122c127-53aa-45d5-997c-4fa2cb84d2ff
using StatsPlots

# ╔═╡ 45dc0462-d63c-11ef-0f62-e5d029618cc1
md"""
# Solving Inverse Problems and Parameter Estimation with Julia's SciML

In these notes we will showcase how to solve inverse problems in Julia. We will use the forward modeling tools like DifferentialEquations.jl for solving nonlinear systems, and then show how to incorporate automatic differentiation, Optimization.jl and SciMLSensitivity.jl for optimizing complicated cost functions.
"""

# ╔═╡ 51c67a4d-f49a-4bb5-a1c2-361ea0e8f20e
md"""
### Setting Up Our Forward Model

Similar to the previous notes on forward modeling, we will use DiffernetialEquations.jl in order to define and solve the differential equation. For simplicity we will simply put the code here for the Lotka-Volterra model, see the other notebook for details on building models and using the ODE solver.

The Lotka-Volterra system is defined as:

```math
\begin{align}
\frac{dx}{dt} &= \alpha x - \beta x y \\
\frac{dy}{dt} &= -\gamma y + \delta x y
\end{align}
```

with initial conditions ``x(0) = 1`` and ``y(0) = 1``, with parameters ``\alpha = 1``, ``\beta = 1.5``, ``\gamma = 3``, and ``\delta = 1``. We define the ODE system by first defining a function `f` that computes the derivative:
"""

# ╔═╡ 86b53570-0f03-4bb8-8c78-3a45c38ad789
function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α * x - β * x * y
    du[2] = dy = -δ * y + γ * x * y
end

# ╔═╡ 34efa390-0d6d-48dd-b08b-0c1bc391ffc2
md"""
We now define an ODEProblem according to the documentation. In Julia you can get the documentation for how to do things by using the help command `?`:
"""

# ╔═╡ 760b16ea-410f-4f46-ae44-f859de9b75e3
lotka_u0 = [1.0,1.0]

# ╔═╡ 27059e1a-5f0c-40ac-85ea-ff6ad8d7b68c
lotka_p = [1.0, 1.5, 3.0, 1.0]

# ╔═╡ 9ff83b76-86dd-4520-ac20-df1eb30cc1dd
lotka_tspan = (0.0, 10.0)

# ╔═╡ 1245a924-10c8-4170-b3a9-ce025c54319d
lotkaprob = ODEProblem(lotka_volterra!, lotka_u0, lotka_tspan,  lotka_p)

# ╔═╡ d2ac7dbe-a47a-4b50-a796-a04462f410cc
md"""
Now let's test out our Lotka-Volterra system and see if it gives the classic periodic pattern. We will use the Plots.jl system to show the results:
"""

# ╔═╡ 9ad843f4-0f54-4a4f-a83d-b29bedad37c3
lotka_sol = solve(lotkaprob, Tsit5())

# ╔═╡ 1e27f7dd-9d41-4fe9-b823-0013ef35927d
plot(lotka_sol)

# ╔═╡ 835c2922-37de-4e1b-b9ba-05a729eca801
plot(lotka_sol, idxs = (1,2))

# ╔═╡ 2ab07370-1cf4-41db-b197-295d4cfd6a68
md"""
We can see that we get a periodic system from both the timeseries solution and the phase plot. 

There is much more that can be said about solving differential equations, but let's leave it here for now and move onto the next part!
"""

# ╔═╡ c725adf8-4bfa-4046-929f-65dfef8c2309
md"""

### Solving Optimization Problems with Optimization.jl

Before we move onto the inverse problems, we will explain one more forward model, and that's via optimization. Optimization models are models of the form:

```math
u^\ast = \min_u f(u)
```

i.e. find the ``u`` vector that minimizes some function. To demonstrate this, we will use Optimization.jl to find the solution of the function:

```math
f(u,p) = (p_1 - u_1)^2 + p_2 (u_2 - u_1^2)^2
```

where `p = [1.0, 100.0]`. This is done via the following definition
"""

# ╔═╡ 2c247d32-e215-48ac-95df-4c562333309f
simpleoptf(u, p) = (p[1] - u[1])^2 + p[2] * (u[2] - u[1]^2)^2

# ╔═╡ 0a83e463-7180-4f7e-8f93-37165d56aed8
simpleopt_p = [1.0, 100.0]

# ╔═╡ bbe5e86a-0e72-4ce2-aa95-1b80a070b3ac
md"""
For most optimizers we will then need to give them an initial condition. In this case we will just guess the solution [0.0,0.0]:
"""

# ╔═╡ ee4e03a6-1187-4de1-816c-9299debe54ed
simpleopt_u0 = zeros(2)

# ╔═╡ 2ff9c802-5562-4669-8045-d3bf40e42cd2
md"""
Now to perform the optimization with the method we wish to use we will need to endow the function with the ability to calculate derivatives, in particular many optimizers will want gradients. We can do this by telling it to use the Zygote automatic differentiation library for reverse-mode AD. We will go over what automatic differentiation is in future lectures, but for now understand this as a description of how the gradient function should be generated.
"""

# ╔═╡ 9d09b14b-af52-48e2-9c6c-446c398ed422
simpleoptfun = OptimizationFunction(simpleoptf, AutoZygote())

# ╔═╡ eaec431c-e7f3-49a7-a2a2-cb18d464348c
md"""
Now like before we build an OptimizationProblem and choose to solve this with some optimization method:
"""

# ╔═╡ 5e533d0a-5aaa-4e42-a420-fb3b3ccb5060
simpleoptprob = OptimizationProblem(simpleoptfun, simpleopt_u0, simpleopt_p)

# ╔═╡ 8b0256ff-c49c-468c-b140-1416e3e311ab
simpleopt_sol = solve(simpleoptprob, OptimizationOptimJL.LBFGS())

# ╔═╡ 34aa161d-4421-4344-b40f-e25a151a98e2
simpleopt_sol.u

# ╔═╡ f8a4e5ed-684e-4242-adce-9114576cbc68
simpleopt_sol.objective

# ╔═╡ 92c24e7c-deee-4b56-a64d-00748d7fe69a
md"""
We can see that it basically found a solution `[1.0,1.0]` to 13 digits of accuracy.
"""

# ╔═╡ d942a2d0-dc61-4b74-8702-789c54044300
md"""
## Solving Inverse Problems with Optimization

Now that we understand how to write models that can be solved, we will now showcase how to solve a few inverse problems. The most basic way to solve inverse problems is via optimization. In the optimization route, what you do is define a cost function ``C(p)`` which is a function of the parameters ``p`` which you wish to optimize. By definition, this cost functions is defined such that the inverse problem solution, i.e. the optimal parameters for the model ``p^\ast``, are given by:

```math
p^\ast = \min_p C(p)
```

or in otherwords, the parameters `p` which minimize the cost function. This is best explained through examples, so let's show a few simple examples.

### Inverse Problems on Differential Equations: Example via Parameter Estimation to Data

For our first inverse problem, we will solve a classic parameter estimation problem. A parameter estimation problem is generally a problem where you are given data which you want your model to fit, and the goal is to find the parameters of the model which case the model's prediction to best fit the data.

To demonstrate the parameter estimation problem, we will use a "test back" example. A test back example is an example of generating data via a model, so that way we know the true solution, and then see if we can recover the true parameters. To do this, let's create a dataset of the solution at the current values:
"""

# ╔═╡ 134afb44-b726-45d3-a2a0-002a7c433cfb
lotka_data_sol = solve(lotkaprob, Tsit5(), abstol=1e-8, reltol=1e-8, saveat = 0.5)

# ╔═╡ 8bbafc21-3f6b-4bce-9b41-56ba36b6a2b4
md"""
Notice that one major difference from the original solving process is that in this case we made sure to cause the solution to `saveat` every 0.5 points: when solving inverse problems it's paramount that we will have a clear definition on what data we are fitting and at what time points this data is to be found.

Now let's take the solution and turn it into a data matrix:
"""

# ╔═╡ 13a104ee-1af5-47f2-acf1-db5cd32ea37e
lotka_data = Array(lotka_data_sol)

# ╔═╡ 80f79d79-0432-46f9-9734-c598f061e87c
md"""
Let's plot these as scatter points over the original solution:
"""

# ╔═╡ 730d88c9-682b-4377-bae6-3b9e51f4c2d1
plot1 = plot(lotka_sol)

# ╔═╡ b9a8e9e3-f17e-425b-86c8-6e836932735c
scatter!(plot1, lotka_data_sol.t, lotka_data')

# ╔═╡ dbfe82fd-6b79-434e-a0c5-591a9f3990c6
md"""
To make this more realistic, let's add some noise to the data
"""

# ╔═╡ 1f2212b0-d6b3-42e1-8864-8588aacab5c4
rng = StableRNG(100)

# ╔═╡ c07e6522-1c78-491f-bffa-70bf0c11dea5
lotka_data_noisy = abs.(Array(lotka_data_sol) + 0.35randn(rng, size(lotka_data)))

# ╔═╡ a5a3d044-eddc-4c79-bf0f-2d6753a16e07
begin
	plot2 = plot(lotka_sol)
	scatter!(plot2, lotka_data_sol.t, lotka_data_noisy')
end

# ╔═╡ de5354a6-b8cd-4871-b9b3-2bd3be6bee17
md"""
Now to write our loss function, we will need to write a routine that changes our parameters and then re-solves the model with new parameters. We can do this using the `remake` method provided in SciML. This allows you to change a `prob` by only overriding what is changed. Our `remake` and `solve` function then looks like:
"""

# ╔═╡ a6668c04-6b54-45df-951e-4dff659ae152
function lotka_prediction(p; saveat = 0.5)
	_prob = remake(lotkaprob, p = p)
	sol = solve(_prob, Tsit5(); saveat, abstol=1e-8, reltol=1e-8)
end

# ╔═╡ 9723e36d-7ec4-4c0d-bab3-13ee372b7515
plot(lotka_prediction([1.0,2.0,3.0,4.0]))

# ╔═╡ 50f8441b-6a5b-4b3b-8846-29894388bf6f
md"""
A small note: the reason why the lines are so jagged is because we're only saving the solution now at 0.5. When used without `saveat`, the ODE solver returns a continuous solution that satisfies the tolerance and includes an interpolation that gives accurate predictions between solver stepping points. When we change to saveat, it only saves the desired points at high accuracy, but for plotting it still contains a linear interpolation by default. To give a better view of what's going on continuously, we can pass an empty `saveat = ()` for better plots.
"""

# ╔═╡ 10338166-3f7a-4479-9089-1437b8a82a6e
begin
	plot3 = plot(lotka_prediction([1.0,2.0,3.0,4.0], saveat = ()))
	scatter!(plot3, lotka_data_sol.t, lotka_data_noisy')
end

# ╔═╡ d8fd5015-6cdc-4f4f-80eb-b16e21f0e908
md"""
We can see that using the dummy parameters `[1.0,2.0,3.0,4.0]` that we are nowhere close to the data points (i.e. the data generated from the model with `[1.0,1.5,3.0,1.5]`). Our goal is to automatically start from these bad parameters and let a mathematical routine find the correct parameters. We can do this by defining a loss function. What we want is that the line goes through each data point. Thus we want `sol(t) - data(t) = 0`. What we can do is define our loss function to be the sum of the normed differences between every output and every data point:
"""

# ╔═╡ 3d580d10-9fe1-48ff-b563-d7e267205e8c
function lotka_loss(p; noise = true)
	if noise
		norm(Array(lotka_prediction(p)) - lotka_data_noisy)
	else
		norm(Array(lotka_prediction(p)) - lotka_data)
	end
end

# ╔═╡ 981ba39c-6328-4577-bc9a-5574dcc634d9
md"""
This gives us a scalar measurement for "how bad our parameters are". As a check, we should have a minimum of near zero when we have the correct parameters:
"""

# ╔═╡ 5765470f-44a3-4911-9aec-d029ee7f87b3
lotka_realp_loss = lotka_loss([1.0,1.5,3.0,1.0])

# ╔═╡ f71e8884-5829-4346-8bde-43ea1947d7f5
md"""
As a sanity check, let's check the loss function when there is no noise in the data:
"""

# ╔═╡ cb5be726-5287-4519-a199-14a27246726e
norm(Array(lotka_prediction([1.0,1.5,3.0,1.0])) - lotka_data)

# ╔═╡ f38d3cec-65ea-4999-867a-984b23b9fe21
md"""
Thus the reason that the loss value with the real parameters that generated the data is $lotka_realp_loss instead of 0.0 is because of the noise in the data.
"""

# ╔═╡ bed65c9b-cad3-48c2-aac5-5dc201a15558
md"""
Now in order to solve this optimization, i.e. find the `p` that case `lotka_loss` to be minimized, we will define an OptimizationProblem like before. But now in this case we will stack our solvers: i.e. our optimization will be defined where the function we are optimizing is a function that is solving our differential equation. This is done simply by:
"""

# ╔═╡ dba58f1a-6340-4242-8b1a-482dd50e8d11
paramest_fun = OptimizationFunction((p,θ)->lotka_loss(p), AutoZygote())

# ╔═╡ 08389217-2791-469e-ae48-e7efdc0de493
paramest_prob = OptimizationProblem(paramest_fun, [1.0,2.0,3.0,4.0], lb = [0.1, 0.1, 0.1, 0.1], ub = [4.0, 4.0, 4.0, 4.0])

# ╔═╡ 49175b07-6c2e-4a13-86f7-8c5e5047e553
md"""
Notice that we had to pull in the SciMLSensitivity.jl library. If you forgot it, then you would've gotten the error message:

```
Compatibility with reverse-mode automatic differentiation requires SciMLSensitivity.jl.

Please install SciMLSensitivity.jl and do `using SciMLSensitivity`/`import SciMLSensitivity`

for this functionality. For more details, see https://sensitivity.sciml.ai/dev/.
```

This is because this library is required for defining the local sensitivity analysis (i.e. rules for calculating gradients of) the SciML solvers, such as the ODE solver. This will be the topic of the future lectures, but for now just add this and automatic differentiation of the solvers will magically work.
"""

# ╔═╡ 7ad1e21f-2c72-415c-9d9c-c0ee35ab44e6
md"""
Notice here that the state ``u`` of our optimization problem is the ``p`` parameters of our ODE. The parameters of our optimization problem, ``\theta``, would be hyperparmaeters of the optimization. In this case we don't have any so we simply ignore them. But this shows how you could define inverse problems over inverse problems, not a topic we will address in this workshop.

But okay, now like before we use a nonlinear optimizer to solve this problem:
"""

# ╔═╡ e6413dda-3ebb-49c1-9dfb-a340321e86c6
paramestopt_sol = solve(paramest_prob, OptimizationOptimJL.LBFGS())

# ╔═╡ a6c0695e-6c29-4992-b7dd-dcab7aeea3f1
md"""
Note: If you didn't set the tolerance to be lower in the ODE solver, this solve might've been unstable. This is because the accuracy of the solve effects the accuracy of the gradient and thus the stability of the optimization process. This is a mathematical detail that will be discussed in a future lecture of the workshop.

But okay, we got a solution! Let's look at it:
"""

# ╔═╡ a5e5b6bf-1fe9-4462-a5e3-c30bf656a6cd
paramestopt_sol.u

# ╔═╡ f9749dde-697a-40aa-a0e7-47edc8fc778f
paramestopt_sol.objective

# ╔═╡ 08617e66-a04f-464a-a1b1-31ad7d44f491
md"""
Oh wow that's not that great!
"""

# ╔═╡ 68fc5732-e059-44a3-a8d5-3428cb6c36d1
md"""
#### Understanding the Difficulties of Optimization

Let's investigate this a bit. How does the fit look?
"""

# ╔═╡ 099f04f0-8f67-425c-a395-0d2dd20925b1
begin
	plot4 = plot(lotka_prediction(paramestopt_sol.u, saveat = ()))
	scatter!(plot4, lotka_data_sol.t, lotka_data_noisy')
end

# ╔═╡ 598a4615-f77a-4167-9bf1-62aa25218b17
md"""
What happened? Why did it stop there? Is the optimizer bad? To investigate, let's try a different optimizer:
"""

# ╔═╡ 0d3e1592-5535-4c62-a1a2-82fb11109ce4
bboparamest_sol = solve(paramest_prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters = 10000)

# ╔═╡ b30eb1b1-b355-4f3e-980a-dd98b2d9d1d4
bboparamest_sol.u

# ╔═╡ 764b48c6-8427-4e1d-8ec7-59126a020e71
begin
	plot5 = plot(lotka_prediction(bboparamest_sol.u, saveat = ()))
	scatter!(plot5, lotka_data_sol.t, lotka_data_noisy')
end

# ╔═╡ d1897917-14ae-418a-a1b5-e7d47e58a8dd
md"""
This optimizer got much closer! In fact, if we didn't have noise in the data it recovers the exact parameters:
"""

# ╔═╡ 3231de71-277a-4a13-b3ae-220650089f8b
noiseless_paramest_fun = OptimizationFunction((p,θ)->lotka_loss(p; noise=false), AutoZygote())

# ╔═╡ c82ea0b1-f5dd-45be-ad89-4746d31109a8
noiseless_paramest_prob = OptimizationProblem(noiseless_paramest_fun, [1.0,2.0,3.0,4.0], lb = [0.1, 0.1, 0.1, 0.1], ub = [4.0, 4.0, 4.0, 4.0])

# ╔═╡ 6572ab63-4428-4d4b-8ba2-37820d1403cc
bbo_noiselessparamest_sol = solve(noiseless_paramest_prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters = 10000)

# ╔═╡ 14cf3698-fb1b-44bc-89b1-98d8fdf29723
bbo_noiselessparamest_sol.u

# ╔═╡ aa7d1c43-0d78-462b-93d0-090d2bbdff31
md"""
That's almost exactly the right solution! Meanwhile our other method:
"""

# ╔═╡ 241cb0fa-9e3e-4291-a3b8-e0d1246c9d2f
local_paramestopt_sol = solve(noiseless_paramest_prob, OptimizationOptimJL.LBFGS())

# ╔═╡ e2ee3937-32f2-4caa-9509-a0418fb36135
local_paramestopt_sol.u

# ╔═╡ a3497c89-f8c8-4468-bfb3-430e3fe07071
md"""
Even without noise this method did not do well. So does that mean `BBO_adaptive_de_rand_1_bin_radiuslimited` is just better? Not necessarily. Let's look at the stats of the solution process:
"""

# ╔═╡ 0fa7450e-4c48-40f5-8785-cd0ab7a8c4f7
bbo_noiselessparamest_sol.stats

# ╔═╡ ff48c782-955d-4e7d-be0f-0f7db2c011ae
local_paramestopt_sol.stats

# ╔═╡ 19f1b4f6-e42c-45c3-8e4a-2f6c623fa65e
md"""
From this we can see that the global optimizer required solving the ODE with $(bbo_noiselessparamest_sol.stats.fevals) evaluations of the cost function. That's solving >10,000 ODEs! The other method is meanwhile using $(local_paramestopt_sol.stats.fevals) evaluations. The reason is because this BBO method is a blackbox global optimization method while `LBFGS` is a local gradient-based optimization method. Local optimization methods are much faster, orders of magnitude faster, but can get stuck in local minima, while Global optimization methods require hundreds of thousands (and more, millions) of iterations in order to converge, but are generally more robust at avoiding local minima. It's a trade-off, but the global optimization methods scale poorly to more expensive problems and in particular high-dimensional parameter spaces. Because our model is sufficiently small this solved perfectly fine, but it's not something that can always be used!

To demonstrate that this is an issue with local minima, we can use the local minimizer with a different loss value. To make this more clear we will use the noiseless data.
"""

# ╔═╡ 66e393f7-4120-46e1-925c-11367c3b6d4c
noiseless_paramest_prob_newu0 = OptimizationProblem(noiseless_paramest_fun, [1.0,2.0,3.0,3.0], lb = [0.1, 0.1, 0.1, 0.1], ub = [4.0, 4.0, 4.0, 4.0])

# ╔═╡ a6b84344-a03d-4575-bb56-9ba2ba37a3f7
local_paramestopt_newu0_sol = solve(noiseless_paramest_prob_newu0, OptimizationOptimJL.LBFGS())

# ╔═╡ 18e5b654-10e9-4df9-95b5-62624d694b9f
local_paramestopt_newu0_sol.u

# ╔═╡ 48f71f49-44c3-4424-824e-63a0f975bd38
local_paramestopt_newu0_sol.objective

# ╔═╡ 3b48ff74-df66-4a6d-aca0-fcf2b855cb23
md"""
Notice I made a small tweak to the initial condition of the parameter estimation: `[1.0,2.0,3.0,3.0]` instead of `[1.0,2.0,3.0,4.0]`, and suddenly we got the right answer! And we get it really fast too:
"""

# ╔═╡ 1c913f13-6a33-4161-a9e4-7e439f1ebf70
local_paramestopt_newu0_sol.stats

# ╔═╡ 757546ff-9c2a-442c-9b71-b4b30c48cdfa
md"""
We can see that by only changing the last initial condition we can "fix" the local optimization. Just $(local_paramestopt_newu0_sol.stats.fevals) solves of our ODE gives us the solution now. This would be the best solution if our initial guess was good enough!

In order to see this local optima phonomena, we can slice the cost function and perturb just one of the parameters, and plot the cost function as that parameter changes:
"""

# ╔═╡ ecaaea3c-f399-42d7-a56a-033a0f6b4a8b
function loss_changing_1st(p1)
	lotka_loss([p1,1.5,3.0,1.0]; noise=false)
end

# ╔═╡ f87f8a17-a706-497e-94d4-e6e6f7601f37
p1s = 0.01:0.01:5.0

# ╔═╡ 79d4d09a-d1a2-45a9-bb02-ae05a24b4f1d
plot(p1s, loss_changing_1st.(p1s), yaxis="Loss Value", xaxis="p1", title="Loss given p1, where all other parameters are exact")

# ╔═╡ 790110fe-ad24-4a71-a2c7-8446e0441985
md"""
This of course raises a bunch of interesting questions:

1. Can we do local methods with many starting points? Yes, these are multistart methods.
2. Can we make global optimization more efficient? Yes there are all sorts of algorithms here.
3. Can we mix global optimization and gradient-based methods? This is one branch of research.
4. Can we make difficult models solve quicker and then just use the global optimizers? This is the space of surrogates and reduced order methods which other lecturers will cover.
5. Can we define loss functions which have less local minima? This will be one of the later lectures.
6. How does the automatic differentiation choice effect scaling, accuracy, and stability? That will be the next lecture.

And so on and so forth. So this is just the most basic of solutions to the parameter estimation problem, but shows all of the steps in full.
"""

# ╔═╡ b79609cd-bb10-49ad-91eb-5de46325d0e0
md"""
### Generalizing to Other Inverse Problems: Inverse Problems on Nonlinear Systems

Now that we have seen inverse problems on differential equations, it should be readily apparent how to generalize inverse problems to all other kinds of mathematical equations. Basically, you take the method for solving that type of equation and put it inside of an optimization loop. As we noted above, there are many mathematical/computational questions to be asked about whether this process will converge to the right solution, whether it will be computationally efficient, etc., but procedurally as code it's "easy" to generalize this to any method.

To show this, let's now do the same for an inverse problem on the Brusselator PDE.

#### Description of the Steady State Brusselator PDE

**If you are not familiar with PDEs, feel free to skip this section. It's simply defining a nonlinear system.**

The Brusselator PDE is defined on a unit square periodic domain as follows:

```math
\begin{align}
\frac{\partial U}{\partial t} &= 1 + U^2V - 4.4U + \alpha \nabla^2 U + f(x, y, t),\\
\frac{\partial V}{\partial t} &= 3.4U - U^2V + \alpha \nabla^2 V,
\end{align}
```

and thus its steady state is:

```math
\begin{align}
0 &= 1 + U^2V - 4.4U + \alpha \nabla^2 U + f(x, y, t),\\
0 &= 3.4U - U^2V + \alpha \nabla^2 V,
\end{align}
```

where

```math
f(x, y, t) = \begin{cases}
5 & \quad \text{if } (x-0.3)^2+(y-0.6)^2 ≤ 0.1^2 \text{ and } t ≥ 1.1\\
0 & \quad \text{else}
\end{cases}, \mathrm{and}
```

$\nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}$ is the two dimensional Laplacian operator. The above equations are to be solved for a time interval $t \in [0, 11.5]$ subject to the initial conditions

```math
\begin{align}
U(x, y, 0) &= 22\cdot (y(1-y))^{3/2} \\
V(x, y, 0) &= 27\cdot (x(1-x))^{3/2}
\end{align},
```

and the periodic boundary conditions

```math
\begin{align}
U(x+1,y,t) &= U(x,y,t) \\
V(x,y+1,t) &= V(x,y,t).
\end{align}
```

To solve this PDE, we will discretize it into a system of ODEs with the finite
difference method. We discretize the unit square domain with `N` grid points in each direction.
`U[i,j]` and `V[i,j]` then represent the value of the discretized field at a given point in time, i.e.

```
U[i,j] = U(i*dx,j*dy)
V[i,j] = V(i*dx,j*dy)
```

where `dx = dy = 1/N`. To implement our ODE system, we collect both `U` and `V` in a single array `u` of size `(N,N,2)` with `u[i,j,1] = U[i,j]` and `u[i,j,2] = V[i,j]`. This approach can be easily generalized to PDEs with larger number of field variables.

Using a three-point stencil, the Laplacian operator discretizes into a tridiagonal matrix with elements `[1 -2 1]` and a `1` in the top, bottom, left, and right corners coming from the periodic boundary conditions. The nonlinear terms are implemented pointwise in a straightforward manner.

tl;dr, the set of nonlinear equations is:
"""

# ╔═╡ 421dbe2a-a689-4390-9d0b-b4528d11f4af
N = 8

# ╔═╡ 8186418e-d332-41da-b6ae-d90964c28c08
xyd_brusselator = range(0, stop = 1, length = N)

# ╔═╡ f3685420-ec96-4cdc-aa9a-f3f0cf948b1e
brusselator_f(x, y) = (((x - 0.3)^2 + (y - 0.6)^2) <= 0.1^2) * 5.0

# ╔═╡ 95bebbd8-7e5d-4e7a-a2a8-68b546614c0e
limit(a, N) = a == N + 1 ? 1 : a == 0 ? N : a

# ╔═╡ 169aa5f4-a85d-4e53-8442-f47757456726
function brusselator_2d_loop(_du, _u, p)
    A, B, alpha = p
	dx = step(xyd_brusselator)
    alpha = alpha / dx^2
	u = reshape(_u, N, N, 2)
	du = reshape(_du, N, N, 2)
	
    @inbounds for I in CartesianIndices((N, N))
        i, j = Tuple(I)
        x, y = xyd_brusselator[I[1]], xyd_brusselator[I[2]]
        ip1, im1, jp1, jm1 = limit(i + 1, N), limit(i - 1, N), limit(j + 1, N),
        limit(j - 1, N)
        du[i, j, 1] = alpha * (u[im1, j, 1] + u[ip1, j, 1] + u[i, jp1, 1] + u[i, jm1, 1] -
                       4u[i, j, 1]) +
                      B +
                      u[i, j, 1]^2 * u[i, j, 2] - (A + 1) * u[i, j, 1] + brusselator_f(x, y)
        du[i, j, 2] = alpha * (u[im1, j, 2] + u[ip1, j, 2] + u[i, jp1, 2] + u[i, jm1, 2] -
                       4u[i, j, 2]) + A * u[i, j, 1] - u[i, j, 1]^2 * u[i, j, 2]
    end
end

# ╔═╡ 490d7195-a878-4634-8f2b-a51463976d44
bruss_p = [3.4, 1.0, 10.0]

# ╔═╡ 4d3dd038-5cdf-44ad-9fcc-22353e55961f
function init_brusselator_2d(xyd)
    N = length(xyd)
    u = zeros(N, N, 2)
    for I in CartesianIndices((N, N))
        x = xyd[I[1]]
        y = xyd[I[2]]
        u[I, 1] = 22 * (y * (1 - y))^(3 / 2)
        u[I, 2] = 27 * (x * (1 - x))^(3 / 2)
    end
    u
end

# ╔═╡ f66493ab-dddb-46fa-9305-80f7625e8f88
bruss_u0 = vec(init_brusselator_2d(xyd_brusselator))

# ╔═╡ 664a2a56-58d6-42bf-89c3-15307834d633
prob_brusselator_2d = NonlinearProblem(
    brusselator_2d_loop, bruss_u0, bruss_p; abstol = 1e-10, reltol = 1e-10
)

# ╔═╡ 8d46da5d-e59d-47e4-af1a-b8b33a8b57a7
md"""
We can solve this system with the chosen parameters:
"""

# ╔═╡ 3bc2d2d7-863c-4f59-9918-4eb3944ac0ec
bruss_sol = solve(prob_brusselator_2d, NewtonRaphson())

# ╔═╡ 14ee0daf-95f5-4e54-8ccc-ea1df5a52157
md"""
But now we can ask a question. That question doesn't have to be "fit the data", it can be anything about the solution. The solution here is the steady state values for `(u,v)`. Let's ask the question: for what values of the parameters is the norm of `u` equal to the norm of `v`? We can see that in our first solution, the `u` is much smaller than the `v`, so for what parameters is that not true?

For this PDE, the steady state that's found is an unstable steady state with a periodic orbit around it, and thus this would be asking when the periodic orbit is relatively circular, but simplified a bit.

But okay, whether or not we justify this question, how would we code up a method to solve it? We can do that as follows:
"""

# ╔═╡ 95cd9037-5985-4c45-bbe8-e20a5e244b5f
function bruss_optcost(p, θ)
	_prob = remake(prob_brusselator_2d, p=p)
	new_bruss_sol = solve(_prob, NewtonRaphson(), verbose = false)
	u = reshape(new_bruss_sol.u, N, N, 2)
	abs2(norm(u[:,:,1]) - norm(u[:,:,2]))
end

# ╔═╡ 8ffc8473-1c74-4ba2-bd6c-5f8aa1223f10
brussoptf = OptimizationFunction(bruss_optcost, AutoZygote())

# ╔═╡ 943a4e6f-1cbb-4800-aade-6bd398d966a3
md"""
Note that we are taking the `abs2` here (squaring the result) in order to ensure that the result is always positive, and thus the minimum is where the norms are equal. Now we setup an optimization to find the parameters for which this works:
"""

# ╔═╡ 22bdc1e6-ce8d-41bd-a257-d015202f468d
bruss_opt = OptimizationProblem(brussoptf, [3.4, 1.0, 10.0], lb = [0.1, 0.1, 0.1], ub = [20.0, 20.0, 20.0])

# ╔═╡ 22f0edf0-d358-4d40-ad1a-78e251cdc809
bruss_opt_sol = solve(bruss_opt, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters=300)

# ╔═╡ 7a49a79c-95b5-4dbc-b979-4c753a75e86e
bruss_opt_sol.objective

# ╔═╡ b0b10ca7-0dca-4359-9fe5-f60b375f3413
bruss_opt_sol.u

# ╔═╡ 522a37fb-a2f7-4758-8e32-02c8058c8df3
md"""
This solution looks really good!
"""

# ╔═╡ db0b2561-7a02-427f-9a9c-a3943947da4c
begin 
	_prob = remake(prob_brusselator_2d, p=bruss_opt_sol.u)
	new_bruss_sol = solve(_prob, NewtonRaphson(), verbose = false)
	u = reshape(new_bruss_sol.u, N, N, 2)
end

# ╔═╡ 0351b05b-c63d-4a7a-a20b-e6e72b835bec
md"""

### Solving Inverse Problems via Bayesian Methods

All of the above shows how to use optimization methods for solving the inverse problem. The other major branch of methods are the Bayesian methods. Bayesian methods or other probabilistic methods give an estimate of uncertainty around the inverse problem solution. This is good because, as seen with the parameter estimation example, it can be tricky to know if your solution is any good. In particular, you don't expect to have exactly the right parameters if you have noise, but how much off are they likely to be? Bayesian probabilistic estimates help give you this information by, instead of giving you an exact value for the parameters, it instead gives you a probability distribution for the best parameters.

Mathematically, the difference between these two approaches can be tremendous. But computationally with a probabilsitic programming framework. For this we will using Turing.jl
"""

# ╔═╡ 0bfd1396-c569-4ac9-b92a-f535e533ba50
md"""
Now, instead of defining an objective function, we define a likelihood function. A likelihood function measures how likely parameters are: the closer to the data the simulation is, the more likely the parameters should be. We can do this using the Turing model syntax as follows:
"""

# ╔═╡ c479f931-f403-4c93-bc63-14a01f19718b
@model function fitlv(data)
    # Prior distributions.
    σ ~ InverseGamma(2, 3)
    α ~ truncated(Normal(1.0, 0.5); lower=0.5, upper=2.5)
    β ~ truncated(Normal(2.0, 0.5); lower=0, upper=2)
    γ ~ truncated(Normal(3.0, 0.5); lower=1, upper=4)
    δ ~ truncated(Normal(3.0, 0.5); lower=0, upper=2)

    # Simulate Lotka-Volterra model. 
    p = [α, β, γ, δ]
	predicted = lotka_prediction(p)

    # Observations.
    for i in 1:size(data, 2)
        data[:, i] ~ MvNormal(predicted[:,i], σ^2 * I)
    end

    return nothing
end

# ╔═╡ 1a5be3f3-9d6c-45f0-963a-8e19e65c0f3a
model = fitlv(lotka_data_noisy)

# ╔═╡ e479984b-579c-4cdd-9f69-6d7b0db8202e
chain = sample(model, NUTS(), MCMCSerial(), 1000, 3; progress=false, chain_type=MCMCChains.Chains)

# ╔═╡ 4db2c565-7c96-41c9-8480-e07d107f12d3
plot(chain)

# ╔═╡ 9677c213-4af4-48bc-9c49-b7907f478f8d
md"""
For more information on understanding the Bayesian results, it is highly recommended you check out the Bayesian estimation of differential equations tutorial at [https://turinglang.org/docs/tutorials/bayesian-differential-equations/](https://turinglang.org/docs/tutorials/bayesian-differential-equations/).

From the plots above we can see how chains have converged, and thus we are getting resonable probability distributions for the parameters. The chains are thus a sample-based estimate of the probability distribution: the probability of being in an area matches the probability of seeing a given parameter. Thus the areas where a given value is in the chain more often has a higher probability. The parameter distributions on the right side above are slides of the marginal density estimates from this information, generated by a kernel density estimate.

We can get summary statistics about the chain as follows:
"""

# ╔═╡ 243b7f0b-2f0c-4373-ae98-ce234562f87a
summarize(chain)

# ╔═╡ 60118730-1525-43fd-b1ff-b675b6a20e42
md"""
What's σ? For the likelihood function we chose to use a normally distributed likelihood, where the variance is σ. Thus σ is a hyperparameter of our cost function for the variance at each data point, or an estimate for the intrinsic noise in the data. We can see it's a fairly good estimate for the size of the noise that we had added!
"""

# ╔═╡ 33adf452-e528-4cbc-88a7-d81bac52e124
md"""
If time allows, we can go over the details about how the HMC/NUTS algorithm works, but in a sense it's the probabilistic generalization of gradient-based optimization. And we can see it gives similar estimates to the parameters on the noisy data as before. To double check how it did without noise we can see:
"""

# ╔═╡ d6001516-314b-4f9c-9929-61cd2e2e4e98
model_nonoise = fitlv(lotka_data)

# ╔═╡ 3c20fb0d-ffba-4df6-bc21-175823894867
chain_nonoise = sample(model_nonoise, NUTS(), MCMCSerial(), 1000, 3; progress=false, chain_type=MCMCChains.Chains)

# ╔═╡ d2868575-d779-436e-97a1-c524b775899d
plot(chain_nonoise)

# ╔═╡ f448e44b-66ee-42db-99a1-1a9ad1b92d23
summarize(chain_nonoise)

# ╔═╡ 583d4aa9-0193-49b1-8350-7b2715041f39
md"""
We can see it's right on the dot!
"""

# ╔═╡ 38de9bfc-ea00-43c2-af5b-08fb031e1861
md"""

## Investigating Identifiability in Inverse Problems

Now that you have the basic techniques for solving inverse problems, you might want to ask the question, how do I know I got "the right answer"? To motivate this question better, let's go back to our parameter estimation problem. In that case, we assumed that we had data for the time series of `[x,y]`. However, in most scientific context that is unreasonable: it's highly unlikely that you have data for every state in your system. Instead you likely only have partial data. If you only observed part of the system, will you still recover the right parameters? 

Let's see this in action with our Lotka-Volterra system. Let's assume that we only observe the ``x`` values. Thus we could do this by simply changing our cost function:
"""

# ╔═╡ bf8cdb2b-e77c-4a9c-97c0-ab57e8fc6e6b
function lotka_loss_partially_observed(p)
	norm(Array(lotka_prediction(p))[1,:] - lotka_data[1,:])
end

# ╔═╡ cd427bf5-afbd-4e92-9def-c5f5e7bea416
paramest_partial_fun = OptimizationFunction((p,θ)->lotka_loss_partially_observed(p), AutoZygote())

# ╔═╡ 5755fb3c-24d5-45b2-a10c-a1010c27df4f
paramest_partial_prob = OptimizationProblem(paramest_partial_fun, [1.0,2.0,3.0,3.0], lb = [0.1, 0.1, 0.1, 0.1], ub = [4.0, 4.0, 4.0, 4.0])

# ╔═╡ 70fbd8fd-fda4-426d-8a6f-d1f8fa303200
md"""
Notice we are using the exact same setup as the local optimization that previously converged, i.e. the "fixed" or better initial conditions. Let's see how this does:
"""

# ╔═╡ bd129656-3273-4725-afe5-f9ff04afda23
local_paramestopt_partial_sol = solve(paramest_partial_prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters = 10000)

# ╔═╡ 97118579-0ae9-4078-b6f4-4ad4a6a54901
local_paramestopt_partial_sol.objective

# ╔═╡ fdd91eab-f5af-43f5-9cf2-3311e7eea56c
md"""
We see that even the global optimization does not recover the correct parameters anymore, but it does give a solution that does exactly what we asked for:
"""

# ╔═╡ e662f173-84ad-438e-afe8-a1e1e82d38c6
begin
	plot7 = plot(lotka_prediction(local_paramestopt_partial_sol.u, saveat = ()))
	scatter!(plot7, lotka_data_sol.t, lotka_data')
end

# ╔═╡ f9ac099c-8fc8-45b4-b0b6-6e47985a5880
md"""
We see that we got the right parameters! That's basically changing things about `y` also changes things about `x` in a way that is uniquely defined, and thus data about the timeseries of `x` is sufficient to understand the full system.

When should we expect this to be true? When should we not expect this to be true? This is the question of identifiability and will be discussed in future lectures if time allows.
"""

# ╔═╡ 3bf58446-ddff-4dd8-a3f6-1ac9822a930f
md"""
## Conclusion

In this lecture we showed how to solving forward and inverse problems using Julia. We see that with Julia's SciML infrastructure we can use methods that are compatible with automatic differentiation in order to generate the gradients automatically in a way that is compatible with both the optimization and the Bayesian inference libraries. We showed how to use both optimization and Bayesian inference to solve inverse problems on ODEs, nonlinear systems, and one can guess how this would generalize to all sorts of codes. Thus it's safe to say that if you understood this lecture, then you understand how to solve inverse problems! Case closed, workshop done.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
NonlinearSolveFirstOrder = "5959db7a-ea39-4486-b5fe-2dd0bf03d60d"
Optimization = "7f7a1694-90dd-40f0-9382-eb1efda571ba"
OptimizationBBO = "3e6eede4-6085-4f62-9a71-46d9bc1eb92b"
OptimizationOptimJL = "36348300-93cb-4f02-beb5-3c3902f8871e"
OrdinaryDiffEqTsit5 = "b1df2697-797e-41e3-8120-5422d3b24e4a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
SciMLSensitivity = "1ed8b502-d754-442c-8d5d-10ac956f44a1"
StableRNGs = "860ef19b-820b-49d6-a774-d7a799459cd3"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
MCMCChains = "~7.7.0"
NonlinearSolveFirstOrder = "~2.8.0"
Optimization = "~5.9.1"
OptimizationBBO = "~0.4.12"
OptimizationOptimJL = "~0.4.21"
OrdinaryDiffEqTsit5 = "~2.1.4"
Plots = "~1.41.7"
SciMLSensitivity = "~7.119.8"
StableRNGs = "~1.0.4"
StatsPlots = "~0.15.8"
Turing = "~0.49.0"
Zygote = "~0.7.13"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.0"
manifest_format = "2.1"
project_hash = "8cabbaaa87fd3df9fe8573edfd261399c47764fa"

[[deps.ADTypes]]
deps = ["PrecompileTools"]
git-tree-sha1 = "629de23e1c16911b439dabd2303c08af9575b226"
registries = "General"
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
registries = "General"
uuid = "14f7f29c-3bd6-536c-9a0b-7339e30b5a3e"
version = "0.5.4"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
registries = "General"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Dates", "Distributed", "LogDensityProblems", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "UUIDs"]
git-tree-sha1 = "328c7d50f307c66308a915abb20d9889e5aab48b"
registries = "General"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "5.16.0"

    [deps.AbstractMCMC.extensions]
    AbstractMCMCOnlineStatsExt = "OnlineStats"
    AbstractMCMCTensorBoardLoggerExt = "TensorBoardLogger"

    [deps.AbstractMCMC.weakdeps]
    OnlineStats = "a15396b6-48d5-5d58-9928-6d29437db91e"
    TensorBoardLogger = "899adc3e-224a-11e9-021f-63837185c80f"

[[deps.AbstractPPL]]
deps = ["ADTypes", "AbstractMCMC", "Accessors", "BangBang", "DensityInterface", "JSON", "LinearAlgebra", "MacroTools", "OrderedCollections", "Random", "StatsBase"]
git-tree-sha1 = "31fb177c9952267ac2dc54a44868cb768390e6ef"
registries = "General"
uuid = "7a57a42e-76ec-4ea3-a279-07e840d6d9cf"
version = "0.15.5"

    [deps.AbstractPPL.extensions]
    AbstractPPLDifferentiationInterfaceExt = ["DifferentiationInterface"]
    AbstractPPLDistributionsExt = ["Distributions", "LinearAlgebra"]
    AbstractPPLForwardDiffExt = ["ForwardDiff", "DiffResults"]
    AbstractPPLMooncakeExt = ["Mooncake"]
    AbstractPPLTestExt = ["Test"]

    [deps.AbstractPPL.weakdeps]
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    DifferentiationInterface = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
registries = "General"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "7063ad1083578215c7c4bf410368150abe8d5524"
registries = "General"
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
registries = "General"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.7.1"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdvancedHMC]]
deps = ["AbstractMCMC", "ArgCheck", "DocStringExtensions", "IrrationalConstants", "LinearAlgebra", "LogDensityProblems", "LogDensityProblemsAD", "LogExpFunctions", "ProgressMeter", "Random", "Setfield", "Statistics", "StatsBase"]
git-tree-sha1 = "230bc5a3a9cd4e2ddb9c7cb1e00f330c88f04e03"
registries = "General"
uuid = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d"
version = "0.8.7"

    [deps.AdvancedHMC.extensions]
    AdvancedHMCADTypesExt = "ADTypes"
    AdvancedHMCCUDAExt = "CUDA"
    AdvancedHMCComponentArraysExt = "ComponentArrays"
    AdvancedHMCMCMCChainsExt = "MCMCChains"
    AdvancedHMCOrdinaryDiffEqSymplecticRKExt = "OrdinaryDiffEqSymplecticRK"

    [deps.AdvancedHMC.weakdeps]
    ADTypes = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
    MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
    OrdinaryDiffEqSymplecticRK = "fa646aed-7ef9-47eb-84c4-9443fc8cbfa8"

[[deps.AdvancedMH]]
deps = ["AbstractMCMC", "Distributions", "DocStringExtensions", "FillArrays", "LinearAlgebra", "LogDensityProblems", "Random", "Requires"]
git-tree-sha1 = "62ddbccf0ce5c26f8ef3cebe4bedef6b1599d616"
registries = "General"
uuid = "5b7e9947-ddc0-4b3f-9b55-0d8042f74170"
version = "0.8.10"
weakdeps = ["DiffResults", "ForwardDiff", "MCMCChains", "StructArrays"]

    [deps.AdvancedMH.extensions]
    AdvancedMHForwardDiffExt = ["DiffResults", "ForwardDiff"]
    AdvancedMHMCMCChainsExt = "MCMCChains"
    AdvancedMHStructArraysExt = "StructArrays"

[[deps.AdvancedVI]]
deps = ["ADTypes", "AbstractPPL", "Accessors", "ChainRulesCore", "DiffResults", "Distributions", "DocStringExtensions", "FillArrays", "Functors", "LinearAlgebra", "LogDensityProblems", "Optimisers", "ProgressMeter", "Random", "StatsBase"]
git-tree-sha1 = "fc42c29943eae508fd28ec620d1faa4d2c1ce044"
registries = "General"
uuid = "b5ca4192-6429-45e5-a2d9-87aec30a685c"
version = "0.7.0"

    [deps.AdvancedVI.extensions]
    AdvancedVIDynamicPPLExt = ["DynamicPPL", "Accessors", "Distributions", "LogDensityProblems"]
    AdvancedVIEnzymeExt = ["Enzyme", "ChainRulesCore"]
    AdvancedVIMooncakeExt = ["Mooncake", "ChainRulesCore"]
    AdvancedVIReverseDiffExt = ["ReverseDiff", "ChainRulesCore"]

    [deps.AdvancedVI.weakdeps]
    DynamicPPL = "366bfd00-2699-11ea-058f-f148b4cae6d8"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
registries = "General"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgCheck]]
git-tree-sha1 = "f9e9a66c9b7be1ad7372bbd9b062d9230c30c5ce"
registries = "General"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
registries = "General"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "libblastrampoline_jll"]
git-tree-sha1 = "7f54761502ff149a9d492e4acefe9805898e29b3"
registries = "General"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.2+0"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "1aec1ff0dcaa83a484e7f72097562fa3102c6722"
registries = "General"
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

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "1bbcf97714051f302b3beafab65bcdeb6044c4df"
registries = "General"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.2.1"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
registries = "General"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "4126b08903b777c88edf1754288144a0492c05ad"
registries = "General"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.8"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "e386db8b4753b42caac75ac81d0a4fe161a68a97"
registries = "General"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.6.1"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "cceb62468025be98d42a5dc581b163c20896b040"
registries = "General"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.9"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Bijectors]]
deps = ["ADTypes", "AbstractPPL", "ArgCheck", "ChangesOfVariables", "Distributions", "DocStringExtensions", "FillArrays", "Functors", "InverseFunctions", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "MappedArrays", "Random", "Reexport", "Roots", "SparseArrays", "Statistics", "Test"]
git-tree-sha1 = "a58fb73182f07a5a561bfadbe40efb9c6c892853"
registries = "General"
uuid = "76274a88-744f-5084-9051-94815aaf08c4"
version = "0.16.3"

    [deps.Bijectors.extensions]
    BijectorsChainRulesCoreExt = "ChainRulesCore"
    BijectorsEnzymeCoreExt = "EnzymeCore"
    BijectorsForwardDiffExt = "ForwardDiff"
    BijectorsLazyArraysExt = "LazyArrays"
    BijectorsMooncakeExt = "Mooncake"
    BijectorsReverseDiffChainRulesExt = ["ChainRules", "ReverseDiff"]
    BijectorsReverseDiffExt = "ReverseDiff"

    [deps.Bijectors.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

[[deps.BinaryHeaps]]
deps = ["PrecompileTools"]
git-tree-sha1 = "0adb6d8d0a4bae93a07d79c8d59eaf617f4c59da"
registries = "General"
uuid = "b2a6c25c-c996-4615-94ab-6e519ffb8690"
version = "1.1.0"

[[deps.BlackBoxOptim]]
deps = ["Compat", "Distributed", "Distributions", "LinearAlgebra", "PrecompileTools", "Printf", "Random", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "67dbdbe9b34f7c74c42bb67ea6e75d8c1d8f48c0"
registries = "General"
uuid = "a134a8b2-14d6-55f6-9291-3336d3ab0209"
version = "0.6.12"

    [deps.BlackBoxOptim.extensions]
    BlackBoxOptimRealtimePlotServerExt = ["HTTP", "JSON", "Sockets"]

    [deps.BlackBoxOptim.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
    JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
    Sockets = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.BracketingNonlinearSolve]]
deps = ["CommonSolve", "ConcreteStructs", "NonlinearSolveBase", "PrecompileTools", "Reexport", "SciMLBase", "SciMLLogging"]
git-tree-sha1 = "027dcb51dcf326023e015902b3d8e88ca59efb8a"
registries = "General"
uuid = "70df07ce-3d50-431d-a3e7-ca6ddb60ac1e"
version = "1.12.7"
weakdeps = ["ChainRulesCore", "ForwardDiff"]

    [deps.BracketingNonlinearSolve.extensions]
    BracketingNonlinearSolveChainRulesCoreExt = ["ChainRulesCore", "ForwardDiff"]
    BracketingNonlinearSolveForwardDiffExt = "ForwardDiff"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
registries = "General"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
registries = "General"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1fa950ebc3e37eccd51c6a8fe1f92f7d86263522"
registries = "General"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.7+0"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "3c190c570fb3108c09f838607386d10c71701789"
registries = "General"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.73.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "12177ad6b3cad7fd50c8b3825ce24a99ad61c18f"
registries = "General"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Chairmarks]]
deps = ["Printf", "Random"]
git-tree-sha1 = "9a49491e67e7a4d6f885c43d00bb101e6e5a434b"
registries = "General"
uuid = "0ca39b1e-fe0b-4e98-acfc-b1656634c4de"
version = "1.3.1"
weakdeps = ["Statistics"]

    [deps.Chairmarks.extensions]
    StatisticsChairmarksExt = ["Statistics"]

[[deps.ChangesOfVariables]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "83ee8183bd8c4a390ae178385e6c7b3aa4e468b2"
registries = "General"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.11"
weakdeps = ["InverseFunctions", "Test"]

    [deps.ChangesOfVariables.extensions]
    ChangesOfVariablesInverseFunctionsExt = "InverseFunctions"
    ChangesOfVariablesTestExt = "Test"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "3e22db924e2945282e70c33b75d4dde8bfa44c94"
registries = "General"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
registries = "General"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
registries = "General"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
registries = "General"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
registries = "General"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.CommonSolve]]
deps = ["PrecompileTools"]
git-tree-sha1 = "6c389fa857f6ca5a95474b52a52023fd77f24cb7"
registries = "General"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.14"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
registries = "General"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
registries = "General"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.5.5+2"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
registries = "General"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcreteStructs]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a72c3b5ce6d2a477f55b5c9b8756e91e695c67f6"
registries = "General"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.8"

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
registries = "General"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
registries = "General"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
registries = "General"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "54b76cbb40d9a0f5368c880725b2f141da77c94f"
registries = "General"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.2.0"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
registries = "General"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "b0bc6d2cad1fed8b7fd59a1551a991cb3d2809e6"
registries = "General"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.6"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
registries = "General"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
registries = "General"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
registries = "General"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
registries = "General"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffEqBase]]
deps = ["ArrayInterface", "BracketingNonlinearSolve", "ConcreteStructs", "DocStringExtensions", "FastBroadcast", "FastClosures", "FastPower", "FunctionWrappers", "FunctionWrappersWrappers", "LinearAlgebra", "Logging", "Markdown", "MuladdMacro", "PrecompileTools", "Printf", "RecursiveArrayTools", "Reexport", "RespecializeParams", "SciMLBase", "SciMLLogging", "SciMLOperators", "SciMLStructures", "Setfield", "StaticArraysCore", "SymbolicIndexingInterface", "TruncatedStacktraces"]
git-tree-sha1 = "e2a0c69c358ed6a8a6ac5ae8f1c7e36150a4dbd9"
registries = "General"
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
registries = "General"
uuid = "459566f4-90b8-5000-8ac3-15dfb0a30def"
version = "4.19.4"
weakdeps = ["Functors"]

    [deps.DiffEqCallbacks.extensions]
    DiffEqCallbacksFunctorsExt = "Functors"

[[deps.DiffEqNoiseProcess]]
deps = ["CommonSolve", "DiffEqBase", "Distributions", "GPUArraysCore", "LinearAlgebra", "Markdown", "PoissonRandom", "PrecompileTools", "QuadGK", "Random", "RecipesBase", "RecursiveArrayTools", "ResettableStacks", "SciMLBase", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "f5901b2fdc18fac5775b832ffa1d5d3f67600368"
registries = "General"
uuid = "77a26b50-5914-5dd7-bc55-306e6241c503"
version = "5.36.3"
weakdeps = ["Optim", "ReverseDiff"]

    [deps.DiffEqNoiseProcess.extensions]
    DiffEqNoiseProcessOptimExt = "Optim"
    DiffEqNoiseProcessReverseDiffExt = "ReverseDiff"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
registries = "General"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "79a2aca180a85c690c58a020d47b426954b590f8"
registries = "General"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.16.0"

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "LinearAlgebra"]
git-tree-sha1 = "0693d8b0a4608ff289d228ab4c598df5894845cd"
registries = "General"
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

[[deps.DimensionalData]]
deps = ["ConstructionBase", "DataAPI", "Dates", "Extents", "Interfaces", "IntervalSets", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "PrecompileTools", "Random", "Statistics", "TableTraits", "Tables"]
git-tree-sha1 = "3e5b58039d35b98f425446c099577d67633a7237"
registries = "General"
uuid = "0703355e-b756-11e9-17c0-8b28908087d0"
version = "0.30.2"

    [deps.DimensionalData.extensions]
    DimensionalDataAbstractFFTsExt = "AbstractFFTs"
    DimensionalDataAdaptExt = "Adapt"
    DimensionalDataAlgebraOfGraphicsExt = "AlgebraOfGraphics"
    DimensionalDataArrayInterfaceExt = "ArrayInterface"
    DimensionalDataCategoricalArraysExt = "CategoricalArrays"
    DimensionalDataChainRulesCoreExt = "ChainRulesCore"
    DimensionalDataDiskArraysExt = "DiskArrays"
    DimensionalDataMakieExt = "Makie"
    DimensionalDataNearestNeighborsExt = "NearestNeighbors"
    DimensionalDataPythonCallExt = "PythonCall"
    DimensionalDataRecipesBaseExt = "RecipesBase"
    DimensionalDataSparseArraysExt = "SparseArrays"
    DimensionalDataStatsBaseExt = "StatsBase"

    [deps.DimensionalData.weakdeps]
    AbstractFFTs = "621f4979-c628-5d54-868e-fcf4e3e8185c"
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    AlgebraOfGraphics = "cbdf2221-f076-402e-a563-3d30da359d67"
    ArrayInterface = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
    CategoricalArrays = "324d7699-5711-5eae-9e2f-1d82baa6b597"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiskArrays = "3c3547ce-8d99-4f5e-a174-61eb10b00ae3"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    NearestNeighbors = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
registries = "General"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "Roots", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "a958ab3a40c755563f5e1405c0846cb0446bf19d"
registries = "General"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.131"
weakdeps = ["ChainRulesCore", "DensityInterface", "SparseConnectivityTracer", "Test"]

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DistributionsTestExt = "Test"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
registries = "General"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.DynamicPPL]]
deps = ["ADTypes", "AbstractMCMC", "AbstractPPL", "Accessors", "BangBang", "Bijectors", "Chairmarks", "Compat", "ConstructionBase", "Distributions", "DocStringExtensions", "FillArrays", "InteractiveUtils", "LinearAlgebra", "LogDensityProblems", "LogExpFunctions", "MacroTools", "OrderedCollections", "PartitionedDistributions", "PrecompileTools", "Preferences", "Printf", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "Test"]
git-tree-sha1 = "8f9dc691ddef45d534e9ca1ddea33101a02283c7"
registries = "General"
uuid = "366bfd00-2699-11ea-058f-f148b4cae6d8"
version = "0.42.13"

    [deps.DynamicPPL.extensions]
    DynamicPPLBridgeStanExt = ["BridgeStan"]
    DynamicPPLComponentArraysExt = ["ComponentArrays"]
    DynamicPPLEnzymeCoreExt = ["EnzymeCore"]
    DynamicPPLForwardDiffExt = ["ForwardDiff"]
    DynamicPPLInputProvenanceExt = ["ForwardDiff"]
    DynamicPPLMCMCChainsExt = ["MCMCChains"]
    DynamicPPLMarginalLogDensitiesExt = ["MarginalLogDensities"]
    DynamicPPLMooncakeExt = ["Mooncake"]
    DynamicPPLReverseDiffExt = ["ReverseDiff"]

    [deps.DynamicPPL.weakdeps]
    BridgeStan = "c88b6f0a-829e-4b0b-94b7-f06ab5908f5a"
    ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
    MarginalLogDensities = "f0c3360a-fb8d-11e9-1194-5521fd7ee392"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

[[deps.EllipticalSliceSampling]]
deps = ["AbstractMCMC", "ArrayInterface", "Distributions", "Random", "Statistics"]
git-tree-sha1 = "e611b7fdfbfb5b18d5e98776c30daede41b44542"
registries = "General"
uuid = "cad2338a-1db2-11e9-3401-43bc07c9ede2"
version = "2.0.0"

[[deps.EnumX]]
git-tree-sha1 = "c49898e8438c828577f04b92fc9368c388ac783c"
registries = "General"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.7"

[[deps.Enzyme]]
deps = ["CEnum", "EnzymeCore", "Enzyme_jll", "GPUCompiler", "InteractiveUtils", "LLVM", "Libdl", "LinearAlgebra", "ObjectFile", "PrecompileTools", "Preferences", "Printf", "Random", "ScopedValues", "SparseArrays"]
git-tree-sha1 = "d242d84323f15c1b4788e9eaaca3f8d5ca08dca3"
registries = "General"
uuid = "7da242da-08ed-463a-9acd-ee780be4f1d9"
version = "0.13.204"

    [deps.Enzyme.extensions]
    EnzymeBFloat16sExt = "BFloat16s"
    EnzymeCUDAExt = "CUDA"
    EnzymeChainRulesCoreExt = "ChainRulesCore"
    EnzymeGPUArraysCoreExt = "GPUArraysCore"
    EnzymeLogExpFunctionsExt = "LogExpFunctions"
    EnzymeOrderedCollectionsExt = "OrderedCollections"
    EnzymeSpecialFunctionsExt = "SpecialFunctions"
    EnzymeStaticArraysExt = "StaticArrays"

    [deps.Enzyme.weakdeps]
    ADTypes = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
    OrderedCollections = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.EnzymeCore]]
git-tree-sha1 = "971d7831cc85f43bc9f51d615a3f7f21270c2f1d"
registries = "General"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.21"
weakdeps = ["Adapt", "ChainRulesCore"]

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"
    EnzymeCoreChainRulesCoreExt = "ChainRulesCore"

[[deps.Enzyme_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "d405b496023e00071fcf874377e7d3e455badd66"
registries = "General"
uuid = "7cc45869-7501-5eee-bdea-0790c847d4ef"
version = "0.0.293+0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
registries = "General"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2bfb1e047e2ad0a5ca94365340bde8005d637568"
registries = "General"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.4+0"

[[deps.ExprTools]]
git-tree-sha1 = "d2e49e7efd29719d6f28b891b0e0e159daa9d2b4"
registries = "General"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.11"

[[deps.Extents]]
git-tree-sha1 = "b309b36a9e02fe7be71270dd8c0fd873625332b4"
registries = "General"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.6"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
registries = "General"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "7a58e45171b63ed4782f2d36fdee8713a469e6e0"
registries = "General"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.1.2+0"

[[deps.FFTA]]
deps = ["AbstractFFTs", "DocStringExtensions", "LinearAlgebra", "MuladdMacro", "Primes", "Random", "Reexport"]
git-tree-sha1 = "65e55303b72f4a567a51b174dd2c47496efeb95a"
registries = "General"
uuid = "b86e33f2-c0db-4aa1-a6e0-ab43e668529e"
version = "0.3.1"

[[deps.FastBroadcast]]
deps = ["ArrayInterface", "LinearAlgebra", "PrecompileTools"]
git-tree-sha1 = "c8f8eefadaa330d982bbf5e395c34e53a13ab668"
registries = "General"
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
registries = "General"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[deps.FastPower]]
deps = ["PrecompileTools"]
git-tree-sha1 = "3c7269c236978d434a16ebe99f9743b9d706270a"
registries = "General"
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
registries = "General"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.17.0"
weakdeps = ["PDMats", "SparseArrays", "StaticArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FindFirstFunctions]]
deps = ["PrecompileTools"]
git-tree-sha1 = "bba475ce782fc77777f539cf0b7c029207798db1"
registries = "General"
uuid = "64ca27bc-2ba2-4a57-88aa-44e436879224"
version = "3.2.1"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "5031f23e040bf17082e5b52422d77b5e844eefb1"
registries = "General"
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
registries = "General"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.FlexiChains]]
deps = ["AbstractMCMC", "AbstractPPL", "DelimitedFiles", "DimensionalData", "DocStringExtensions", "LinearAlgebra", "MCMCDiagnosticTools", "OrderedCollections", "PrecompileTools", "Printf", "Random", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "587c76bae96afb16314651cfd79aa73024c6b3df"
registries = "General"
uuid = "4a37a8b9-6e57-4b92-8664-298d46e639f7"
version = "0.6.39"

    [deps.FlexiChains.extensions]
    FlexiChainsAdvancedHMCExt = ["AdvancedHMC", "DimensionalData"]
    FlexiChainsComponentArraysExt = ["ComponentArrays"]
    FlexiChainsDynamicPPLExt = ["DynamicPPL", "Random", "DimensionalData", "OrderedCollections", "Distributions"]
    FlexiChainsInferenceObjectsExt = ["InferenceObjects", "DimensionalData", "OrderedCollections"]
    FlexiChainsMCMCChainsExt = ["MCMCChains", "OrderedCollections"]
    FlexiChainsMakieExt = ["Makie", "StatsBase", "KernelDensity"]
    FlexiChainsMonteCarloMeasurementsExt = ["MonteCarloMeasurements", "OrderedCollections"]
    FlexiChainsPairPlotsExt = ["PairPlots", "Makie"]
    FlexiChainsPigeonsDynamicPPLExt = ["Pigeons", "DynamicPPL", "AbstractMCMC"]
    FlexiChainsPigeonsExt = ["Pigeons"]
    FlexiChainsPosteriorDBExt = ["PosteriorDB", "OrderedCollections"]
    FlexiChainsPosteriorStatsDynamicPPLExt = ["PosteriorStats", "DynamicPPL"]
    FlexiChainsPosteriorStatsExt = ["PosteriorStats", "DimensionalData", "IntervalSets"]
    FlexiChainsRecipesBaseExt = ["RecipesBase", "StatsBase"]
    FlexiChainsStanSampleExt = ["StanSample"]
    FlexiChainsStatsPlotsExt = ["StatsPlots", "DimensionalData"]

    [deps.FlexiChains.weakdeps]
    AdvancedHMC = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d"
    ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    DynamicPPL = "366bfd00-2699-11ea-058f-f148b4cae6d8"
    InferenceObjects = "b5cf5a8d-e756-4ee3-b014-01d49d192c00"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    KernelDensity = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
    MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    PairPlots = "43a3c2be-4208-490b-832a-a21dcd55d7da"
    Pigeons = "0eb8d820-af6a-4919-95ae-11206f830c31"
    PosteriorDB = "1c4bc282-d2f5-44f9-b6d1-8c4424a23ad4"
    PosteriorStats = "7f36be82-ad55-44ba-a5c0-b8b5480d7aa5"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    StanSample = "c1514b29-d3a0-5178-b312-660c88baa699"
    StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
registries = "General"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
registries = "General"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "3b0f72e2ffef1a139ac450725c9ecd01c0a3c050"
registries = "General"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.4.6"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
registries = "General"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
registries = "General"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.FunctionProperties]]
deps = ["PrecompileTools"]
git-tree-sha1 = "eb42e8cd7bf092337ca385be61b4ca35e8854ea2"
registries = "General"
uuid = "f62d2435-5019-4c03-9749-2d4c77af0cbc"
version = "1.2.0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
registries = "General"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "2bcce3ad6f6977d617928d7707fdc86ac83cce03"
registries = "General"
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
registries = "General"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.5.3"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "64bbbb7d1499297751b536dd39c58b20750ab1db"
registries = "General"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.5.1+0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
registries = "General"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "PrecompileTools", "Preferences", "REPL", "Scratch", "Serialization", "TOML", "Tracy", "UUIDs"]
git-tree-sha1 = "5e54ec63c34bcc878558b173c411b8efe6b08344"
registries = "General"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "1.23.0"

    [deps.GPUCompiler.weakdeps]
    AMDGPU_LLVM_Backend_jll = "cc5c0156-bd05-5a77-8a68-bb0aafb29019"
    LLVMDowngrader_jll = "f52de702-fb25-5922-94ba-81dd59b07444"
    NVPTX_LLVM_Backend_jll = "ef6e0fe3-e6ef-59c0-bde6-4989574699e0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "4d777f73c46b46b8b5276206059cf8a195499314"
registries = "General"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.27"

    [deps.GR.extensions]
    IJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f8eb8f7ba13ea75083531647fc8faeda8d541f07"
registries = "General"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.27+0"

[[deps.Gamma]]
git-tree-sha1 = "86f86b6168a016ed88e4ae4e64577b98c3b59e8e"
registries = "General"
uuid = "a0844989-3bd2-4988-8bea-c9407ab0941b"
version = "1.1.0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
registries = "General"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
registries = "General"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "090526e65de8f69648ac156daae153de8b56df62"
registries = "General"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.88.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
registries = "General"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "9d9531a9cb63a9edc33836414e82a07e81710de2"
registries = "General"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "100.14004.0+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
registries = "General"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HypergeometricFunctions]]
deps = ["Gamma", "LinearAlgebra"]
git-tree-sha1 = "31bb6c92405c084617facc1d7ed9eb6c402d061e"
registries = "General"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.30"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "88d07a6b68b8fffb13cacd49e23ea73c571859b2"
registries = "General"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.20"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
registries = "General"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "c72458f1962faeb003bf23cbdb75164fe6280906"
registries = "General"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.4"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
registries = "General"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interfaces]]
git-tree-sha1 = "331ff37738aea1a3cf841ddf085442f31b84324f"
registries = "General"
uuid = "85a1e053-f937-4924-92a5-1367d23b7b87"
version = "0.3.2"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "48922d06068130f87e43edef52382e6a94305ae6"
registries = "General"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.16.3"

    [deps.Interpolations.extensions]
    InterpolationsForwardDiffExt = "ForwardDiff"
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.IntervalSets]]
git-tree-sha1 = "79d6bd28c8d9bccc2229784f1bd637689b256377"
registries = "General"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.14"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
registries = "General"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
registries = "General"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
registries = "General"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
registries = "General"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
registries = "General"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
registries = "General"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
registries = "General"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "88352712893ec50bee3680605891eaf0e9ed6368"
registries = "General"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.8.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "037babc10853eeb8e585418922246cb97b8e5b74"
registries = "General"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.2.0+1"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "a5b87110fa95d711355af44832497745aa93fb52"
registries = "General"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.42"
weakdeps = ["EnzymeCore", "LinearAlgebra", "SparseArrays"]

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTA", "Interpolations", "StatsBase"]
git-tree-sha1 = "9eda8292dd3268b3b7ec9df21bbfac24e177ec52"
registries = "General"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.12"

[[deps.Krylov]]
deps = ["LinearAlgebra", "Printf", "SparseArrays"]
git-tree-sha1 = "909890a0d1486c392128cf4bd6e038ee0c6a7af4"
registries = "General"
uuid = "ba0b0d4f-ebba-5204-a429-3ac8c609bfb7"
version = "0.10.10"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
registries = "General"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "39bca05343661c347aae0bca57a5994a0bf4f08d"
registries = "General"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.2.0+0"

[[deps.LHLFactorization]]
deps = ["LinearAlgebra", "PrecompileTools"]
git-tree-sha1 = "51d99d9892d0f95b4c9d4d7310785971f2d48298"
registries = "General"
uuid = "2faa5264-e118-4071-8864-e10559f68c7c"
version = "2.2.2"

    [deps.LHLFactorization.extensions]
    LHLFactorizationPolyesterExt = "Polyester"
    LHLFactorizationSparseExt = ["PureKLU", "SparseArrays"]

    [deps.LHLFactorization.weakdeps]
    Polyester = "f517fe37-dbe3-4b94-8317-1923a5111588"
    PureKLU = "0c0d3e7f-3a8b-4f7e-b6f1-9a4d2e7c1f01"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "PrecompileTools", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "d4bfee24427f4f441bd9212a107e375c39663aab"
registries = "General"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.13.1"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "d77aea19c9a71059a021acd99b0a4343e9661d94"
registries = "General"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.47+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e5b100780d4d30d63b4618d7930d48af409c1772"
registries = "General"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "23.1.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f88f3ccef05a6a72a0cf0ed417c8fd68530f4ab2"
registries = "General"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.1"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "df7566479bd64f20bd16b09960145e70160ffb3b"
registries = "General"
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
registries = "General"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.3.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "1.0.0"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "Zstd_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.18.0+1"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "PCRE2_jll", "Zlib_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.1+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "OpenSSL_jll", "Zlib_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.103+0"

[[deps.LibTracyClient_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d4e20500d210247322901841d4eafc7a0c52642d"
registries = "General"
uuid = "ad6e5548-8b26-5c9f-8ef3-ef0ad883f3a5"
version = "0.13.1+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
registries = "General"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
registries = "General"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
registries = "General"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
registries = "General"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Libtask]]
deps = ["MistyClosures", "Test"]
git-tree-sha1 = "d045ad5c965fc46c5d4c459d19a99271887de0e9"
registries = "General"
uuid = "6f1fad26-d15e-5dc8-ae53-837a1d7b8c9f"
version = "0.9.19"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "aebd334d06cee9f24cea70bd19a39749daf73881"
registries = "General"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.3+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d620582b1f0cbe2c72dd1d5bd195a9ce73370ab1"
registries = "General"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.0+0"

[[deps.LineSearch]]
deps = ["ADTypes", "CommonSolve", "ConcreteStructs", "FastClosures", "LinearAlgebra", "MaybeInplace", "PrecompileTools", "SciMLBase", "SciMLJacobianOperators", "StaticArraysCore"]
git-tree-sha1 = "e8fb8395e72f73cb8432572dbafcd941514fb94d"
registries = "General"
uuid = "87fe0de2-c867-4266-b59a-2f0a94fc965b"
version = "0.1.18"
weakdeps = ["LineSearches"]

    [deps.LineSearch.extensions]
    LineSearchLineSearchesExt = "LineSearches"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Printf"]
git-tree-sha1 = "a09a85e94e681def2446752cddaf9401a4d82b6d"
registries = "General"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.8.2"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.13.0"

[[deps.LinearSolve]]
deps = ["AMD", "ArrayInterface", "ConcreteStructs", "DocStringExtensions", "EnumX", "GPUArraysCore", "InteractiveUtils", "Krylov", "LHLFactorization", "Libdl", "LinearAlgebra", "MKL_jll", "Markdown", "OpenBLAS_jll", "PrecompileTools", "Preferences", "PureKLU", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLLogging", "SciMLOperators", "SciMLStructures", "Setfield", "SparseArrays", "SparseColumnPivotedQR", "StaticArraysCore"]
git-tree-sha1 = "b57acc4a96a542efd0a8c9c7262f9168b8822788"
registries = "General"
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

[[deps.LogDensityProblems]]
deps = ["ArgCheck", "DocStringExtensions", "Random"]
git-tree-sha1 = "d9625f27ded4ad726ceca7819394a4cc77ed25b3"
registries = "General"
uuid = "6fdf6af0-433a-55f7-b3ed-c6c6e0b8df7c"
version = "2.2.0"

[[deps.LogDensityProblemsAD]]
deps = ["DocStringExtensions", "LogDensityProblems"]
git-tree-sha1 = "7b83f3ad0a8105f79a067cafbfd124827bb398d0"
registries = "General"
uuid = "996a588d-648d-4e1f-a8f0-a84b347e47b1"
version = "1.13.1"

    [deps.LogDensityProblemsAD.extensions]
    LogDensityProblemsADADTypesExt = "ADTypes"
    LogDensityProblemsADDifferentiationInterfaceExt = ["ADTypes", "DifferentiationInterface"]
    LogDensityProblemsADEnzymeExt = "Enzyme"
    LogDensityProblemsADFiniteDifferencesExt = "FiniteDifferences"
    LogDensityProblemsADForwardDiffBenchmarkToolsExt = ["BenchmarkTools", "ForwardDiff"]
    LogDensityProblemsADForwardDiffExt = "ForwardDiff"
    LogDensityProblemsADReverseDiffExt = "ReverseDiff"
    LogDensityProblemsADTrackerExt = "Tracker"
    LogDensityProblemsADZygoteExt = "Zygote"

    [deps.LogDensityProblemsAD.weakdeps]
    ADTypes = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
    BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
    DifferentiationInterface = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
registries = "General"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"
weakdeps = ["ChainRulesCore", "ChangesOfVariables", "InverseFunctions"]

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
registries = "General"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.MCMCChains]]
deps = ["AbstractMCMC", "AxisArrays", "DataAPI", "Dates", "Distributions", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "MLJModelInterface", "NaturalSort", "OrderedCollections", "PrettyTables", "Random", "RecipesBase", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "060d6bc7cf60e621dfd056ed2c1a2db1e68db0fe"
registries = "General"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "7.7.0"

[[deps.MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "DataStructures", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "345bedeaf7d650f673fefa042dd8384c63de6c68"
registries = "General"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.3.19"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
registries = "General"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MLJModelInterface]]
deps = ["InteractiveUtils", "REPL", "Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "c275fae2e693206b4527dd9d2382aa15359ef3ed"
registries = "General"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.12.1"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
registries = "General"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.MappedArrays]]
git-tree-sha1 = "0ee4497a4e80dbd29c058fcee6493f5219556f40"
registries = "General"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.3"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MaybeInplace]]
deps = ["ArrayInterface", "LinearAlgebra", "MacroTools", "PrecompileTools"]
git-tree-sha1 = "f2cde0ae772162f20287b803e623966d0d94dea9"
registries = "General"
uuid = "bb5d69b7-63fc-4a16-80bd-7e42200c7bdb"
version = "0.1.8"
weakdeps = ["SparseArrays"]

    [deps.MaybeInplace.extensions]
    MaybeInplaceSparseArraysExt = "SparseArrays"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
registries = "General"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
registries = "General"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.MistyClosures]]
git-tree-sha1 = "d1a692e293c2a0dc8fda79c04cad60582f3d4de3"
registries = "General"
uuid = "dbe65cb8-6be2-42dd-bbc5-4196aaced4f4"
version = "2.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2026.8.13"

[[deps.MuladdMacro]]
deps = ["PrecompileTools"]
git-tree-sha1 = "283bf85d4a767481dd924dff0eee1735e95f449e"
registries = "General"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.7"

[[deps.MultivariateStats]]
deps = ["Arpack", "Distributions", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "8bcdc90d5c361a73635ad1c19d9c39a52bf3e0a3"
registries = "General"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.5"

[[deps.NLSolversBase]]
deps = ["ADTypes", "DifferentiationInterface", "FiniteDiff", "LinearAlgebra"]
git-tree-sha1 = "f96d38936d92d610318dec4d3b5ef37b0373c20f"
registries = "General"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "8.0.1"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "BFloat16s", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "ScopedValues", "Statistics"]
git-tree-sha1 = "d450844d195714a2d29b38c231193c0297cf90e8"
registries = "General"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.45"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreCUDNNExt = ["EnzymeCore", "CUDA", "cuDNN"]
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"
    NNlibMetalExt = "Metal"
    NNlibMooncakeCUDAExt = ["Mooncake", "CUDA"]
    NNlibSpecialFunctionsExt = "SpecialFunctions"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "dbd2e8cd2c1c27f0b584f6661b4309609c5a685e"
registries = "General"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.4"

[[deps.NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
registries = "General"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[deps.NearestNeighbors]]
deps = ["AbstractTrees", "Distances", "StaticArrays"]
git-tree-sha1 = "576eb4656529c12e77a46b17c23103dfba9fa570"
registries = "General"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.29"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.NonlinearSolveBase]]
deps = ["ADTypes", "Adapt", "ArrayInterface", "CommonSolve", "Compat", "ConcreteStructs", "DifferentiationInterface", "EnumX", "EnzymeCore", "FastClosures", "FunctionWrappers", "FunctionWrappersWrappers", "LinearAlgebra", "LogExpFunctions", "Markdown", "MaybeInplace", "PreallocationTools", "PrecompileTools", "Preferences", "Printf", "RecursiveArrayTools", "RespecializeParams", "SciMLBase", "SciMLJacobianOperators", "SciMLLogging", "SciMLOperators", "SciMLStructures", "Setfield", "StaticArraysCore", "SymbolicIndexingInterface", "TimerOutputs"]
git-tree-sha1 = "a166907a5f5d6722aa09f74172a51332e43267f0"
registries = "General"
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
registries = "General"
uuid = "5959db7a-ea39-4486-b5fe-2dd0bf03d60d"
version = "2.8.0"

[[deps.ObjectFile]]
deps = ["Reexport", "StructIO"]
git-tree-sha1 = "23c4d109603f7a7bfe888c1f63c5ab7be6183d0b"
registries = "General"
uuid = "d8793406-e978-5875-9003-1fc021f44a92"
version = "0.5.1"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
registries = "General"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
registries = "General"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
registries = "General"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.30+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.6+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
registries = "General"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optim]]
deps = ["ADTypes", "EnumX", "FillArrays", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "PositiveFactorizations", "Printf", "SparseArrays", "Statistics"]
git-tree-sha1 = "81f338ad984b25bc82471db4d235e7b8bf21b8d6"
registries = "General"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "2.3.2"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Compat", "ConstructionBase", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "b6a586b581eccc60a181145ffd4382099e32e2df"
registries = "General"
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
registries = "General"
uuid = "7f7a1694-90dd-40f0-9382-eb1efda571ba"
version = "5.9.1"

[[deps.OptimizationBBO]]
deps = ["BlackBoxOptim", "OptimizationBase", "Reexport", "SciMLBase", "SciMLLogging"]
git-tree-sha1 = "dc927d30eeac77988ed8121bfca02b8ebe424621"
registries = "General"
uuid = "3e6eede4-6085-4f62-9a71-46d9bc1eb92b"
version = "0.4.12"
weakdeps = ["LogExpFunctions"]

[[deps.OptimizationBase]]
deps = ["ADTypes", "ArrayInterface", "DifferentiationInterface", "DocStringExtensions", "FastClosures", "LinearAlgebra", "PrecompileTools", "SciMLBase", "SciMLLogging", "SparseArrays", "SparseConnectivityTracer", "SparseMatrixColorings", "SymbolicIndexingInterface"]
git-tree-sha1 = "c390a9e18ec082cee22136a42f50b208b8312d0b"
registries = "General"
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

[[deps.OptimizationOptimJL]]
deps = ["Optim", "OptimizationBase", "SciMLBase", "SparseArrays"]
git-tree-sha1 = "6bc626d531967889260563d7cd42294047fd91e8"
registries = "General"
uuid = "36348300-93cb-4f02-beb5-3c3902f8871e"
version = "0.4.21"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
registries = "General"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
registries = "General"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

[[deps.OrdinaryDiffEqCore]]
deps = ["ADTypes", "Accessors", "Adapt", "ArrayInterface", "BinaryHeaps", "CommonSolve", "ConstructionBase", "DiffEqBase", "DocStringExtensions", "EnumX", "EnzymeCore", "FastBroadcast", "FastClosures", "FastPower", "FindFirstFunctions", "FunctionWrappers", "FunctionWrappersWrappers", "InteractiveUtils", "LinearAlgebra", "Logging", "MacroTools", "MuladdMacro", "PrecompileTools", "Preferences", "Printf", "Random", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLLogging", "SciMLOperators", "SciMLStructures", "SymbolicIndexingInterface", "TruncatedStacktraces"]
git-tree-sha1 = "b4351cee919a8312d0fa6186b9e90d464d62a397"
registries = "General"
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

[[deps.OrdinaryDiffEqTsit5]]
deps = ["CommonSolve", "DiffEqBase", "FastBroadcast", "LinearAlgebra", "MuladdMacro", "OrdinaryDiffEqCore", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "SciMLBase", "TruncatedStacktraces"]
git-tree-sha1 = "2fdf22d85a4c35e913d8b746480167e0206cf0c3"
registries = "General"
uuid = "b1df2697-797e-41e3-8120-5422d3b24e4a"
version = "2.1.4"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.46.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "123266c25174ef6c8d4718920abc206452cf8de6"
registries = "General"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.41"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1912a9f1b9ca55005b03ba075f8e19993583e237"
registries = "General"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.58.2+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools"]
git-tree-sha1 = "663e8b48b789916221e0765393b289ca6c88f24e"
registries = "General"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "3.0.0"

[[deps.PartitionedDistributions]]
deps = ["Distributions", "FillArrays", "InvertedIndices", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "PDMats", "SpecialFunctions", "StatsBase"]
git-tree-sha1 = "621944b594bec8df388aeecd2d694824cbe71ecd"
registries = "General"
uuid = "569bd051-8d7b-4221-bcb8-d78512b5866a"
version = "0.1.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
registries = "General"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "Zstd_jll", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.13.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
registries = "General"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
registries = "General"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "83bd514e8ff16b5858ac54c53fa0bcf6002a3b00"
registries = "General"
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
registries = "General"
uuid = "e409e4f3-bfea-5376-8464-e040bb5c01ab"
version = "0.4.13"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
registries = "General"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "cd00e28071a75c98664663f87c2ae447d25c0503"
registries = "General"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "1.7.1"
weakdeps = ["EnzymeCore", "ForwardDiff", "ReverseDiff"]

    [deps.PreallocationTools.extensions]
    PreallocationToolsEnzymeCoreExt = "EnzymeCore"
    PreallocationToolsForwardDiffExt = "ForwardDiff"
    PreallocationToolsReverseDiffExt = "ReverseDiff"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
registries = "General"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "5005266de4bfe50e53ff44a5cb5c540b6e47a254"
registries = "General"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.6.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1b8aa19f229b1cea7fc93874a52e49db6a854450"
registries = "General"
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
registries = "General"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "f0803bc1171e455a04124affa9c21bba5ac4db32"
registries = "General"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.6"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "fbb92c6c56b34e1a2c4c36058f68f332bec840e7"
registries = "General"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
registries = "General"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.PureKLU]]
deps = ["LinearAlgebra", "PrecompileTools", "SparseArrays"]
git-tree-sha1 = "6719681d517f9b02fc3e3c3a0befbde5e4a2320e"
registries = "General"
uuid = "0c0d3e7f-3a8b-4f7e-b6f1-9a4d2e7c1f01"
version = "1.5.0"
weakdeps = ["ForwardDiff"]

    [deps.PureKLU.extensions]
    PureKLUForwardDiffExt = "ForwardDiff"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "144895f6166994730ee7ff8113b981fc360638f1"
registries = "General"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.10.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll", "Qt6Svg_jll"]
git-tree-sha1 = "159d253ab126d5b29230cf53521899bea4ef4648"
registries = "General"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.10.2+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "4d85eedf69d875982c46643f6b4f66919d7e157b"
registries = "General"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.10.2+1"

[[deps.Qt6Svg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "81587ff5ff25a4e1115ce191e36285ede0334c9d"
registries = "General"
uuid = "6de9746b-f93d-5813-b365-ba18ad4a9cf3"
version = "6.10.2+0"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "672c938b4b4e3e0169a07a5f227029d4905456f2"
registries = "General"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.10.2+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "5e8e8b0ab68215d7a2b14b9921a946fee794749e"
registries = "General"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.3"
weakdeps = ["Enzyme"]

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

[[deps.REPL]]
deps = ["Base64", "Dates", "FileWatching", "InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "dbe5fd0b334694e905cb9fda73cd8554333c46e2"
registries = "General"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.7.1"

[[deps.RandomNumbers]]
deps = ["Random"]
git-tree-sha1 = "c6ec94d2aaba1ab2ff983052cf6a606ca5985902"
registries = "General"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.6.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
registries = "General"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
registries = "General"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
registries = "General"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
registries = "General"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
registries = "General"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "GPUArraysCore", "LinearAlgebra", "PrecompileTools", "RecipesBase", "SciMLPublic", "SciMLStructures", "StaticArraysCore", "SymbolicIndexingInterface"]
git-tree-sha1 = "775b6023c34c401e35b9a159568d2f6d051280d5"
registries = "General"
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
registries = "General"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
registries = "General"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
registries = "General"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.ResettableStacks]]
deps = ["PrecompileTools", "StaticArrays"]
git-tree-sha1 = "ed16b7ff60555aa33a8013b9a165c404da3685b3"
registries = "General"
uuid = "ae5879a3-cd67-5da8-be7f-38c6eb64a37b"
version = "1.4.0"

[[deps.RespecializeParams]]
deps = ["FunctionWrappersWrappers", "PrecompileTools"]
git-tree-sha1 = "140c074bb63518a0e1ec2c8aba2a8a5719bcf21c"
registries = "General"
uuid = "9fe22ead-9e00-4db2-8b46-706a60d40f5e"
version = "1.3.0"

[[deps.ReverseDiff]]
deps = ["ChainRulesCore", "DiffResults", "DiffRules", "ForwardDiff", "FunctionWrappers", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "Random", "SpecialFunctions", "StaticArrays", "Statistics"]
git-tree-sha1 = "00113c71356ea1da3e90ca1569a746fc043b1b4c"
registries = "General"
uuid = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
version = "1.17.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
registries = "General"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d40b2fe70437b01397d2a4d5b020008da4e7019"
registries = "General"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.2+0"

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "4db094d5e079abbda658acfe1c4d098430417717"
registries = "General"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "3.0.8"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"
    RootsUnitfulExt = "Unitful"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "PrecompileTools", "SHA", "Serialization"]
git-tree-sha1 = "1719b26eee2c5e40d4b41c647b91dbaac1dcbf05"
registries = "General"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.26"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "1.0.0"

[[deps.SciMLBase]]
deps = ["ADTypes", "Accessors", "Adapt", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FindFirstFunctions", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "LoggingExtras", "Markdown", "PreallocationTools", "PrecompileTools", "Preferences", "Printf", "Random", "RecipesBase", "RecursiveArrayTools", "RuntimeGeneratedFunctions", "SciMLOperators", "SciMLPublic", "SciMLStructures", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface"]
git-tree-sha1 = "8a14bac10be51ecf8cb167e483e39c068ca24bfd"
registries = "General"
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
registries = "General"
uuid = "19f34311-ddf3-4b8b-af20-060888a46c0e"
version = "0.1.19"

[[deps.SciMLLogging]]
deps = ["Logging", "LoggingExtras", "PrecompileTools", "Preferences"]
git-tree-sha1 = "3aafee7edc2ec7e71f50dcc6816cb031fac390a0"
registries = "General"
uuid = "a6db7da4-7206-11f0-1eab-35f2a5dbe1d1"
version = "2.1.0"
weakdeps = ["Tracy"]

    [deps.SciMLLogging.extensions]
    SciMLLoggingTracyExt = "Tracy"

[[deps.SciMLOperators]]
deps = ["Accessors", "Adapt", "ArrayInterface", "DocStringExtensions", "LinearAlgebra", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "9eba48f738160e04268c0770e01dae2451199bc2"
registries = "General"
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
registries = "General"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.3.0"

[[deps.SciMLSensitivity]]
deps = ["ADTypes", "Accessors", "Adapt", "ArrayInterface", "ChainRulesCore", "ConstructionBase", "DiffEqBase", "DiffEqCallbacks", "DiffEqNoiseProcess", "Distributions", "Enzyme", "FastBroadcast", "FiniteDiff", "ForwardDiff", "FunctionProperties", "FunctionWrappersWrappers", "Functors", "GPUArraysCore", "IntervalSets", "LinearAlgebra", "LinearSolve", "Markdown", "NonlinearSolveBase", "OrdinaryDiffEqCore", "PreallocationTools", "PrecompileTools", "QuadGK", "Random", "RandomNumbers", "RecursiveArrayTools", "Reexport", "ReverseDiff", "SciMLBase", "SciMLJacobianOperators", "SciMLLogging", "SciMLOperators", "SciMLStructures", "SparseArrays", "StaticArrays", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tracker", "Zygote"]
git-tree-sha1 = "eccd4496eea5457dd9cd0d36bf4786f890b382d8"
registries = "General"
uuid = "1ed8b502-d754-442c-8d5d-10ac956f44a1"
version = "7.119.8"

    [deps.SciMLSensitivity.extensions]
    SciMLSensitivityMooncakeExt = "Mooncake"
    SciMLSensitivityReactantExt = "Reactant"
    SciMLSensitivitySundialsExt = "Sundials"

    [deps.SciMLSensitivity.weakdeps]
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    Sundials = "c3572dad-4567-51f8-b174-8c6c989267f4"

[[deps.SciMLStructures]]
deps = ["ArrayInterface", "PrecompileTools"]
git-tree-sha1 = "5c2f9dbf6f07eea6bc9e93f117b00b7939a79f9e"
registries = "General"
uuid = "53ae85a6-f571-4167-b2af-e1d143709226"
version = "1.10.5"

[[deps.ScientificTypesBase]]
deps = ["InteractiveUtils"]
git-tree-sha1 = "e785eaa35a0f5518a388f9010e66fda64ea95ede"
registries = "General"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.1.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "67a144433c4ce877ee6d1ada69a124d6b1ecf7be"
registries = "General"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.6.2"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
registries = "General"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "084c47c7c5ce5cfecefa0a98dff69eb3646b5a80"
registries = "General"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.10"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
registries = "General"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates"]
git-tree-sha1 = "8238217340ad0aaabe11afe39c1098b5bc9f4c8e"
registries = "General"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.1.1"

[[deps.Sobol]]
deps = ["DelimitedFiles", "Random"]
git-tree-sha1 = "5a74ac22a9daef23705f010f72c81d6925b19df8"
registries = "General"
uuid = "ed01d8cd-4d21-5b2a-85b4-cc3bdc58bad4"
version = "1.5.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
registries = "General"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.13.0"

[[deps.SparseColumnPivotedQR]]
deps = ["LinearAlgebra", "PrecompileTools", "SparseArrays"]
git-tree-sha1 = "8cf1a59c78dd6f900feb366e5a574f99ed278ef1"
registries = "General"
uuid = "a57abbd0-fea5-4d57-96be-5e525945e8e4"
version = "2.1.8"
weakdeps = ["AMD"]

    [deps.SparseColumnPivotedQR.extensions]
    SparseColumnPivotedQRAMDExt = "AMD"

[[deps.SparseConnectivityTracer]]
deps = ["ADTypes", "DocStringExtensions", "FillArrays", "LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "5c127b6b5e36e1ba38556b578c6346b2375ec0b4"
registries = "General"
uuid = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
version = "1.2.3"
weakdeps = ["ChainRulesCore", "LogExpFunctions", "NNlib", "NaNMath", "SpecialFunctions"]

    [deps.SparseConnectivityTracer.extensions]
    SparseConnectivityTracerChainRulesCoreExt = "ChainRulesCore"
    SparseConnectivityTracerLogExpFunctionsExt = "LogExpFunctions"
    SparseConnectivityTracerNNlibExt = "NNlib"
    SparseConnectivityTracerNaNMathExt = "NaNMath"
    SparseConnectivityTracerSpecialFunctionsExt = "SpecialFunctions"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "eec446511ab8c3293dd846c61c15128392fceed5"
registries = "General"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.3"

[[deps.SparseMatrixColorings]]
deps = ["ADTypes", "DocStringExtensions", "LinearAlgebra", "PrecompileTools", "Random", "SparseArrays"]
git-tree-sha1 = "63e1776f40bbd5e7394edce08e62f852b1384baf"
registries = "General"
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
registries = "General"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.9.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
registries = "General"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "e206cf4850fd7ac4255ffd2b98922f563e18ac53"
registries = "General"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.20"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
registries = "General"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "89f86d9376acd18a1a4fbef66a56335a3a7633b8"
registries = "General"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.5.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e2b53ce13a53367e96601081e33d34746b571bad"
registries = "General"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.5"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
registries = "General"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "adb9da019510162e67a4493fc235c23203d8b09e"
registries = "General"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.13"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "0aa97471d55945556e8d2859568b8317cef98351"
registries = "General"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.3"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "88cf3587711d9ad0a55722d339a013c4c56c5bbc"
registries = "General"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.8"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "773065c6e0e903924a9d838259be74338422aef2"
registries = "General"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.5.0"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "ad8002667372439f2e3611cfd14097e03fa4bccd"
registries = "General"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.3"
weakdeps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

[[deps.StructIO]]
git-tree-sha1 = "c581be48ae1cbf83e899b14c07a807e1787512cc"
registries = "General"
uuid = "53d494c1-5632-5724-8f4c-31dff12d585f"
version = "0.3.1"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "2d0fc55c61321ba245c47be599570d11bac50303"
registries = "General"
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

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.10.1+0"

[[deps.SymbolicIndexingInterface]]
deps = ["Accessors", "ArrayInterface", "PrecompileTools", "RuntimeGeneratedFunctions", "StaticArraysCore"]
git-tree-sha1 = "7eb6da9656581ac9fbffaef3ef7950a090a5002a"
registries = "General"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.55"
weakdeps = ["PrettyTables"]

    [deps.SymbolicIndexingInterface.extensions]
    SymbolicIndexingInterfacePrettyTablesExt = "PrettyTables"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
registries = "General"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
registries = "General"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "a94d9bdda1b7bed0046cea645639ab3f62196fac"
registries = "General"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.14.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
registries = "General"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "81c9b4137edfe56a56efcdcb35d721b2ce3e2416"
registries = "General"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.8"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TimerOutputs]]
deps = ["ExprTools", "PrettyTables", "Printf", "Tables"]
git-tree-sha1 = "03271b02dd1c8f87281271575448b9cf6326a961"
registries = "General"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "1.2.1"

    [deps.TimerOutputs.extensions]
    FlameGraphsExt = "FlameGraphs"

    [deps.TimerOutputs.weakdeps]
    FlameGraphs = "08572546-2f56-4bcf-ba4e-bab62c3a3f89"

[[deps.Tracker]]
deps = ["Adapt", "ChainRulesCore", "DiffRules", "ForwardDiff", "Functors", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NNlib", "NaNMath", "Optimisers", "Printf", "Random", "Requires", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "19b3f57d1b9f9aeff1f905a41818c75e467d07a2"
registries = "General"
uuid = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
version = "0.2.39"
weakdeps = ["PDMats"]

    [deps.Tracker.extensions]
    TrackerPDMatsExt = "PDMats"

[[deps.Tracy]]
deps = ["ExprTools", "LibTracyClient_jll", "Libdl"]
git-tree-sha1 = "73e3ff50fd3990874c59fef0f35d10644a1487bc"
registries = "General"
uuid = "e689c965-62c8-4b79-b2c5-8359227902fd"
version = "0.1.6"

    [deps.Tracy.extensions]
    TracyProfilerExt = "TracyProfiler_jll"

    [deps.Tracy.weakdeps]
    TracyProfiler_jll = "0c351ed6-8a68-550e-8b79-de6f926da83c"

[[deps.TruncatedStacktraces]]
deps = ["InteractiveUtils", "MacroTools", "Preferences"]
git-tree-sha1 = "ea3e54c2bdde39062abf5a9758a23735558705e1"
registries = "General"
uuid = "781d530d-4396-4725-bb49-402e4bee1e77"
version = "1.4.0"

[[deps.Turing]]
deps = ["ADTypes", "AbstractMCMC", "AbstractPPL", "Accessors", "AdvancedHMC", "AdvancedMH", "AdvancedVI", "BangBang", "Bijectors", "Compat", "DataStructures", "Distributions", "DocStringExtensions", "DynamicPPL", "EllipticalSliceSampling", "FlexiChains", "ForwardDiff", "Libtask", "LinearAlgebra", "LogDensityProblems", "Optimization", "OptimizationOptimJL", "OrderedCollections", "Printf", "Random", "Random123", "Reexport", "SciMLBase", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "92b456a228ba157b0d8ee4309f82769e6f672671"
registries = "General"
uuid = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
version = "0.49.0"

    [deps.Turing.extensions]
    TuringDistributionsExt = "Distributions"
    TuringDynamicHMCExt = "DynamicHMC"
    TuringMCMCChainsExt = "MCMCChains"

    [deps.Turing.weakdeps]
    DynamicHMC = "bbc10e6e-7c05-544b-b16e-64fede858acb"
    MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
registries = "General"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "21b39bfb1fab6156b61fbcba4c86c57b6216d2c3"
registries = "General"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.2"
weakdeps = ["LLVM"]

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
registries = "General"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
registries = "General"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
registries = "General"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "cab29ed70d9355a28b95b1798007ff3a00eefc91"
registries = "General"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.8"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "248a7031b3da79a127f14e5dc5f417e26f9f6db7"
registries = "General"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.1.0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e52eca002a11c30a858185efdfb15311e1c7a6bf"
registries = "General"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.4+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
registries = "General"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
registries = "General"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
registries = "General"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
registries = "General"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
registries = "General"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
registries = "General"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
registries = "General"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
registries = "General"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "dcb316b3ce0941f195537dda56bea4517fcd3ff5"
registries = "General"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.4+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
registries = "General"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
registries = "General"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
registries = "General"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
registries = "General"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
registries = "General"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
registries = "General"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
registries = "General"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
registries = "General"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
registries = "General"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
registries = "General"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
registries = "General"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
registries = "General"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
registries = "General"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "2e59214e017a55cb87474a00fa76035c82ac0e17"
registries = "General"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.47.0+2"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
registries = "General"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["CompilerSupportLibraries_jll", "Libdl"]
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "a1ec45a8a0adee4d581a37f7e1a0fb401c1cb113"
registries = "General"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.7.13"

    [deps.Zygote.extensions]
    ZygoteAtomExt = "Atom"
    ZygoteCUDAExt = "CUDA"
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Atom = "c52e3926-4ff0-5f6e-af25-54175e0327b1"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "c6a86c133861234450ab260dee01b42abd604095"
registries = "General"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.8"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
registries = "General"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
registries = "General"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ef17c47d22224aaecc76e597ab21a072e025cf7b"
registries = "General"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.14.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "cb007192783c56d8249db4cf0e3495001edfe414"
registries = "General"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.5+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
registries = "General"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "28e57478e8a160d346a19c28b3fffb9273bcc9c2"
registries = "General"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.134+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
registries = "General"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
registries = "General"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
registries = "General"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e51150d5ab85cee6fc36726850f0e627ad2e4aba"
registries = "General"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.58+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
registries = "General"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
registries = "General"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
registries = "General"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.67.1+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "da8c1f6eee04831f14edcfa5dae611d309807e57"
registries = "General"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.3.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.8.2+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
registries = "General"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
registries = "General"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
registries = "General"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"

[registries.General]
url = "https://github.com/JuliaRegistries/General.git"
uuid = "23338594-aafe-5451-b93e-139f81909106"
"""

# ╔═╡ Cell order:
# ╟─45dc0462-d63c-11ef-0f62-e5d029618cc1
# ╟─51c67a4d-f49a-4bb5-a1c2-361ea0e8f20e
# ╠═885f5685-5239-42aa-942c-2b9bdf7df220
# ╠═86b53570-0f03-4bb8-8c78-3a45c38ad789
# ╟─34efa390-0d6d-48dd-b08b-0c1bc391ffc2
# ╠═760b16ea-410f-4f46-ae44-f859de9b75e3
# ╠═27059e1a-5f0c-40ac-85ea-ff6ad8d7b68c
# ╠═9ff83b76-86dd-4520-ac20-df1eb30cc1dd
# ╠═1245a924-10c8-4170-b3a9-ce025c54319d
# ╟─d2ac7dbe-a47a-4b50-a796-a04462f410cc
# ╠═9ad843f4-0f54-4a4f-a83d-b29bedad37c3
# ╠═f5c9ea83-6dba-4848-8340-1a8c772bd941
# ╠═1e27f7dd-9d41-4fe9-b823-0013ef35927d
# ╠═835c2922-37de-4e1b-b9ba-05a729eca801
# ╟─2ab07370-1cf4-41db-b197-295d4cfd6a68
# ╟─c725adf8-4bfa-4046-929f-65dfef8c2309
# ╠═c7193526-d32f-4f2a-95aa-09e98b88b30b
# ╠═2c247d32-e215-48ac-95df-4c562333309f
# ╠═0a83e463-7180-4f7e-8f93-37165d56aed8
# ╟─bbe5e86a-0e72-4ce2-aa95-1b80a070b3ac
# ╠═ee4e03a6-1187-4de1-816c-9299debe54ed
# ╟─2ff9c802-5562-4669-8045-d3bf40e42cd2
# ╠═36985121-9fe5-4961-9ec2-3f5ee57cfaa1
# ╠═9d09b14b-af52-48e2-9c6c-446c398ed422
# ╟─eaec431c-e7f3-49a7-a2a2-cb18d464348c
# ╠═5e533d0a-5aaa-4e42-a420-fb3b3ccb5060
# ╠═8b0256ff-c49c-468c-b140-1416e3e311ab
# ╠═34aa161d-4421-4344-b40f-e25a151a98e2
# ╠═f8a4e5ed-684e-4242-adce-9114576cbc68
# ╟─92c24e7c-deee-4b56-a64d-00748d7fe69a
# ╟─d942a2d0-dc61-4b74-8702-789c54044300
# ╠═134afb44-b726-45d3-a2a0-002a7c433cfb
# ╟─8bbafc21-3f6b-4bce-9b41-56ba36b6a2b4
# ╠═13a104ee-1af5-47f2-acf1-db5cd32ea37e
# ╟─80f79d79-0432-46f9-9734-c598f061e87c
# ╠═730d88c9-682b-4377-bae6-3b9e51f4c2d1
# ╠═b9a8e9e3-f17e-425b-86c8-6e836932735c
# ╟─dbfe82fd-6b79-434e-a0c5-591a9f3990c6
# ╠═ba06d75c-9f29-48c0-9db3-b3822568bf39
# ╠═1f2212b0-d6b3-42e1-8864-8588aacab5c4
# ╠═c07e6522-1c78-491f-bffa-70bf0c11dea5
# ╠═a5a3d044-eddc-4c79-bf0f-2d6753a16e07
# ╟─de5354a6-b8cd-4871-b9b3-2bd3be6bee17
# ╠═a6668c04-6b54-45df-951e-4dff659ae152
# ╠═9723e36d-7ec4-4c0d-bab3-13ee372b7515
# ╟─50f8441b-6a5b-4b3b-8846-29894388bf6f
# ╠═10338166-3f7a-4479-9089-1437b8a82a6e
# ╟─d8fd5015-6cdc-4f4f-80eb-b16e21f0e908
# ╠═6f04db2c-3c38-4244-8d70-22a15a7dd72e
# ╠═3d580d10-9fe1-48ff-b563-d7e267205e8c
# ╟─981ba39c-6328-4577-bc9a-5574dcc634d9
# ╠═5765470f-44a3-4911-9aec-d029ee7f87b3
# ╟─f71e8884-5829-4346-8bde-43ea1947d7f5
# ╠═cb5be726-5287-4519-a199-14a27246726e
# ╟─f38d3cec-65ea-4999-867a-984b23b9fe21
# ╟─bed65c9b-cad3-48c2-aac5-5dc201a15558
# ╠═8a3108d7-0bec-40d1-b5a9-e4deec2f88c0
# ╠═dba58f1a-6340-4242-8b1a-482dd50e8d11
# ╠═08389217-2791-469e-ae48-e7efdc0de493
# ╟─49175b07-6c2e-4a13-86f7-8c5e5047e553
# ╟─7ad1e21f-2c72-415c-9d9c-c0ee35ab44e6
# ╠═e6413dda-3ebb-49c1-9dfb-a340321e86c6
# ╟─a6c0695e-6c29-4992-b7dd-dcab7aeea3f1
# ╠═a5e5b6bf-1fe9-4462-a5e3-c30bf656a6cd
# ╠═f9749dde-697a-40aa-a0e7-47edc8fc778f
# ╟─08617e66-a04f-464a-a1b1-31ad7d44f491
# ╟─68fc5732-e059-44a3-a8d5-3428cb6c36d1
# ╠═099f04f0-8f67-425c-a395-0d2dd20925b1
# ╟─598a4615-f77a-4167-9bf1-62aa25218b17
# ╠═f860ea33-eb7c-43f6-a1b5-859b0b33d367
# ╠═0d3e1592-5535-4c62-a1a2-82fb11109ce4
# ╠═b30eb1b1-b355-4f3e-980a-dd98b2d9d1d4
# ╠═764b48c6-8427-4e1d-8ec7-59126a020e71
# ╟─d1897917-14ae-418a-a1b5-e7d47e58a8dd
# ╠═3231de71-277a-4a13-b3ae-220650089f8b
# ╠═c82ea0b1-f5dd-45be-ad89-4746d31109a8
# ╠═6572ab63-4428-4d4b-8ba2-37820d1403cc
# ╠═14cf3698-fb1b-44bc-89b1-98d8fdf29723
# ╟─aa7d1c43-0d78-462b-93d0-090d2bbdff31
# ╠═241cb0fa-9e3e-4291-a3b8-e0d1246c9d2f
# ╠═e2ee3937-32f2-4caa-9509-a0418fb36135
# ╟─a3497c89-f8c8-4468-bfb3-430e3fe07071
# ╠═0fa7450e-4c48-40f5-8785-cd0ab7a8c4f7
# ╠═ff48c782-955d-4e7d-be0f-0f7db2c011ae
# ╟─19f1b4f6-e42c-45c3-8e4a-2f6c623fa65e
# ╠═66e393f7-4120-46e1-925c-11367c3b6d4c
# ╠═a6b84344-a03d-4575-bb56-9ba2ba37a3f7
# ╠═18e5b654-10e9-4df9-95b5-62624d694b9f
# ╠═48f71f49-44c3-4424-824e-63a0f975bd38
# ╟─3b48ff74-df66-4a6d-aca0-fcf2b855cb23
# ╠═1c913f13-6a33-4161-a9e4-7e439f1ebf70
# ╟─757546ff-9c2a-442c-9b71-b4b30c48cdfa
# ╠═ecaaea3c-f399-42d7-a56a-033a0f6b4a8b
# ╠═f87f8a17-a706-497e-94d4-e6e6f7601f37
# ╠═79d4d09a-d1a2-45a9-bb02-ae05a24b4f1d
# ╟─790110fe-ad24-4a71-a2c7-8446e0441985
# ╟─b79609cd-bb10-49ad-91eb-5de46325d0e0
# ╠═421dbe2a-a689-4390-9d0b-b4528d11f4af
# ╠═8186418e-d332-41da-b6ae-d90964c28c08
# ╠═f3685420-ec96-4cdc-aa9a-f3f0cf948b1e
# ╠═95bebbd8-7e5d-4e7a-a2a8-68b546614c0e
# ╠═169aa5f4-a85d-4e53-8442-f47757456726
# ╠═490d7195-a878-4634-8f2b-a51463976d44
# ╠═4d3dd038-5cdf-44ad-9fcc-22353e55961f
# ╠═f66493ab-dddb-46fa-9305-80f7625e8f88
# ╠═664a2a56-58d6-42bf-89c3-15307834d633
# ╟─8d46da5d-e59d-47e4-af1a-b8b33a8b57a7
# ╠═5550a2d3-bfc7-433e-a3d8-4c4791ecd7fc
# ╠═3bc2d2d7-863c-4f59-9918-4eb3944ac0ec
# ╟─14ee0daf-95f5-4e54-8ccc-ea1df5a52157
# ╠═95cd9037-5985-4c45-bbe8-e20a5e244b5f
# ╠═8ffc8473-1c74-4ba2-bd6c-5f8aa1223f10
# ╟─943a4e6f-1cbb-4800-aade-6bd398d966a3
# ╠═22bdc1e6-ce8d-41bd-a257-d015202f468d
# ╠═22f0edf0-d358-4d40-ad1a-78e251cdc809
# ╠═7a49a79c-95b5-4dbc-b979-4c753a75e86e
# ╠═b0b10ca7-0dca-4359-9fe5-f60b375f3413
# ╟─522a37fb-a2f7-4758-8e32-02c8058c8df3
# ╠═db0b2561-7a02-427f-9a9c-a3943947da4c
# ╟─0351b05b-c63d-4a7a-a20b-e6e72b835bec
# ╠═6402bee8-18dd-4c1b-8616-d57030310503
# ╟─0bfd1396-c569-4ac9-b92a-f535e533ba50
# ╠═c479f931-f403-4c93-bc63-14a01f19718b
# ╠═1a5be3f3-9d6c-45f0-963a-8e19e65c0f3a
# ╠═e479984b-579c-4cdd-9f69-6d7b0db8202e
# ╠═b122c127-53aa-45d5-997c-4fa2cb84d2ff
# ╠═4db2c565-7c96-41c9-8480-e07d107f12d3
# ╟─9677c213-4af4-48bc-9c49-b7907f478f8d
# ╠═243b7f0b-2f0c-4373-ae98-ce234562f87a
# ╟─60118730-1525-43fd-b1ff-b675b6a20e42
# ╟─33adf452-e528-4cbc-88a7-d81bac52e124
# ╠═d6001516-314b-4f9c-9929-61cd2e2e4e98
# ╠═3c20fb0d-ffba-4df6-bc21-175823894867
# ╠═d2868575-d779-436e-97a1-c524b775899d
# ╠═f448e44b-66ee-42db-99a1-1a9ad1b92d23
# ╟─583d4aa9-0193-49b1-8350-7b2715041f39
# ╟─38de9bfc-ea00-43c2-af5b-08fb031e1861
# ╠═bf8cdb2b-e77c-4a9c-97c0-ab57e8fc6e6b
# ╠═cd427bf5-afbd-4e92-9def-c5f5e7bea416
# ╠═5755fb3c-24d5-45b2-a10c-a1010c27df4f
# ╟─70fbd8fd-fda4-426d-8a6f-d1f8fa303200
# ╠═bd129656-3273-4725-afe5-f9ff04afda23
# ╠═97118579-0ae9-4078-b6f4-4ad4a6a54901
# ╟─fdd91eab-f5af-43f5-9cf2-3311e7eea56c
# ╠═e662f173-84ad-438e-afe8-a1e1e82d38c6
# ╟─f9ac099c-8fc8-45b4-b0b6-6e47985a5880
# ╟─3bf58446-ddff-4dd8-a3f6-1ac9822a930f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
