# High-Performance Scientific Modeling with Julia and SciML

[Link to the Github Repository Source Code](https://github.com/SciML/Julia_Modeling_Workshop)

This workshop provides comprehensive training in high-performance scientific computing using the Julia programming language and the SciML (Scientific Machine Learning) ecosystem. Designed for researchers and computational modelers, this material covers everything from Julia basics to advanced techniques in scientific modeling, including differential equation solving, parameter estimation, and the integration of machine learning with mechanistic models.

The workshop emphasizes practical, hands-on learning through interactive Pluto notebooks, progressing from fundamental Julia concepts through sophisticated modeling techniques. You'll learn to build performant scientific models, solve large-scale stiff systems, perform inverse problem solving, and leverage automatic differentiation for machine learning integration. Whether you're modeling chemical kinetics, biological systems, or physical processes, these materials provide the tools and knowledge needed to deploy advanced computational models on high-performance computing systems.

This workshop bridges the gap between traditional numerical methods and modern machine learning approaches, demonstrating how Julia's unique features enable performance and expressiveness in scientific computing.

## About the Author

Dr. Chris Rackauckas is the VP of Modeling and Simulation at JuliaHub, the Director of Scientific Research at Pumas-AI, Co-PI of the Julia Lab at MIT, and the lead developer of the Julia SciML Open Source Software Organization. For his work in mechanistic machine learning, his work is credited for the 15,000x acceleration of NASA Launch Services simulations and recently demonstrated a 60x-570x acceleration over Modelica tools in HVAC simulation, earning Chris the US Air Force Artificial Intelligence Accelerator Scientific Excellence Award. See more at https://chrisrackauckas.com/. He was the original architect of the Pumas software for nonlinear mixed effects (NLME) modeling in industrial pharmaceutical applications and received a top presentation award at every ACoP from 2019-2021 for improving methods for uncertainty quantification, automated GPU acceleration of nonlinear mixed effects modeling (NLME), and machine learning assisted construction of NLME models with DeepNLME. This software was known for being used for the Moderna Covid-19 vaccine. For these achievements, Chris received the Emerging Scientist award from ISoP.

## Quick Start: Running the Workshop Notebooks

To open the Pluto notebooks, first install Pluto locally via:

```julia
using Pkg
Pkg.add("Pluto")
```

and then run the following command to get Pluto running:

```julia
import Pluto; Pluto.run()
```

Then simply paste the URLs below into the URL bar and hit enter. Then click "Edit" to make the notebook live on your computer.

## Quick Reference Guide

For a comprehensive overview of Julia syntax and scientific modeling concepts:

* [Scientific Modeling Cheatsheet (MATLAB – Python – Julia Quick Reference)](https://sciml.github.io/Scientific_Modeling_Cheatsheet/scientific_modeling_cheatsheet) - A cheatsheet for converting between MATLAB, Python, and Julia
* [Julia Scientific Modeling Quick Reference](https://sciml.github.io/Julia_Modeling_Workshop/Julia_Scientific_Modeling_Quick_Reference) ([Pluto Notebook Source](https://github.com/SciML/Julia_Modeling_Workshop/blob/main/Julia_Scientific_Modeling_Quick_Reference.jl)) - A complete reference guide covering Julia basics, data structures, control flow, plotting, optimization, differential equations, and more

## Workshop Notebooks

The notebooks for the workshop are available as HTML pages with links to the Pluto notebook source code:

* [Getting Started with Julia: Installation, Tooling, and Community](https://sciml.github.io/Julia_Modeling_Workshop/Getting_Started_Julia) ([Pluto Notebook Source](https://github.com/SciML/Julia_Modeling_Workshop/blob/main/Getting_Started_Julia.jl))
* [Introduction to Julia for People Who Already Do Scientific Computing](https://sciml.github.io/Julia_Modeling_Workshop/Introduction_to_Julia) ([Pluto Notebook Source](https://github.com/SciML/Julia_Modeling_Workshop/blob/main/Introduction_to_Julia.jl))
* [More Details on Arrays and Matrices (Optional)](https://sciml.github.io/Julia_Modeling_Workshop/More_Details_Arrays_Matrices) ([Pluto Notebook Source](https://github.com/SciML/Julia_Modeling_Workshop/blob/main/More_Details_Arrays_Matrices.jl))
* [Introduction to Mathematical Modeling in Julia: Numerical Solvers for Differential Equations and Nonlinear Systems](https://sciml.github.io/Julia_Modeling_Workshop/Introduction_Mathematical_Modeling_Julia) ([Pluto Notebook Source](https://github.com/SciML/Julia_Modeling_Workshop/blob/main/Introduction_Mathematical_Modeling_Julia.jl))
* [Symbolic-Numeric Model Definitions with Symbolics.jl and ModelingToolkit.jl](https://sciml.github.io/Julia_Modeling_Workshop/Symbolic_Numeric_ModelingToolkit) ([Pluto Notebook Source](https://github.com/SciML/Julia_Modeling_Workshop/blob/main/Symbolic_Numeric_ModelingToolkit.jl))
* [Introduction to Biological Modeling with Catalyst.jl](https://sciml.github.io/Julia_Modeling_Workshop/Introduction_to_Catalyst) ([Pluto Notebook Source](https://github.com/SciML/Julia_Modeling_Workshop/blob/main/Introduction_to_Catalyst.jl))
* [Solving Inverse Problems and Parameter Estimation with Julia's SciML](https://sciml.github.io/Julia_Modeling_Workshop/Solving_Inverse_Problems_Julia) ([Pluto Notebook Source](https://github.com/SciML/Julia_Modeling_Workshop/blob/main/Solving_Inverse_Problems_Julia.jl))
* [Scientific Machine Learning (SciML) and Equation Discovery with Julia](https://sciml.github.io/Julia_Modeling_Workshop/Missing_Physics_UDE) ([Pluto Notebook Source](https://github.com/SciML/Julia_Modeling_Workshop/blob/main/Missing_Physics_UDE.jl))

### Exercises

Practice problems and solutions:

* [Workshop Exercises](https://sciml.github.io/Julia_Modeling_Workshop/Workshop_Exercises) ([Pluto Notebook Source](https://github.com/SciML/Julia_Modeling_Workshop/blob/main/Workshop_Exercises.jl)) - Problems to practice Julia concepts
* [Workshop Solutions](https://sciml.github.io/Julia_Modeling_Workshop/Workshop_Solutions) ([Pluto Notebook Source](https://github.com/SciML/Julia_Modeling_Workshop/blob/main/Workshop_Solutions.jl)) - Complete solutions with explanations

# Detailed Getting Started with Julia and the Workshop

## Installing Julia

If you have never used Julia before, then the first thing is to get started with Julia! There are two recommended ways to do this. The simplest is to go to the Julia webpage [https://julialang.org/downloads/](https://julialang.org/downloads/) and download the binary. There will be a latest current release binary for every operating system: pop that in and you are good to go. 

Alternatively, you can download Julia via juliaup. Juliaup is a system that makes controlling and updating Julia very simple, and thus this is what we would recommend for most users. To download Juliaup, you can:

1. On Windows, download julia from the Julia store
2. On MacOSX, Linux, or FreeBSD, you can use the command `curl -fsSL https://install.julialang.org | sh`

Once juliaup is installed, you can manage your Julia installation from the terminal. The main commands to know are:

1. `juliaup help`: shows all of the commands
2. `juliaup update`: updates your Julia version
3. `juliaup default release`: defaults your Julia to use the latest released version.
4. `juliaup default lts`: defaults your Julia to use the long-term stable (LTS) release, currently 1.10.x.

We recommend that you stick to the current default `release`, though production scenarios may want to use LTS. Either way, once this is setup you can just use the command `julia` in your terminal and it should pop right up.

## Using the Package Manager

To use the package manager in Julia, simply hit `]`. You will see your terminal marker change to a blue `pkg>`. The main commands to know in the package manager mode are:

1. `help`: shows all of the commands
2. `st`: show all of your currently installed packages
3. `add PackageX`: adds package X
4. `rm PackageX`: removes package X
5. `activate MyEnv`: activates an environment
6. `instantiate .`: instantiates an environment at a given point.

You can also use the Julia Pkg.jl package, for example `using Pkg; Pkg.add("MyPackage")`.

## Using Pluto Notebooks

Pluto notebooks, such as the one these lecture notes are written in, are a notebook system designed specifically for Julia. Unlike Jupyter notebooks (which, reminder stands for Julia Python R notebooks!), Pluto notebooks are made with an emphasis on reproducible science. As such, it fixes a lot of the issues that arise with the irreproducibility commonly complained about with Jupyter notebooks, such as:

1. Being environment dependent: Jupyter notebook environments depend on your current installation and packages. Pluto notebooks tie the package management into the file. This means that a Pluto notebook is its own package environment, keeps track of package versions, and when opened will automatically install all of the packages to the right version to make sure it fully reproduces on the new computer.
2. Being run-order dependent: in a Jupyter notebook the user is in control of execution and may evaluate blocks out of order. This means that if you pick up a Jupyter notebook and have all of the same packages you can run the notebook from top to bottom and might not get the same answer as what was previously rendered. This is not possible with Pluto (except for differences in pseudorandom numbers of course), as Pluto is a fully reactive execution engine, and thus any change in the entire notebook ensures that all downstream cells are automatically updated. This means it has more limitations than a normal Julia program, for example every name can only be used exactly once, but it means that every Pluto notebook is always guaranteed to be up to date.

Thus for reproducibility we will be using Pluto notebooks for these notes.

To get started with Pluto, simply install Pluto (`using Pkg; Pkg.add("Pluto")`) and then get it started: `using Pluto; Pluto.run()`. This will open the Pluto runner in your browser, and you can put the URL for a Pluto notebook in to open it on your computer. When you click edit it will make the document then live.

Note: when you first make this notebook live, it will download all of the required packages. This notebook has a lot of things in there so that might take a while! Be patient as it's installing ~200 big packages, but they will be reused for the later lectures.

# Workshop Outline

## Getting Started with Julia

* Installing Julia and package management
* Using the REPL and Pluto notebooks
* Julia ecosystem and community resources

## Introduction to Julia for People Who Already Do Scientific Computing

Core Julia concepts for scientific computing:

* Arrays: The workhorse of scientific computing
  - Column-major storage and 1-indexing
  - Array comprehensions and broadcasting
  - Views vs copies for memory efficiency
* Loops: Fast and natural in Julia
  - No vectorization needed
  - Writing efficient nested loops
* Structs: Custom types for organization
  - Immutable vs mutable structs
  - Parametric types for flexibility
* Multiple Dispatch: Julia's superpower
  - Methods for different type combinations
  - Generic programming and extensibility
* Making Code Fast:
  - Golden rule: Put code in functions
  - JIT compilation and type specialization
  - Type stability for optimal performance
  - Function specialization and limiting it

## More Details on Arrays and Matrices (Optional)

### Advanced array concepts for performance optimization:

* Array Memory Layout and Column-Major Order
* Broadcasting and Vectorization
* Linear Algebra Operations and BLAS
* Sparse Matrices
* Static Arrays for Performance
* Views and Memory Efficiency
* Custom Array Types

## Introduction to Mathematical Modeling in Julia

Building and solving mathematical models:

* Ordinary Differential Equations (ODEs)
  - Setting up ODE systems with DifferentialEquations.jl
  - Mass matrix formulations
  - Solver selection and tolerances
* Finding Steady States
  - Using NonlinearSolve.jl for equilibrium solutions
  - Bifurcation analysis basics
* Numerical Methods
  - Understanding stiff vs non-stiff problems
  - Adaptive time-stepping
  - Event handling and callbacks

## Symbolic-Numeric Model Definitions with ModelingToolkit.jl

Combining symbolic and numeric approaches:

* Symbolic Computing with Symbolics.jl
  - Creating symbolic variables and expressions
  - Symbolic differentiation and simplification
* ModelingToolkit.jl for Model Building
  - Defining ODEs symbolically
  - Automatic Jacobian and Hessian generation
  - Index reduction for DAEs
  - Model composition and hierarchical modeling

## Introduction to Biological Modeling with Catalyst.jl

Specialized tools for biological systems:

* Chemical Reaction Networks
  - DSL for defining reaction systems
  - Mass action kinetics
  - Hill functions and Michaelis-Menten
* Stochastic Simulation
  - Gillespie algorithms via JumpProcesses.jl
  - Chemical Langevin equations
  - Hybrid stochastic-deterministic models
* Network Analysis
  - Conservation laws
  - Reaction network visualization

## Solving Inverse Problems and Parameter Estimation

Data-driven modeling and inference:

* Optimization Fundamentals
  - Local vs global optimization
  - Choosing appropriate loss functions
  - Multi-start strategies
* Sensitivity Analysis
  - Forward sensitivity analysis
  - Adjoint methods for gradient computation
  - Automatic differentiation through ODE solvers
* Parameter Estimation
  - Maximum likelihood estimation
  - Regularization techniques
  - Handling experimental data and measurement error
* Bayesian Inference
  - MCMC sampling with Turing.jl
  - Prior selection and posterior analysis
  - Uncertainty quantification
* Identifiability Analysis
  - Structural vs practical identifiability
  - Profile likelihood methods
  - Fisher information matrix

## Scientific Machine Learning and Universal Differential Equations

Combining machine learning with scientific models:

* Neural ODEs and UDEs
  - Embedding neural networks in differential equations
  - Training strategies and regularization
  - Handling stiff neural ODEs
* Physics-Informed Neural Networks (PINNs)
  - Soft vs hard constraints
  - Collocation methods
* Symbolic Regression and Model Discovery
  - Sparse regression (SINDy)
  - Genetic programming approaches
  - Equation learning from data

## Workshop Exercises

Practice problems based on the UCI Data Science Initiative materials:

* Basic Problems: Arrays, loops, and Monte Carlo methods
* Integration Problems: Time series and dynamical systems
* Intermediate Problems: Custom types, operators, and regression
* Advanced Problems: Metaprogramming and numerical analysis

# References and Additional Resources

## Original Workshop Materials

* [UCI Data Science Initiative - Introduction to Julia](https://github.com/UCIDataScienceInitiative/IntroToJulia) - The original workshop materials this course builds upon

## Related SciML Workshops and Tutorials

* [A Deep Dive Into DifferentialEquations.jl (JuliaCon 2025)](https://github.com/SciML/2025-JuliaCon-DifferentialEquations-Workshop) [(Video)](https://www.youtube.com/watch?v=lSGFAmXKIsE) - Advanced techniques in differential equations
* [ModelingToolkit Workshop (Hierarchical Component-Based Modeling with ModelingToolkit.jl) (JuliaCon 2024)](https://github.com/SciML/ModelingToolkitWorkshop_JuliaCon2024) [(Video)](https://www.youtube.com/watch?v=OMn9FeVM8NA) - Deep dive into symbolic-numeric modeling
* [Parallel Computing and Scientific Machine Learning](https://book.sciml.ai/) - Comprehensive online book [(Youtube Playlist)](https://www.youtube.com/@scimlorg)
* [Catalyst Workshop](https://github.com/SciML/JuliaCon2022_Catalyst_Workshop) [(Video)](https://www.youtube.com/watch?v=tVfxT09AtWQ) - Biological modeling tutorials (Old)

## Documentation and Learning Resources

* [SciML Documentation](https://docs.sciml.ai/) - Comprehensive documentation for the SciML ecosystem
* [DifferentialEquations.jl Documentation](https://docs.sciml.ai/DiffEqDocs/stable/) - ODE/SDE/DDE/DAE solver documentation
* [ModelingToolkit.jl Documentation](https://docs.sciml.ai/ModelingToolkit/stable/) - Symbolic-numeric modeling framework
* [Catalyst.jl Documentation](https://docs.sciml.ai/Catalyst/stable/) - Chemical reaction network modeling
* [Optimization.jl Documentation](https://docs.sciml.ai/Optimization/stable/) - Unified optimization interface
* [SciMLSensitivity.jl Documentation](https://docs.sciml.ai/SciMLSensitivity/stable/) - Sensitivity analysis and AD

## Julia Resources

* [Julia Documentation](https://docs.julialang.org/) - Official Julia language documentation
* [Julia Academy](https://juliaacademy.com/) - Free Julia courses
* [JuliaHub](https://juliahub.com/) - Package discovery and cloud computing
* [Julia Discourse](https://discourse.julialang.org/) - Community forum
* [Julia Slack](https://julialang.org/slack/) - Real-time chat
* [Julia Zulip](https://julialang.zulipchat.com/) - Alternative chat platform

## Academic Papers and Books

* Rackauckas, C., & Nie, Q. (2017). DifferentialEquations.jl – A Performant and Feature-Rich Ecosystem for Solving Differential Equations in Julia. Journal of Open Research Software.
* Rackauckas et al. (2020). Universal Differential Equations for Scientific Machine Learning. arXiv:2001.04385
* Ma, Y. et al. (2021). ModelingToolkit: A Composable Graph Transformation System For Equation-Based Modeling. arXiv:2103.05244
