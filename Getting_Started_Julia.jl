### A Pluto.jl notebook ###
# v0.20.18

using Markdown
using InteractiveUtils

# ╔═╡ 1a2b3c4d-5e6f-7890-abcd-ef1234567890
begin
    using Pkg
    Pkg.activate(temp=true)
    Pkg.add("PlutoUI")
    using PlutoUI
end

# ╔═╡ 2b3c4d5e-6f78-90ab-cdef-234567890123
md"""
# Getting Started with Julia: Installation, Tooling, and Community

*An introduction to the Julia ecosystem for scientific computing*

This notebook provides a comprehensive guide to getting started with Julia, including:
- Installation and setup
- Understanding Julia's mental model
- Development tools (VS Code, Pluto, Jupyter)
- Package management and Git workflows
- Community resources and documentation
- Best practices for scientific computing

Based on materials from the [UCI Data Science Initiative](https://github.com/UCIDataScienceInitiative/IntroToJulia) and modern Julia workflows.
"""

# ╔═╡ 3c4d5e6f-7890-abcd-ef12-345678901234
md"""
## 📥 Installing Julia

There are two recommended ways to install Julia:

### Option 1: Juliaup
Juliaup makes it easy to manage and update Julia versions.
It's also the easiest way to make sure you get the right version, and it is installed the right way etc.

**Windows:** Install from the Microsoft Store

**Mac/Linux:** Run in terminal:
```bash
curl -fsSL https://install.julialang.org | sh
```

After installation, type `julia` in your terminal to start the REPL.
You can use the `juliaup` command on your commandline to manage julia versions
See `juliaup --help` for details.

### Option 2: Direct Download (Simple)
Visit [julialang.org/downloads](https://julialang.org/downloads/) and download the binary for your operating system.

### Bad Option: Install with your package manager
We strongly recommend against using your system package manager (apt, homebrew etc) to install julia.


For various reasons they tend to distribute a version that is either:
 - wildly out of date (like 5-10 years),
 - linked to subtly broken versions of libraries;
 - both!
"""

# ╔═╡ 4d5e6f78-90ab-cdef-2345-678901234567
md"""
## 🧠 Julia Mental Model: Talking to a Scientist

Understanding Julia's philosophy helps you write better code:

### Julia vs Other Languages

**Python/R/MATLAB:** Like talking to a politician
- They try to give you what you want
- May hide complexity behind the scenes
- Prioritize ease over speed
- Using library code written in C is much faster than anything you can write yourself in the language.

**C/Fortran:** Like talking to a philosopher
- Demand extreme specificity
- Require deep understanding of details
- Fast but verbose
- Everything might catch fire if you do it wrong

**Julia:** Like talking to a scientist
- Surface simplicity with specific underlying details
- Context determines meaning (multiple dispatch)
- Nothing is hidden - you can inspect everything
- Expects precision but rewards with performance

### Key Principles
1. **Write generic code, get specific performance** - The compiler specializes your code.
   - Regardless of if you put type annotation on functions it always specialized on types of inputs.
2. **Two-language problem solved** - Prototype and production in the same language.
   - Not all julia code it fast, you can write quick sloppy code. But you can then start to take care and write fast efficient code only in the critical parts. And at all times you just write Julia.
3. **Don't reinvent the wheel** - Julia has a built in package manager and a lot of excellent libraries. Further, it has excellent ability to call code written in C, python, R etc.
4. **Composability** - Packages work together naturally through shared abstractions.
5. **Speed without sacrifice** - Fast code that's still readable
"""

# ╔═╡ 5e6f7890-abcd-ef12-3456-789012345678
md"""
## 🛠️ Development Tools

### VS Code (Recommended IDE)
The Julia VS Code extension provides a full-featured development environment:

1. **Install VS Code:** Download from [code.visualstudio.com](https://code.visualstudio.com/)
2. **Install Julia Extension:** Search for "Julia" in extensions (Ctrl/Cmd+Shift+X)
3. **Features:**
   - IntelliSense and autocomplete
   - Integrated REPL and debugger
   - Plot viewer and workspace explorer
   - Git integration
   - Linting and formatting

### Pluto Notebooks (Interactive Documents)
Perfect for reproducible research and teaching:

```julia
using Pkg
Pkg.add("Pluto")
using Pluto
Pluto.run()
```

**Why Pluto?**
- **Reproducible:** Package environments are built into the file
- **Reactive:** Changes propagate automatically
- **Simple:** No hidden state or execution order issues

### Jupyter Notebooks (Data Science)
For data analysis and visualization:

```julia
using Pkg
Pkg.add("IJulia")
using IJulia
notebook()
```

**When to use what:**
- **VS Code:** Package development, large projects
- **Pluto:** Teaching, reproducible research, reactive programming
- **Jupyter:** Data exploration, machine learning workflows
"""

# ╔═╡ 6f789012-3456-789a-bcde-f12345678901
md"""
## 📦 Package Management

Julia's package manager is built into the REPL. Press `]` to enter package mode:

### Creating Environments
Environments isolate package dependencies per project:

```julia
# In package mode (])
activate MyProject
add DataFrames Plots
```

This creates `Project.toml` and `Manifest.toml` files specify your dependencies.
The `Project.toml` specifies dependencies, including version compatibility at a high level.
from it Pkg generates a `Manifest.toml` which says exactly what version is being installed -- including of indirect dependencies.
You may or may not want to version control your `Manifest.toml`.

It is normal to create one julia environment per project you are working on, to keep things isolated and manage your dependencies and make sure things stay compatible.
(And required to do so if that project is a julia package).

### Essential Pkg Commands
- `add PackageName` - Install a package
- `rm PackageName` - Remove a package
- `update` - Update all packages
- `resolve - Determine versions of all packages specified in `Project.toml` generating a `Manifest.toml` (implictly called by above.)
- `instantiate` - Install all packages from Manifest.toml (implictly called by above)

- `st` or `status` - Show installed packages
- `activate .` - Activate environment in current directory
- `help` - Show all commands

### Starting/Change Julia to a particular environment

#### From commandline
```bash
julia --project="."
```

#### With-in Julia REPL:
As per above for creating an environment.
```
activate .
```

#### In VS-Code
Either from the open REPL as above
In bottom right click the text that will start out saying `(Main)`.

#### In JuPyTer
```julia
using Pkg: @pkg_str
pkg"activate ."
```

#### In Pluto:
Each file in pluto has its own internal hidden Project.toml and Manifest.toml,
with things being automatically installed when you first import them.
but if you want to use external one, it is as per JuPyTer.


## Global Environment

If you start julia without specifying an environment you are placed in the global environment.
Things installed here are available no matter what environment you have activated.

It's common to install development tools here, like `Pluto`, `IJulia`, `BenchmarkTools` and `Revise`.

It's discouraged from installing tools you need for your science here since you want to keep a nice isolated environment per project so it is easy to give to others.

"""

# ╔═╡ 7890abcd-ef12-3456-789a-bcdef1234567
md"""
## 🌐 Git and Package Development

### Using Git with Julia Projects

1. **Initialize repository:**
```bash
git init
git add Project.toml Manifest.toml src/
git commit -m "Initial commit"
```

2. **Clone and instantiate:**
```bash
git clone https://github.com/username/MyPackage.jl
cd MyPackage.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

### Developing Packages
```julia
# In package mode
dev path/to/MyPackage    # Use local version
dev MyPackage            # Clone from registry
```
"""

# ╔═╡ 890abcde-f123-4567-89ab-cdef12345678
md"""
## 👥 Community Resources

### Primary Communication Channels

📱 **[Julia Slack](https://julialang.org/slack/)** (Most Active)
- Real-time chat with core developers
- Channels for specific topics (#helpdesk, #machine-learning, #sciml)
- Friendly and responsive community

💬 **[Discourse Forum](https://discourse.julialang.org/)**
- In-depth discussions and questions
- Package announcements
- Performance help and code reviews

### Additional Resources

💬 **[Discord](https://discord.gg/julia)** - Community chat and discussions

📧 **[Zulip](https://julialang.zulipchat.com/)** - Threaded conversations

🐙 **[GitHub](https://github.com/JuliaLang/julia)** - Issues and development

📚 **[JuliaHub](https://juliahub.com/)** - Package search and documentation

### Getting Help

1. **Read the error message carefully** - Julia's errors are informative
2. **Search Discourse** - Your question may be answered
3. **Ask on Slack #helpdesk** - For quick questions
4. **Post on Discourse** - For complex issues needing code examples
5. **Check package issues** - For package-specific problems

### Community Guidelines
- Be respectful and patient
- Provide minimal working examples (MWE)
- Share what you've tried
- Help others when you can!
"""

# ╔═╡ 9abcdef0-1234-5678-9abc-def012345678
md"""
## 📚 Documentation and Learning Resources

### Official Documentation
- **[Julia Documentation](https://docs.julialang.org/)** - Comprehensive manual
- **[Julia by Example](https://juliabyexample.helpmanual.io/)** - Learn through code
- **[JuliaAcademy](https://juliaacademy.com/)** - Free courses

### Essential Reading
1. **[Noteworthy Differences](https://docs.julialang.org/en/v1/manual/noteworthy-differences/)** - Coming from other languages
2. **[Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/)** - Write fast code
3. **[Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/)** - Julia conventions
4. **[MATLAB-Python-Julia Cheatsheet](http://cheatsheets.quantecon.org/)** - Syntax comparison

### Modern Julia Workflows
- **[Modern Julia Workflows](https://modernjuliaworkflows.github.io/)** - Best practices and tools
- **[SciML Tutorials](https://tutorials.sciml.ai/)** - Scientific machine learning
- **[Julia Data Science](https://juliadatascience.io/)** - Data science with Julia

### Getting Documentation in the REPL
```julia
?println  # Shows documentation for println
```

Press `?` to enter help mode, then type any function or type name!
"""

# ╔═╡ 3456789a-bcde-f012-3456-789abcdef012
md"""
---
*This notebook is part of the Julia Modeling Workshop. For more materials, visit the [workshop repository](https://github.com/SciML/Julia_Modeling_Workshop).*
"""

# ╔═╡ Cell order:
# ╟─1a2b3c4d-5e6f-7890-abcd-ef1234567890
# ╟─2b3c4d5e-6f78-90ab-cdef-234567890123
# ╟─3c4d5e6f-7890-abcd-ef12-345678901234
# ╟─4d5e6f78-90ab-cdef-2345-678901234567
# ╟─5e6f7890-abcd-ef12-3456-789012345678
# ╟─6f789012-3456-789a-bcde-f12345678901
# ╟─7890abcd-ef12-3456-789a-bcdef1234567
# ╟─890abcde-f123-4567-89ab-cdef12345678
# ╟─9abcdef0-1234-5678-9abc-def012345678
# ╠═3456789a-bcde-f012-3456-789abcdef012
