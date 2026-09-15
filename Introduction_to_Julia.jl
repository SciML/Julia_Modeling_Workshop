### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ a5d7b3e0-4e2a-11ef-1234-5b6c8d9a7f12
using BenchmarkTools

# ╔═╡ b2c4d5f6-3a8b-11ef-0912-6d7e9c8a4b23
md"""
# Introduction to Julia for People Who Already Do Scientific Computing

This notebook provides a focused introduction to Julia for experienced scientific computing practitioners. We'll cover the essential language features and performance concepts that make Julia uniquely suited for computational science.

## Why Julia?

Julia solves the "two-language problem" - you can write high-level, readable code that also runs at C/Fortran speeds. This is achieved through:

- **Just-In-Time (JIT) compilation** to native code via LLVM
- **Type specialization** that generates optimized code for each type combination
- **Multiple dispatch** as a core organizing principle
- **Type stability** enabling aggressive compiler optimizations

Let's dive into the essentials.
"""

# ╔═╡ c3d5e6f7-4b9c-11ef-2345-7c8d9e0a1b34
md"""
## Arrays: The Workhorse of Scientific Computing

Julia's arrays are column-major (like Fortran/MATLAB, unlike C/Python) and 1-indexed. They're mutable and support a rich set of operations.
"""

# ╔═╡ d4e6f708-5cab-11ef-3456-8d9e0f1a2b45
# Creating arrays
A = [1.0 2.0 3.0; 4.0 5.0 6.0]  # 2×3 matrix

# ╔═╡ e5f70819-6dbc-11ef-4567-9e0f1a2b3c56
# Vector creation
v = [1.0, 2.0, 3.0]  # Column vector

# ╔═╡ f6081920-7ecd-11ef-5678-0f1a2b3c4d67
# Array comprehensions - powerful and fast
squares = [x^2 for x in 1:10]

# ╔═╡ 07192a31-8fde-11ef-6789-1a2b3c4d5e78
# Broadcasting - element-wise operations with .
result = sin.(A) .+ cos.(v')  # v' is transpose

# ╔═╡ 18203b42-90ef-11ef-7890-2b3c4d5e6f89
md"""
### Array Performance Tips

- Preallocate arrays when possible
- Use views instead of copies with `@view`
- Broadcasting (`.`) fuses operations for efficiency
"""

# ╔═╡ 29314c53-a1f0-11ef-8901-3c4d5e6f7089
# In-place operations save memory
function update_array!(A, B)
    A .= A .+ 2 .* B  # Updates A in-place
    return nothing
end

# ╔═╡ 3a425d64-b201-11ef-9012-4d5e6f708190
# Example of view vs copy
B = rand(1000, 1000)

# ╔═╡ 4b536e75-c312-11ef-0123-5e6f70819201
@btime $B[1:10, 1:10];  # Creates a copy

# ╔═╡ 5c647f86-d423-11ef-1234-6f70819202a3
@btime @view $B[1:10, 1:10];  # Creates a view (no copy)

# ╔═╡ 6d758097-e534-11ef-2345-708192030b14
md"""
## Loops: Fast and Natural

Unlike Python/MATLAB, loops in Julia are fast! Write them naturally without vectorization gymnastics.
"""

# ╔═╡ 7e869108-f645-11ef-3456-8192030b1c25
function sum_squares(arr)
    total = 0.0
    for x in arr
        total += x^2
    end
    return total
end

# ╔═╡ 8f970219-0756-11ef-4567-92030b1c2d36
# Nested loops - write them in the natural order
function matrix_multiply!(C, A, B)
    m, n = size(A)
    n2, p = size(B)
    @assert n == n2 "Dimension mismatch"

    for j in 1:p
        for i in 1:m
            Cij = 0.0
            for k in 1:n
                Cij += A[i,k] * B[k,j]
            end
            C[i,j] = Cij
        end
    end
    return C
end

# ╔═╡ a0a8132a-1867-11ef-5678-a30b1c2d3e47
# Loops with enumerate and zip
data = rand(100)

# ╔═╡ b1b9243b-2978-11ef-6789-b41c2d3e4f58
for (i, val) in enumerate(data[1:5])
    println("Element $i: $val")
end

# ╔═╡ c2ca354c-3a89-11ef-7890-c52d3e4f5069
md"""
## Structs: Custom Types for Organization

Julia's type system is powerful yet simple. Define custom types with `struct` for better code organization and performance.
"""

# ╔═╡ d3db465d-4b9a-11ef-8901-d63e4f506170
# Immutable struct (default, preferred for performance)
struct Point2D
    x::Float64
    y::Float64
end

# ╔═╡ e4ec576e-5cab-11ef-9012-e74f50617281
# Mutable struct (when you need to modify fields)
mutable struct Particle
    position::Point2D
    velocity::Point2D
    mass::Float64
end

# ╔═╡ f5fd6880-6dbc-11ef-0123-f850617283a4
# Parametric types for flexibility
struct Vector3D{T<:Real}
    x::T
    y::T
    z::T
end

# ╔═╡ 06097991-7ecd-11ef-1234-0961728394a5
# Creating instances
p1 = Point2D(1.0, 2.0)

# ╔═╡ 171a8a02-8fde-11ef-2345-1a728394b5c6
v3_float = Vector3D(1.0, 2.0, 3.0)

# ╔═╡ 282b9b13-90ef-11ef-3456-2b8394c5d6e7
v3_int = Vector3D(1, 2, 3)  # Automatically Vector3D{Int64}

# ╔═╡ 393cac24-a1f0-11ef-4567-3c9495d6e7f8
md"""
## Multiple Dispatch: Julia's Superpower

Multiple dispatch allows functions to behave differently based on the types of ALL arguments, not just the first one (like OOP).
"""

# ╔═╡ 4a4dbd35-b201-11ef-5678-4da506e7f809
# Define methods for different type combinations
distance(p1::Point2D, p2::Point2D) = sqrt((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

# ╔═╡ 5b5ece46-c312-11ef-6789-5eb617f80910
distance(v1::Vector3D, v2::Vector3D) = sqrt((v1.x - v2.x)^2 + (v1.y - v2.y)^2 + (v1.z - v2.z)^2)

# ╔═╡ 6c6fdf57-d423-11ef-7890-6fc7280901a1
# Generic programming with abstract types
abstract type Shape end

# ╔═╡ 7d70e068-e534-11ef-8901-70d8391012b3
struct Circle <: Shape
    center::Point2D
    radius::Float64
end

# ╔═╡ 8e81f179-f645-11ef-9012-81e9401123c4
struct Rectangle <: Shape
    corner::Point2D
    width::Float64
    height::Float64
end

# ╔═╡ 9f92028a-0756-11ef-0123-92fa512234d5
# Define area for each shape
area(c::Circle) = π * c.radius^2

# ╔═╡ a0a3139b-1867-11ef-1234-a30b623345e6
area(r::Rectangle) = r.width * r.height

# ╔═╡ b1b424ac-2978-11ef-2345-b41c734456f7
# Multiple dispatch in action
shapes = [
    Circle(Point2D(0.0, 0.0), 1.0),
    Rectangle(Point2D(0.0, 0.0), 2.0, 3.0)
]

# ╔═╡ c2c535bd-3a89-11ef-3456-c52d845567f8
[area(s) for s in shapes]

# ╔═╡ d3d646ce-4b9a-11ef-4567-d63e956678f9
md"""
### The Power of Generic Programming

Multiple dispatch enables writing generic algorithms that work with any type that implements required methods.
"""

# ╔═╡ e4e757df-5cab-11ef-5678-e74f067789fa
# Generic norm function works for anything with + and *
function mynorm(x)
    sqrt(sum(xi^2 for xi in x))
end

# ╔═╡ f5f868f0-6dbc-11ef-6789-f8506789a0fb
mynorm([1.0, 2.0, 3.0])  # Works with arrays

# ╔═╡ 0607a001-7ecd-11ef-7890-096189ab1c2d
mynorm((1, 2, 3))  # Works with tuples too!

# ╔═╡ 1718b112-8fde-11ef-8901-1a789abc2d3e
md"""
## Making Code Fast: Writing Functions

**Golden Rule: Put performance-critical code in functions!**

Julia's JIT compiler optimizes functions, not global scope code.
"""

# ╔═╡ b0e46118-d9be-47f3-9060-715c630d9a3a
# BAD: Global scope computation
begin
	global_sum = 0.0
	data_array = rand(1000)
	for x in data_array
    	global global_sum += x  # Slow due to type instability
	end
end

# ╔═╡ 4a4be445-b201-11ef-1234-4da5ef567189
# GOOD: Function-based computation
function compute_sum(data)
    s = 0.0
    for x in data
        s += x
    end
    return s
end

# ╔═╡ 5b5cf556-c312-11ef-2345-5eb6f06789ab
@btime compute_sum($data_array);  # Much faster!

# ╔═╡ 6c6e0667-d423-11ef-3456-6fc7189abcd0
md"""
## JIT Compilation: Understanding the Magic

Julia compiles functions on first call for each unique set of argument types. This is why:
1. First call is slower (compilation time)
2. Subsequent calls are fast (using compiled code)
3. Different types trigger new compilations
"""

# ╔═╡ 7d7f1778-e534-11ef-4567-70d8ab12de01
function add_numbers(a, b)
    return a + b
end

# ╔═╡ 8e802889-f645-11ef-5678-81e9bc23ef12
# First call with Float64 - includes compilation
@time add_numbers(1.0, 2.0)

# ╔═╡ 9f90399a-0756-11ef-6789-92facd34f023
# Second call with Float64 - uses compiled code
@time add_numbers(3.0, 4.0)

# ╔═╡ a0a14aab-1867-11ef-7890-a30bde450134
# First call with Int64 - new compilation
@time add_numbers(1, 2)

# ╔═╡ b1b25bbc-2978-11ef-8901-b41cef561245
# Inspect generated code
@code_llvm add_numbers(1.0, 2.0)

# ╔═╡ c2c36ccd-3a89-11ef-9012-c52df0672356
md"""
## Type Stability: The Key to Performance

Type-stable functions are those where the compiler can infer the output type from input types. This enables optimal code generation.
"""

# ╔═╡ d3d47dde-4b9a-11ef-0123-d63e01783467
# Type UNSTABLE - return type depends on value
function unstable_function(x)
    if x > 0
        return 1.0  # Float64
    else
        return 1    # Int64
    end
end

# ╔═╡ e4e58eef-5cab-11ef-1234-e74f12894578
# Type STABLE - return type consistent
function stable_function(x)
    if x > 0
        return 1.0
    else
        return 0.0  # Always Float64
    end
end

# ╔═╡ f5f6a000-6dbc-11ef-2345-f85023905689
# Check type stability with @code_warntype
@code_warntype unstable_function(1.0)  # Shows type instability warnings

# ╔═╡ 0605d611-7ecd-11ef-3456-09623a016789
@code_warntype stable_function(1.0)  # Clean, no warnings

# ╔═╡ 1716e722-8fde-11ef-4567-1a78ab127890
# Performance impact
test_array = rand(1000)

# ╔═╡ 2827f833-90ef-11ef-5678-2b89cd238901
@btime [unstable_function(x) for x in $test_array];

# ╔═╡ 39390944-a1f0-11ef-6789-3c9aef349012
@btime [stable_function(x) for x in $test_array];

# ╔═╡ 4a4a1a55-b201-11ef-7890-4da5f045a123
md"""
### Container Type Annotations

Be careful with containers - specify types when needed for stability.
"""

# ╔═╡ 5b5b2666-c312-11ef-8901-5eb6f156b234
# Type-unstable container
function make_unstable_array()
    result = []  # Type is Vector{Any}
    push!(result, 1.0)
    push!(result, 2.0)
    return result
end

# ╔═╡ 6c6c3777-d423-11ef-9012-6fc72367c345
# Type-stable container
function make_stable_array()
    result = Float64[]  # Type is Vector{Float64}
    push!(result, 1.0)
    push!(result, 2.0)
    return result
end

# ╔═╡ 7d7d4888-e534-11ef-0123-70d8bc478456
@code_warntype make_unstable_array()

# ╔═╡ 8e7e5999-f645-11ef-1234-81e9cd589567
@code_warntype make_stable_array()

# ╔═╡ 9f8f6aaa-0756-11ef-2345-92fade69a678
md"""
## Function Specialization: Automatic Optimization

Julia automatically specializes functions for each combination of argument types, generating optimized code paths.
"""

# ╔═╡ a09f7bbb-1867-11ef-3456-a30bef7ab789
# Generic function
function process_data(x::T) where T
    result = zero(eltype(T))  # Type-appropriate zero
    for val in x
        result += val^2
    end
    return sqrt(result)
end

# ╔═╡ b1b08ccc-2978-11ef-4567-b41cf08bc890
# Specialized for Float64 arrays
float_data = rand(1000)

# ╔═╡ c2c19ddd-3a89-11ef-5678-c52df189cd9a
@btime process_data($float_data)

# ╔═╡ d3d2aeee-4b9a-11ef-6789-d63e02abde01
# Specialized for Int arrays
int_data = rand(ComplexF64, 1000)

# ╔═╡ e4e3bfff-5cab-11ef-7890-e74f23bcef12
@btime process_data($int_data)

# ╔═╡ f5f4d110-6dbc-11ef-8901-f8503cd0f023
# Each gets its own optimized version
methods(process_data)

# ╔═╡ 39373a54-a1f0-11ef-2345-3c9af0454689
md"""
## Performance Best Practices Summary

1. **Write functions** - The compiler optimizes functions, not global code
2. **Type stability** - Ensure output types are inferrable from input types
3. **Use concrete types in performance-critical code** - But keep interfaces generic
4. **Preallocate arrays** - Avoid growing arrays dynamically
5. **Use `@views` for slicing** - Avoid unnecessary copies
6. **Broadcast operations** - Use `.` for element-wise operations (they fuse!)
7. **Profile before optimizing** - Use `@btime` and `@profview`

## Key Differences from Other Languages

### From Python/NumPy:
- Loops are fast - use them freely
- 1-indexed arrays (not 0-indexed)
- Column-major order (not row-major)
- No GIL - true parallelism is possible

### From MATLAB:
- Free and open source
- Multiple dispatch instead of OOP
- Better performance without vectorization tricks
- True parallelism support

### From C/Fortran:
- Interactive development with REPL
- No separate compilation step
- Automatic memory management
- Rich ecosystem of packages

## Next Steps

You now have the foundational knowledge to write efficient Julia code for scientific computing. The key insights are:

1. Julia's JIT compilation means you write high-level code that runs fast
2. Type stability and specialization are your friends
3. Multiple dispatch enables elegant, extensible code
4. Put computational code in functions for best performance

As you continue, remember that Julia rewards writing clear, type-stable functions. The performance will follow naturally!
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"

[compat]
BenchmarkTools = "~1.8.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "4ceb3e6560d57a4c938e7aac1f17b6b66f7eab1b"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "PrecompileTools", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "9670d3febc2b6da60a0ae57846ba74670290653f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.8.0"

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
version = "1.3.0+1"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "88352712893ec50bee3680605891eaf0e9ed6368"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.8.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools"]
git-tree-sha1 = "663e8b48b789916221e0765393b289ca6c88f24e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "3.0.0"

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

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
deps = ["StyledStrings"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e2b53ce13a53367e96601081e33d34746b571bad"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.5"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

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

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"
"""

# ╔═╡ Cell order:
# ╟─b2c4d5f6-3a8b-11ef-0912-6d7e9c8a4b23
# ╠═a5d7b3e0-4e2a-11ef-1234-5b6c8d9a7f12
# ╟─c3d5e6f7-4b9c-11ef-2345-7c8d9e0a1b34
# ╠═d4e6f708-5cab-11ef-3456-8d9e0f1a2b45
# ╠═e5f70819-6dbc-11ef-4567-9e0f1a2b3c56
# ╠═f6081920-7ecd-11ef-5678-0f1a2b3c4d67
# ╠═07192a31-8fde-11ef-6789-1a2b3c4d5e78
# ╟─18203b42-90ef-11ef-7890-2b3c4d5e6f89
# ╠═29314c53-a1f0-11ef-8901-3c4d5e6f7089
# ╠═3a425d64-b201-11ef-9012-4d5e6f708190
# ╠═4b536e75-c312-11ef-0123-5e6f70819201
# ╠═5c647f86-d423-11ef-1234-6f70819202a3
# ╟─6d758097-e534-11ef-2345-708192030b14
# ╠═7e869108-f645-11ef-3456-8192030b1c25
# ╠═8f970219-0756-11ef-4567-92030b1c2d36
# ╠═a0a8132a-1867-11ef-5678-a30b1c2d3e47
# ╠═b1b9243b-2978-11ef-6789-b41c2d3e4f58
# ╟─c2ca354c-3a89-11ef-7890-c52d3e4f5069
# ╠═d3db465d-4b9a-11ef-8901-d63e4f506170
# ╠═e4ec576e-5cab-11ef-9012-e74f50617281
# ╠═f5fd6880-6dbc-11ef-0123-f850617283a4
# ╠═06097991-7ecd-11ef-1234-0961728394a5
# ╠═171a8a02-8fde-11ef-2345-1a728394b5c6
# ╠═282b9b13-90ef-11ef-3456-2b8394c5d6e7
# ╟─393cac24-a1f0-11ef-4567-3c9495d6e7f8
# ╠═4a4dbd35-b201-11ef-5678-4da506e7f809
# ╠═5b5ece46-c312-11ef-6789-5eb617f80910
# ╠═6c6fdf57-d423-11ef-7890-6fc7280901a1
# ╠═7d70e068-e534-11ef-8901-70d8391012b3
# ╠═8e81f179-f645-11ef-9012-81e9401123c4
# ╠═9f92028a-0756-11ef-0123-92fa512234d5
# ╠═a0a3139b-1867-11ef-1234-a30b623345e6
# ╠═b1b424ac-2978-11ef-2345-b41c734456f7
# ╠═c2c535bd-3a89-11ef-3456-c52d845567f8
# ╟─d3d646ce-4b9a-11ef-4567-d63e956678f9
# ╠═e4e757df-5cab-11ef-5678-e74f067789fa
# ╠═f5f868f0-6dbc-11ef-6789-f8506789a0fb
# ╠═0607a001-7ecd-11ef-7890-096189ab1c2d
# ╟─1718b112-8fde-11ef-8901-1a789abc2d3e
# ╠═b0e46118-d9be-47f3-9060-715c630d9a3a
# ╠═4a4be445-b201-11ef-1234-4da5ef567189
# ╠═5b5cf556-c312-11ef-2345-5eb6f06789ab
# ╟─6c6e0667-d423-11ef-3456-6fc7189abcd0
# ╠═7d7f1778-e534-11ef-4567-70d8ab12de01
# ╠═8e802889-f645-11ef-5678-81e9bc23ef12
# ╠═9f90399a-0756-11ef-6789-92facd34f023
# ╠═a0a14aab-1867-11ef-7890-a30bde450134
# ╠═b1b25bbc-2978-11ef-8901-b41cef561245
# ╟─c2c36ccd-3a89-11ef-9012-c52df0672356
# ╠═d3d47dde-4b9a-11ef-0123-d63e01783467
# ╠═e4e58eef-5cab-11ef-1234-e74f12894578
# ╠═f5f6a000-6dbc-11ef-2345-f85023905689
# ╠═0605d611-7ecd-11ef-3456-09623a016789
# ╠═1716e722-8fde-11ef-4567-1a78ab127890
# ╠═2827f833-90ef-11ef-5678-2b89cd238901
# ╠═39390944-a1f0-11ef-6789-3c9aef349012
# ╟─4a4a1a55-b201-11ef-7890-4da5f045a123
# ╠═5b5b2666-c312-11ef-8901-5eb6f156b234
# ╠═6c6c3777-d423-11ef-9012-6fc72367c345
# ╠═7d7d4888-e534-11ef-0123-70d8bc478456
# ╠═8e7e5999-f645-11ef-1234-81e9cd589567
# ╟─9f8f6aaa-0756-11ef-2345-92fade69a678
# ╠═a09f7bbb-1867-11ef-3456-a30bef7ab789
# ╠═b1b08ccc-2978-11ef-4567-b41cf08bc890
# ╠═c2c19ddd-3a89-11ef-5678-c52df189cd9a
# ╠═d3d2aeee-4b9a-11ef-6789-d63e02abde01
# ╠═e4e3bfff-5cab-11ef-7890-e74f23bcef12
# ╠═f5f4d110-6dbc-11ef-8901-f8503cd0f023
# ╟─39373a54-a1f0-11ef-2345-3c9af0454689
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
