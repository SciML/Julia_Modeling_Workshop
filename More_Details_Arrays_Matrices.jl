### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000001
using PlutoUI

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000037
using LinearAlgebra

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000038
using SparseArrays

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000039
using BenchmarkTools

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000040
using StaticArrays

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000042
A_test_perf = rand(1000, 1000)

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000008
x_vec_broadcast = [1, 2, 3]

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000043
y_vec_broadcast = [4, 5, 6]'  # Row vector (1×3)

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000044
broadcast_result = x_vec_broadcast .+ y_vec_broadcast

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000010
function unfused_operations(x, y, z)
    tmp1 = x .+ y
    tmp2 = tmp1 .* z
    return sin.(tmp2)
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000045
function fused_operations(x, y, z)
    return sin.((x .+ y) .* z)
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000046
function fused_with_macro(x, y, z)
    @. sin((x + y) * z)
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000047
test_x_fusion = rand(1000)

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000048
test_y_fusion = rand(1000)

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000049
test_z_fusion = rand(1000)

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000011
with_terminal() do
    println("Unfused operations:")
    @btime unfused_operations($test_x_fusion, $test_y_fusion, $test_z_fusion)
    println("\nFused operations:")
    @btime fused_operations($test_x_fusion, $test_y_fusion, $test_z_fusion)
    println("\nFused with @. macro:")
    @btime fused_with_macro($test_x_fusion, $test_y_fusion, $test_z_fusion)
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000013
M1_linalg = rand(100, 100)

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000050
M2_linalg = rand(100, 100)

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000051
v_linalg = rand(100)

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000016
A_sym_decomp = let
    tmp = rand(5, 5)
    tmp + tmp'  # Make symmetric
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000053
lu_fact_decomp = lu(A_sym_decomp)

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000054
eigen_fact_decomp = eigen(A_sym_decomp)

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000055
qr_fact_decomp = qr(A_sym_decomp)

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000018
n_sparse = 1000

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000057
sparse_mat_example = spdiagm(-1 => ones(n_sparse-1), 0 => -2*ones(n_sparse), 1 => ones(n_sparse-1))

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000058
dense_mat_comparison = Matrix(sparse_mat_example)

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000059
vec_sparse_mult = rand(n_sparse)

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000019
with_terminal() do
    println("Memory usage:")
    println("Sparse: ", Base.summarysize(sparse_mat_example), " bytes")
    println("Dense: ", Base.summarysize(dense_mat_comparison), " bytes")

    println("\nMatrix-vector multiplication:")
    println("Sparse:")
    @btime $sparse_mat_example * $vec_sparse_mult
    println("Dense:")
    @btime $dense_mat_comparison * $vec_sparse_mult
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000021
I_idx_sparse = [1, 2, 3, 4, 5]

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000060
J_idx_sparse = [1, 2, 3, 4, 5]

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000061
V_vals_sparse = [1.0, 2.0, 3.0, 4.0, 5.0]

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000062
S1_sparse = sparse(I_idx_sparse, J_idx_sparse, V_vals_sparse, 5, 5)

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000063
S2_sparse = let
    S2_temp = spzeros(5, 5)
    for (i, j, v) in zip(I_idx_sparse, J_idx_sparse, V_vals_sparse)
        S2_temp[i, j] = v
    end
    S2_temp
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000023
regular_vec_static = [1.0, 2.0, 3.0]

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000064
static_vec_example = SVector(1.0, 2.0, 3.0)

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000065
regular_mat_static = [1.0 2.0; 3.0 4.0]

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000066
static_mat_example = SMatrix{2,2}(1.0, 3.0, 2.0, 4.0)

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000067
function sum_regular(x, y)
    return x + y
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000068
function sum_static(x::SVector, y::SVector)
    return x + y
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000024
with_terminal() do
    println("Regular array addition:")
    @btime sum_regular($regular_vec_static, $regular_vec_static)
    println("\nStatic array addition:")
    @btime sum_static($static_vec_example, $static_vec_example)
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000026
function rotate_3d(θ)
    return SMatrix{3,3}(
        cos(θ), -sin(θ), 0,
        sin(θ), cos(θ), 0,
        0, 0, 1
    )
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000069
point_3d_example = SVector(1.0, 0.0, 0.0)

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000070
rotated_point = rotate_3d(π/4) * point_3d_example

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000028
large_array_views = rand(1000, 1000)

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000071
function process_with_copy(A)
    sub = A[1:100, 1:100]  # Creates a copy
    return sum(sub)
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000072
function process_with_view(A)
    sub = @view A[1:100, 1:100]  # Creates a view
    return sum(sub)
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000029
with_terminal() do
    println("Processing with copy:")
    @btime process_with_copy($large_array_views)
    println("\nProcessing with view:")
    @btime process_with_view($large_array_views)
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000031
vec_data_reshape = collect(1:12)

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000073
matrix_view_reshape = reshape(vec_data_reshape, 3, 4)

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000074
int_array_reinterpret = Int32[1, 2, 3, 4]

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000075
byte_view_reinterpret = reinterpret(UInt8, int_array_reinterpret)

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000033
struct CircularArray{T} <: AbstractVector{T}
    data::Vector{T}
    start::Ref{Int}

    CircularArray{T}(n::Int) where T = new(Vector{T}(undef, n), Ref(1))
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000077
Base.size(c::CircularArray) = size(c.data)

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000004
function col_major_sum(A)
    s = 0.0
    for j in 1:size(A,2)
        for i in 1:size(A,1)
            s += A[i,j]
        end
    end
    return s
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000041
function row_major_sum(A)
    s = 0.0
    for i in 1:size(A,1)
        for j in 1:size(A,2)
            s += A[i,j]
        end
    end
    return s
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000006
with_terminal() do
    println("Column-major iteration:")
    @btime col_major_sum($A_test_perf)
    println("\nRow-major iteration:")
    @btime row_major_sum($A_test_perf)
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000052
function manual_matmul(A, B)
    C = zeros(size(A,1), size(B,2))
    for i in 1:size(A,1)
        for j in 1:size(B,2)
            for k in 1:size(A,2)
                C[i,j] += A[i,k] * B[k,j]
            end
        end
    end
    return C
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000014
with_terminal() do
    println("Manual matrix multiplication:")
    @btime manual_matmul($M1_linalg, $M2_linalg)
    println("\nBuilt-in matrix multiplication:")
    @btime $M1_linalg * $M2_linalg
    println("\nIn-place multiplication with mul!:")
    C_temp = similar(M1_linalg)
    @btime mul!($C_temp, $M1_linalg, $M2_linalg)
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000078
Base.getindex(c::CircularArray, i::Int) = c.data[mod1(c.start[] + i - 1, length(c.data))]

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000002
md"""
# More Details on Arrays and Matrices

This optional notebook provides deeper insights into working with arrays and matrices in Julia, covering advanced topics for performance optimization and specialized array types.

## Table of Contents
1. Array Memory Layout and Performance
2. Broadcasting and Vectorization
3. Linear Algebra Operations
4. Sparse Matrices
5. Static Arrays for Performance
6. Views and Memory Efficiency
7. Custom Array Types
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000003
md"""
## 1. Array Memory Layout and Performance

Julia arrays are stored in column-major order (like Fortran), which affects performance when iterating.
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000005
md"""
Compare performance of column-major vs row-major access:
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000007
md"""
## 2. Broadcasting and Vectorization

Broadcasting allows element-wise operations on arrays of different shapes efficiently.
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000009
md"""
### Fusion of Broadcast Operations

Julia can fuse multiple broadcast operations into a single loop for efficiency:
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000012
md"""
## 3. Linear Algebra Operations

Julia provides optimized BLAS operations for linear algebra:
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000015
md"""
### Common Linear Algebra Operations
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000056
md"""
- LU decomposition: $(lu_fact_decomp.L[1:3,1:3])
- Eigenvalues: $(round.(eigen_fact_decomp.values, digits=3))
- QR Q matrix: $(round.(qr_fact_decomp.Q[1:3,1:3], digits=3))
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000017
md"""
## 4. Sparse Matrices

For matrices with mostly zeros, sparse representations save memory and computation:
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000020
md"""
### Sparse Matrix Construction Patterns
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000022
md"""
## 5. Static Arrays for Performance

StaticArrays provide stack-allocated arrays with compile-time known sizes:
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000025
md"""
### When to Use Static Arrays

Static arrays are beneficial when:
- Array size is small (typically < 100 elements)
- Size is known at compile time
- Performance is critical
- You want to avoid heap allocations
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000027
md"""
## 6. Views and Memory Efficiency

Views provide a way to work with subarrays without copying data:
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000030
md"""
### Reshape and Reinterpret

Reshape and reinterpret provide zero-copy ways to change array structure:
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000076
md"""
Original vector: $vec_data_reshape

Reshaped to 3×4: $matrix_view_reshape

Int32 array: $int_array_reinterpret

Reinterpreted as bytes: $byte_view_reinterpret
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000032
md"""
## 7. Custom Array Types

Julia allows defining custom array types for specialized behavior:
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000079
function Base.setindex!(c::CircularArray, v, i::Int)
    c.data[mod1(c.start[] + i - 1, length(c.data))] = v
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000080
function rotate!(c::CircularArray, n::Int)
    c.start[] = mod1(c.start[] + n, length(c.data))
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000081
circ_array_example = let
    circ_tmp = CircularArray{Int}(5)
    for i in 1:5
        circ_tmp[i] = i
    end
    rotate!(circ_tmp, 2)
    circ_tmp[1]
end

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000082
md"""
Circular array example:
- Initial: [1, 2, 3, 4, 5]
- After rotating by 2, accessing element 1 gives: $circ_array_example
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000034
md"""
## Performance Tips Summary

1. **Memory Layout**: Iterate in column-major order for best cache performance
2. **Broadcasting**: Use `.` operators and `@.` macro for fused operations
3. **Preallocate**: Use `similar()` or preallocate arrays when possible
4. **In-place Operations**: Use functions ending in `!` (like `mul!`) to avoid allocations
5. **Static Arrays**: Use for small, fixed-size arrays in performance-critical code
6. **Views**: Use `@view` to avoid unnecessary copies
7. **Sparse Matrices**: Use for matrices with many zeros
8. **Type Stability**: Ensure functions return consistent types

## Benchmarking Best Practices

Always use `@btime` with interpolation (`$`) for accurate benchmarks:
```julia
# Good
@btime sum($my_array)

# Bad (includes compilation time)
@btime sum(my_array)
```
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000035
md"""
## Exercises

1. **Memory Layout**: Write a function that transposes a matrix efficiently by reading in column-major order

2. **Broadcasting**: Implement a normalized exponential function using broadcasting:
   - Take a vector `x`
   - Compute `exp.(x) ./ sum(exp.(x))`
   - Compare with a loop-based version

3. **Sparse Matrices**: Create a sparse representation of a finite difference operator for the 1D heat equation

4. **Static Arrays**: Implement a particle simulation where each particle has position and velocity as `SVector{3}`

5. **Views**: Process a large image by dividing it into tiles using views, applying a filter to each tile
"""

# ╔═╡ 1a2b3c4d-0000-0000-0000-000000000036
md"""
## Additional Resources

- [Julia Arrays Documentation](https://docs.julialang.org/en/v1/manual/arrays/)
- [Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/)
- [LinearAlgebra Standard Library](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/)
- [SparseArrays Documentation](https://docs.julialang.org/en/v1/stdlib/SparseArrays/)
- [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl)
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[compat]
BenchmarkTools = "~1.8.0"
PlutoUI = "~0.7.83"
StaticArrays = "~1.9.20"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.7"
manifest_format = "2.0"
project_hash = "568ab421671819e11ef08dcfe632eac780dec046"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "e71ee7b4aa06b045259a7d6101e1cb45ad140bce"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "PrecompileTools", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "9670d3febc2b6da60a0ae57846ba74670290653f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.8.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

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

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
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

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

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

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.6+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools"]
git-tree-sha1 = "663e8b48b789916221e0765393b289ca6c88f24e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "3.0.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

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

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "e206cf4850fd7ac4255ffd2b98922f563e18ac53"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.20"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

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

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "908fec9df6c5de98548ead82a468c95ccf6cd263"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.7.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"
"""

# ╔═╡ Cell order:
# ╟─1a2b3c4d-0000-0000-0000-000000000002
# ╠═1a2b3c4d-0000-0000-0000-000000000001
# ╠═1a2b3c4d-0000-0000-0000-000000000037
# ╠═1a2b3c4d-0000-0000-0000-000000000038
# ╠═1a2b3c4d-0000-0000-0000-000000000039
# ╠═1a2b3c4d-0000-0000-0000-000000000040
# ╟─1a2b3c4d-0000-0000-0000-000000000003
# ╠═1a2b3c4d-0000-0000-0000-000000000004
# ╠═1a2b3c4d-0000-0000-0000-000000000041
# ╠═1a2b3c4d-0000-0000-0000-000000000042
# ╟─1a2b3c4d-0000-0000-0000-000000000005
# ╠═1a2b3c4d-0000-0000-0000-000000000006
# ╟─1a2b3c4d-0000-0000-0000-000000000007
# ╠═1a2b3c4d-0000-0000-0000-000000000008
# ╠═1a2b3c4d-0000-0000-0000-000000000043
# ╠═1a2b3c4d-0000-0000-0000-000000000044
# ╟─1a2b3c4d-0000-0000-0000-000000000009
# ╠═1a2b3c4d-0000-0000-0000-000000000010
# ╠═1a2b3c4d-0000-0000-0000-000000000045
# ╠═1a2b3c4d-0000-0000-0000-000000000046
# ╠═1a2b3c4d-0000-0000-0000-000000000047
# ╠═1a2b3c4d-0000-0000-0000-000000000048
# ╠═1a2b3c4d-0000-0000-0000-000000000049
# ╠═1a2b3c4d-0000-0000-0000-000000000011
# ╟─1a2b3c4d-0000-0000-0000-000000000012
# ╠═1a2b3c4d-0000-0000-0000-000000000013
# ╠═1a2b3c4d-0000-0000-0000-000000000050
# ╠═1a2b3c4d-0000-0000-0000-000000000051
# ╠═1a2b3c4d-0000-0000-0000-000000000052
# ╠═1a2b3c4d-0000-0000-0000-000000000014
# ╟─1a2b3c4d-0000-0000-0000-000000000015
# ╠═1a2b3c4d-0000-0000-0000-000000000016
# ╠═1a2b3c4d-0000-0000-0000-000000000053
# ╠═1a2b3c4d-0000-0000-0000-000000000054
# ╠═1a2b3c4d-0000-0000-0000-000000000055
# ╠═1a2b3c4d-0000-0000-0000-000000000056
# ╟─1a2b3c4d-0000-0000-0000-000000000017
# ╠═1a2b3c4d-0000-0000-0000-000000000018
# ╠═1a2b3c4d-0000-0000-0000-000000000057
# ╠═1a2b3c4d-0000-0000-0000-000000000058
# ╠═1a2b3c4d-0000-0000-0000-000000000059
# ╠═1a2b3c4d-0000-0000-0000-000000000019
# ╟─1a2b3c4d-0000-0000-0000-000000000020
# ╠═1a2b3c4d-0000-0000-0000-000000000021
# ╠═1a2b3c4d-0000-0000-0000-000000000060
# ╠═1a2b3c4d-0000-0000-0000-000000000061
# ╠═1a2b3c4d-0000-0000-0000-000000000062
# ╠═1a2b3c4d-0000-0000-0000-000000000063
# ╟─1a2b3c4d-0000-0000-0000-000000000022
# ╠═1a2b3c4d-0000-0000-0000-000000000023
# ╠═1a2b3c4d-0000-0000-0000-000000000064
# ╠═1a2b3c4d-0000-0000-0000-000000000065
# ╠═1a2b3c4d-0000-0000-0000-000000000066
# ╠═1a2b3c4d-0000-0000-0000-000000000067
# ╠═1a2b3c4d-0000-0000-0000-000000000068
# ╠═1a2b3c4d-0000-0000-0000-000000000024
# ╟─1a2b3c4d-0000-0000-0000-000000000025
# ╠═1a2b3c4d-0000-0000-0000-000000000026
# ╠═1a2b3c4d-0000-0000-0000-000000000069
# ╠═1a2b3c4d-0000-0000-0000-000000000070
# ╟─1a2b3c4d-0000-0000-0000-000000000027
# ╠═1a2b3c4d-0000-0000-0000-000000000028
# ╠═1a2b3c4d-0000-0000-0000-000000000071
# ╠═1a2b3c4d-0000-0000-0000-000000000072
# ╠═1a2b3c4d-0000-0000-0000-000000000029
# ╟─1a2b3c4d-0000-0000-0000-000000000030
# ╠═1a2b3c4d-0000-0000-0000-000000000031
# ╠═1a2b3c4d-0000-0000-0000-000000000073
# ╠═1a2b3c4d-0000-0000-0000-000000000074
# ╠═1a2b3c4d-0000-0000-0000-000000000075
# ╠═1a2b3c4d-0000-0000-0000-000000000076
# ╟─1a2b3c4d-0000-0000-0000-000000000032
# ╠═1a2b3c4d-0000-0000-0000-000000000033
# ╠═1a2b3c4d-0000-0000-0000-000000000077
# ╠═1a2b3c4d-0000-0000-0000-000000000078
# ╠═1a2b3c4d-0000-0000-0000-000000000079
# ╠═1a2b3c4d-0000-0000-0000-000000000080
# ╠═1a2b3c4d-0000-0000-0000-000000000081
# ╠═1a2b3c4d-0000-0000-0000-000000000082
# ╟─1a2b3c4d-0000-0000-0000-000000000034
# ╟─1a2b3c4d-0000-0000-0000-000000000035
# ╟─1a2b3c4d-0000-0000-0000-000000000036
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
