using QuantumToolbox
using LinearAlgebra
using LinearSolve
using SparseArrays
using OffsetArrays
using MKL
using IncompleteLU

function steadystate_fourier_2d(
    H_0::QuantumObject{OpType1},
    H_p1_00::QuantumObject{OpType2},
    H_m1_00::QuantumObject{OpType3},
    H_p1_p1::QuantumObject{OpType4},
    H_p1_m1::QuantumObject{OpType5},
    H_m1_p1::QuantumObject{OpType6},
    H_m1_m1::QuantumObject{OpType7},
    ωd1::Number,
    ωd2::Number,
    c_ops::Union{Nothing,AbstractVector,Tuple}=nothing;
    n_max1::Integer=2,
    n_max2::Integer=2,
    tol::R=1e-8,
    kwargs...,
) where {
    OpType1<:Union{Operator,SuperOperator},
    OpType2<:Union{Operator,SuperOperator},
    OpType3<:Union{Operator,SuperOperator},
    OpType4<:Union{Operator,SuperOperator},
    OpType5<:Union{Operator,SuperOperator},
    OpType6<:Union{Operator,SuperOperator},
    OpType7<:Union{Operator,SuperOperator},
    R<:Real,
}
    L_0 = liouvillian(H_0, c_ops)
    L_p1_00 = liouvillian(H_p1_00)
    L_m1_00 = liouvillian(H_m1_00)
    L_p1_p1 = liouvillian(H_p1_p1)
    L_p1_m1 = liouvillian(H_p1_m1)
    L_m1_p1 = liouvillian(H_m1_p1)
    L_m1_m1 = liouvillian(H_m1_m1)

    return _steadystate_fourier_2d(L_0, L_p1_00, L_m1_00, L_p1_p1, L_p1_m1, L_m1_p1, L_m1_m1, ωd1, ωd2; n_max1=n_max1, n_max2=n_max2, tol=tol, kwargs...)
end

function _steadystate_fourier_2d(
    L_0::QuantumObject{SuperOperator},
    L_p1_00::QuantumObject{SuperOperator},
    L_m1_00::QuantumObject{SuperOperator},
    L_p1_p1::QuantumObject{SuperOperator},
    L_p1_m1::QuantumObject{SuperOperator},
    L_m1_p1::QuantumObject{SuperOperator},
    L_m1_m1::QuantumObject{SuperOperator},
    ωd1::Number,
    ωd2::Number;
    n_max1::Integer=1,
    n_max2::Integer=1,
    tol::R=1e-8,
    kwargs...,
) where {R<:Real}

    T1 = eltype(L_0)
    T2 = eltype(L_p1_00)
    T3 = eltype(L_m1_00)
    T4 = eltype(L_p1_p1)
    T5 = eltype(L_p1_m1)
    T6 = eltype(L_m1_p1)
    T7 = eltype(L_m1_m1)
    T = promote_type(T1, T2, T3, T4, T5, T6, T7)

    L_0_mat = get_data(L_0)
    L_p1_00_mat = get_data(L_p1_00)
    L_m1_00_mat = get_data(L_m1_00)
    L_p1_p1_mat = get_data(L_p1_p1)
    L_p1_m1_mat = get_data(L_p1_m1)
    L_m1_p1_mat = get_data(L_m1_p1)
    L_m1_m1_mat = get_data(L_m1_m1)

    N = size(L_0_mat, 1) # dimension of the density matrix
    Ns = isqrt(N) # dimension of the Hilbert space
    n_fourier_1 = 2 * n_max1 + 1 # number of Fourier coefficients for the first dimension
    n_fourier_2 = 2 * n_max2 + 1 # number of Fourier coefficients for the second dimension
    n_fourier = n_fourier_1 * n_fourier_2 # number of Fourier coefficients
    n_list_1 = -n_max1:n_max1
    n_list_2 = -n_max2:n_max2

    weight = 1
    Mn = sparse(ones(Ns), [Ns * (j - 1) + j for j in 1:Ns], fill(weight, Ns), N, N)
    # population condition
    L = L_0_mat + Mn

    M = spzeros(T, n_fourier * N, n_fourier * N)

    M += kron(
        kron(spdiagm(+1 => ones(n_fourier_2 - 1)), spdiagm(-1 => ones(n_fourier_1 - 1))),
        L_p1_m1_mat,
    )
    M += kron(
        kron(spdiagm(-1 => ones(n_fourier_2 - 1)), spdiagm(+1 => ones(n_fourier_1 - 1))),
        L_m1_p1_mat,
    )

    M += kron(
        kron(spdiagm(+1 => ones(n_fourier_2 - 1)), spdiagm(+1 => ones(n_fourier_1 - 1))),
        L_m1_m1_mat,
    )
    M += kron(
        kron(spdiagm(-1 => ones(n_fourier_2 - 1)), spdiagm(-1 => ones(n_fourier_1 - 1))),
        L_p1_p1_mat,
    )
    M += kron(
        kron(Matrix(I, n_fourier_2, n_fourier_2), spdiagm(+1 => ones(n_fourier_1 - 1))),
        L_m1_00_mat,
    )
    M += kron(
        kron(Matrix(I, n_fourier_2, n_fourier_2), spdiagm(-1 => ones(n_fourier_1 - 1))),
        L_p1_00_mat,
    )
    # julia> kron(spdiagm(1 => ones(5)), Matrix(I, 6, 6))
    # 36×36 SparseMatrixCSC{Float64, Int64} with 30 stored entries:
    # ⎡⠀⠀⠀⠑⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎤
    # ⎢⠀⠀⠀⠀⠀⠑⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
    # ⎢⠀⠀⠀⠀⠀⠀⠀⠑⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
    # ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⢄⠀⠀⠀⠀⠀⠀⠀⎥
    # ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⢄⠀⠀⠀⠀⠀⎥
    # ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⢄⠀⠀⠀⎥
    # ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⢄⠀⎥
    # ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⎥
    # ⎣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎦
    # second one is small cycle, first one is big cycle
    # 第二个是L的第一指标，第一个是L的第二指标，以此类推

    for i in 1:n_fourier_1
        for j in 1:n_fourier_2
            n1 = n_list_1[i]
            n2 = n_list_2[j]
            index = (j - 1) * n_fourier_1 + i
            M += kron(
                sparse([index], [index], one(T), n_fourier, n_fourier),
                L - 1im * ωd1 * n1 * Matrix(I, size(L)) - 1im * ωd2 * n2 * Matrix(I, size(L)),
            )
        end
    end

    v0 = zeros(T, n_fourier * N)
    v0[((n_max2+1-1)*n_fourier_1+n_max1+1-1)*N+1] = weight

    # ρtot = M \ v0
    prob = LinearProblem(M, v0)
    sol = solve(prob, KrylovJL_GMRES(), Pl=IncompleteLU.ilu(M))
    ρtot = sol.u

    offset1 = ((n_max2 + 1 - 1) * n_fourier_1 + n_max1 + 1 - 1) * N
    offset2 = offset1 + N
    ρ0 = reshape(ρtot[(offset1+1):offset2], Ns, Ns)
    # 把flatten修复回矩阵
    ρ0_tr = tr(ρ0)
    ρ0 = ρ0 / ρ0_tr
    ρ0 = QuantumObject((ρ0 + ρ0') / 2, type=Operator(), dims=L_0.dimensions)
    ρtot = ρtot / ρ0_tr

    ρ_matrix = Matrix{typeof(ρ0)}(undef, n_max1 * 2 + 1, n_max2 * 2 + 1)
    ρ_matrix = OffsetArray(ρ_matrix, -n_max1:n_max1, -n_max2:n_max2)
    ρ_matrix[0, 0] = ρ0

    for i in -n_max1:n_max1
        for j in -n_max2:n_max2
            i == j == 0 && continue
            offset1 = ((n_max2 + 1 - 1 + j) * n_fourier_1 + n_max1 + 1 - 1 + i) * N
            offset2 = offset1 + N
            ρ_matrix[i, j] = QuantumObject(reshape(ρtot[(offset1+1):offset2], Ns, Ns), type=Operator(), dims=L_0.dimensions)
        end
    end
    # ρ_matrix[0, 0] e^{0 * i * ωd1 * t} e^{0 * i * ωd2 * t}
    # ρ_matrix[1, 0] e^{1 * i * ωd1 * t} e^{0 * i * ωd2 * t}
    # ρ_matrix[0, 1] e^{0 * i * ωd1 * t} e^{1 * i * ωd2 * t}
    return ρ_matrix
end