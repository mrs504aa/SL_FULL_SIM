using CairoMakie
using QuantumToolbox
using WignerSymbols
using HDF5
using ProgressMeter
using JLD2
using RationalRoots
include("Floquet_2D_Solver_II.jl")

Base.@kwdef mutable struct HamiltonianParas
    Omega_p::Float64 = 0.01
    Delta_p::Float64 = 5.0
    theta_p = 1 / 2 * pi

    Omega_cf::Float64 = 20.0 / sqrt(3 / 10)
    Omega_cb::Float64 = 0.0 / sqrt(3 / 10)
    Delta_c::Float64 = -0.0
    theta_c = 0 / 2 * pi

    Omega_b::Float64 = 0.1
    Delta_b::Float64 = 0.0
    theta_b = 0 / 2 * pi
end

#P D 11 12 13 14 15
#P B    08 09 10
#S A 03 04 05 06 07
#S C    00 01 02

j_list = fill(1 // 2, 16)
i_list = fill(3 // 2, 16)
f_list = Vector{Any}(undef, 16)
m_list = Vector{Any}(undef, 16)
for i in 1:3
    f_list[i] = 1
    m_list[i] = i - 2
end
for i in 4:8
    f_list[i] = 2
    m_list[i] = i - 6
end
for i in 9:11
    f_list[i] = 1
    m_list[i] = i - 10
end
for i in 12:16
    f_list[i] = 2
    m_list[i] = i - 14
end

function transition(a::Int64, b::Int64)
    return basis(16, a) * basis(16, b)'
end

function cg_c(a::Int64, b::Int64)
    if a < b
        a, b = b, a
    end
    f1 = f_list[b+1]
    f2 = f_list[a+1]
    m1 = m_list[b+1]
    m2 = m_list[a+1]
    abs(m2 - m1) < 2 || return 0
    return clebschgordan(f1, m1, 1, m2 - m1, f2, m2)
end

function dipole_jtf_c(a::Int64, b::Int64)
    if a < b
        a, b = b, a
    end
    F = f_list[b+1]
    F1 = f_list[a+1]
    J = j_list[b+1]
    J1 = j_list[a+1]
    I1 = i_list[b+1]
    s = signedroot((2 * F1 + 1) * (2 * J + 1))
    s *= -2 * isodd(F1 + J + 2 + I1) + 1
    s *= wigner6j(J, J1, 1, F1, F, I1)
    return s
end

function Jumping_Operators(P::HamiltonianParas)
    J_Set = Any[]
    sqrtGamma = sqrt(5.746)
    c22 = 0.00
    c11 = 0.00
    c12 = 0.1 * sqrt(5)
    c21 = 0.1 * sqrt(3)

    J = 0.0 * transition(0, 0)
    for i in 11:15
        J += sqrtGamma * transition(i - 8, i) * abs(cg_c(i - 8, i) * dipole_jtf_c(i - 8, i))
    end
    push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 11:14
        J += sqrtGamma * transition(i - 7, i) * abs(cg_c(i - 7, i) * dipole_jtf_c(i - 7, i))
    end
    push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 12:15
        J += sqrtGamma * transition(i - 9, i) * abs(cg_c(i - 9, i) * dipole_jtf_c(i - 9, i))
    end
    push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 12:14
        J += sqrtGamma * transition(i - 12, i) * abs(cg_c(i - 12, i) * dipole_jtf_c(i - 12, i))
    end
    push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 11:13
        J += sqrtGamma * transition(i - 11, i) * abs(cg_c(i - 11, i) * dipole_jtf_c(i - 11, i))
    end
    push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 13:15
        J += sqrtGamma * transition(i - 13, i) * abs(cg_c(i - 13, i) * dipole_jtf_c(i - 13, i))
    end
    push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 8:10
        J += sqrtGamma * transition(i - 8, i) * abs(cg_c(i - 8, i) * dipole_jtf_c(i - 8, i))
    end
    push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 8:9
        J += sqrtGamma * transition(i - 7, i) * abs(cg_c(i - 7, i) * dipole_jtf_c(i - 7, i))
    end
    push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 9:10
        J += sqrtGamma * transition(i - 9, i) * abs(cg_c(i - 9, i) * dipole_jtf_c(i - 9, i))
    end
    push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 8:10
        J += sqrtGamma * transition(i - 4, i) * abs(cg_c(i - 4, i) * dipole_jtf_c(i - 4, i))
    end
    push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 8:10
        J += sqrtGamma * transition(i - 5, i) * abs(cg_c(i - 5, i) * dipole_jtf_c(i - 5, i))
    end
    push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 8:10
        J += sqrtGamma * transition(i - 3, i) * abs(cg_c(i - 3, i) * dipole_jtf_c(i - 3, i))
    end
    push!(J_Set, J)

    # collision terms

    for i in 3:7
        for j in 3:7
            J = c22 * transition(i, j) * (i != j)
            push!(J_Set, J)
        end
    end

    for i in 0:2
        for j in 0:2
            J = c11 * transition(i, j) * (i != j)
            push!(J_Set, J)
        end
    end

    for i in 0:2
        for j in 3:7
            J = c12 * transition(i, j)
            push!(J_Set, J)
            J = c21 * transition(j, i)
            push!(J_Set, J)
        end
    end

    return J_Set
end

function Type_II_Hamiltonian(P::HamiltonianParas)
    H_0 = 0.0 * transition(0, 0)

    for i in 3:7
        H_0 += +P.Delta_c * transition(i, i)
    end
    for i in 11:15
        H_0 += -P.Delta_b * transition(i, i)
    end

    H_p1 = 0.0 * transition(0, 0)

    for i in 4:6
        H_p1 += P.Omega_cf * transition(i + 4, i) * cg_c(i + 4, i)
        H_p1 += P.Omega_cb * transition(i, i + 4) * cg_c(i, i + 4)
    end
    for i in 0:2
        H_p1 += P.Omega_b * cos(P.theta_b) * transition(i + 12, i) * cg_c(i + 12, i)
        H_p1 += P.Omega_b * sin(P.theta_b) / sqrt(2) * transition(i + 11, i) * cg_c(i + 11, i)
        H_p1 += P.Omega_b * sin(P.theta_b) / sqrt(2) * transition(i + 13, i) * cg_c(i + 13, i)
    end

    H_m1 = 0.0 * transition(0, 0)

    for i in 4:6
        H_m1 += P.Omega_cf * transition(i, i + 4) * cg_c(i, i + 4)
        H_m1 += P.Omega_cb * transition(i + 4, i) * cg_c(i + 4, i)
    end
    for i in 0:2
        H_m1 += P.Omega_b * cos(P.theta_b) * transition(i, i + 12) * cg_c(i, i + 12)
        H_m1 += P.Omega_b * sin(P.theta_b) / sqrt(2) * transition(i, i + 11) * cg_c(i, i + 11)
        H_m1 += P.Omega_b * sin(P.theta_b) / sqrt(2) * transition(i, i + 13) * cg_c(i, i + 13)
    end

    H_p12 = 0.0 * transition(0, 0)

    for i in 4:6
        H_p12 += P.Omega_p * cos(P.theta_p) * transition(i + 4, i) * cg_c(i + 4, i)
    end
    for i in 3:5
        H_p12 += P.Omega_p * sin(P.theta_p) / sqrt(2) * transition(i + 5, i) * cg_c(i + 5, i)
    end
    for i in 5:7
        H_p12 += P.Omega_p * sin(P.theta_p) / sqrt(2) * transition(i + 3, i) * cg_c(i + 3, i)
    end

    H_m12 = 0.0 * transition(0, 0)

    for i in 4:6
        H_m12 += P.Omega_p * cos(P.theta_p) * transition(i, i + 4) * cg_c(i, i + 4)
    end
    for i in 3:5
        H_m12 += P.Omega_p * sin(P.theta_p) / sqrt(2) * transition(i, i + 5) * cg_c(i, i + 5)
    end
    for i in 5:7
        H_m12 += P.Omega_p * sin(P.theta_p) / sqrt(2) * transition(i, i + 3) * cg_c(i, i + 3)
    end

    return H_0, H_p1, H_m1, H_p12, H_m12
end

function Type_II_Absorption_Operators(P::HamiltonianParas)
    Ops = fill(0.0 * transition(0, 0), 3)

    for i in 8:10
        Ops[1] += transition(i - 4, i) * cg_c(i, i - 4)
        Ops[2] += transition(i - 5, i) * cg_c(i, i - 5)
        Ops[3] += transition(i - 3, i) * cg_c(i, i - 3)
    end

    return Ops
end

function Calculate_Obs(P::HamiltonianParas, Index::Int64)
    P.Omega_b = 0.0
    J_Set = Jumping_Operators(P)
    Absorption_Ops = Type_II_Absorption_Operators(P)

    Delta_d_list = collect(range(-50, -20, 41))
    Delta_p_list = collect(range(-50, -50, 1))

    Abs_1_ref_set = zeros(length(Delta_p_list), length(Delta_d_list))
    Abs_2_ref_set = zeros(length(Delta_p_list), length(Delta_d_list))
    Abs_3_ref_set = zeros(length(Delta_p_list), length(Delta_d_list))

    Abs_1_diff_set = zeros(length(Delta_p_list), length(Delta_d_list))
    Abs_2_diff_set = zeros(length(Delta_p_list), length(Delta_d_list))
    Abs_3_diff_set = zeros(length(Delta_p_list), length(Delta_d_list))

    Abs_1_obs_set = zeros(length(Delta_p_list), length(Delta_d_list))
    Abs_2_obs_set = zeros(length(Delta_p_list), length(Delta_d_list))
    Abs_3_obs_set = zeros(length(Delta_p_list), length(Delta_d_list))

    Prog = Progress(length(Delta_d_list) * length(Delta_p_list))
    for i in eachindex(Delta_d_list)
        for j in eachindex(Delta_p_list)
            P.Delta_p = Delta_p_list[j]
            H_0, H_p1, H_m1, H_p2, H_m2 = Type_II_Hamiltonian(P)
            ρ_ss = steadystate_fourier_2d(
                H_0,
                H_p1,
                H_m1,
                H_p2,
                H_m2,
                Delta_d_list[i],
                -Delta_p_list[j] + P.Delta_c,
                J_Set;
                n_max1=8,
                n_max2=8,
            )

            Abs_1_ref_set[j, i] = imag(tr(ρ_ss[1, 1] * Absorption_Ops[1]))
            Abs_2_ref_set[j, i] = imag(tr(ρ_ss[1, 1] * Absorption_Ops[2]))
            Abs_3_ref_set[j, i] = imag(tr(ρ_ss[1, 1] * Absorption_Ops[3]))

            ProgressMeter.next!(Prog)
        end
    end

    Prog = Progress(length(Delta_d_list) * length(Delta_p_list))
    P.Omega_b = 0.1
    for i in eachindex(Delta_d_list)
        for j in eachindex(Delta_p_list)
            P.Delta_p = Delta_p_list[j]
            P.Delta_b = Delta_d_list[i]
            H_0, H_p1, H_m1, H_p2, H_m2 = Type_II_Hamiltonian(P)
            ρ_ss = steadystate_fourier_2d(
                H_0,
                H_p1,
                H_m1,
                H_p2,
                H_m2,
                Delta_d_list[i],
                -Delta_p_list[j] + P.Delta_c,
                J_Set;
                n_max1=8,
                n_max2=8,
            )

            Abs_1_diff_set[j, i] = imag(tr(ρ_ss[1, 1] * Absorption_Ops[1]))
            Abs_2_diff_set[j, i] = imag(tr(ρ_ss[1, 1] * Absorption_Ops[2]))
            Abs_3_diff_set[j, i] = imag(tr(ρ_ss[1, 1] * Absorption_Ops[3]))

            ProgressMeter.next!(Prog)
        end
    end

    Abs_1_obs_set = -Abs_1_diff_set .+ Abs_1_ref_set
    Abs_2_obs_set = -Abs_2_diff_set .+ Abs_2_ref_set
    Abs_3_obs_set = -Abs_3_diff_set .+ Abs_3_ref_set


    File = h5open("Type_II_Obs_$(Index).h5", "w")
    write(File, "Delta_p_list", Delta_p_list)
    write(File, "Delta_d_list", Delta_d_list)
    write(File, "A1", Abs_1_obs_set)
    write(File, "A2", Abs_2_obs_set)
    write(File, "A3", Abs_3_obs_set)
    close(File)
end

P = HamiltonianParas()
P.Omega_cf = 20.0 * cg_c(3, 11) / cg_c(4, 8)
P.Omega_cb = 0.0
P.Delta_c = 0.0
Calculate_Obs(P, 1)

# P.Omega_cf = 0.0
# P.Omega_cb = 20.0 * cg_c(3, 11) / cg_c(4, 8)
# P.Delta_c = 0.0
# Calculate_Obs(P, 2)

# P.Omega_cf = 15.0 * cg_c(3, 11) / cg_c(4, 8)
# P.Omega_cb = 15.0 * cg_c(3, 11) / cg_c(4, 8)
# P.Delta_c = 0.0
# Calculate_Obs(P, 3)

# P.Omega_cf = 10.0 * cg_c(3, 11) / cg_c(4, 8)
# P.Omega_cb = 10.0 * cg_c(3, 11) / cg_c(4, 8)
# P.Delta_c = -20.0
# Calculate_Obs(P, 4)
