using CairoMakie
using QuantumToolbox
using WignerSymbols
using HDF5
using ProgressMeter
using JLD2
using RationalRoots
using Statistics
include("Floquet_2D_Solver_I.jl")

Base.@kwdef mutable struct HamiltonianParas
    Omega_p::Float64 = 0.01
    Delta_p::Float64 = 5.0
    theta_p = 1 / 2 * pi

    Omega_cf::Float64 = 20.0
    Omega_cb::Float64 = 0.0
    Delta_c::Float64 = -0.0
    theta_c = 0 / 2 * pi

    Omega_b::Float64 = 0.1
    Delta_b::Float64 = 0.0
    theta_b = 1 / 2 * pi
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
    c12 = 0.01 * sqrt(5)
    c21 = 0.01 * sqrt(3)
    t1 = 1.0

    J = 0.0 * transition(0, 0)
    for i in 11:15
        J += sqrtGamma * transition(i - 8, i) * abs(cg_c(i - 8, i) * dipole_jtf_c(i - 8, i))
        # push!(J_Set, sqrtGamma * transition(i - 8, i) * abs(cg_c(i - 8, i) * dipole_jtf_c(i - 8, i)))
    end
    push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 11:14
        J += sqrtGamma * transition(i - 7, i) * abs(cg_c(i - 7, i) * dipole_jtf_c(i - 7, i))
        # push!(J_Set, sqrtGamma * transition(i - 7, i) * abs(cg_c(i - 7, i) * dipole_jtf_c(i - 7, i)))
    end
    push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 12:15
        J += sqrtGamma * transition(i - 9, i) * abs(cg_c(i - 9, i) * dipole_jtf_c(i - 9, i))
        # push!(J_Set, sqrtGamma * transition(i - 9, i) * abs(cg_c(i - 9, i) * dipole_jtf_c(i - 9, i)))
    end
    push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 12:14
        J += sqrtGamma * transition(i - 12, i) * abs(cg_c(i - 12, i) * dipole_jtf_c(i - 12, i))
        # push!(J_Set, sqrtGamma * transition(i - 12, i) * abs(cg_c(i - 12, i) * dipole_jtf_c(i - 12, i)))
    end
    push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 11:13
        J += sqrtGamma * transition(i - 11, i) * abs(cg_c(i - 11, i) * dipole_jtf_c(i - 11, i))
        # push!(J_Set, sqrtGamma * transition(i - 11, i) * abs(cg_c(i - 11, i) * dipole_jtf_c(i - 11, i)))
    end
    push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 13:15
        J += sqrtGamma * transition(i - 13, i) * abs(cg_c(i - 13, i) * dipole_jtf_c(i - 13, i))
        # push!(J_Set, sqrtGamma * transition(i - 13, i) * abs(cg_c(i - 13, i) * dipole_jtf_c(i - 13, i)))
    end
    push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 8:10
        J += t1 * sqrtGamma * transition(i - 8, i) * abs(cg_c(i - 8, i) * dipole_jtf_c(i - 8, i))
        # push!(J_Set, sqrtGamma * transition(i - 8, i) * abs(cg_c(i - 8, i) * dipole_jtf_c(i - 8, i)))
    end
    push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 8:9
        J += t1 * sqrtGamma * transition(i - 7, i) * abs(cg_c(i - 7, i) * dipole_jtf_c(i - 7, i))
        # push!(J_Set, sqrtGamma * transition(i - 7, i) * abs(cg_c(i - 7, i) * dipole_jtf_c(i - 7, i)))
    end
    push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 9:10
        J += t1 * sqrtGamma * transition(i - 9, i) * abs(cg_c(i - 9, i) * dipole_jtf_c(i - 9, i))
        # push!(J_Set, sqrtGamma * transition(i - 9, i) * abs(cg_c(i - 9, i) * dipole_jtf_c(i - 9, i)))
    end
    push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 8:10
        J += t1 * sqrtGamma * transition(i - 4, i) * abs(cg_c(i - 4, i) * dipole_jtf_c(i - 4, i))
        # push!(J_Set, sqrtGamma * transition(i - 4, i) * abs(cg_c(i - 4, i) * dipole_jtf_c(i - 4, i)))
    end
    push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 8:10
        J += t1 * sqrtGamma * transition(i - 5, i) * abs(cg_c(i - 5, i) * dipole_jtf_c(i - 5, i))
        # push!(J_Set, sqrtGamma * transition(i - 5, i) * abs(cg_c(i - 5, i) * dipole_jtf_c(i - 5, i)))
    end
    push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 8:10
        J += t1 * sqrtGamma * transition(i - 3, i) * abs(cg_c(i - 3, i) * dipole_jtf_c(i - 3, i))
        # push!(J_Set, sqrtGamma * transition(i - 3, i) * abs(cg_c(i - 3, i) * dipole_jtf_c(i - 3, i)))
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

function Type_I_Hamiltonian(P::HamiltonianParas)
    H_0 = 0.0 * transition(0, 0)
    iso_coeff = 1.0
    for i in 11:15
        H_0 += -P.Delta_p * transition(i, i)
    end
    for i in 3:7
        H_0 += (P.Delta_c - P.Delta_p) * transition(i, i)
    end
    for i in 8:10
        H_0 += -P.Delta_b * transition(i, i)
    end

    H_p1 = 0.0 * transition(0, 0)

    for i in 3:7
        H_p1 += P.Omega_cf * cos(P.theta_c) * transition(i + 8, i) * cg_c(i + 8, i)
        H_p1 += P.Omega_cb * cos(P.theta_c) * transition(i, i + 8) * cg_c(i, i + 8)
    end
    for i in 3:6
        H_p1 += P.Omega_cf / 2^0.5 * sin(P.theta_c) * transition(i + 9, i) * cg_c(i + 9, i) * iso_coeff
        H_p1 += P.Omega_cb / 2^0.5 * sin(P.theta_c) * transition(i, i + 9) * cg_c(i, i + 9) * iso_coeff
    end
    for i in 4:7
        H_p1 += P.Omega_cf / 2^0.5 * sin(P.theta_c) * transition(i + 7, i) * cg_c(i + 7, i) * iso_coeff
        H_p1 += P.Omega_cb / 2^0.5 * sin(P.theta_c) * transition(i, i + 7) * cg_c(i, i + 7) * iso_coeff
    end
    for i in 0:2
        H_p1 += P.Omega_p * cos(P.theta_p) * transition(i + 12, i) * cg_c(i + 12, i)
        H_p1 += P.Omega_p / 2^0.5 * sin(P.theta_p) * transition(i + 13, i) * cg_c(i + 13, i) * iso_coeff
        H_p1 += P.Omega_p / 2^0.5 * sin(P.theta_p) * transition(i + 11, i) * cg_c(i + 11, i) * iso_coeff

        H_p1 += P.Omega_b * cos(P.theta_b) * transition(i + 8, i) * cg_c(i + 8, i)
    end
    for i in 0:1
        H_p1 += P.Omega_b / 2^0.5 * sin(P.theta_b) * transition(i + 9, i) * cg_c(i + 9, i) * iso_coeff
    end
    for i in 1:2
        H_p1 += P.Omega_b / 2^0.5 * sin(P.theta_b) * transition(i + 7, i) * cg_c(i + 7, i) * iso_coeff
    end

    H_m1 = 0.0 * transition(0, 0)

    for i in 3:7
        H_m1 += P.Omega_cf * cos(P.theta_c) * transition(i, i + 8) * cg_c(i, i + 8)
        H_m1 += P.Omega_cb * cos(P.theta_c) * transition(i + 8, i) * cg_c(i + 8, i)
    end
    for i in 3:6
        H_m1 += P.Omega_cf / 2^0.5 * sin(P.theta_c) * transition(i, i + 9) * cg_c(i, i + 9) * iso_coeff
        H_m1 += P.Omega_cb / 2^0.5 * sin(P.theta_c) * transition(i + 9, i) * cg_c(i + 9, i) * iso_coeff
    end
    for i in 4:7
        H_m1 += P.Omega_cf / 2^0.5 * sin(P.theta_c) * transition(i, i + 7) * cg_c(i, i + 7) * iso_coeff
        H_m1 += P.Omega_cb / 2^0.5 * sin(P.theta_c) * transition(i + 7, i) * cg_c(i + 7, i) * iso_coeff
    end
    for i in 0:2
        H_m1 += P.Omega_p * cos(P.theta_p) * transition(i, i + 12) * cg_c(i, i + 12)
        H_m1 += P.Omega_p / 2^0.5 * sin(P.theta_p) * transition(i, i + 13) * cg_c(i, i + 13) * iso_coeff
        H_m1 += P.Omega_p / 2^0.5 * sin(P.theta_p) * transition(i, i + 11) * cg_c(i, i + 11) * iso_coeff

        H_m1 += P.Omega_b * cos(P.theta_b) * transition(i, i + 8) * cg_c(i, i + 8)
    end
    for i in 0:1
        H_m1 += P.Omega_b / 2^0.5 * sin(P.theta_b) * transition(i, i + 9) * cg_c(i, i + 9) * iso_coeff
    end
    for i in 1:2
        H_m1 += P.Omega_b / 2^0.5 * sin(P.theta_b) * transition(i, i + 7) * cg_c(i, i + 7) * iso_coeff
    end

    H_p2 = 0.0 * transition(0, 0)
    H_m2 = 0.0 * transition(0, 0)

    return H_0, H_p1, H_m1, H_p2, H_m2
end

function Type_I_Absorption_Operators(P::HamiltonianParas)
    Ops = fill(0.0 * transition(0, 0), 3)

    for i in 0:2
        Ops[1] += transition(i, i + 12) * cg_c(i + 12, i)
        Ops[2] += transition(i, i + 13) * cg_c(i + 13, i)
        Ops[3] += transition(i, i + 11) * cg_c(i + 11, i)
    end

    return Ops
end

function Type_I_Absorption_Operators_SP(P::HamiltonianParas)
    Ops = fill(0.0 * transition(0, 0), 3)

    Ops[1] += transition(0, 11) * cg_c(11, 0)
    Ops[2] += transition(1, 12) * cg_c(12, 1)
    Ops[3] += transition(0, 13) * cg_c(13, 0)

    Ops[1] += transition(2, 15) * cg_c(15, 2)
    Ops[2] += transition(1, 14) * cg_c(14, 1)
    Ops[3] += transition(2, 13) * cg_c(13, 2)

    return Ops
end

function Calculate_Obs(P::HamiltonianParas, FileName::String)
    P.Omega_b = 0.0
    J_Set = Jumping_Operators(P)
    Absorption_Ops = Type_I_Absorption_Operators(P)

    Delta_d_list = collect(range(-50, 50, 100))
    Delta_p_list = collect(range(-50, 50, 100))
    Abs_1_ref_set = zeros(length(Delta_p_list), length(Delta_d_list))
    Abs_2_ref_set = zeros(length(Delta_p_list), length(Delta_d_list))
    Abs_3_ref_set = zeros(length(Delta_p_list), length(Delta_d_list))
    Pop_pm1_wb_set = zeros(length(Delta_p_list), length(Delta_d_list))
    Pop_0_wb_set = zeros(length(Delta_p_list), length(Delta_d_list))

    Prog = Progress(length(Delta_d_list) * length(Delta_p_list))
    for i in eachindex(Delta_d_list)
        for j in eachindex(Delta_p_list)
            P.Delta_p = Delta_p_list[j]
            H_0, H_p1, H_m1, H_p2, H_m2 = Type_I_Hamiltonian(P)
            ρ_ss = steadystate_fourier_2d(
                H_0,
                H_p1,
                H_m1,
                H_p2,
                H_m2,
                Delta_d_list[i],
                0.0,
                J_Set;
                n_max1=10,
                n_max2=0,
            )

            Abs_1_ref_set[j, i] = abs.(imag(tr(ρ_ss[1, 0] * Absorption_Ops[1])))
            Abs_2_ref_set[j, i] = abs.(imag(tr(ρ_ss[1, 0] * Absorption_Ops[2])))
            Abs_3_ref_set[j, i] = abs.(imag(tr(ρ_ss[1, 0] * Absorption_Ops[3])))
            Pop_pm1_wb_set[j, i] = real(ρ_ss[0, 0][1, 1])
            Pop_0_wb_set[j, i] = real(ρ_ss[0, 0][2, 2])

            ProgressMeter.next!(Prog)
        end
    end

    Abs_1_diff_set = zeros(length(Delta_d_list), length(Delta_p_list))
    Abs_2_diff_set = zeros(length(Delta_d_list), length(Delta_p_list))
    Abs_3_diff_set = zeros(length(Delta_d_list), length(Delta_p_list))

    Abs_1_obs_set = zeros(length(Delta_p_list), length(Delta_d_list))
    Abs_2_obs_set = zeros(length(Delta_p_list), length(Delta_d_list))
    Abs_3_obs_set = zeros(length(Delta_p_list), length(Delta_d_list))

    Pop_pm1_set = zeros(length(Delta_p_list), length(Delta_d_list))
    Pop_0_set = zeros(length(Delta_p_list), length(Delta_d_list))

    Prog = Progress(length(Delta_d_list) * length(Delta_p_list))
    P.Omega_b = 0.1
    for i in eachindex(Delta_d_list)
        for j in eachindex(Delta_p_list)
            P.Delta_p = Delta_p_list[j]
            P.Delta_b = Delta_d_list[i]
            H_0, H_p1, H_m1, H_p2, H_m2 = Type_I_Hamiltonian(P)
            ρ_ss = steadystate_fourier_2d(
                H_0,
                H_p1,
                H_m1,
                H_p2,
                H_m2,
                Delta_d_list[i],
                0.0,
                J_Set;
                n_max1=10,
                n_max2=0,
            )

            Abs_1_diff_set[j, i] = abs.(imag(tr(ρ_ss[1, 0] * Absorption_Ops[1])))
            Abs_2_diff_set[j, i] = abs.(imag(tr(ρ_ss[1, 0] * Absorption_Ops[2])))
            Abs_3_diff_set[j, i] = abs.(imag(tr(ρ_ss[1, 0] * Absorption_Ops[3])))

            Pop_pm1_set[j, i] = real(ρ_ss[0, 0][1, 1])
            Pop_0_set[j, i] = real(ρ_ss[0, 0][2, 2])

            ProgressMeter.next!(Prog)
        end
    end

    Abs_1_obs_set = -Abs_1_diff_set .+ Abs_1_ref_set
    Abs_2_obs_set = -Abs_2_diff_set .+ Abs_2_ref_set
    Abs_3_obs_set = -Abs_3_diff_set .+ Abs_3_ref_set

    File = h5open("$FileName", "w")
    write(File, "Delta_p_list", Delta_p_list)
    write(File, "Delta_d_list", Delta_d_list)
    write(File, "A1", Abs_1_obs_set)
    write(File, "A2", Abs_2_obs_set)
    write(File, "A3", Abs_3_obs_set)
    write(File, "Ppm1wb", Pop_pm1_wb_set)
    write(File, "P0wb", Pop_0_wb_set)
    write(File, "Ppm1", Pop_pm1_set)
    write(File, "P0", Pop_0_set)
    close(File)
end

for a in 0:1
    for b in 0:1
        P = HamiltonianParas()
        P.theta_p = a * 0.5 * pi
        P.theta_b = b * 0.5 * pi
        sp = "z"
        sb = "z"
        if a == 1
            sp = "y"
        end
        if b == 1
            sb = "y"
        end
        P.Omega_cf = 20.0
        P.Omega_cb = 0.0
        P.Delta_c = 0.0
        Calculate_Obs(P, "Type_I_z$(sp)$(sb)_eit.h5")

        P.Omega_cf = 0.0
        P.Omega_cb = 20.0
        P.Delta_c = 0.0
        Calculate_Obs(P, "Type_I_z$(sp)$(sb)_reit.h5")

        P.Omega_cf = 15.0
        P.Omega_cb = 15.0
        P.Delta_c = 0.0
        Calculate_Obs(P, "Type_I_z$(sp)$(sb)_sw.h5")
        
        P.Omega_cf = 10.0
        P.Omega_cb = 10.0
        P.Delta_c = -20.0
        Calculate_Obs(P, "Type_I_z$(sp)$(sb)_det.h5")
    end
end