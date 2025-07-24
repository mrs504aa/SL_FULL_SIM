using CairoMakie
using QuantumToolbox
using WignerSymbols
using HDF5
using ProgressMeter
using JLD2
using RationalRoots
using Statistics
include("Floquet_2D_Solver_III.jl")

Base.@kwdef mutable struct HamiltonianParas
    Omega_p::Float64 = 0.01
    Delta_p::Float64 = 5.0

    Omega_c1f::Float64 = 5.0 / sqrt(3 / 5)
    Omega_c1b::Float64 = 0.0
    Delta_c1::Float64 = -0.0

    Omega_c2f::Float64 = 5.0 / sqrt(1 / 10)
    Omega_c2b::Float64 = 0.0
    Delta_c2::Float64 = -0.0

    Omega_b::Float64 = 0.1
    Delta_b::Float64 = 0.0
    b_couple::Int64 = -1

    magnetic_B::Float64 = 100.0
end

#P D 11 12 13 14 15
#P B    08 09 10
#S A 03 04 05 06 07
#S C    00 01 02

beta_S_F1 = -0.7
beta_S_F2 = 0.7
beta_P_F1 = -0.23
beta_P_F2 = 0.23

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
        # J += sqrtGamma * transition(i - 8, i) * abs(cg_c(i - 8, i) * dipole_jtf_c(i - 8, i))
        push!(J_Set, sqrtGamma * transition(i - 8, i) * abs(cg_c(i - 8, i) * dipole_jtf_c(i - 8, i)))
    end
    # push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 11:14
        # J += sqrtGamma * transition(i - 7, i) * abs(cg_c(i - 7, i) * dipole_jtf_c(i - 7, i))
        push!(J_Set, sqrtGamma * transition(i - 7, i) * abs(cg_c(i - 7, i) * dipole_jtf_c(i - 7, i)))
    end
    # push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 12:15
        # J += sqrtGamma * transition(i - 9, i) * abs(cg_c(i - 9, i) * dipole_jtf_c(i - 9, i))
        push!(J_Set, sqrtGamma * transition(i - 9, i) * abs(cg_c(i - 9, i) * dipole_jtf_c(i - 9, i)))
    end
    # push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 12:14
        # J += sqrtGamma * transition(i - 12, i) * abs(cg_c(i - 12, i) * dipole_jtf_c(i - 12, i))
        push!(J_Set, sqrtGamma * transition(i - 12, i) * abs(cg_c(i - 12, i) * dipole_jtf_c(i - 12, i)))
    end
    # push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 11:13
        # J += sqrtGamma * transition(i - 11, i) * abs(cg_c(i - 11, i) * dipole_jtf_c(i - 11, i))
        push!(J_Set, sqrtGamma * transition(i - 11, i) * abs(cg_c(i - 11, i) * dipole_jtf_c(i - 11, i)))
    end
    # push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 13:15
        # J += sqrtGamma * transition(i - 13, i) * abs(cg_c(i - 13, i) * dipole_jtf_c(i - 13, i))
        push!(J_Set, sqrtGamma * transition(i - 13, i) * abs(cg_c(i - 13, i) * dipole_jtf_c(i - 13, i)))
    end
    # push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 8:10
        # J += t1 * sqrtGamma * transition(i - 8, i) * abs(cg_c(i - 8, i) * dipole_jtf_c(i - 8, i))
        push!(J_Set, sqrtGamma * transition(i - 8, i) * abs(cg_c(i - 8, i) * dipole_jtf_c(i - 8, i)))
    end
    # push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 8:9
        # J += t1 * sqrtGamma * transition(i - 7, i) * abs(cg_c(i - 7, i) * dipole_jtf_c(i - 7, i))
        push!(J_Set, sqrtGamma * transition(i - 7, i) * abs(cg_c(i - 7, i) * dipole_jtf_c(i - 7, i)))
    end
    # push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 9:10
        # J += t1 * sqrtGamma * transition(i - 9, i) * abs(cg_c(i - 9, i) * dipole_jtf_c(i - 9, i))
        push!(J_Set, sqrtGamma * transition(i - 9, i) * abs(cg_c(i - 9, i) * dipole_jtf_c(i - 9, i)))
    end
    # push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 8:10
        # J += t1 * sqrtGamma * transition(i - 4, i) * abs(cg_c(i - 4, i) * dipole_jtf_c(i - 4, i))
        push!(J_Set, sqrtGamma * transition(i - 4, i) * abs(cg_c(i - 4, i) * dipole_jtf_c(i - 4, i)))
    end
    # push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 8:10
        # J += t1 * sqrtGamma * transition(i - 5, i) * abs(cg_c(i - 5, i) * dipole_jtf_c(i - 5, i))
        push!(J_Set, sqrtGamma * transition(i - 5, i) * abs(cg_c(i - 5, i) * dipole_jtf_c(i - 5, i)))
    end
    # push!(J_Set, J)

    J = 0.0 * transition(0, 0)
    for i in 8:10
        # J += t1 * sqrtGamma * transition(i - 3, i) * abs(cg_c(i - 3, i) * dipole_jtf_c(i - 3, i))
        push!(J_Set, sqrtGamma * transition(i - 3, i) * abs(cg_c(i - 3, i) * dipole_jtf_c(i - 3, i)))
    end
    # push!(J_Set, J)

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

function Type_III_Hamiltonian(P::HamiltonianParas)
    H_0 = 0.0 * transition(0, 0)
    for i in 0:2
        m = i - 1
        H_0 += m * P.magnetic_B * beta_S_F1 * transition(i, i)
    end
    for i in 3:7
        m = i - 5
        H_0 += m * P.magnetic_B * beta_S_F2 * transition(i, i)
        H_0 += (P.Delta_c1 - P.Delta_p) * transition(i, i)
    end
    for i in 8:10
        m = i - 9
        H_0 += m * P.magnetic_B * beta_P_F1 * transition(i, i)
        H_0 += -P.Delta_p * transition(i, i)
    end
    for i in 11:15
        m = i - 13
        H_0 += m * P.magnetic_B * beta_P_F2 * transition(i, i)
        H_0 += -P.Delta_b * transition(i, i)
    end

    H_p1_00 = 0.0 * transition(0, 0)
    for i in 3:5
        H_p1_00 += -P.Omega_c1f * cg_c(i + 5, i) * transition(i + 5, i)
        H_p1_00 += -P.Omega_c1b * cg_c(i, i + 5) * transition(i, i + 5)
    end
    for i in 0:1
        H_p1_00 += -P.Omega_p * cg_c(i + 9, i) * transition(i + 9, i)
    end
    for i in 1:2
        H_p1_00 += -P.Omega_p * cg_c(i + 7, i) * transition(i + 7, i)
    end
    for i in 0:2
        H_p1_00 += -P.Omega_b * cg_c(i + 11, i) * transition(i + 11, i) * (P.b_couple == -1)
        H_p1_00 += -P.Omega_b * cg_c(i + 13, i) * transition(i + 13, i) * (P.b_couple == +1)
    end

    H_m1_00 = 0.0 * transition(0, 0)
    for i in 3:5
        H_m1_00 += -P.Omega_c1f * cg_c(i, i + 5) * transition(i, i + 5)
        H_m1_00 += -P.Omega_c1b * cg_c(i + 5, i) * transition(i + 5, i)
    end
    for i in 0:1
        H_m1_00 += -P.Omega_p * cg_c(i, i + 9) * transition(i, i + 9)
    end
    for i in 1:2
        H_m1_00 += -P.Omega_p * cg_c(i, i + 7) * transition(i, i + 7)
    end
    for i in 0:2
        H_m1_00 += -P.Omega_b * cg_c(i, i + 11) * transition(i, i + 11) * (P.b_couple == -1)
        H_m1_00 += -P.Omega_b * cg_c(i, i + 13) * transition(i, i + 13) * (P.b_couple == +1)
    end

    H_p1_p1 = 0.0 * transition(0, 0)
    for i in 3:5
        H_p1_p1 += -P.Omega_c2f * cg_c(i + 5, i) * transition(i + 5, i)
    end
    H_p1_m1 = 0.0 * transition(0, 0)
    for i in 3:5
        H_p1_m1 += -P.Omega_c2b * cg_c(i, i + 5) * transition(i, i + 5)
    end
    H_m1_p1 = 0.0 * transition(0, 0)
    for i in 3:5
        H_m1_p1 += -P.Omega_c2b * cg_c(i + 5, i) * transition(i + 5, i)
    end
    H_m1_m1 = 0.0 * transition(0, 0)
    for i in 3:5
        H_m1_m1 += -P.Omega_c2f * cg_c(i, i + 5) * transition(i, i + 5)
    end

    return H_0, H_p1_00, H_m1_00, H_p1_p1, H_p1_m1, H_m1_p1, H_m1_m1
end

function Type_III_Absorption_Operators(P::HamiltonianParas)
    Ops = fill(0.0 * transition(0, 0), 3)

    for i in 9:10
        Ops[1] += transition(i - 9, i) * cg_c(i - 9, i)
    end
    for i in 8:9
        Ops[2] += transition(i - 7, i) * cg_c(i - 7, i)
    end
    for i in 8:10
        Ops[3] += transition(i - 8, i) * cg_c(i - 8, i)
    end

    return Ops
end

P = HamiltonianParas()
P.Omega_c1b = 10.0
P.Omega_c2b = 10.0
P.Omega_c1f = 10.0
P.Omega_c2b = 10.0
P.b_couple = -1
J_Set = Jumping_Operators(P)
Absorption_Ops = Type_III_Absorption_Operators(P)
P.Omega_b = 0.0

P.Delta_c1 = (0.23 * 1 + 0.7 * 2) * P.magnetic_B
P.Delta_c2 = (0.23 * -1 + 0.7 * 0) * P.magnetic_B

Delta_d_list = collect(range(-200, 200, 201))
Delta_p_list = collect(range(-200, 200, 201))

Abs_wb_set = zeros(3, length(Delta_p_list), length(Delta_d_list))
Abs_set = zeros(3, length(Delta_p_list), length(Delta_d_list))

Prog = Progress(length(Delta_d_list) * length(Delta_p_list))
for i in eachindex(Delta_d_list)
    P.Delta_b = Delta_d_list[i]
    Threads.@threads for j in eachindex(Delta_p_list)
        PT = deepcopy(P)
        PT.Delta_p = Delta_p_list[j]
        H_0, H_p1_00, H_m1_00, H_p1_p1, H_p1_m1, H_m1_p1, H_m1_m1 = Type_III_Hamiltonian(PT)
        ρ_ss = steadystate_fourier_2d(
            H_0,
            H_p1_00,
            H_m1_00,
            H_p1_p1,
            H_p1_m1,
            H_m1_p1,
            H_m1_m1,
            Delta_d_list[i],
            -(PT.Delta_c2 - PT.Delta_c1),
            J_Set;
            n_max1=1,
            n_max2=0,
        )

        Abs_wb_set[1, j, i] = abs.(imag(tr(ρ_ss[1, 0] * Absorption_Ops[1])))
        Abs_wb_set[2, j, i] = abs.(imag(tr(ρ_ss[1, 0] * Absorption_Ops[2])))
        Abs_wb_set[3, j, i] = abs.(imag(tr(ρ_ss[1, 0] * Absorption_Ops[3])))

        ProgressMeter.next!(Prog)
    end
end

P.Omega_b = 0.1
Prog = Progress(length(Delta_d_list) * length(Delta_p_list))
for i in eachindex(Delta_d_list)
    P.Delta_b = Delta_d_list[i] + 0.7 * P.magnetic_B
    Threads.@threads for j in eachindex(Delta_p_list)
        PT = deepcopy(P)
        PT.Delta_p = Delta_p_list[j]
        H_0, H_p1_00, H_m1_00, H_p1_p1, H_p1_m1, H_m1_p1, H_m1_m1 = Type_III_Hamiltonian(PT)
        ρ_ss = steadystate_fourier_2d(
            H_0,
            H_p1_00,
            H_m1_00,
            H_p1_p1,
            H_p1_m1,
            H_m1_p1,
            H_m1_m1,
            Delta_d_list[i],
            -(PT.Delta_c2 - PT.Delta_c1),
            J_Set;
            n_max1=1,
            n_max2=0,
        )

        Abs_set[1, j, i] = abs.(imag(tr(ρ_ss[1, 0] * Absorption_Ops[1])))
        Abs_set[2, j, i] = abs.(imag(tr(ρ_ss[1, 0] * Absorption_Ops[2])))
        Abs_set[3, j, i] = abs.(imag(tr(ρ_ss[1, 0] * Absorption_Ops[3])))

        ProgressMeter.next!(Prog)
    end
end

Fig = Figure(size=(1000, 800))
max = maximum(abs.(Abs_wb_set))
for i in 1:2
    Axe = Axis(
        Fig[1, i],
        aspect=1,
        xlabel="Δ_p (MHz)",
        ylabel="Δ_d (MHz)",
    )

    
    # max == 0 ? max = 1.0 : max

    heatmap!(
        Axe,
        Delta_p_list,
        Delta_d_list,
        Abs_wb_set[i, :, :],
        colormap=:seaborn_icefire_gradient,
        colorrange=(-max, max)
    )
end

max = maximum(abs.(Abs_wb_set .- Abs_set))
for i in 1:2
    Axe = Axis(
        Fig[2, i],
        aspect=1,
        xlabel="Δ_p (MHz)",
        ylabel="Δ_d (MHz)",
    )

    heatmap!(
        Axe,
        Delta_p_list,
        Delta_d_list,
        Abs_wb_set[i, :, :] .- Abs_set[i, :, :],
        colormap=:seaborn_icefire_gradient,
        colorrange=(-max, max)
    )
end

display(Fig)
save("Type_III_Absorption_wb.png", Fig)

File = h5open("Type_III_Absorption.h5", "w")
write(File, "Abs_wb_set", Abs_wb_set)
write(File, "Abs_set", Abs_set)
write(File, "Delta_d_list", Delta_d_list)
write(File, "Delta_p_list", Delta_p_list)
close(File)