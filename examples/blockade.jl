using NCTSSoS, NCTSSoS.FastPolynomials

using JSON
using JuMP
using MosekTools
using Graphs


function cs_nctssos_with_blockade(pop::OP, solver_config::SolverConfig, blockade_constraints::Vector{P}; dualize::Bool=true) where {P, OP<:NCTSSoS.OptimizationProblem{P}}
    # temporarily add blockade constraints as algebraic constraints
   for c in blockade_constraints
       push!(pop.eq_constraints, c)
   end

   sa = SimplifyAlgorithm(comm_gps=pop.comm_gps, is_projective=pop.is_projective, is_unipotent=pop.is_unipotent)
   order = iszero(solver_config.order) ? maximum([ceil(Int, maxdegree(poly) / 2) for poly in [pop.objective; pop.eq_constraints; pop.ineq_constraints]]) : solver_config.order

   corr_sparsity = NCTSSoS.correlative_sparsity(pop, order, solver_config.cs_algo)

   cliques_objective = [reduce(+, [issubset(sort!(variables(mono)), clique) ? coef * mono : zero(coef) * one(mono) for (coef, mono) in zip(coefficients(pop.objective), monomials(pop.objective))]) for clique in corr_sparsity.cliques]

   initial_activated_supps = map(zip(cliques_objective, corr_sparsity.clq_cons, corr_sparsity.clq_mom_mtx_bases)) do (partial_obj, cons_idx, mom_mtx_base)
        NCTSSoS.init_activated_supp(partial_obj, corr_sparsity.cons[cons_idx], mom_mtx_base, sa)
   end

   cliques_term_sparsities = map(zip(initial_activated_supps, corr_sparsity.clq_cons, corr_sparsity.clq_mom_mtx_bases, corr_sparsity.clq_localizing_mtx_bases)) do (init_act_supp, cons_idx, mom_mtx_bases, localizing_mtx_bases)
        NCTSSoS.term_sparsities(init_act_supp, corr_sparsity.cons[cons_idx], mom_mtx_bases, localizing_mtx_bases, solver_config.ts_algo, sa)
   end

   moment_problem = NCTSSoS.moment_relax(pop, corr_sparsity, cliques_term_sparsities)

   # recover blockade constraints
   for (type, cons) in moment_problem.constraints
       type == :HPSD && continue
       (cons[1, 1] in blockade_constraints) && (cons[2:end, 2:end] .*= zero(ComplexF64))
   end

   (pop isa NCTSSoS.ComplexPolyOpt{P} && !dualize) && error("Solving Moment Problem for Complex Poly Opt is not supported")
   problem_to_solve = !dualize ? moment_problem : NCTSSoS.sos_dualize(moment_problem)

   set_optimizer(problem_to_solve.model, solver_config.optimizer)
   optimize!(problem_to_solve.model)
   return NCTSSoS.PolyOptResult(objective_value(problem_to_solve.model), corr_sparsity, cliques_term_sparsities, problem_to_solve.model)
end

data_folder = "examples/data"
data_files = readdir(data_folder)

results = Dict{String, Any}[]
for file in data_files
    Lx, Ly = map(match(r"Lx(\d+)-Ly(\d+)-.*\.json", file).captures) do m
        parse(Int, m)
    end
    
    #!(Lx == 2 && Ly == 2) && continue

    data = JSON.parsefile(joinpath(data_folder, file))

    N = Lx * Ly
    @ncpolyvar x[1:N] y[1:N] z[1:N]

    H = zero(ComplexF64) - sum(data["Detuning"] ./2 .* (ones(ComplexF64, N) .- z)) + sum(data["Rabi"] ./2 .* x) + sum(map(V -> V[3] / 4 * (1-z[V[1]]) * (1-z[V[2]]), data["Vanderwaals"]); init=zero(ComplexF64))
    @show H

    Pauli_algebra = reduce(vcat, [[x[i] * y[i] - im * z[i], y[i] * x[i] + im * z[i], y[i] * z[i] - im * x[i], z[i] * y[i] + im * x[i], z[i] * x[i] - im * y[i], x[i] * z[i] + im * y[i]] for i in 1:N])
    blockade_constraints = [one(ComplexF64) - z[e[1]] - z[e[2]] + z[e[1]] * z[e[2]] for e in data["PXP"]]
    @show blockade_constraints

    pop = cpolyopt(H; eq_constraints=Pauli_algebra, comm_gps=[[x[i], y[i], z[i]] for i in 1:N], is_unipotent=true)
    solver_config = SolverConfig(optimizer=Mosek.Optimizer, order=2)

    elapsed_time = @elapsed res = cs_nctssos_with_blockade(pop, solver_config, blockade_constraints)
    @show res.objective
    @show data["Exact GSE"]

    push!(results, Dict(
        "test case" => file,
        "GSE by SDP" => res.objective,
        "Exact GSE by subspace" => data["Exact GSE"],
        "elapsed time" => elapsed_time
    ))
end
