# %% Cell 1
using RxInfer, Distributions, PyPlot, LaTeXStrings, Random, LinearAlgebra

# setting pwd to root directory of this project.
current_dir = last(split(pwd(), "/"))
if current_dir == "src"
  cd("..")
end

# %% Cell 2
# Defining the hack wich allows for the usage of q(z)
using SpecialFunctions
@rule Categorical(:p, Marginalisation) (m_out::Categorical, q_out::PointMass) = begin
  @logscale -SpecialFunctions.logfactorial(length(probvec(q_out)))
  return Dirichlet(probvec(q_out) .+ one(eltype(probvec(q_out))))
end

function ReactiveMP.constrain_form(
  ::PointMassFormConstraint,
  distribution::Categorical
)
  k = mode(distribution)
  v = zeros(length(distribution.support))
  v[k] = 1
  return PointMass(v)
end

struct EnforceMarginalFunctionalDependency <: ReactiveMP.FunctionalDependencies
  edge::Symbol
end

function ReactiveMP.collect_functional_dependencies(
  ::MixtureNode, enforce::EnforceMarginalFunctionalDependency
)
  return enforce
end

function ReactiveMP.functional_dependencies(enforce::EnforceMarginalFunctionalDependency, factornode, interface, iindex)
  message_dependencies, default = functional_dependencies(
    ReactiveMP.collect_functional_dependencies(factornode, nothing),
    factornode,
    interface,
    iindex
  )

  index = ReactiveMP.findnext(
    i -> ReactiveMP.name(i) === enforce.edge,
    ReactiveMP.getinterfaces(factornode),
    1
  )

  if index === iindex
    return message_dependencies, default
  end

  extra_localmarginal = ReactiveMP.FactorNodeLocalMarginal(enforce.edge)
  vmarginal = ReactiveMP.getmarginal(
    ReactiveMP.getvariable(ReactiveMP.getinterfaces(factornode)[index]), ReactiveMP.IncludeAll()
  )
  extra_stream = ReactiveMP.MarginalObservable()
  ReactiveMP.connect!(extra_stream, vmarginal)
  ReactiveMP.setmarginal!(extra_localmarginal, extra_stream)

  # Find insertion position (probably might be implemented more efficiently)
  insertafter = sum(first(el) < iindex ? 1 : 0 for el in default; init=0)
  marginal_dependencies = ReactiveMP.TupleTools.insertafter(
    default, insertafter, (extra_localmarginal,)
  )

  return message_dependencies, marginal_dependencies
end

# function for using hard switching
function ReactiveMP.functional_dependencies(::EnforceMarginalFunctionalDependency, factornode::MixtureNode{N}, interface, iindex::Int) where {N}
  message_dependencies = if iindex === 1
    # output depends on:
    (factornode.inputs,)
  elseif iindex === 2
    # switch depends on:
    (factornode.out, factornode.inputs)
  elseif 2 < iindex <= N + 2
    # k'th input depends on:
    (factornode.out,)
  else
    error("Bad index in functional_dependencies for SwitchNode")
  end

  marginal_dependencies = if iindex === 1
    # output depends on:
    (factornode.switch,)
  elseif iindex == 2
    #  switch depends on
    ()
  elseif 2 < iindex <= N + 2
    # k'th input depends on:
    (factornode.switch,)
  else
    error("Bad index in function_dependencies for SwitchNode")
  end
  # println(marginal_dependencies)
  return message_dependencies, marginal_dependencies
end

# create an observable that is used to compute the switch with pipeline constraints
function ReactiveMP.collect_latest_messages(::EnforceMarginalFunctionalDependency, factornode::MixtureNode{N}, messages::Tuple{ReactiveMP.NodeInterface,NTuple{N,ReactiveMP.IndexedNodeInterface}}) where {N}
  switchinterface = messages[1]
  inputsinterfaces = messages[2]

  msgs_names = Val{(ReactiveMP.name(switchinterface), ReactiveMP.name(inputsinterfaces[1]))}()
  msgs_observable =
    combineLatest((ReactiveMP.messagein(switchinterface), combineLatest(map((input) -> ReactiveMP.messagein(input), inputsinterfaces), PushNew())), PushNew()) |>
    map_to((ReactiveMP.messagein(switchinterface), ReactiveMP.ManyOf(map((input) -> ReactiveMP.messagein(input), inputsinterfaces))))
  return msgs_names, msgs_observable
end

# create an observable that is used to compute the output with pipeline constraints
function ReactiveMP.collect_latest_messages(::EnforceMarginalFunctionalDependency, factornode::MixtureNode{N}, messages::Tuple{NTuple{N,ReactiveMP.IndexedNodeInterface}}) where {N}
  inputsinterfaces = messages[1]

  msgs_names = Val{(ReactiveMP.name(inputsinterfaces[1]),)}()
  msgs_observable =
    combineLatest(map((input) -> ReactiveMP.messagein(input), inputsinterfaces), PushNew()) |>
    map_to((ReactiveMP.ManyOf(map((input) -> ReactiveMP.messagein(input), inputsinterfaces)),))
  return msgs_names, msgs_observable
end

# create an observable that is used to compute the input with pipeline constraints
function ReactiveMP.collect_latest_messages(::EnforceMarginalFunctionalDependency, factornode::MixtureNode{N}, messages::Tuple{ReactiveMP.NodeInterface}) where {N}
  outputinterface = messages[1]

  msgs_names = Val{(ReactiveMP.name(outputinterface),)}()
  msgs_observable = combineLatestUpdates((ReactiveMP.messagein(outputinterface),), PushNew())
  return msgs_names, msgs_observable
end

function ReactiveMP.collect_latest_marginals(::EnforceMarginalFunctionalDependency, factornode::MixtureNode{N}, marginals::Tuple{ReactiveMP.NodeInterface}) where {N}
  switchinterface = marginals[1]

  marginal_names = Val{(ReactiveMP.name(switchinterface),)}()
  marginals_observable = combineLatestUpdates((getmarginal(ReactiveMP.getvariable(switchinterface), IncludeAll()),), PushNew())

  return marginal_names, marginals_observable
end

# %% Cell 3: Loading the data
using CSV, Tables
N_classes_yolo = 80

result_destination = "results/7038"
csv_source = "data/measurements/7038/global_xy.csv"
image_source = "data/images/DSCF7038.JPG"

data = CSV.File(csv_source) |> Tables.matrix
data = data[2:size(data)[1], :]
img = plt.imread(image_source)


# %% Cell 3.1: Scaling data with image size
img_size_y = size(img)[1]
img_size_x = size(img[0])[1]
data[:, 1] = data[:, 1] / img_size_x
data[:, 2] = data[:, 2] / img_size_y
println("Scalled the data")

# %% Cell 3.2: Putting the data in the correct type
nr_samples = size(data)[1]
xy_data = Vector{Vector{Float64}}(undef, nr_samples)
for i in 1:nr_samples
  xy_data[i] = Vector(data[i, 1:2])
end

println(xy_data[1:4, :])
println(typeof(xy_data))


# %% Cell 3.3: Data visualization
plt.figure()
plt.imshow(img)
plt.scatter(data[:, 1] * img_size_x, data[:, 2] * img_size_y, alpha=0.4)
plt.savefig(result_destination*"/data_visualization.png")


# %% Cell 3.4: Getting ground truth clusters
# ground_truth_cluster = data[:, 7]
# println(ground_truth_cluster[1:10])

# %% Cell 3.5: Creating measurement error arrays
data_σu = ones(nr_samples) / 800
data_σv = ones(nr_samples) / 800
println("Created measurement uncertainty vectors")


# %% Cell 4: Model defintion
nr_components = 20

@model function model_dirichlet_process(y, σu, σv, α, μu_θ, Λu_θ, μv_θ, Λv_θ)
  # `y` is specify experimental outcomes
  local θuk
  local θvk

  # specify initial distribution over clusters
  π ~ Dirichlet(α)

  # prior over model selection variable
  z ~ Categorical(π) where {
    dependencies=EnforceMarginalFunctionalDependency(:out)
  }

  # specify prior models over θ
  for k in 1:nr_components
    θuk[k] ~ NormalMeanPrecision(μu_θ[k], Λu_θ[k])
    θvk[k] ~ NormalMeanPrecision(μv_θ[k], Λv_θ[k])
  end

  # specify mixture distribution
  θu ~ Mixture(switch=z, inputs=θuk) where {
    dependencies=EnforceMarginalFunctionalDependency(:switch)
  }
  θv ~ Mixture(switch=z, inputs=θvk) where {
    dependencies=EnforceMarginalFunctionalDependency(:switch)
  }

  # specify observation noise
  y[1] ~ NormalMeanVariance(θu, σu)
  y[2] ~ NormalMeanVariance(θv, σv)
end

@constraints function constraints_dirichlet_process()
  q(z)::PointMassFormConstraint()
end;

# %% Cell 5: Some helper functions
alpha = 1e-2
# base_measure = MvNormalMeanPrecision(zeros(2), 0.1 * diagm(ones(2)));

function update_alpha_vector(α_prev)
  ind = findfirst(x -> isapprox(1e-10, x; rtol=0.1), α_prev)
  if isnothing(ind)
    α_new = α_prev
    @error "upper bound reached"
  elseif ind == 2 && α_prev[1] != 10.0^alpha
    α_new = α_prev
    α_new[ind-1] = 1
    α_new[ind] = 10.0^alpha
  elseif ind > 2 && α_prev[ind-1] ≈ 1 + 10.0^alpha
    α_new = α_prev
    α_new[ind-1] = 1
    α_new[ind] = 10.0^alpha
  else
    α_new = α_prev
  end
  return α_new
end

function update_alpha(dist)
  return update_alpha_vector(probvec(dist))
end

function broadcast_mean_precision(dist)
  tmp = mean_precision.(dist)
  return first.(tmp), last.(tmp)
end

@rule Mixture((:inputs, k), Marginalisation) (m_out::Any, q_switch::PointMass) = begin
  # check whether mean is one-hot
  p = mean(q_switch)
  @assert sum(p) ≈ 1 "The selector variable connected to the Mixture node is not normalized."
  @assert all(x -> x == 1 || x == 0, p) "The selector variable connected to the Mixture node is not one-hot encoded."

  # get selected cluster
  kmax = argmax(p)

  if k == kmax
    @logscale 0
    return m_out
  else
    @logscale missing
    return missing
  end
end

# RxInfer.is_data(vector::Vector{RxInfer.GraphVariableRef}) = all(RxInfer.is_data.(vector))
# RxInfer.is_data(vector::AbstractVector{RxInfer.GraphVariableRef}) = all(RxInfer.is_data.(vector))
# GraphPPL.is_data(collection::AbstractArray{RxInfer.GraphVariableRef}) = all(GraphPPL.is_data, collection)


# %% Cell 6: Perfoming Inference
autoupdates_dirichlet_process = @autoupdates begin
  α = update_alpha(q(π))
  μu_θ, Λu_θ = broadcast_mean_precision(q(θuk))
  μv_θ, Λv_θ = broadcast_mean_precision(q(θvk))
end;

alpha_start = 1e-10 * ones(nr_components)
alpha_start[1] = 10.0^alpha

results_dirichlet_process = infer(
  model=model_dirichlet_process(),
  data=(y=xy_data, σu=data_σu, σv=data_σv),
  constraints=constraints_dirichlet_process(),
  autoupdates=autoupdates_dirichlet_process,
  initialization=@initialization(begin
    q(π) = Dirichlet(alpha_start; check_args=false)
    q(θuk) = NormalMeanPrecision(0.5, 1)
    q(θvk) = NormalMeanPrecision(0.5, 1)
  end),
  returnvars=(:π, :θuk, :θvk),
  keephistory=nr_samples,
  historyvars=(z=KeepLast(), π=KeepLast(), θuk=KeepLast(), θvk=KeepLast()),
  autostart=true,
  addons=AddonLogScale()
)


# %% Cell 7: Result visualization
N = 10
# N = nr_samples - 1
plt.figure()
plt.imshow(img)
plt.scatter(data[1:N, 1] * img_size_x, data[1:N, 2] * img_size_y, alpha=0.5, c=argmax.(mean.(results_dirichlet_process.history[:z][1:N])))
for k in findall(x -> x >= 1, probvec(results_dirichlet_process.history[:π][N]))[1:end-1]
  plt.scatter(img_size_x * mean(results_dirichlet_process.history[:θuk][N][k]), img_size_y * mean(results_dirichlet_process.history[:θvk][N][k]), marker="x", color="black")
end
plt.xlabel(L"y_1")
plt.ylabel(L"y_2")
plt.savefig(result_destination*"/inference_result.png")
# plt.grid()

# %% Cell 7.1: Storing cluster means
open(result_destination*"/cluster_centers.txt", "w") do file
  for k in findall(x -> x >= 1, probvec(results_dirichlet_process.history[:π][N]))
    txt = string(img_size_x * mean(results_dirichlet_process.history[:θuk][N][k])) * " " * string(img_size_y * mean(results_dirichlet_process.history[:θvk][N][k]))*"\n"
    write(file, txt)
  end
end


# %% Cell 8: Visualization to set correlation_vector
# inferred_clusters = argmax.(mean.(results_dirichlet_process.history[:z][:]))
# nr_inferred_clusters = length(findall(x -> x >= 1, probvec(results_dirichlet_process.history[:π][N])))

# for cluster_id in 1:nr_inferred_clusters
#   idxs = findall(x -> x==cluster_id, inferred_clusters)
#   plt.figure()
#   plt.imshow(img)
#   plt.title("Inferred Cluster ID: "*string(cluster_id))
#   plt.scatter(data[idxs, 1] * img_size_x, data[idxs, 2] * img_size_y, alpha=0.4)
#   plt.savefig(result_destination*"/result_cluster_"*string(cluster_id)*".png")
#   # plt.gcf()
# end


# %% Cell 8.1: MORE Visualization to set correlation_vector
# nr_true_clusters = maximum(ground_truth_cluster)

# for cluster_id in 1:nr_true_clusters
#   idxs = findall(x -> x==cluster_id, ground_truth_cluster)
#   plt.figure()
#   plt.imshow(img)
#   plt.title("True Cluster ID: "*string(cluster_id))
#   plt.scatter(data[idxs, 1] * img_size_x, data[idxs, 2] * img_size_y, alpha=0.4)
#   plt.gcf()
# end

# %% Cell 9: Scoring the result
# println(inferred_clusters)
# println(length(ground_truth_cluster))
# correlation_vector = [9, 4, 5, 7, 2, 3, 8, 1, 99, 99, 10, 99]
# correlation_vector_explanation = "i is the infered cluster. This vectors is such that the number vector[i]=cluster in labeling.\n"
# open(result_destination*"/correlation_vector.txt", "w") do file
#   write(file, correlation_vector_explanation)
#   write(file, string(correlation_vector))
# end

# score = 0

# for point in 1:nr_samples
#   inferred_cluster = inferred_clusters[point]
#   correlated_cluster = correlation_vector[inferred_cluster]
#   if correlated_cluster == ground_truth_cluster[point]
#     score += 1
#   end
# end

# score = score/nr_samples
# open(result_destination*"/score.txt", "w") do file
#   write(file, string(score))
# end
# println(score)


# # %% Cell 10: Janky correlation matrix
# correlation = zeros(Int8, (Int8(nr_true_clusters), nr_inferred_clusters))
# for point in 1:nr_samples
#   inferred_cluster = inferred_clusters[point]
#   true_cluster = Int8(ground_truth_cluster[point])
#   if true_cluster != -1
#     correlation[true_cluster, inferred_cluster] += 1
#   end
# end
# print(correlation)
# open(result_destination*"/correlation.txt", "w") do file
#   write(file, "True cluster on the rows, inferred on the collums.\n")
#   write(file, string(correlation))
# end
# plt.figure()
# plt.matshow(correlation)
