# %%
using RxInfer, Distributions, CSV, Tables, PyPlot
K = 80 # Number of  yolo classes

# setting pwd to root directory of this project.
current_dir = last(split(pwd(), "/"))
if current_dir == "src"
  cd("..")
end


# %% Importing data
data = CSV.File("data/measurements/7038/global_xy.csv") |> Tables.matrix
label_data = convert(Vector{Int32}, data[:, 5])[1:10]
# label_data = [21, 22, 21, 22, 21, 22, 21, 22, 21, 22]
confidence_data = convert(Vector{Float64}, data[:, 6])
print(label_data)


# %% Creating a OneHot version of the data
onehot_data = Vector{Vector{Int32}}(undef, length(label_data))

for i in 1:length(label_data)
  data_point = label_data[i] + 1
  # println(data_point)
  onehot_data_point = zeros(K)
  onehot_data_point[data_point] = 1
  # print(onehot_data_point)
  onehot_data[i] = onehot_data_point
end

print(onehot_data[1])


# %% Alternatifve vectorized data
# Measured class has value of the confidence and
# uncertainty on other classes
onehot_like_data = Vector{Vector{Float64}}(undef, length(label_data))

for i in 1:length(label_data)
  confidence = confidence_data[i]
  onehot_like_data_point = ones(K)*(1-confidence)/79
  onehot_like_data_point[label_data[i]+1] = confidence
  onehot_like_data[i] = onehot_like_data_point
end

println(onehot_like_data[1])
println(sum(onehot_like_data[1]))


# %% Model
@model function my_model(y, π)
  local θk

  for k in 1:K
    β = 1 * ones(K)
    β[k] += 1
    θk[k] ~ Dirichlet(β)
  end

  C ~ Categorical(π)

  θ ~ Mixture(switch=C, inputs=θk)

  y ~ Categorical(θ)
end


# %% AutoUpdates
function update_prior(dist)
  return ReactiveMP.getdata(dist).p
end

auto_updates = @autoupdates begin
  π = update_prior(q(C))
end


# %% Defining some needed functions
println("So far, no extra funtions are needed.")


# %%
N_history = length(label_data)

result = infer(
  model=my_model(),
  data=(y=onehot_like_data,),
  autoupdates=auto_updates,
  initialization=@initialization(begin
    q(C) = Categorical(ones(K) / K)
  end),
  returnvars=(:C,),
  keephistory=N_history,
  historyvars=(C=KeepLast(),),
  autostart=true,
  addons=AddonLogScale(),
  #postprocess=UnpackMarginalPostprocess()
)
# %% Unpacking results
N_show = 5
idx_step = Int32(round(N_history / N_show))
println("idx_setp: ", idx_step)

# println(result.history[:C][1])
n = 0
while N_history - n > 0
  idx = N_history - # %%
  using RxInfer, Distributions, CSV, Tables, PyPlot
  K = 80
n
  plt.figure()
  plt.title("Inference after $idx measurements")
  plt.bar(1:K, ReactiveMP.getdata(result.history[:C][idx]).p)
  plt.xlabel("Classes")
  plt.ylabel("Probability")
  # plt.savefig("result-"*string(idx)*".png")
  plt.show()
  n += idx_step
end

println(argmax(ReactiveMP.getdata(result.history[:C][N_history]).p))
