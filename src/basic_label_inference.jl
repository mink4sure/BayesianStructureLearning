# %%
using RxInfer, Distributions, CSV, Tables, PyPlot
K = 80


# %% Importing data
# data = CSV.File("../data/global_xy.csv") |> Tables.matrix
# label_data = convert(Vector{Int32}, data[:, 5])
label_data = [21, 22, 21, 22, 21, 22, 21, 22, 21, 22]
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


# %% Model
@model function my_model(y, π)
  β ~ Dirichlet(π)
  y ~ Categorical(β)
end


# %% AutoUpdates
function update_prior(dist)
  # print(ReactiveMP.getdata(dist))
  return ReactiveMP.getdata(dist).alpha
end

auto_updates = @autoupdates begin
  π = update_prior(q(β))
end


# %% Defining some needed functions
println("So far, no extra funtions are needed.")


# %%
N_history = 10

result = infer(
  model=my_model(),
  data=(y=onehot_data,),
  autoupdates=auto_updates,
  initialization=@initialization(begin
    q(β) = Dirichlet(ones(K))
  end),
  returnvars=(:β,),
  keephistory=N_history,
  historyvars=(β=KeepLast(),),
  autostart=true,
  addons=AddonLogScale(),
  #postprocess=UnpackMarginalPostprocess()
)
# %% Unpacking results
N_show = 3
idx_step = Int32(round(N_history / N_show))
println("idx_setp: ", idx_step)

# println(result.history[:C][1])
n = 0
while N_history - n > 0
  idx = N_history - n
  plt.figure()
  plt.title("Inference after $idx measurements")
  plt.bar(1:K, ReactiveMP.getdata(result.history[:β][idx]).alpha)
  plt.xlabel("Classes")
  plt.ylabel("Probability")
  # plt.savefig("result-"*string(idx)*".png")
  plt.show()
  n += idx_step
end
