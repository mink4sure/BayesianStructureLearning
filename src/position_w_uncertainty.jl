# %%
using Pkg; Pkg.add("Plots")
using RxInfer, Distributions, Random, Plots


# %% Generating data
true_position = 1
noise = 5
N = 200

data = Array{Float64}(undef, N, 2)
position_measurements = rand(Normal(true_position, noise), N)
data[:, 1] = position_measurements
measurement_uncertainties = rand(Normal(5, 1), N)
data[:, 2] = measurement_uncertainties

println(data)
println(measurement_uncertainties)
r = range(0, 10, length=1000)
p = plot(r, (y)->pdf(5+Normal(0, 1), y))


# %% The Model
# @model function my_model(y, μ, σ)
@model function my_model(y, μ, σ, σ_θ)
  x ~ Normal(mean=μ, variance=σ)
  y ~ Normal(mean=x, variance=σ_θ)
  # y[1] ~ Normal(mean=x, variance=y[2])
end

my_model_autoupdates = @autoupdates begin
  μ = mean(q(x))
  σ = var(q(x))
end


# %% Performing the Inference
results = infer(
  model=my_model(),
  data=(y=position_measurements, σ_θ=measurement_uncertainties),
  # data=(y=data,),
  autoupdates=my_model_autoupdates,
  initialization=@initialization(begin
    q(x) = vague(NormalMeanPrecision)
  end),
  returnvars=(:x,),
  keephistory=N,
  historyvars=(x=KeepLast(),),
  autostart=true
)
# %%
x_estimated = results.history[:x]
# print(x_estimated)

anim = @animate for x in x_estimated
  r = range(-5, 5, length=1000)
  p = plot(r, (y)->pdf(x, y), fillalpha=.3, fillrange=0)
end

gif(anim, "../results/test.gif", fps=5)
