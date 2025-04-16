module SolveInitial

import LinearAlgebra
import CSV
import DataFrames
import NonlinearSolve
import NonlinearSolveFirstOrder
import NonlinearSolveBase
import CommonSolve
import ADTypes
import LogExpFunctions

function main(A0, L0, tau0, theta, len)

  # dfA0 = CSV.read("A_0.csv", DataFrame, header=0)
  # A0 = dfA0.Column2
  # 
  # max_index = maximum(dfA0.Column1)
  # @assert dfA0.Column1 == collect(0:max_index)
  # 
  # dfL0 = CSV.read("L_0.csv", DataFrame, header=0)
  # L0 = dfL0.Column2
  # @assert dfL0.Column1 == collect(0:max_index)
  # 
  # dftau = CSV.read("tau_0.csv", DataFrame, header=0)
  # tau0 = dftau.Column3
  # 
  # tau0 = zeros(Float64, max_index + 1, max_index + 1)
  # for (i, j, tau) in eachrow(dftau)
  #   tau0[i+1, j+1] = tau
  # end
  
  # calculate tau^(-theta)
  taumtheta = tau0 .^ (-theta)
  
  
  function objfun(u, p)

    logA, logL, taumtheta, len = p
  
    logPi = u[1:len]
    logP = u[len+1:2len]
  
    logY = (theta / (theta + 1.0)) * (logA + logL - logPi)
  
    # first equation residual
    ret1 = exp.((-1.0 / theta) * LogExpFunctions.logsumexp(log.(taumtheta) + repeat((theta .* logP + logY)', len), dims=2)) - exp.(logPi)
    ret2 = exp.((-1.0 / theta) * LogExpFunctions.logsumexp(log.(taumtheta) + repeat((theta .* logPi + logY), 1, len), dims=1)') - exp.(logP)
  
    # ret1 = (-1.0/theta) * log.(taumtheta * (logP.^(theta) .* exp.(logY))) - logPi
    # ret2 = (-1.0/theta) * log.(taumtheta' * (logPi.^(-theta) .* exp.(logY))) - logP
  
    return [ret1; ret2]
  
  end
  
  # f(u, p) = u .* u .- p
  # u0 = [1.0 ./ A0 ; 1.0 ./ A0]
  u0 = ones(Float64, 2 * len)
  p = (log.(A0), log.(L0), taumtheta, len)
  
  prob = NonlinearSolve.NonlinearProblem(objfun, u0, p)
  @time sol = CommonSolve.solve(prob, NonlinearSolveFirstOrder.NewtonRaphson(autodiff=ADTypes.AutoForwardDiff()); show_trace=Val(true), trace_level=NonlinearSolveBase.TraceAll(1))
  # success! 1.4k seconds
  
  # visualize solution
  
  logPi = sol.u[1:len]
  logP = sol.u[len+1:2len]
  
  exp.(logP)
  exp.(logPi)
  
  # CSV.write("log_Pi_0.csv", df_logPi)
  # CSV.write("log_P_0.csv", df_logP)
  
  return (logPi, logP)

end

export main

end # module
