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

  # calculate tau^(-theta)
  tau_theta = tau0 .^ (-theta)

  function objfun(u, p)

    logA, logL, tau_theta, len = p

    logPi = u[1:len]
    logP = u[len+1:2len]

    logY = (theta / (theta + 1.0)) * (logA + logL - logPi)

    # first equation residual
    ret1 = exp.((-1.0 / theta) * LogExpFunctions.logsumexp(log.(tau_theta) + repeat((theta .* logP + logY)', len), dims=2)) - exp.(logPi)
    ret2 = exp.((-1.0 / theta) * LogExpFunctions.logsumexp(log.(tau_theta) + repeat((theta .* logPi + logY), 1, len), dims=1)') - exp.(logP)

    # ret1 = (-1.0/theta) * log.(taumtheta * (logP.^(theta) .* exp.(logY))) - logPi
    # ret2 = (-1.0/theta) * log.(taumtheta' * (logPi.^(-theta) .* exp.(logY))) - logP

    return [ret1; ret2]

  end

  # f(u, p) = u .* u .- p
  # u0 = [1.0 ./ A0 ; 1.0 ./ A0]
  u0 = ones(Float64, 2 * len)
  p = (log.(A0), log.(L0), tau_theta, len)

  prob = NonlinearSolve.NonlinearProblem(objfun, u0, p)
  @time sol = CommonSolve.solve(prob, NonlinearSolveFirstOrder.NewtonRaphson(autodiff=ADTypes.AutoForwardDiff()); show_trace=Val(true), trace_level=NonlinearSolveBase.TraceAll(1))

  logPi = sol.u[1:len]
  logP = sol.u[len+1:2len]

  exp.(logP)
  exp.(logPi)

  return (logPi, logP)

end

export main

end # module
