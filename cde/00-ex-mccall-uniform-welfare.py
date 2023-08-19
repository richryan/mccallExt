# Program: 00-ex-mccall-uniform-welfare.py
# Purpose: Examples of the McCall model of sequential job search. The
# program boosts confidence that the code is doing what it should be
# doing by comparing simulated statistics with theoretical statistics.
#
# The aim of this program is comparing the expected welfare of a
# McCall worker calculated numerically and calculated
# theoretically. Theoretically the worker expects to spend
#    F(w_R) / [1 - F(w_R)]
# periods unemployed and then get the expected wage (conditional on
# the wage being above the reservation wage thereafter).
#
# The flow value of non-work is adjusted so that the worker expects to
# be unemployed 8 periods.
#
# Comparisons are made between sequences of reservation wages computed
# using MCMC integration and theoretical, closed-form solutions.
#
# E-mail: RichRyan@csub.edu
# Date Started: 2022-06-15
# Date Revised: 2023-07-17
from pathlib import Path
import seaborn as sns
# Apply the default theme
sns.set_theme()
import matplotlib.pyplot as plt
plt.set_loglevel('warning')
import numpy as np
import sys
import random
from numba import jit, njit, float64, int64 # njit is an alias for @jit(nopython=True)
from numba.experimental import jitclass
import quantecon as qe
from scipy import stats 
from scipy import optimize as optim

qe.tic()

# import the mccall module comprising functions written for this project
import mccall
from importlib import reload
reload(mccall)

# File parameters
filePrg = '00-ex-mccall-uniform-welfare'

csubBlue = (0 / 255, 53 / 255, 148 / 255)
bpink1 = '#e0218a'
bpink2 = '#ff68bf'


# Log file
import logging
logFile = Path.cwd().parent / Path('log') / Path(filePrg + '.log')
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.basicConfig(filename=str(logFile),
                    filemode='w',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')

# Plotting parameters
golden = 0.5*(1 + np.sqrt(5))
mywidth = 11
plt.rcParams["figure.figsize"] = (mywidth, mywidth/golden)  #set default figure size
plt.rcParams.update({'font.size': 16})

# === For guidance on seeding, see: https://numba.readthedocs.io/en/stable/reference/numpysupported.html#random
logging.info('START of program')
logging.info(" ")
logging.info("=======================================")
logging.info("   Guidance on random variables:")
logging.info("=======================================")
logging.info(" ")
logging.info(">>> mccall.seed(1234)")
logging.info(">>> mccall.rand(3)")
logging.info("generates:")
mccall.seed(1234)
logging.info(mccall.rand(3))

mccall.seed(1234)
logging.info("And again:")
logging.info((mccall.rand(3)))

logging.info(" ")
logging.info("Calling")
logging.info(">>> np.random.seed(1234)")
logging.info(">>> mccall.rand(3)")
logging.info("results in different output:")
np.random.seed(1234)
logging.info(mccall.rand(3))

np.random.seed(1234)
logging.info(mccall.rand(3))

logging.info("=== END Gudiance on random variables ==")
logging.info(" ")
logging.info(" ")
logging.info(" ")
logging.info(" ==============================================================")
logging.info(" === Determining periods unemployed / flow value of nonwork ===")
logging.info(" ==============================================================")
logging.info(" ")

# === Theoretical Welfare for Uniform[0,1] distribution
β = 0.95
vFlowNonwork = np.linspace(0.05, 0.9*(1 + 1 / β) / 2, 50)
vNperiodsUnemployed = np.empty_like(vFlowNonwork)
periodsUnempTarget = 8
fun_match_periods_unemployed = mccall.fun_make_match_periods_unemployed_unif(β=β, nperiods=periodsUnempTarget)    

for i in range(len(vFlowNonwork)):
    iFlowNonwork = vFlowNonwork[i]
    # Theoretical reseravtion wage is computed using closed-form solution.
    reservationWage = 1/β - ((1 - β)*(1 + β - 2 * β * iFlowNonwork))**(1 / 2) / β
    # Theoretical expected number of periods unemployed computed,
    # using the fact that the cdf of a uniform[0,1] RV is x.
    vNperiodsUnemployed[i] = reservationWage / (1 - reservationWage)
        
# Define parameter space to search over flow value of nonwork.
# The maximum wage draw is 1, so flow nonwork is between 0 and 1.
brac_lo = 0.001
brac_hi = np.min([0.9*(1 + 1/β)/2 , 0.99])

res = optim.minimize_scalar(fun_match_periods_unemployed, bracket=(brac_lo, brac_hi), method='Brent', tol=1e-12)

fig, ax = plt.subplots(1,1)
ax.hlines(y=periodsUnempTarget, xmin=vFlowNonwork[0], xmax=vFlowNonwork[-1], color=bpink2, linewidth=1.5, linestyle='dotted', label='Targeted number of periods')
ax.vlines(x=res.x, ymin=np.min(vNperiodsUnemployed), ymax=np.max(vNperiodsUnemployed), color=bpink1, linewidth=1.5, linestyle='dashdot', label='Optimal parameter value')
ax.plot(vFlowNonwork, vNperiodsUnemployed, color=csubBlue, linewidth=2.5, linestyle='solid', label='Theoretical expectation')
ax.set_xlabel("Flow value of nonwork")
ax.set_ylabel("Expected periods unemployed")
# Add a note outside of the frame.
ax.annotate('Expected periods unemployed increases as the flow value of nonwork increases',
            xy = (1.0, -0.2),
            xycoords='axes fraction',
            ha='right',
            va='center',
            fontsize=10)
ax.legend(fancybox=False, frameon=False, fontsize=12)
plt.show()
figOutMatchUnempDur = Path.cwd().parent / Path('out') / Path('fig_' + filePrg + '-match.pdf')
fig.savefig(figOutMatchUnempDur, bbox_inches='tight')

logging.info(" === END Determining periods unemployed / flow value of nonwork ===")

# === Theoretical expected welfare
# This section of the code computes the theoretical expected welfare
# of the searching worker.
flowNonwork = res.x
reservationWageTheory = 1/β - (((1 - β)*(1 + β - 2*β*flowNonwork))**(1/2))/β
nunemp = periodsUnempTarget
welfareTheory = reservationWageTheory**2 / (1 - β) + (1 - reservationWageTheory**2)/(1 - β)/2

# === Welfare for standard McCall model ===
logging.info(" ")
logging.info(" ")
logging.info("==========================================")
logging.info("=== Welfare for standard McCall model ===")
logging.info("==========================================")
logging.info(" ")
logging.info("The standard model features sequential search and benefits that do not expire.")

# nreps = 1_000_000
nreps = 100_000

# Initialize instance of class.
mcmc_unif = mccall.McCallModelUniform(c=flowNonwork, z=flowNonwork, β=β, mc_size=2_000_000)

reservationWageMCMC = mccall.compute_reservation_wage0(mcmc_unif)

statsTheory = mccall.compute_mean_accept_uniform_parallel(β=β, c=flowNonwork, z=flowNonwork, reservation_wages=np.array([reservationWageTheory]), num_reps=nreps)
statsMCMC = mccall.compute_mean_accept_uniform_parallel(β=β, c=flowNonwork, z=flowNonwork, reservation_wages=reservationWageMCMC, num_reps=nreps)

Fwr = stats.uniform.cdf(reservationWageTheory)

# Print results to the log.
logging.info("Based on a simulation of size " + str(nreps))
logging.info(" ")
logging.info("Theoretical expected number of unemployed periods = " + str(Fwr / (1 - Fwr)))
logging.info("(based on varying flow value of nonwork)")
logging.info("Avg number of unemployed periods using theoretical reservation wage = " + str(statsTheory[0]))
logging.info("Avg number of unemployed periods using MCMC-computed reservation wage = " + str(statsMCMC[0]))
logging.info(" ")
logging.info("Theoretical welfare is " + str(welfareTheory))
logging.info("Simulated welfare using theoretical reservation wage: " + str(statsTheory[2]))
logging.info("Simulated welfare using MCMC-computed reservation wage: " + str(statsMCMC[2]))
logging.info(" ")
logging.info(" === END Welfare for standard McCall model ===")
logging.info(" ")
logging.info(" ")

# ===================================================
# === Work with Finite periods of UI compensation ===
# ===================================================
logging.info(" ")
logging.info(" ")
logging.info("================================================================")
logging.info("=== Welfare for McCall model with finite periods of benefits ===")
logging.info("================================================================")
logging.info(" ")

z = flowNonwork / 2.0
b = flowNonwork / 2.0

mcmc_unif_finite = mccall.McCallModelUniform(c=b, z=z, β=β, mc_size=1_000_000)

nperiodsUI = 26
reservationWagesTheory = mccall.fun_compute_reservation_wages_uniform(β=β, c=b, z=z, n=nperiodsUI)
reservationWagesMCMC = mccall.fun_compute_reservation_wages_mcmc(mcmc_unif_finite, n=nperiodsUI, tol=1e-8, max_iter=10_000)

horizon = np.arange(0, nperiodsUI + 1, step = 1)

fig, ax = plt.subplots(2, 1, sharex='col', sharey='none')
ax[0].plot(horizon, reservationWagesTheory, color=csubBlue, linewidth=3.5, linestyle='solid', label='Theory')
ax[0].plot(horizon, reservationWagesMCMC, color=bpink1, linewidth=3.5, linestyle='dotted', label='MCMC')
ax[1].plot(horizon, reservationWagesTheory - reservationWagesMCMC, color=csubBlue, linestyle='solid', linewidth=1.5)
ax[0].set_xticks(np.arange(0, nperiodsUI + 1, step=1))
ax[1].set_xticks(np.arange(0, nperiodsUI + 1, step=1))
ax[1].set_xlabel("Periods of remaining UI benefits")
ax[0].set_ylabel("Reservation wage")
ax[1].set_ylabel("Error")
ax[1].annotate('Workers become less selective as UI benefits expire.',
            xy = (1.0, -0.4),
            xycoords='axes fraction',
            ha='right',
            va='center',
            fontsize=10)
ax[0].legend(loc='lower right', frameon=False)
# ax[0].set_title("Sequence of reservation wages")
# ax[1].set_title("Difference between theory and simulation")
# Label panels as in https://matplotlib.org/2.0.2/users/transforms_tutorial.html
ax[0].text(-0.1, 1.1, 'A', transform=ax[0].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
ax[0].text(-0.1, 1.1, 'B', transform=ax[1].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
plt.show()

figOutSeqReservationWages = Path.cwd().parent / Path('out') / Path('fig_' + filePrg + '-sequence-reservation-wages.pdf')
fig.savefig(figOutSeqReservationWages, bbox_inches='tight')

# Compute welfare for the case where UI benefits are paid for finite period
reservationWageN = reservationWagesTheory[nperiodsUI]
welfareTheoryN = (1 + reservationWageN**2) / (1 - β) / 2

nrepsN = 20_000_000


simReservationWagesTheory = mccall.compute_mean_accept_uniform_parallel(β=mcmc_unif_finite.β,
                                                                        c=mcmc_unif_finite.c,
                                                                        z=mcmc_unif_finite.z,
                                                                        reservation_wages=reservationWagesTheory,
                                                                        num_reps=nrepsN)
simReservationWagesTheoryWelfare = simReservationWagesTheory[2]

simReservationWagesMCMC = mccall.compute_mean_accept_uniform_parallel(β=mcmc_unif_finite.β,
                                                                      c=mcmc_unif_finite.c,
                                                                      z=mcmc_unif_finite.z,
                                                                      reservation_wages=reservationWagesMCMC,
                                                                      num_reps=nrepsN)
simReservationWagesMCMCwelfare = simReservationWagesMCMC[2]

logging.info("Based on a simulation of size " + str(nrepsN))
logging.info(" ")
logging.info("The difference in welfare computed using the seq of")
logging.info("theoretically computed wages and theoretical welfare is")
logging.info(str(100 * (simReservationWagesTheoryWelfare / welfareTheoryN - 1)) + " percent")
logging.info(" ")
logging.info("The difference in welfare computed using the seq of")
logging.info("MCMC-computed wages and theoretical welfare is")
logging.info(str(100 * (simReservationWagesMCMCwelfare / welfareTheoryN - 1)) + " percent")
logging.info("(Note: 1.0 is one percent.)")
logging.info(" ")
logging.info(" === END Welfare for McCall model with finite periods of benefits ===")

# ==================================
# === Matching expected duration ===
# ==================================
logging.info(" ")
logging.info(" ")
logging.info("==================================================================")
logging.info("=== Matching expected duration with finite periods of benefits ===")
logging.info("==================================================================")
logging.info(" ")

nperiodsUI = 3

reservationWages3 = mccall.fun_compute_reservation_wages_uniform(β=β, c=b, z=z, n=nperiodsUI)
reservationWages3numerical = mccall.fun_compute_reservation_wages_mcmc(mcmc_unif_finite, n=nperiodsUI, tol=1e-8, max_iter=10_000)


horizon = np.arange(0, nperiodsUI + 1, step = 1)
fig, ax = plt.subplots(1,1)
ax.plot(horizon, reservationWages3, color=csubBlue, linestyle='solid', linewidth=3.5)
ax.plot(horizon, reservationWages3numerical, color=bpink1, linestyle='dotted', linewidth=3.5)
ax.set_title('Theoretical vs numerical sequence of reservation wages')
plt.show()

# Add up the infite sum numerically untill convergence
expectedDuration = (stats.uniform.cdf(reservationWages3[3]) * (1 - stats.uniform.cdf(reservationWages3[2])) * 1.0
                    + stats.uniform.cdf(reservationWages3[3]) * stats.uniform.cdf(reservationWages3[2])
                    * (1 - stats.uniform.cdf(reservationWages3[1])) * 2.0
                    + stats.uniform.cdf(reservationWages3[3]) * stats.uniform.cdf(reservationWages3[2]) * stats.uniform.cdf(reservationWages3[1])
                    * (1 - stats.uniform.cdf(reservationWages3[0])) * 3.0)

error = 1.0
tol = 1e-12
max_iter = 1_000
i = 0
term = stats.uniform.cdf(reservationWages3[3]) * stats.uniform.cdf(reservationWages3[2]) * stats.uniform.cdf(reservationWages3[1])
k = 4
while i < max_iter and error > tol:
    expectedDurationNew = expectedDuration + term * (stats.uniform.cdf(reservationWages3[0])**(k - 3)) * (1 - stats.uniform.cdf(reservationWages3[0])) * k
    error = np.abs(expectedDurationNew - expectedDuration)    
    expectedDuration = expectedDurationNew
    k += 1
    i += 1                                                                                                                               

simDuration3 = mccall.compute_mean_accept_uniform_parallel(β=β, z=z, c=b, reservation_wages=reservationWages3, num_reps=1_000_000)
simDuration3Duration = simDuration3[0]

duration3Theory = (stats.uniform.cdf(reservationWages3[3]) * (1 - stats.uniform.cdf(reservationWages3[2])) 
                    + stats.uniform.cdf(reservationWages3[3]) * stats.uniform.cdf(reservationWages3[2])
                    * (1 - stats.uniform.cdf(reservationWages3[1])) * 2
                    + stats.uniform.cdf(reservationWages3[3]) * stats.uniform.cdf(reservationWages3[2]) * stats.uniform.cdf(reservationWages3[1])
                    * (1 - stats.uniform.cdf(reservationWages3[0])) * 3
                    + stats.uniform.cdf(reservationWages3[3]) * stats.uniform.cdf(reservationWages3[2]) * stats.uniform.cdf(reservationWages3[1])
                    * stats.uniform.cdf(reservationWages3[0])  
                    * (4.0 * (1 - stats.uniform.cdf(reservationWages3[0])) + stats.uniform.cdf(reservationWages3[0]))
                    / (1 - stats.uniform.cdf(reservationWages3[0])))

logging.info("    (Aside: The difference between the closed-form solution and the while loop is " + str(duration3Theory - expectedDuration) + ")")
logging.info(" ")
logging.info(" ")
logging.info("The difference in duration between the simulation and theory, using seq of numerical wages, is " + str(100 * (simDuration3Duration / duration3Theory - 1)) + " percent")
logging.info(" ")
logging.info("=== END Matching expected duration with finite periods of benefits ===")
logging.info(' ')

# ================================================
# === Comparison of reservation-wage sequences ===
# ================================================
logging.info("====================================================")
logging.info("=== Comparison of sequences of reservation-wages ===")
logging.info("====================================================")

mcmc_unif_finite = mccall.McCallModelUniform(c=b, z=z, β=β, mc_size=2_000_000)

nperiodsCompensation = 20
Δ = 25
δ = 0.5

# Max number of periods with benefits = nperiods (which includes the current period)
# + an extension between now and next period, which is nperiods - 1
# (Includes wages when there are no remaining periods of UI compensation)
maxPeriodsCompensation = nperiodsCompensation - 1 + Δ 
reservationWagesExtended = mccall.fun_compute_reservation_wages_uniform(β=β, c=b, z=z, n=maxPeriodsCompensation)
reservationWagesδ = mccall.fun_compute_reservation_wages_uniform_ext(β=β, c=b, z=z, n=nperiodsCompensation, δ=δ, Δ=Δ)

reservationWagesExtendedNumeric = mccall.fun_compute_reservation_wages_mcmc(mcmc_unif_finite, n=maxPeriodsCompensation)
reservationWagesδNumeric = mccall.fun_compute_reservation_wages_ext_mcmc(mcmc_unif_finite,
                                                                         reservation_wages_extended=reservationWagesExtendedNumeric,
                                                                         n=nperiodsCompensation,
                                                                         δ=δ, Δ=Δ,
                                                                         tol=1e-10)

horizon = np.arange(0, nperiodsCompensation + 1, step = 1)
fig, ax = plt.subplots(2, 1, sharex='col', sharey='none')
ax[0].plot(horizon, reservationWagesδ, color=csubBlue, linewidth=3.5, linestyle='solid', label='Theoretical')
ax[0].plot(horizon, reservationWagesδNumeric, color=bpink1, linewidth=3.5, linestyle='dotted', label='MCMC')
ax[1].plot(horizon, reservationWagesδ - reservationWagesδNumeric, color=csubBlue)
ax[1].set_xlabel("Periods of remaining UI benefits")
ax[0].set_xticks(np.arange(0, nperiodsCompensation + 1, step=1))
ax[1].set_xticks(np.arange(0, nperiodsCompensation + 1, step=1))
ax[0].set_ylabel("Reservation wage")
ax[1].set_ylabel("Error")
ax[0].legend(loc='lower right', frameon=False)
# Label panels as in https://matplotlib.org/2.0.2/users/transforms_tutorial.html
ax[0].text(-0.1, 1.1, 'A', transform=ax[0].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
ax[0].text(-0.1, 1.1, 'B', transform=ax[1].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

figOutSeqReservationWagesComparison = Path.cwd().parent / Path('out') / Path('fig_' + filePrg + '-comparison-seq-reservation-wages-chance-ext.pdf')
fig.savefig(figOutSeqReservationWagesComparison, bbox_inches='tight')

logging.info('Elapsed time:')
logging.info(' ')
logging.info(qe.toc())
# logging.shutdown()

# Local Variables:
# coding: utf-8
# eval: (set-input-method 'TeX)
# End:    
