# Program: 01-get-sequence-reservation-wages.py
# Purpose: Generate sequences of reservation wages when there is a
# chance that UI benefits are extended.
#
# E-mail: RichRyan@csub.edu
# Date Started: 2023-07-17
# Date Revised: 2023-07-17
from pathlib import Path
import seaborn as sns
# Apply the default theme
sns.set_theme()
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
from numba import jit, njit, float64, int64 # njit is an alias for @jit(nopython=True)
from numba.experimental import jitclass
import quantecon as qe
from scipy import stats 
from scipy import optimize as optim
import pandas as pd

# import the mccall module comprising functions written for this project
import mccall
from importlib import reload
reload(mccall)

print(' ')
print('=== Start Program ===')
print(' ')
qe.tic()

# File parameters
filePrg = '01-get-sequence-reservation-wages'

csubBlue = (0 / 255, 53 / 255, 148 / 255)
bpink1 = '#e0218a'
bpink2 = '#ff68bf'

# Plotting parameters
golden = 0.5*(1 + np.sqrt(5))
mywidth = 11
plt.rcParams["figure.figsize"] = (mywidth, mywidth/golden)  #set default figure size
plt.rcParams.update({'font.size': 16})

# ====================================================
# === Get reasonable value for the flow of nonwork ===
# ====================================================
β = 0.95
vFlowNonwork = np.linspace(0.05, 0.9*(1 + 1 / β) / 2, 50)
vNperiodsUnemployed = np.empty_like(vFlowNonwork)
periodsUnempTarget = 10
fun_match_periods_unemployed = mccall.fun_make_match_periods_unemployed_unif(β=β, nperiods=periodsUnempTarget)    

for i in range(len(vFlowNonwork)):
    iFlowNonwork = vFlowNonwork[i]
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

# === Theoretical expected welfare
# This section of the code computes the theoretical expected welfare
# of the searching worker.
flowNonwork = res.x

# ==========================================
# === Compute sequences of reservation wages
# ==========================================

z = flowNonwork / 2.0
c = flowNonwork / 2.0

nperiodsUI = 15
Δ = 13
δ = 0.5

maxPeriodsUI = nperiodsUI - 1 + Δ
reservationWagesExtended = mccall.fun_compute_reservation_wages_uniform(β=β, c=c, z=z, n=maxPeriodsUI)


vPrExt = np.arange(start=0.1, stop=1.0, step=0.1)
horizon = np.arange(0, maxPeriodsUI + 1, step = 1)

dat = {"horizon" : horizon,
       "reservation_wages" : reservationWagesExtended,
       "pr_ext" : "extended"} 
df = pd.DataFrame(dat)

# pd.Series(reservationWagesExtended, index = horizon)

horizon = np.arange(0, nperiodsUI + 1, step = 1)
for i, δ in enumerate(vPrExt):
    iReservationWages = mccall.fun_compute_reservation_wages_uniform_ext(β=β, c=c, z=z, n=nperiodsUI, δ=δ, Δ=Δ)
    iDat = {"horizon" : horizon,
            "reservation_wages" : iReservationWages,
            "pr_ext" : δ}
    idf = pd.DataFrame(iDat)
    df = pd.concat([df, idf])

fout = Path.cwd().parent / Path('out') / Path('dat_' + filePrg + '-seq-reservation-wages.csv')
df.to_csv(fout, index=False)

# =======================
# Properties of the model
# =======================

# Higher β
βhi = 0.97

maxPeriodsUI = nperiodsUI - 1 + Δ
reservationWagesExtended_βhi = mccall.fun_compute_reservation_wages_uniform(β=βhi, c=c, z=z, n=maxPeriodsUI)

vPrExt = np.arange(start=0.1, stop=1.0, step=0.1)
horizon = np.arange(0, maxPeriodsUI + 1, step = 1)

dat_properties = {"horizon" : horizon,
                  "reservation_wages" : reservationWagesExtended_βhi,
                  "bbeta" : βhi * np.ones_like(horizon),
                  "ui_compensation" : c * np.ones_like(horizon),
                  "pr_ext" : "extended"} 
df_properties = pd.DataFrame(dat_properties)


# Higher UI compensation
chi = c * 1.1

reservationWagesExtended_chi = mccall.fun_compute_reservation_wages_uniform(β=β, c=chi, z=z, n=maxPeriodsUI)

dat_chi = {"horizon" : horizon,
           "reservation_wages" : reservationWagesExtended_chi,
           "bbeta" : β * np.ones_like(horizon),
           "ui_compensation" : chi * np.ones_like(horizon),           
           "pr_ext" : "extended"} 
df_chi = pd.DataFrame(dat_chi)

df_properties = pd.concat([df_properties, df_chi])

horizon = np.arange(0, nperiodsUI + 1, step = 1)
for i, δ in enumerate(vPrExt):
    iReservationWages_βhi = mccall.fun_compute_reservation_wages_uniform_ext(β=βhi, c=c, z=z, n=nperiodsUI, δ=δ, Δ=Δ)
    iDat_βhi = {"horizon" : horizon,
                "reservation_wages" : iReservationWages_βhi,
                "bbeta" : βhi * np.ones_like(horizon),
                "ui_compensation" : c * np.ones_like(horizon),                       
                "pr_ext" : δ}
    idf_βhi = pd.DataFrame(iDat_βhi)
    df_properties = pd.concat([df_properties, idf_βhi])

    iReservationWages_chi = mccall.fun_compute_reservation_wages_uniform_ext(β=β, c=chi, z=z, n=nperiodsUI, δ=δ, Δ=Δ)
    iDat_chi = {"horizon" : horizon,
                "reservation_wages" : iReservationWages_βhi,
                "bbeta" : β * np.ones_like(horizon),
                "ui_compensation" : chi * np.ones_like(horizon),                       
                "pr_ext" : δ}
    idf_chi = pd.DataFrame(iDat_chi)
    df_properties = pd.concat([df_properties, idf_chi])


fout_properties = Path.cwd().parent / Path('out') / Path('dat_' + filePrg + '-seq-reservation-wages-properties.csv')
df.to_csv(fout_properties, index=False)

print(' ')
print('Elapsed time:')
print(' ')
qe.toc()

# Local Variables:
# coding: utf-8
# eval: (set-input-method 'TeX)
# End:    
