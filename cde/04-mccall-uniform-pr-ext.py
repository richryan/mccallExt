# Program: 04-mccall-uniform-pr-ext.py
# Purpose: Compute a McCall worker's
#   1. expected welfare,
#   2. duration of unemployment, and
#   3. accepted wage
# when the wage-offer distribution is uniform and the worker
# can mis-evaluate
#   ** the probability that benefits are extended
#
# E-mail: RichRyan@csub.edu
# Date Started: 2023-01-12
# Date Revised: 2023-07-18
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
# njit is an alias for @jit(nopython=True)
from numba import jit, njit, float64, int64 
from numba.experimental import jitclass
import quantecon as qe
from scipy import stats 
from scipy import optimize as optim
import pandas as pd
from datetime import date
# import the mccall module comprising functions written for this project
import mccall
from importlib import reload
reload(mccall)

# File parameters
file_prg = '04-mccall-uniform-pr-ext'

aws = 0

print('START OF PROGRAM: ' + file_prg)
qe.tic()

# ============================
# === Load deep parameters ===
# ============================

fin = Path.cwd().parent / Path('out') / Path('dat_03-mccall-uniform-parameters.npz')
dat = np.load(fin)

β = dat['β'].item()
c = dat['c'].item()
z = dat['z'].item()
target_nperiods = dat['target_nperiods'].item()

# ============================
# === Chance for extension ===
# ============================

nperiods_ui_compensation = target_nperiods
Δ = 13
δ = 0.5

# =======================================================================
# === Welfare costs when incorrect assessment of extension likelihood ===
# =======================================================================
print(" ")
print(" ")
print(" =======================================================================")
print(" === Welfare costs when incorrect assessment of extension likelihood ===")
print(" =======================================================================")
print(" ")

if aws == 1:
    n_reps = 500_000_000
else:
    n_reps = 1_000

# Define true probability that wages are extended.
pr_ext = 0.5

probs_gridSize = 20
# Add in pr_ext to grid of probabilities.
probs = np.concatenate([np.array([pr_ext]),
                        np.linspace(0.3, 0.7, num=probs_gridSize)
                        ])
probs.sort()
# Row position of true prob of ext
pos_true = np.intc(probs_gridSize / 2)

vWelfare = np.empty((probs_gridSize + 1, 3))

# Regardless of assessed likelihood, the extended benefits are agree upon.
nperiods_ui_compensation_max = nperiods_ui_compensation - 1 + Δ
reservation_wages_extended = mccall.fun_compute_reservation_wages_uniform(β=β, c=c, z=z, n=nperiods_ui_compensation_max)

# Save sequences of reservation wages. 
horizon = np.arange(0, nperiods_ui_compensation_max + 1, step = 1)
dat = {"horizon" : horizon,
       "reservation_wages" : reservation_wages_extended,
       "pr_ext" : "extended",
       "periods_ui_compensation" : nperiods_ui_compensation_max,
       "length_extension" : Δ} 
df = pd.DataFrame(dat)

horizon = np.arange(0, nperiods_ui_compensation + 1, step = 1)
for i, δ in enumerate(probs):
    mccall.progressbar(i, tot=probs_gridSize+1)
    reservation_wages_chance_extδ = mccall.fun_compute_reservation_wages_uniform_ext(β=β, c=c, z=z, n=nperiods_ui_compensation, δ=δ, Δ=Δ)
    # Save sequence of reservation wages
    if δ == pr_ext:
        true_pr_ext = "yes"
    else:
        true_pr_ext = "no"
    iDat = {"horizon" : horizon,
            "reservation_wages" : reservation_wages_chance_extδ,
            "pr_ext" : δ,
            "true_pr_ext" : true_pr_ext,
            "periods_ui_compensation" : nperiods_ui_compensation,
            "length_extension" : Δ}
    idf = pd.DataFrame(iDat)
    df = pd.concat([df, idf])    
    

    i_stopping_time, i_accepted_wage, i_welfare = mccall.compute_mean_accept_uniform_ext_parallel(β=β,
                                                                                                  z=z,
                                                                                                  c=c,
                                                                                                  reservation_wages=reservation_wages_chance_extδ,
                                                                                                  # Extended are always the same
                                                                                                  reservation_wages_ext=reservation_wages_extended,
                                                                                                  # True pr of extension
                                                                                                  δ=pr_ext,
                                                                                                  Δ=Δ,
                                                                                                  num_reps=n_reps)

    
    vWelfare[i, 0] = i_stopping_time
    vWelfare[i, 1] = i_accepted_wage
    vWelfare[i, 2] = i_welfare


# Compute relative welfare
vWelfare_relative_welfare = (vWelfare[:, 2] / vWelfare[pos_true, 2] - 1) * 100

if aws == 0:
    # Plot relative welfares
    fig, ax = plt.subplots(1,1)
    ax.plot(probs, vWelfare_relative_welfare, '-.m')
    ax.set_title('Welfare')
    plt.show()

# =================
# === Save data ===
# =================

today = date.today()
day = today.strftime("%Y-%m-%d")            

# Save reservation wages to .csv file.
fout_reservation_wages = Path.cwd().parent / Path('out') / Path('dat_' + file_prg + '-seq-reservation-wages.csv')
df.to_csv(fout_reservation_wages, index=False)

vWelfare_relative_welfare = np.expand_dims(vWelfare_relative_welfare, axis=1)
# Concatenate welfares with probabilities to write
vWelfare = np.hstack([probs.reshape(probs_gridSize + 1, 1), vWelfare])
vWelfare = np.hstack([vWelfare, vWelfare_relative_welfare])

# Save welfare data to NumPy array.
fout_numpy = Path.cwd().parent / Path('out') / Path('dat_' + file_prg + '_' + day + '.npz')
np.savez(fout_numpy, vWelfare=vWelfare, β=β, c=c, z=z, nperiods_ui_compensation=nperiods_ui_compensation, Δ=Δ, δ=δ)

# Write welfare data to .csv file.
fout_welfare = Path.cwd().parent / Path('out') / Path('dat_' + file_prg + '_' + day + '.csv')
fout_header = 'probs_perceived, stopping_time, accepted_wage, welfare, relative_welfare'
np.savetxt(fout_welfare, vWelfare, header=fout_header, delimiter=',', comments='')

print(' ')
print('Elapsed time:')
print(' ')
qe.toc()
print(" === END Welfare costs when incorrect assessment of extension likelihood ===")

# Local Variables:
# coding: utf-8
# eval: (set-input-method 'TeX)
# End:    
