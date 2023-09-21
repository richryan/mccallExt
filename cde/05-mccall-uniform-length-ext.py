# Program: 04-mccall-uniform-length-ext.py
# Purpose: Compute a McCall worker's
#   1. expected welfare,
#   2. duration of unemployment, and
#   3. accepted wage
# when the wage-offer distribution is uniform and the worker
# can mis-evaluate
#   ** the length of the extension
#
# Note: The variable aws denotes whether a large, time-consuming
# simulation is conducted.
#
#   aws = 0 : small simulation
#   aws = 1 : large simulation
#
# E-mail: RichRyan@csub.edu
# Date Started: 2023-05-30
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

aws = 0

# File parameters
file_prg = '05-mccall-uniform-length-ext'

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

# =======================================================================
# === Welfare costs when incorrect assessment of extension likelihood ===
# =======================================================================
print(" ")
print(" ")
print(" =======================================================================")
print(" === Welfare costs when incorrect assessment of extension length =======")
print(" =======================================================================")
print(" ")

nperiods_ui_compensation = target_nperiods
δ = 0.5
if aws == 1:
    n_reps = 500_000_000
else:
    n_reps = 1_000

# True number of extensions
Δ = 13
lengthExtension_halfGridSize = 10
lengthExtension_gridSize = 2 * lengthExtension_halfGridSize + 1
vLengthExtension = np.arange(Δ - lengthExtension_halfGridSize, Δ + lengthExtension_halfGridSize + 1)
# Row position of true prob of ext
pos_true = lengthExtension_halfGridSize
vWelfare = np.empty((lengthExtension_gridSize, 3))

# Regardless of assessed likelihood, the extended benefits are agree upon
nperiods_ui_compensation_max = nperiods_ui_compensation - 1 + Δ
reservation_wages_extended = mccall.fun_compute_reservation_wages_uniform(β=β, c=c, z=z, n=nperiods_ui_compensation_max)

# Save sequences of reservation wages. 
horizon = np.arange(0, nperiods_ui_compensation_max + 1, step = 1)
dat = {"horizon" : horizon,
       "reservation_wages" : reservation_wages_extended,
       "length_ext" : "extended",
       "periods_ui_compensation" : nperiods_ui_compensation_max,
       "length_extension" : Δ} 
df = pd.DataFrame(dat)

horizon = np.arange(0, nperiods_ui_compensation + 1, step = 1)
for i, iΔ in enumerate(vLengthExtension):
    mccall.progressbar(i, tot=lengthExtension_gridSize)
    reservation_wages_chance_extΔ = mccall.fun_compute_reservation_wages_uniform_ext(β=β, c=c, z=z, n=nperiods_ui_compensation, δ=δ, Δ=iΔ)

    if iΔ == Δ:
        true_length_ext = "yes"
    else:
        true_length_ext = "no"
    iDat = {"horizon" : horizon,
            "reservation_wages" : reservation_wages_chance_extΔ,
            "pr_ext" : δ,
            "true_length_ext" : true_length_ext,
            "periods_ui_compensation" : nperiods_ui_compensation,
            "length_extension" : iΔ}
    idf = pd.DataFrame(iDat)
    df = pd.concat([df, idf])            

    i_stopping_time, i_accepted_wage, i_welfare = mccall.compute_mean_accept_uniform_ext_parallel(β=β,
                                                                                                  z=z,
                                                                                                  c=c,
                                                                                                  reservation_wages=reservation_wages_chance_extΔ,
                                                                                                  # Extended seq of reservation wages are always the same
                                                                                                  reservation_wages_ext=reservation_wages_extended,
                                                                                                  # True pr of extension
                                                                                                  δ=δ,
                                                                                                  Δ=Δ,
                                                                                                  num_reps=n_reps)

    
    vWelfare[i, 0] = i_stopping_time
    vWelfare[i, 1] = i_accepted_wage
    vWelfare[i, 2] = i_welfare


# Compute relative welfare
vWelfare_relative_welfare = (vWelfare[:, 2] / vWelfare[pos_true, 2] - 1) * 100

if aws == 0:
    # Plot welfares
    fig, ax = plt.subplots(1,1)
    ax.plot(vLengthExtension, vWelfare_relative_welfare, '-.m')
    ax.set_title('Welfare')
    plt.show()

vWelfare_relative_welfare = np.expand_dims(vWelfare_relative_welfare, axis=1)
# Concatenate welfares with probabilities to write
vWelfare = np.hstack([vLengthExtension.reshape(lengthExtension_gridSize, 1), vWelfare])
vWelfare = np.hstack([vWelfare, vWelfare_relative_welfare])

# =================
# === Save data ===
# =================
today = date.today()
day = today.strftime("%Y-%m-%d")

# Save reservation wages to .csv file.
fout_reservation_wages = Path.cwd().parent / Path('out') / Path('dat_' + file_prg + '-seq-reservation-wages.csv')
df.to_csv(fout_reservation_wages, index=False)

# Save data to NumPy array.
fout_numpy = Path.cwd().parent / Path('out') / Path('dat_' + file_prg + '_' + day + '.npz')
np.savez(fout_numpy, vWelfare=vWelfare, β=β, c=c, z=z, nperiods_ui_compensation=nperiods_ui_compensation, Δ=Δ, δ=δ)

# Write data to CSV file.
fout = Path.cwd().parent / Path('out') / Path('dat_' + file_prg + '_' + day + '.csv')
fout_header = 'length_ext_perceived, stopping_time, accepted_wage, welfare, relative_welfare'
np.savetxt(fout, vWelfare, header=fout_header, delimiter=',', comments='')

print('Elapsed time:')
print(qe.toc())
print(' ')
print('END OF PROGRAM: ' + file_prg)

# Local Variables:
# coding: utf-8
# eval: (set-input-method 'TeX)
# End:    
