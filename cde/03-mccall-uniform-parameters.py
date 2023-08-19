# Program: 03-mccall-uniform-parameters.py
# Purpose: Determine and set parameters for the simulations. 
#
# E-mail: RichRyan@csub.edu
# Date Started: 2023-07-18
# Date Revised: 2023-07-18

from pathlib import Path
# import matplotlib.pyplot as plt
import numpy as np
# import sys
# import random
# # njit is an alias for @jit(nopython=True)
# from numba import jit, njit, float64, int64 
# from numba.experimental import jitclass
# import quantecon as qe
# from scipy import stats 
from scipy import optimize as optim
# from datetime import date
# # import the mccall module comprising functions written for this project
import mccall
from importlib import reload
reload(mccall)

# aws = 0

# File parameters
file_prg = '03-mccall-uniform-parameters'

# =======================
# === Deep parameters ===
# =======================
β = 0.95
target_nperiods = 10
fun_match_periods_unemployed = mccall.fun_make_match_periods_unemployed_unif(β=β, nperiods=target_nperiods)    

# Flow is between 0 and 1 (the max wage is 1)
brac_lo = 0.001
brac_hi = np.min([0.9*(1 + 1/β)/2 , 0.99])

res = optim.minimize_scalar(fun_match_periods_unemployed, bracket=(brac_lo, brac_hi), method='Brent', tol=1e-12)

flowNonwork = res.x

z = flowNonwork / 2.0
c = flowNonwork / 2.0

fout = Path.cwd().parent / Path('out') / Path('dat_' + file_prg + '.npz')
np.savez(fout, β=β, c=c, z=z, target_nperiods=target_nperiods)

# Local Variables:
# coding: utf-8
# eval: (set-input-method 'TeX)
# End:    
