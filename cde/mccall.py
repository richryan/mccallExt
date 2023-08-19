# Program: mccall.py
# Purpose: Keep user-defined functions in the same place, a module.

# Date Started: 2022-12-28
# Date Revised: 2023-07-19

import numpy as np
import sys
import random
from numba import jit, njit, float64, int64, prange, vectorize # njit is an alias for @jit(nopython=True)
from numba.experimental import jitclass

# === Progress bar ===
# See https://stackoverflow.com/questions/3160699/python-progress-bar
def progressbar(i, tot, prefix="", size=60, out=sys.stdout): # Python3.3+
    # count = len(it)
    def show(j):
        x = int(size * j / tot)
        print("{}[{}{}] {}/{}".format(prefix, "="*x, "."*(size-x), j, tot), 
                end='\r', file=out, flush=True)
    show(i)
    print("\n", flush=True, file=out)

# === Random number generation ===
@njit
def seed(a):
    np.random.seed(a)

@njit
def rand(a):
    return np.random.random(size=a)        

def fun_make_match_periods_unemployed_unif(β=0.95, nperiods=8):
    """Return function used to match periods unemployed when wage offers are uniform [0,1].

    The wage offer is uniform[0,1]. The returned function can be used to
    search over the flow value of nonwork to match expected periods
    unemployed.

    Parameters
    ---------- 
    β : double    
        Discount factor.
    nperiods : integer 
        Number of periods expected to be unemployed.

    Returns
    –------
    function
        Function that takes flow value of nonwork as input and returns sqaured
        difference from nperiods.
    """
    def fun_match_periods_unemployed_unif(c):
        reservationWage = 1/β - ((1 - β)*(1 + β - 2 * β * c))**(1 / 2) / β
        return (reservationWage / (1 - reservationWage) - nperiods)**2
    return fun_match_periods_unemployed_unif

@njit
def uniform():
    """Return draw from uniform distribution, [0, 1).

    Note: Naming conventions between numba and numpy run into a
    problem here. See https://github.com/numba/numba/issues/8161 
    """
    return np.random.uniform(0.0, 1.0)

@njit
def random_binomial(n=1, p=0.5):
    # Note: The size argument is not supported in numpy.random.binomial.
    return np.random.binomial(n=n, p=p)

# === Classes === 
mccall_data_uniform = [
    ('c', float64),          # unemployment compensation
    ('z', float64),          # flow benefit of nonwork
    ('β', float64),          # discount factor
    ('lo', float64),         # lower end of interval for uniform RV
    ('hi', float64),         # upper end of interval for uniform RV
    ('mc_size', int64),      # number of draws
    ('w_draws', float64[:])  # draws of wages for Monte Carlo
]

@jitclass(mccall_data_uniform)
class McCallModelUniform:
    """Implements the McCall sequential search model with uniform wage offers.

    Parameters
    ---------- 
    z : double    
        Flow value of nonwork.
    c : double
        Flow value of unemployment-insurance compensation.
    β : double
        The discount factor.
    lo : double
        The lower end of the support for the uniform wage-offer distribution.
    hi : double
        The upper end of the support for the uniform wage-offer distribution.
    mc_size : int
        Size of the array of wage-offer draws.
    """

    def __init__(self, z=0.5, c=0.5, β=0.95, lo=0.0, hi=1.0, mc_size=1_000_000):

        self.z, self.c, self.β, self.lo, self.hi, self.mc_size = z, c, β, lo, hi, mc_size

        # Draw and store shocks
        seed(612)
        self.w_draws = (hi - lo) * rand(mc_size) + lo
    
# =================================================
# === Functions for computing reservation wages ===
# =================================================

@jit(nopython=True)
def compute_reservation_wage0(mcmc, max_iter=1000, tol=1e-5):
    """Compute the reservation wage with no remaining periods of unemployment insurance.

    Parameters
    ---------- 
    mcmc : 
        instance of mccall class
    max_iter : int
        Maximum number of iterations for the fixed-point algorithm.
    tol : 
        Tolerance for convergence. 

    Returns
    –------
    numpy.float64
        Reservation wage when unemployment benefits have expired.
    """

    z, β, w_draws = mcmc.z, mcmc.β, mcmc.w_draws

    # Start with an initial guess.
    h = np.mean(w_draws) / (1 - β)  
    i = 0
    error = tol + 1
    while i < max_iter and error > tol:

        integral = np.mean(np.maximum(w_draws / (1 - β), h))
        h_next = z + β * integral

        error = np.abs(h_next - h)
        i += 1

        h = h_next

    # == Now compute the reservation wage == #
    # Return 1-dimensional array
    return np.array([(1 - β) * h])

def fun_compute_reservation_wages_uniform(β, c, z, n):
    """Compute sequence of reservation wages.

    Compute sequence of reservation wages when wage offers are
    uniformly distributed ~ [0, 1] and the worker faces a finite
    number of periods of UI benefits. The sequence of reservation
    wages is computed using closed-form solutions.

    Parameters
    ----------
    β : float
        discount factor
    c : float
        unemployment insuarnce compensation
    z : float
        flow value of nonwork
    n : int
        numer of periods of UI benefits compensation

    Returns
    -------
    numpy ndarray
        Sequence of researvation wages, including reservation wage when no
        periods of UI benefits remaining.  Size is n + 1.
    """
    reservation_wages = np.zeros(n + 1)
    reservation_wages[0] = (1 - np.sqrt((1 - β) * (1 + β - 2 * β * z))) / β
    for i in range(1, reservation_wages.size):
        reservation_wages[i] = (z + c) * (1 - β) + β * (1 + reservation_wages[i - 1]**2) / 2

    return reservation_wages

def fun_compute_reservation_wages_uniform_ext(β, c, z, n, δ, Δ):
    """Compute sequence of reservation wages when chance of extension.

    Compute sequence of reservation wages when wage offers are uniformly
    distributed ~ [0, 1] and the worker faces a finite number of periods
    of UI benefits and there is a chance that benefits are extended.

    Parameters
    ----------
    β : float
        discount factor
    c : float
        unemployment insuarnce compensation
    z : float
        flow value of nonwork
    n : int
        numer of periods of UI benefits compensation
    δ : float
        chance that benefits are extended
    Δ : int
        number of periods benefits are extended

    Returns
    -------
    numpy array
        Sequence of researvation wages, including reservation wage when no
        periods of UI benefits remaining.  Size is n + 1.
    """

    # Max number of benefits: n - 1 +  Δ
    # There are n periods of benefits, which can be extended between
    # this period and the next period
    nperiods_ui_compensation_max = n - 1 + Δ
    reservation_wages_extended = fun_compute_reservation_wages_uniform(β=β, c=c, z=z, n=nperiods_ui_compensation_max)
    
    reservation_wages = np.empty(1 + n)
    reservation_wages[0] = (1 - np.sqrt(1 - β * (1 - δ) * (β + 2 * z * (1 - β) + β * δ * (reservation_wages_extended[Δ]**2)))) / (β * (1 - δ))
    for n in np.arange(1, reservation_wages.size, step=1):
        # reservation wage when extended
        rw_ext = reservation_wages_extended[n - 1 + Δ]
        reservation_wages[n] = (z + c) * (1 - β) + β * δ * (1 + rw_ext**2) / 2 + β * (1 - δ) * (1 + reservation_wages[n - 1]**2) / 2

    return reservation_wages        
        
@jit(nopython=True)
def fun_compute_reservation_wages_mcmc(mcmc, n=26, max_iter=1000, tol=1e-5):
    """Compute sequence of reservation wages using MCMC integration.

    Parameters
    ----------
    mcmc  
        Instance of mccall.
    n : int
        Periods worker is entitled to unemployment-insurance compensation.
    max_iter : int
        Maximum number iterations.
    tol : float
        Tolerance for convergence.

    Returns
    -------
    numpy ndarray
        Sequence of researvation wages, including reservation wage when no
        periods of UI benefits remaining.  Size is n + 1.
    """

    z, c, β, w_draws = mcmc.z, mcmc.c, mcmc.β, mcmc.w_draws
    
    # Reservation wage when benefits have expired
    reservation_wages = np.empty(n + 1)
    reservation_wages[0] = compute_reservation_wage0(mcmc)[0]

    for i in range(1, n + 1):
        # Compute current reseravtion wage as a function of previous reservation wage
        reservation_wages[i]  = (z + c) * (1 - β) + β * np.mean(np.maximum(reservation_wages[i - 1], w_draws))        

    return reservation_wages

# =====================
# === Simulated welfare
# =====================

@jit(nopython=True)
def compute_accept_uniform(β, z, c, reservation_wages, seed_stopping_time=1234):
    """Compute statistics associated with searching and accepting a job.

    The reservation wages should include the reservation wage when the
    person does not receive unemployment-insurance benefits.

    Parameters
    ----------
    β
        Discount factor.
    z
        Flow value of nonwork.
    c 
        Unemployment-insurance compensation.
    reservation_wages
        Sequence of reservation wages used to make decisions. 
    seed_stopping_time
        Seed to replicate results.

    Returns
    -------
    stopping_time 
        How long the agent was unemployed.
    w             
        The accepted wage.
    welfare
        The welfare experienced by the agent. 
    """

    # The object reservation_wages[0] contains the reservation wage with no
    # remaining periods of UI benefits.
    # reservation_wages[1], ..., reservation_wages[N] are reservation
    # wages when there are remaining benefits.
    nperiods_ui_compensation = reservation_wages.size - 1
    nperiods_remaining = nperiods_ui_compensation

    seed(seed_stopping_time)    # defined above
    t = 0                       # stopping time

    welfare = 0.0 
    while True:
        # Generate a wage draw
        reservation_wage_t = reservation_wages[nperiods_remaining] 
        w = uniform()
        # Stop when the draw is above the reservation wage.
        if w >= reservation_wage_t:
            stopping_time = t
            break
        else:
            if nperiods_remaining > 0:
                welfare = welfare + (β**t) * (z + c) 
            else:
                welfare = welfare + (β**t) * (z)
            # Update state
            t += 1
            nperiods_remaining = np.maximum(0, nperiods_remaining - 1)
    # Add last wage, which is permanent
    welfare = welfare + (β**stopping_time) * w / (1 - β) # Keep the wage indefinitely
            
    # Return a tuple (dictionary support seems unavailable in numba):
    # Note: return {'stopping_time': stopping_time, 'welfare': welfare} did not work
    return stopping_time, w, welfare

# === 
@jit(nopython=True, parallel=True)
def compute_mean_accept_uniform_parallel(β, z, c, reservation_wages, num_reps=10_000):
    """Computes welfare using prange."""
    obs_stopping_time = np.empty(num_reps, dtype=np.int64)
    obs_accepted_wage = np.empty(num_reps, dtype=np.float64)
    obs_welfare       = np.empty(num_reps, dtype=np.float64)

    for i in prange(num_reps):
        obs_stopping_time[i], obs_accepted_wage[i], obs_welfare[i] = compute_accept_uniform(β, z, c, reservation_wages, seed_stopping_time=i)
        
    return obs_stopping_time.mean(), obs_accepted_wage.mean(), obs_welfare.mean()
    
# === Functions for added periods of UI compenssation
@jit(nopython=True)
def fun_compute_reservation_wages_ext_mcmc(mcmc, reservation_wages_extended, n=26, δ=0.2, Δ=13, max_its=1000, tol=1e-5):
    """Compute sequence of reservation wages when there is a chance of extension. 

    Because there is only the one-time chance that wages are extended,
    the extended wages are passed as an input. The extended wages can
    be computed for the case where there is no chance of extension.

    Parameters
    ----------
    mcmc 
        Instance of mccall.
    reservation_wages_extended : ndarray
        Sequence of reservation wages once an extension occurs.
    n : int
        Periods of UI compensation before extension.
    δ 
        Probability that benefits are extended.
    Δ 
        Number of periods that benefits are extended.
    max_its
        Maximum number of iterations.
    tol
        Tolerance for convergence of reservation wages.
    """

    z, c, β, w_draws = mcmc.z, mcmc.c, mcmc.β, mcmc.w_draws

    # Includes reservation wages when 0 periods remaining
    reservation_wages = np.zeros(n + 1)

    # Reservation wage when no periods remaining
    h = reservation_wages_extended[0] # initial guess
    i = 0
    error = tol + 1
    reservation_wage0_T1 = β * δ * np.mean(np.maximum(reservation_wages_extended[Δ], w_draws))
    while i < max_its and error > tol:
        reservation_wage0_T2 = β * (1 - δ) * np.mean(np.maximum(h, w_draws))
        h_next = z * (1 - β) + reservation_wage0_T1 + reservation_wage0_T2
        error = np.abs(h_next - h)
        i += 1
        h = h_next

    reservation_wages[0] = h

    # Reservation wages for periods 1 through n
    for i in range(1, n + 1):

        periods_remaining = i
        # Compute current reseravtion wage as a function of previous reservation wage

        h = reservation_wages[i - 1] # initial guess
        j = 0
        error = tol + 1        
        i_reservation_wage_extended = reservation_wages_extended[periods_remaining - 1 + Δ]

        reservation_wage0_T1 = β * δ * np.mean(np.maximum(i_reservation_wage_extended, w_draws))
        while j < max_its and error > tol:
            reservation_wage0_T2 = β * (1 - δ) * np.mean(np.maximum(reservation_wages[i - 1], w_draws))
            h_next = (z + c) * (1 - β) + reservation_wage0_T1 + reservation_wage0_T2
            error = np.abs(h_next - h)
            j += 1
            h = h_next

        reservation_wages[i]  = h

    return reservation_wages

@jit(nopython=True)
def compute_accept_uniform_ext(β, z, c, reservation_wages, reservation_wages_ext, δ=0.5, Δ=13, accept_ext_seed=612):
    """Compute statistics associated with searching and accepting a job when there is the possibility of extension.

    The statistics are associated with a McCall worker engaged in
    sequential search. Wage offers are uniform[0,1].

    A note on timing: The person's state is a wage draw they can accept or reject.
    Then unemployment-insurance compensation can be extended.

    The sequence of reservation wages can be computed under false beliefs.
    For example, the agent may mis-perceive the likelihood that benefits
    are extended.  Or they may believe that benefits will be extended 13
    periods when they are extended 26 periods.

    Parameters
    ----------
    β
        Discount factor.
    z
        Flow value of nonwork.
    c 
        Unemployment-insurance compensation.
    reservation_wages
        Perceived sequence of reservation wages. 
    reservation_wages_ext
        Sequence of reservation wages once there is an extension.
    δ 
        True probability of extension 
    accept_ext_seed
        Seed to replicate results.
    
    Returns
    -------
    stopping_time 
        How long the agent was unemployed.
    w             
        The accepted wage.
    welfare
        The welfare experienced by the agent. 
    """

    seed(accept_ext_seed)    # defined above

    # reservation_wagesδ includes reservation wage when 0 remaining periods
    nperiods_ui_compensation = reservation_wages.size - 1
    nperiods_remaining = nperiods_ui_compensation

    t = 0
    extend = 0

    welfare = 0.0 
    while True:        
        while extend == 0:
            # Generate a wage draw, a state variable for W.
            w = uniform()
            reservation_wage_t = reservation_wages[nperiods_remaining]

            # Stop when the draw is above the reservation wage
            if w >= reservation_wage_t:
                stopping_time = t
                break
            else:
                if nperiods_remaining > 0:
                    add_to_welfare = z + c
                else:
                    add_to_welfare = z
                welfare = welfare + (β**t) * add_to_welfare
                t += 1

            # Update the state
            extend = random_binomial(n=1, p=δ)
            if extend == 1:
                # Add in extended periods.
                nperiods_remaining = np.maximum(0, nperiods_remaining - 1 + Δ)
            else:
                # Else subtract one
                nperiods_remaining = np.maximum(0, nperiods_remaining - 1)
        # now extend == 1, now trapped here
        else:                       
            while True:
                reservation_wage_ext_t = reservation_wages_ext[nperiods_remaining]
                w = uniform()

                # Stop when the draw is above the reservation wage
                if w >= reservation_wage_ext_t:
                    stopping_time = t
                    break
                else:
                    if nperiods_remaining > 0:
                        add_to_welfare = z + c
                    else:
                        add_to_welfare = z
                    welfare = welfare + (β**t) * add_to_welfare
                    # Update statistics and state.
                    t += 1
                    nperiods_remaining = np.maximum(0, nperiods_remaining - 1)

        break                       # only executed if "while extend == 0" breaks

    # Keep the accepted wage indefinitely and add to welfare.
    add_to_welfare = w / (1 - β) 
    welfare = welfare + (β**stopping_time) * add_to_welfare
            
    # Return a tuple (dictionary support seems unavailable in numba)
    # Not >>> return {'stopping_time': stopping_time, 'welfare': welfare}
    # print("Returning")
    return stopping_time, w, welfare

# ===
@jit(nopython=True, parallel=True)
def compute_mean_accept_ext_parallel(mcmc, reservation_wagesδ, reservation_wages_ext, δ=0.5, Δ=13, num_reps=10_000):
    
    obs_stopping_time = np.empty(num_reps, dtype=np.int64)
    obs_accepted_wage = np.empty(num_reps, dtype=np.float64)
    obs_welfare       = np.empty(num_reps, dtype=np.float64)

    for i in prange(num_reps):
        obs_stopping_time[i], obs_accepted_wage[i], obs_welfare[i] = compute_accept_ext(mcmc,
                                                                                        reservation_wagesδ,
                                                                                        reservation_wages_ext,
                                                                                        δ=δ, Δ=Δ,
                                                                                        accept_ext_seed=i)
        
    return obs_stopping_time.mean(), obs_accepted_wage.mean(), obs_welfare.mean()

# ===
@jit(nopython=True, parallel=True)
def compute_mean_accept_uniform_ext_parallel(β, z, c, reservation_wages, reservation_wages_ext, δ=0.5, Δ=13, num_reps=10_000):
    
    obs_stopping_time = np.empty(num_reps, dtype=np.int64)
    obs_accepted_wage = np.empty(num_reps, dtype=np.float64)
    obs_welfare       = np.empty(num_reps, dtype=np.float64)

    for i in prange(num_reps):
        obs_stopping_time[i], obs_accepted_wage[i], obs_welfare[i] = compute_accept_uniform_ext(β,
                                                                                                z,
                                                                                                c,
                                                                                                reservation_wages,
                                                                                                reservation_wages_ext,
                                                                                                δ=δ,
                                                                                                Δ=Δ,
                                                                                                accept_ext_seed=i)
        
    return obs_stopping_time.mean(), obs_accepted_wage.mean(), obs_welfare.mean()

# Local Variables:
# coding: utf-8
# eval: (set-input-method 'TeX)
# End:    
