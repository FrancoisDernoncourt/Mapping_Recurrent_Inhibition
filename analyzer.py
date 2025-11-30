##################
# IMPORT LIBRARIES
##################
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import h5py
import matplotlib.gridspec as gridspec
import logging
import pprint # for debugging
from dataclasses import dataclass, field, asdict
from brian2.units import Unit # import the units
import cmasher as cmr # more color maps
import itertools
from itertools import combinations_with_replacement
import random
from scipy.optimize import curve_fit, least_squares
from scipy.signal import butter, filtfilt, windows, csd, detrend
from scipy.stats import norm
from pathlib import Path
import pickle

##################
# CLASS WITH SIMULATIONS
##################
@dataclass
class AnalyzesParams:
    analysis_output_name: str = None # default is None
    # IS THE LOADED HDF5 FILE A SIMULATION OR EXPERIMENTAL DATA?
    is_simulation: bool = True # If False, the data is experimental
    # SELECTION OF MNs FOR ANALYSIS
    select_random_subset_of_MUs_per_pool_for_analyses: int = 0 # if 0, will consider all motor units. Otherwise, will randomly select (subsample) N motor units from each pool
    remove_discontinuous_MNs: bool = True
    ISI_above_which_MNs_are_considered_discontinuous: float = 0.5 # seconds
    # FIRING RATES
    get_firing_rates: bool = True
    firing_rates_output_figures: bool = True
    # GROUND TRUTH CONECTIVITY
    get_ground_truth_RI_connectivity: bool = True # firing rates need to be available
    ground_truth_RI_connectivity_output_figures: bool = True
    # GRAPH THEORY CONNECTIVITY ANALYSIS (from ground truth)
    get_graph_theory_connectivity_measures: bool = False # A bit long to compute
    graph_theory_output_figures: bool = True
    # CROSS HISTOGRAM ANALYSIS
    get_cross_histogram_measures: bool = True # Spiking probability trough magnitude + Synchrony peak height
    cross_histogram_ignore_homonymous_pool: bool = False
    cross_histogram_ignore_heteronymous_pool: bool = False
    cross_histogram_measures_histogram_kind: str = 'normalized' # 'raw' or 'normalized'
    cross_histogram_measures_lowpass_filter_prob_dist: float = 60 # Smoothing of the cross-histogram
    cross_histogram_measures_min_spikes: int = 1e4 # Minimum nb of spikes required for the analysis to be carried on
    cross_histogram_measures_min_r2: float = 0.75 # Minimum R² required for the fit to be considered good enough
    cross_histogram_measures_min_plateau: float = 0.05 # in seconds, minimum "plateau" selected part of the fitted curve
    cross_histogram_measures_null_distrib_nb_iter: int = 100 # nb of iterations of "sampling rounds" to estimate the distribution of spiking probability trough given the null curve
    cross_histogram_output_figures: bool = False # False by default, because it can take a long time
    cross_histogram_save_cross_hists: bool = False # False by default. Saving all histograms/probability distributions generated for the analysis takes up memory space but allows for plotting later
    # COHERENCE ANALYSIS
    get_coherence: bool = False # Long to compute
    coherence_window_length: float = 1  # in s
    coherence_windows_overlap: float = 0.5 # 50% relative to coh_window_length
    # frequency resolution will be = 1/coh_window_length. If coh_window_length = 10s for instance, resolution will be 0.1hz
    coherence_upsampling_frequency_resolution: int = 2 # for the 'nfft' parameter of scipy.csd(). Interpolates the coherence values over the frequencies, to smooth the figure a bit.
    # For example, if the resolution defined by coh_window_length is 1hz, and if upsampling_frequency_resolution = 2, the resolution will be upsampled to 0.5hz
    coherence_max_freq: float = 100 # maximum frequency that is kept
    coherence_calc_max_iteration_nb_per_group_size: int = 100 # 100 # 100 # More iteration for smaller group sizes, because the value obtained is very dependent upon the exact neurons selected, especially when only a few MNs are used to create the CST
    coherence_between_CST_and_common_input: bool = False
    coherence_output_figures: bool = False

###################
# HELPER FUNCTIONS
###################
def get_last_folder(path_str):
    p = Path(path_str)
    # If there's a suffix, assume it's a file → grab its parent folder’s name
    if p.suffix:
        return p.parent.name
    # Otherwise assume it's a folder path → grab its own name
    return p.name

def _ensure_logging(): # Define logger to keep track of each simulation's progress despite parallelization
    """Make sure each process has a file+console logger attached."""
    root = logging.getLogger()
    if not root.handlers:
        # file handler
        fh = logging.FileHandler("simulations_progress_log.log", mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s %(processName)s %(levelname)s: %(message)s"))
        fh.setLevel(logging.INFO)
        root.addHandler(fh)
        # console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(processName)s: %(message)s"))
        root.addHandler(ch)
        root.setLevel(logging.INFO)
        
def h5_to_dict(grp):
    """
    Recursively turn an HDF5 group into a nested dict of numpy arrays.
    """
    out = {}
    for name, item in grp.items():
        if isinstance(item, h5py.Dataset):
            out[name] = item[()]    # read the entire dataset
        else:  # subgroup
            out[name] = h5_to_dict(item)
    return out

def remove_spikes_in_windows_to_ignore(spike_times, duration, window_to_ignore_duration):
    """
    Keep only the spikes that fall within the “analysis window,” i.e.
    between window_to_ignore_duration and (duration + window_to_ignore_duration),
    then re-zero them so time starts at 0.
    """
    # make a 1D numpy array
    t = np.asarray(spike_times, float)

    t0 = window_to_ignore_duration
    t1 = duration + window_to_ignore_duration

    # keep only spikes in [t0, t1]
    mask = (t >= t0) & (t <= t1)
    t = t[mask]

    # shift so that the first analysis time is 0
    t = t - t0
    return t

def remove_discontinuous_MNs(MN_spike_trains, sim_duration, max_ISI=0.5):
    """
    Drop any MN whose **within-analysis** interspike interval ever exceeds max_ISI.
    Also enforce that the gap from t=0→first spike and last spike→sim_duration
    is ≤ max_ISI.
    MNs with <1 real spike (after window‐trimming) are dropped.
    """
    continuous = {}
    idx_kept = []
    # logger = logging.getLogger(__name__) # for debugging purposes
    for mn_id, spike_times in MN_spike_trains.items():
        # sort the spikes
        t = np.sort(np.asarray(spike_times, float))
        # if no spikes at all, skip
        if t.size == 0:
            continue
        # build a full timeline including the edges
        t_full = np.concatenate(([0.0], t, [sim_duration]))
        # compute all inter‐spike intervals (including edges)
        isis = np.diff(t_full)
        # keep only those MNs whose *every* interval is ≤ max_ISI
        if np.all(isis <= max_ISI):
            continuous[mn_id] = t  # store only the real spikes (edges not needed downstream)
            # logger.info(f"MN {mn_id} is continuous\n    Spike times: {t}\n    ISIs: {isis}") # for debugging purposes
            idx_kept.append(int(mn_id.split('_')[1])) # keep the index of the MN

    return continuous, idx_kept

def directed_clustering_with_nulls( # Used to compute the graph theory measures of the MN->MN connectivity matrix
    W,
    num_random=100,
    null_model="ER",
    random_seed=None
):
    """
    Compute directed clustering on a weighted adjacency matrix W,
    and compare against a random‐graph null.

    Parameters
    ----------
    W : array_like, shape (N,N)
        Weighted adjacency (W[i,j] > 0 means a directed edge i->j).
    num_random : int
        How many null graphs to sample.
    null_model : {'ER'}
        Currently only Erdős–Rényi (same density) is supported.
    random_seed : int or None
        For reproducible null sampling.

    Returns
    -------
    results : dict with keys
       'C_emp'       float, clustering of the real graph  
       'C_null'      ndarray, length num_random, clustering of each null  
       'C_null_mean' float, np.mean(C_null)  
       'C_null_std'  float, np.std(C_null)  
       'Z'           float, (C_emp–C_null_mean)/C_null_std  
       'density'     float, edge density of the real graph  
    """
    W = np.asarray(W, float)
    N = W.shape[0]
    if W.shape[1] != N:
        raise ValueError("W must be square")

    # 1) Binarize
    A = (W > 0).astype(int)
    np.fill_diagonal(A, 0)

    # helper: directed clustering per Fagiolo (2007)
    def _directed_clustering(A_bin):
        S = A_bin + A_bin.T
        S3 = np.linalg.matrix_power(S, 3)
        triangles = np.diag(S3).astype(float)
        k = S.sum(axis=1).astype(float)
        Ci = np.zeros(N, float)
        mask = k >= 2
        Ci[mask] = triangles[mask] / (2 * k[mask] * (k[mask] - 1))
        return Ci.mean()

    # empirical clustering
    C_emp = _directed_clustering(A)

    # density
    density = A.sum() / (N*(N-1))

    # sample nulls
    rng = np.random.default_rng(random_seed)
    C_null = np.zeros(num_random, float)

    for t in range(num_random):
        if null_model == "ER":
            # ER with same density
            A_rand = (rng.random((N,N)) < density).astype(int)
            np.fill_diagonal(A_rand, 0)
        else:
            raise ValueError("Only null_model='ER' currently supported")

        C_null[t] = _directed_clustering(A_rand)

    C_null_mean = C_null.mean()
    C_null_std  = C_null.std(ddof=1) if num_random>1 else 0.0
    Z = (C_emp - C_null_mean) / C_null_std if C_null_std>0 else np.nan
    ratio_emp_null = C_emp / C_null_mean

    return {
        "C_emp":        C_emp,
        "C_null":       C_null,
        "C_null_mean":  C_null_mean,
        "C_null_std":   C_null_std,
        "Z":            Z,
        "density":      density,
        "ratio_emp_null": ratio_emp_null
    }

def PSTH_MU_pair_computation(ref_spike_train, comp_spike_train, psth_bins): # ref = inhibiting MU, comp = inhibited MU
    '''
    Inputs:
    - ref_spike_train (inhibiting MU), array of spike times (in s)
    - comp_spike_train (inhibited MU), array of spike times (in s)
    - psth_bins, array corresponding to the time bins to which to add the counts of spikes
    '''
    comp_MN_counts = []
    # First spike after ref spike; and last spike before ref spike
    for spiki in ref_spike_train:
        # Next spike
        next_spikes_in_mn_comp = comp_spike_train[comp_spike_train >= spiki]
        if len(next_spikes_in_mn_comp) > 0:
            next_spike_in_mn_comp = next_spikes_in_mn_comp[0]
            comp_MN_counts.append(next_spike_in_mn_comp - spiki)
        # Previous spike
        previous_spikes_in_mn_comp = comp_spike_train[comp_spike_train < spiki] # not <= because already using >= for "next spike". This avoids counting twice the spikes at delay t=0
        if (len(previous_spikes_in_mn_comp) > 0):
            comp_MN_counts.append(previous_spikes_in_mn_comp[-1] - spiki)
    # Get counts (histogram) for the current REF MU -> inhibit -> COMP MU pair
    MU_pair_hist, _ = np.histogram(comp_MN_counts, bins=psth_bins)
    return MU_pair_hist

def compute_cross_histograms(spike_times, corresponding_MN_idx, fsamp,
                             MU_corresponding_pool_list, list_of_MUs_by_pool,
                             ignore_homonymous_pool = False, ignore_heteronymous_pool = False,
                             spike_delays_count_window_limits = 0.2 # in seconds
                             ):
    # Initialize variables
    histograms_per_MU_pair = {}
    histograms_per_MU = {}
    psth_bins = np.arange(-spike_delays_count_window_limits,
                      spike_delays_count_window_limits, 1/fsamp)
    reference_hist_size = len(psth_bins) - 1
    # Remove indices of discontinuous MNs
    MU_corresponding_pool_list = np.array(MU_corresponding_pool_list)[corresponding_MN_idx]
    for pooli in list_of_MUs_by_pool.keys():
        list_of_MUs_by_pool[pooli] = np.intersect1d(corresponding_MN_idx, list_of_MUs_by_pool[pooli])
    muscles_set = set(MU_corresponding_pool_list)
    muscle_pairings_bidirectional = ["->inhib->".join(pair) for pair in itertools.product(muscles_set, repeat=2)]
    # Start loop through pairs of pools
    for muscle_pairing in muscle_pairings_bidirectional:
        # PER MU PAIR #####
        # Initialize empty lists to store histograms
        histograms_per_MU_pair[muscle_pairing] = {"inhibited_MU_inhibiting_MU_pair_hist_raw": {},
                                        "inhibited_MU_inhibiting_MU_pair_hist_normalized": {}}
        # Get corresponding MU indices
        inhibited_muscle = muscle_pairing.split('->inhib->')[0]
        inhibiting_muscle = muscle_pairing.split('->inhib->')[1]
        MU_inhibited_list = list_of_MUs_by_pool[inhibited_muscle]
        MU_inhibiting_list = list_of_MUs_by_pool[inhibiting_muscle]
        # Loop through all MU pairs corresponding to the selected muscle pair of muscles
        for mu_comp in MU_inhibited_list: # The MU being compared is the INHIBITED MU
            for mu_ref in MU_inhibiting_list: # The ref MU is the INHIBITING MU
                current_MU_pair_psth = np.zeros(reference_hist_size)
                current_MU_pair_psth_normalized = np.zeros(reference_hist_size)
                # special case = assign the MU_pair_psth to zero if specific muscle pairs combinations
                if ((ignore_homonymous_pool and inhibited_muscle == inhibiting_muscle) or \
                (ignore_heteronymous_pool and inhibited_muscle != inhibiting_muscle)):
                    current_MU_pair_psth = np.zeros(reference_hist_size)
                # normal case = not ignoring specific muscle pairs combinations
                else:
                    ref_spike_train = np.copy(spike_times[f'MN_{mu_ref}']) # Ref spike = inhibiting MU
                    comp_spike_train = np.copy(spike_times[f'MN_{mu_comp}']) # Comp spilke = inhibited MU
                    # Compute histograms (raw and normalized)
                    if mu_ref!=mu_comp:
                        current_MU_pair_psth = PSTH_MU_pair_computation(ref_spike_train,comp_spike_train,psth_bins)
                total_hist_count_temp = np.sum(current_MU_pair_psth)
                if total_hist_count_temp > 0:
                    current_MU_pair_psth_normalized = current_MU_pair_psth / total_hist_count_temp
                # Add to the counts of the corresponding MU pair histogram
                MU_pair_key = (mu_comp, mu_ref)
                histograms_per_MU_pair[muscle_pairing]["inhibited_MU_inhibiting_MU_pair_hist_raw"][MU_pair_key] = current_MU_pair_psth
                histograms_per_MU_pair[muscle_pairing]["inhibited_MU_inhibiting_MU_pair_hist_normalized"][MU_pair_key] = current_MU_pair_psth_normalized
                # end of 'for each inhibiting MU'
            # end of 'for each inhibited MU'
        # End of "for each muscle pair"

        # PER MU (inhibited by / inhibiting all other MUs in the pool) ############################
        # Generate all pairings (including self-pairing)
        # muscle_pairings_unidirectional = ["<->".join(pair) for pair in list(itertools.combinations_with_replacement(muscles_set, 2))]
        muscle_pairings_bidirectional = ["<->".join(pair) for pair in itertools.product(muscles_set, repeat=2)]
        # For example, VL<->VL, VL<->VM and VM<->VM
        # Then we will loop through each case and see each case (VL inhibits VL, VL inhibited by VL; VM inhibits VL, VM inhibited by VL; etc.)
        for muscle_pairing in muscle_pairings_bidirectional: # muscle_pairings_unidirectional:    
            # Identify the inhibiting muscle, and the muscle being inhibited
           for inhib_direction in ['A->inhibiting->B','A->inhibited_by->B']:
                # A corresponds to the considered MU's muscle ; B corresponds to the muscle of the other MUs (can be the same muscle as A)
                if inhib_direction == 'A->inhibiting->B':
                    # Here, we want to sum over the inhibiting MUs.
                    inhibiting_muscle = muscle_pairing.split('<->')[0]
                    inhibited_muscle  = muscle_pairing.split('<->')[1]
                    dict_key = f'{inhibiting_muscle}->inhibiting->{inhibited_muscle}'
                    # Compute both lists for clarity:
                    MU_inhibited_list = list_of_MUs_by_pool[inhibited_muscle]
                    MU_inhibiting_list = list_of_MUs_by_pool[inhibiting_muscle]
                    # Initialize the dictionary for inhibiting MUs:
                    histograms_per_MU[dict_key] = {"MU_hist_raw": {},
                                                "MU_hist_normalized": {}}
                    for mu_ref in MU_inhibiting_list: # mu ref is the inhibiting MU here
                        histograms_per_MU[dict_key]["MU_hist_raw"][mu_ref] = np.zeros(reference_hist_size)
                        histograms_per_MU[dict_key]["MU_hist_normalized"][mu_ref] = np.zeros(reference_hist_size)
                    # Now, loop over all pairs and for each pair, if the current mu_ref is found in the pair,
                    # add its contribution to histograms_per_MU[dict_key]["inhibiting_MU_hist_raw"][mu_ref].
                    for mu_comp in MU_inhibited_list:
                        for mu_ref in MU_inhibiting_list:
                            if mu_comp == mu_ref:
                                continue
                            MU_pair_key = (mu_comp, mu_ref)
                            for key in histograms_per_MU_pair.keys():
                                if MU_pair_key in histograms_per_MU_pair[key]["inhibited_MU_inhibiting_MU_pair_hist_raw"]:
                                    histograms_per_MU[dict_key]["MU_hist_raw"][mu_ref] += \
                                        histograms_per_MU_pair[key]["inhibited_MU_inhibiting_MU_pair_hist_raw"][MU_pair_key]
                                    histograms_per_MU[dict_key]["MU_hist_normalized"][mu_ref] += \
                                        histograms_per_MU_pair[key]["inhibited_MU_inhibiting_MU_pair_hist_normalized"][MU_pair_key]
                elif inhib_direction == 'A->inhibited_by->B':
                    # Here, we want to sum over the inhibited MUs.
                    inhibiting_muscle = muscle_pairing.split('<->')[1]
                    inhibited_muscle  = muscle_pairing.split('<->')[0]
                    dict_key = f'{inhibited_muscle}->inhibited_by->{inhibiting_muscle}'
                    # Compute both lists for clarity:
                    MU_inhibited_list = list_of_MUs_by_pool[inhibited_muscle]
                    MU_inhibiting_list = list_of_MUs_by_pool[inhibiting_muscle]
                    histograms_per_MU[dict_key] = {"MU_hist_raw": {},
                                                "MU_hist_normalized": {}}
                    for mu_comp in MU_inhibited_list: # mu comp is the inhibited MU here
                        histograms_per_MU[dict_key]["MU_hist_raw"][mu_comp] = np.zeros(reference_hist_size)
                        histograms_per_MU[dict_key]["MU_hist_normalized"][mu_comp] = np.zeros(reference_hist_size)
                    for mu_comp in MU_inhibited_list:
                        for mu_ref in MU_inhibiting_list:
                            if mu_comp == mu_ref:
                                continue
                            MU_pair_key = (mu_comp, mu_ref)
                            for key in histograms_per_MU_pair.keys():
                                if MU_pair_key in histograms_per_MU_pair[key]["inhibited_MU_inhibiting_MU_pair_hist_raw"]:
                                    histograms_per_MU[dict_key]["MU_hist_raw"][mu_comp] += \
                                        histograms_per_MU_pair[key]["inhibited_MU_inhibiting_MU_pair_hist_raw"][MU_pair_key]
                                    histograms_per_MU[dict_key]["MU_hist_normalized"][mu_comp] += \
                                        histograms_per_MU_pair[key]["inhibited_MU_inhibiting_MU_pair_hist_normalized"][MU_pair_key]
                    # End of "for each inhibiting MU"
                # End of "for each inhibited MU"
            # End of "for each inhibition direction"
    cross_histograms_return_dict = {"histograms_per_MU_pair": histograms_per_MU_pair,
                                    "histograms_per_MU": histograms_per_MU}
    return cross_histograms_return_dict, psth_bins

def convert_histogram_into_smoothed_probability_distribution(histogram, lowpass_cutoff=60, bin_fsamp=2048):
    kernel_duration = ((1/lowpass_cutoff)/2)
    kernel_width = np.round(kernel_duration*bin_fsamp).astype(int) # calculated based on the sampling frequency and the nyquist formula
    spike_count_smoothing_kernel = windows.hann(kernel_width)  # Hanning window filter
    spike_count_smoothing_kernel = spike_count_smoothing_kernel / np.sum(spike_count_smoothing_kernel)  # Normalize window to have a unitary area
    smoothed_probability_distribution = filtfilt(
        spike_count_smoothing_kernel, 1, histogram)
    smoothed_probability_distribution /= np.sum(smoothed_probability_distribution)
    return smoothed_probability_distribution

def fit_trapezoid_windows(x, baseline,
                          min_plateau_duration = 0.05,
                          width_guess=0.1):
    """
    Fit a trapezoid to normalized 'baseline'.  Return:
      (window_backward, window_forward, trapezoid_curve, (t1,t2,t3,t4))
    """
    # 1) normalize baseline to [0,1]
    y = baseline.copy()
    y = (y - y.min()) / (y.max() - y.min() + 1e-12)

    # 2) define trapezoid function
    def T(p, x):
        t1, t2, t3, t4 = p
        out = np.zeros_like(x)
        # rising edge
        mask = (x >= t1) & (x < t2)
        out[mask] = (x[mask] - t1) / (t2 - t1)
        # plateau
        mask = (x >= t2) & (x <= t3)
        out[mask] = 1.0
        # falling edge
        mask = (x > t3) & (x <= t4)
        out[mask] = (t4 - x[mask]) / (t4 - t3)
        return out

    # 3) initial guess
    midx = np.argmin(np.abs(x))
    xc = x[midx]
    p0 = [xc - width_guess, xc - width_guess/2, xc + width_guess/2, xc + width_guess]

    # enforce t1 < t2 < t3 < t4 within the range of x
    lb = [x[0],    x[0],    x[midx],   x[midx]]
    ub = [xc,      xc,      x[-1],     x[-1]]

    def resid_trap(p):
        return T(p, x) - y

    sol = least_squares(resid_trap, p0, bounds=(lb, ub))
    t1, t2, t3, t4 = sol.x

    # Enforce minimum plateau length
    plateau_len = t3 - t2
    if plateau_len < min_plateau_duration:
        half = min_plateau_duration / 2.0
        t2 = -half
        t3 =  half
        t2 = max(t2, x[0])
        t3 = min(t3, x[-1])
        if t2 >= t3:
            t2 = x[midx] - half
            t3 = x[midx] + half

    # Ensure 0 ∈ [t2, t3]
    if not (t2 < 0 < t3):
        t2, t3 = min(t2, t3), max(t2, t3)

    idx = np.arange(len(x))
    window_backward = idx[(x >= t2) & (x <= 0)]
    window_forward  = idx[(x >= 0)  & (x <= t3)]

    trap_curve = T((t1, t2, t3, t4), x)
    return window_backward, window_forward, trap_curve, (t1, t2, t3, t4)

def fit_null_curve_and_extract_windows_from_probability_distrib(
        x, y, fsamp,
        min_plateau_duration = 0.05,
        dl_bounds = (
            # # # Generalizes logistic lower bounds:
            [0.0,  # A = amplitude
            1*1e2,  0.0,  -0.5, # lower bound params for the generalized logistic curve : a, nu, midpoint # [1*1e2, 0, -0.5] works okay too - constrains to a more "rectangular" baseline curve
            0.0],   # vertical offset
            # # # Generalizes logistic upper bounds:
            [0.02,  # A = amplitude
            1*1e4,  100.0,  -0.02, # higher bound params for the generalized logistic curve : a, nu, midpoint
            0.001],   # vertical offset
        ),
        mh_bounds = (
            [0.0, 0.0001, 0.0001, -0.1, 0, -0.01],       # Mexcian hat lower bounds: amplitude, decay, width, center, gauss_std, v_offset
            [0.02, 0.3, 0.3, 0.1, 0.1, 0.01]        # Mexican hat upper bounds:  amplitude, decay, width, center, gauss_std, v_offset
        ),
        p0_dl=None, p0_mh=None,
        n_iter=3
    ):
    """
    1) Fit double‐generalized‐logistic + mexican‐hat by alternating least_squares.
    2) At each DL fit, extract (t2, t3) via trapezoid‐fit and clamp MH width ≤ (t3 - t2)/2.
    3) Return final baseline, full fit, residuals, windows, trapezoid, params, R².
    """
    dl_lb, dl_ub = dl_bounds
    mh_lb, mh_ub = mh_bounds

    if p0_dl is None:
        # initial guess for DL: [amplitude, (a, nu, midpoint), offset]
        p0_dl = [np.mean(y)*2.0,
                 1*1e3, 1.0, -0.1,
                 0.0]
    if p0_mh is None:
        # initial guess for MH: [amplitude, decay, width, center, gauss_std, v_offset]
        p0_mh = [0, 0.001, 0.001, 0, 0.05, 0]

    p_dl = np.array(p0_dl)
    p_mh = np.array(p0_mh)

    def double_generalized_logistic(x,
            amplitude,    # overall vertical scaling
            a, nu, u,     # Richards (generalized logistic curve) params
            offset):      # vertical offset
        # R1 = (1 + nu * np.exp(-a * (x - u)))**(-1.0/nu)
        # R2 = (1 + nu * np.exp(-a * (-x - u) ))**(-1.0/nu)-1
        # Rising plateau (center at x = u):
        z1 = -a * (x - u)
        z1_clipped = np.clip(z1, None, 500)                  # clip any z1 > 500
        R1 = (1.0 + nu * np.exp(z1_clipped)) ** (-1.0/nu)
        # Falling plateau (center at x = -u):
        z2 = -a * (-x - u)    # = a*(x + u)
        z2_clipped = np.clip(z2, None, 500)                  # clip any z2 > 500
        R2 = (1.0 + nu * np.exp(z2_clipped)) ** (-1.0/nu) - 1
        return amplitude * (R1 + R2) + offset

    def mexican_hat_with_gaussian(x,
        amplitude, decay, width, center, gauss_std, v_offset):
        """
        Standard Mexican‐hat with amplitude A, width w, decay tau, center mu, multiplied by a gaussian, and added to a negative gaussian
        """
        xs = x - center
        mex_hat = amplitude * (1.0 - (xs**2) / (width**2)) * np.exp(- (xs**2) / (2.0 * decay**2))
        gauss = ((2.5/(gauss_std*np.sqrt(2*np.pi)))*np.exp(-0.5*(xs/gauss_std)**2))*gauss_std # normalized Gaussian (max height = 1)
        mex_hat_full = mex_hat*gauss + (gauss*v_offset)
        return mex_hat_full

    def full_model(x, pd, pm):
        return double_generalized_logistic(x, *pd) + mexican_hat_with_gaussian(x, *pm)


    # === 1) Alternating least_squares ===
    for _ in range(n_iter):
        # ---- (a) Fit Generalized-logistic curve (=R ichard's curve) baseline (freeze Mexican hat curve) ----
        # Taking into account the negative part of the Mexican curve (can cause the baseline to "float up")
        # def resid_dl(params_dl):
        #     return y - (double_generalized_logistic(x, *params_dl) + mexican_hat(x, *p_mh))

        # Taking into account only the positive part of the mexican hat curve - so that the baseline will just ignore the peak, but appropriately fit the rest
        def resid_dl(params_dl):
            mh_vals = mexican_hat_with_gaussian(x, *p_mh)
            mh_pos  = np.clip(mh_vals, 0, None)   # keep only the positive lobe of MH
            return y - (double_generalized_logistic(x, *params_dl) + mh_pos)

        sol_dl = least_squares(resid_dl, p_dl, bounds=(dl_lb, dl_ub), max_nfev=2000)
        p_dl = sol_dl.x

        # Build the current baseline for trapezoid‐fit
        fitted_baseline_current = double_generalized_logistic(x, *p_dl)

        # ---- (b) Fit trapezoid to extract (t2, t3) ----
        try:
            w_bwd, w_fwd, trap_cur_iter, (t1_iter, t2_iter, t3_iter, t4_iter) = \
                fit_trapezoid_windows(x=x,
                                      baseline=fitted_baseline_current,
                                      min_plateau_duration=min_plateau_duration,
                                      width_guess=(t3_iter - t2_iter) 
                                          if 't2_iter' in locals() and 't3_iter' in locals() 
                                          else min_plateau_duration)
        except ValueError:
            # Fallback if trapezoid‐fit fails (degenerates to triangle/no plateau)
            half = min_plateau_duration / 2.0
            t2_iter = -half
            t3_iter =  half
            w_bwd = np.where((x >= t2_iter) & (x <= 0))[0]
            w_fwd = np.where((x >= 0) & (x <= t3_iter))[0]
            def T_fallback(p, xx):
                t1_, t2_, t3_, t4_ = p
                out_ = np.zeros_like(xx)
                mask_rise  = (xx >= t1_) & (xx < t2_)
                out_[mask_rise]  = (xx[mask_rise] - t1_)/(t2_ - t1_)
                mask_plate = (xx >= t2_) & (xx <= t3_)
                out_[mask_plate] = 1.0
                mask_fall  = (xx > t3_) & (xx <= t4_)
                out_[mask_fall]  = (t4_ - xx[mask_fall])/(t4_ - t3_)
                return out_
            trap_cur_iter = T_fallback((t2_iter, t2_iter, t3_iter, t3_iter), x)

        # logger.info(fitted_baseline_current)
        # Compute plateau_duration as a FLOAT
        plateau_duration = t3_iter - t2_iter
        max_width_allowed = plateau_duration / 2.0
        if max_width_allowed <= 0:
            max_width_allowed = min_plateau_duration / 2.0

        # ---- (c) Clamp MH “width” ≤ max_width_allowed ----
        mh_ub_iter = mh_ub.copy()
        mh_ub_iter[2] = min(mh_ub_iter[2], max_width_allowed)

        # # Ensure p_mh[2] is within new bounds ───────
        # if p_mh[2] > mh_ub_iter[2]:
        #     p_mh[2] = (mh_lb[2] + mh_ub_iter[2]) / 2.0
        # if p_mh[2] < mh_lb[2]:
        #     p_mh[2] = (mh_lb[2] + mh_ub_iter[2]) / 2.0

        # ---- (d) Fit mexican hat (synchrony peak) ----
        def resid_mh(p_mh):
            return y - (double_generalized_logistic(x, *p_dl) + mexican_hat_with_gaussian(x, *p_mh))
        try:
            sol_mh = least_squares(resid_mh, p_mh, bounds=(mh_lb, mh_ub), max_nfev=2000,)
            p_mh = sol_mh.x
        except ValueError as e:
            # if the solver complains "x is not within the trust region",
            # just log and fall back to a zero‐hat (or previous p_mh).
            # print("Warning: mexican‐hat fit failed, falling back to zeros:", e)
            # option A: zero‐out the hat completely
            # p_mh = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            # option B: or keep the last good parameters:
            p_mh = p_mh  # literally do nothing
            
    # End of iterative least square fit (one curve after the other)

    # === 2) Final parameters & outputs ===
    popt_dl = p_dl.copy()
    popt_mh = p_mh.copy()
    popt_all = np.concatenate((popt_dl, popt_mh))

    fitted_baseline = double_generalized_logistic(x, *popt_dl)
    fitted_full     = full_model(x, popt_dl, popt_mh)

    residuals_base = y - fitted_baseline
    residuals_full = y - fitted_full

    ss_tot  = np.sum((y - y.mean())**2)
    r2_full = 1 - np.sum((y - fitted_full)**2) / ss_tot
    r2_base = 1 - np.sum((y - fitted_baseline)**2) / ss_tot

    # === 3) One final trapezoid on the final baseline; use a safe default if t2/t3 not defined ===
    try:
        gg = t3 - t2
    except NameError:
        gg = min_plateau_duration

    window_backward, window_forward, trap_curve, (t1, t2, t3, t4) = \
        fit_trapezoid_windows(x=x,
                              baseline=fitted_baseline,
                              min_plateau_duration=min_plateau_duration,
                              width_guess=gg)

    return (
        fitted_baseline,     # the Richards‐based double‐logistic
        fitted_full,         # baseline + Mexican‐hat
        residuals_base,
        residuals_full,
        window_forward,
        window_backward,
        trap_curve,
        popt_all,            # [amplitude_DL, a, nu, midpoint, offset, amp_MH, width, tau, mu]
        r2_full,
        r2_base
    )

def compute_trough_areas(residuals, window_forward, window_backward): # For the spiking probability trough area
    """
    Integrate only the negative‐residual bits in each window.
    Returns (area_forward, area_backward)
    """
    # forward
    rf = residuals[window_forward]
    area_fwd = np.sum(rf[rf < 0])
    # backward
    rb = residuals[window_backward]
    area_bwd = np.sum(rb[rb < 0])
    return area_fwd, area_bwd

def compute_synchrony_peak_height(psth_bins, fitted_full, fitted_baseline):
    """
    The 'mexican‐hat' synchrony curve is full – baseline.
    We return:
      • peak_height = max(synch_curve)
      • peak_time   = psth_bins[argmax(synch_curve)]
    """
    synch_curve = fitted_full - fitted_baseline
    idx = np.argmax(synch_curve)
    height = float(synch_curve[idx])
    offset = float(psth_bins[idx])
    return height, offset

def estimate_null_trough_distribution( # Sample from the fitted null model
    fitted_curve, psth_bins, window_forward, window_backward,
    N_spikes, lowpass_cutoff, n_iter=100, fsamp=2048):
    """
    For i in 1..n_iter:
      • draw N_spikes samples from the probability mass function (pmf) 'fitted_full'
      • form surrogate residuals = (counts/N_spikes) - fitted_full
      • compute trough areas in each window
    Returns:
      (mean_fwd, std_fwd, mean_bwd, std_bwd)
    """
    if n_iter < 1: # case where no iterations to calculate p values
        return np.nan, np.nan, np.nan, np.nan
    else:
        pmf = np.clip(fitted_curve, 0, None) # Making sure it doesn't go below zero
        pmf = pmf / pmf.sum() # Making sure it sums to 1
        areas_fwd = np.zeros(n_iter)
        areas_bwd = np.zeros(n_iter)
        for i in range(n_iter):
            draws = np.random.choice(psth_bins[0:-1], size=N_spikes, p=pmf) # sample
            # build surrogate probability distribution
            null_hist, _ = np.histogram(draws, bins=psth_bins)
            prob_surr = null_hist / N_spikes
            prob_surr = convert_histogram_into_smoothed_probability_distribution(prob_surr, lowpass_cutoff, bin_fsamp=fsamp)
            # surrogate residuals relative to full fit
            res_surr = prob_surr - fitted_curve
            # compute trough areas
            areas_fwd[i], areas_bwd[i] = compute_trough_areas(
                res_surr, window_forward, window_backward)
        # # Display last iteration hsit for debugging purpose
        # plt.plot(psth_bins[:-1], prob_surr)
        # plt.plot(psth_bins[:-1], pmf, '--')
        # plt.savefig("ZZZ_debug_null_distrib.png")
        # plt.close()
        return (areas_fwd.mean(), areas_fwd.std(),
            areas_bwd.mean(), areas_bwd.std())

def compute_corrected_spiking_probability_trough(raw_area, null_mean, null_std):
    """
    Given:
      • raw_area : the trough from your data
      • null_mean, null_std : from estimate_null_trough_distribution()
    Returns:
      (raw_area, corrected_area, z_score, p_val)
    """
    if np.isnan(null_mean) or np.isnan(null_std):
        raw_area = raw_area
        corrected = raw_area
        z = np.nan
        p_val = np.nan
    else:
        corrected = raw_area - null_mean
        # add tiny epsilon in case null_std==0
        z = (raw_area - null_mean) / (null_std + 1e-10)
        # one-sided p-value: probability that null ≤ observed
        p_val = norm.cdf(z)
    return raw_area, corrected, z, p_val

def estimate_trough_delay_from_autocorr(residuals, window_idx, fsamp, max_lag_ms=100):
    """
    Compute the normalized autocorrelation of 'residuals' restricted to window_idx,
    then find the positive lag (in samples) at which the autocorr reaches its minimum.
    Return that lag both in seconds and in samples.
    """
    # 1) extract the segment
    seg = residuals[window_idx]
    n = len(seg)
    if n < 2:
        return np.nan, None, None, None

    # 2) compute full autocorr (unbiased)
    ac_full = np.correlate(seg - seg.mean(), seg - seg.mean(), mode='full')
    ac_full /= (np.var(seg) * n)

    # 3) build lag vectors
    lags_samples = np.arange(-n + 1, n)
    lags_sec = lags_samples / fsamp

    # 4) restrict to positive lags up to max_lag_ms
    max_lag_samp = int(max_lag_ms * fsamp / 1000)
    center = n - 1  # index of zero‐lag in ac_full
    # consider lags 1…max_lag_samp after the center
    pos_offsets = np.arange(1, min(max_lag_samp, n - 1) + 1)
    pos_indices = center + pos_offsets

    ac_pos = ac_full[pos_indices]
    lag_pos = pos_offsets / fsamp

    # 5) find the trough (minimum) within those positive lags
    trough_idx_in_pos = np.argmin(ac_pos)
    trough_delay_s = float(lag_pos[trough_idx_in_pos])
    trough_delay_samples = int(pos_offsets[trough_idx_in_pos])

    return trough_delay_s, trough_delay_samples, ac_full, lags_sec

def compute_coherence_between_signals(signal1, signal2, fsamp=2048, coh_window_length=1, coh_windows_overlap = 0.5, max_freq=100, upsampling_frequency_resolution=1):
    n_per_seg = coh_window_length * fsamp # convert window length to samples instead of s
    n_overlap = np.round(n_per_seg * coh_windows_overlap).astype(int)
    n_fft = np.round(upsampling_frequency_resolution * n_per_seg).astype(int)
    # Compute intra-group coherence for group 1
    f, COH_intragroup_X = csd(detrend(signal1), detrend(signal1), window=windows.hann(n_per_seg), noverlap=n_overlap, nfft=n_fft, fs=fsamp)
    # Compute intra-group coherence for group 2
    f, COH_intragroup_Y = csd(detrend(signal2), detrend(signal2), window=windows.hann(n_per_seg), noverlap=n_overlap, nfft=n_fft, fs=fsamp)
    # Compute inter-group coherence
    f, COH_intergroup = csd(detrend(signal1), detrend(signal2), window=windows.hann(n_per_seg), noverlap=n_overlap, nfft=n_fft, fs=fsamp)

    # Limiting the frequencies up to the max frequency we are interested in (for efficiency)
    max_index = np.where(f[f<=max_freq])[0][-1]
    COH_intragroup_X = COH_intragroup_X[:max_index]
    COH_intragroup_Y = COH_intragroup_Y[:max_index]
    COH_intergroup = COH_intergroup[:max_index]
    frequency_bins = f[:max_index]

    coherence_over_frequencies = (np.abs(COH_intergroup) ** 2) / (COH_intragroup_X * COH_intragroup_Y) # Welch's method of coherence calculation

    return frequency_bins, coherence_over_frequencies

###################
# ANALYZIS FUNCTIONS
###################

def get_firing_rate(
    spike_trains_MN,
    spike_trains_RC=None,
    generate_figure=False,
    savepath=None):
    """
    Calculate mean/std/max/min firing rates and ISI statistics for spike trains,
    and optionally plot two rows of results:
      Row 1 (Firing Rates):
        – (a) line + shaded‐std plot of mean/std/max/min firing‐rate across MN index
        – (b) histogram of the mean firing rates (MNs)
        – (c) histogram of the mean firing rates (RCs, if provided)
      Row 2 (Inter‐Spike Intervals):
        – (d) line + shaded‐std plot of mean/std/max/min ISI across MN index
        – (e) histogram of the ISI‐COV (coefficient of variation) for MNs
        – (f) histogram of the ISI‐COV for RCs (if provided)
    Returns
    -------
    firing_rates_MN : dict of dicts
        {"mean":{mn_id:…}, "std":{…}, "max":{…}, "min":{…}}
    firing_rates_RC : dict of dicts  (empty if spike_trains_RC is None)
    cov_MN : dict
        ISI coefficient of variation (std(ISI)/mean(ISI)) for each MN
    cov_RC : dict
        ISI coefficient of variation for each RC (or empty dict if None)
    """
    # logger = logging.getLogger(__name__)
    # logger.info(f"      Computing firing rates and inter-spike-intervals...")
    # --- 1) Compute firing‐rate stats and ISI‐COV for MNs ---
    firing_rates_MN = {"mean": {}, "std": {}, "max": {}, "min": {}}
    cov_MN          = {}
    for mn_id, spike_times in spike_trains_MN.items():
        # If fewer than 2 spikes → no valid ISI or firing-rate
        if len(spike_times) < 2:
            firing_rates_MN["mean"][mn_id] = 0.0
            firing_rates_MN["std"][mn_id]  = 0.0
            firing_rates_MN["max"][mn_id]  = 0.0
            firing_rates_MN["min"][mn_id]  = 0.0
            cov_MN[mn_id] = np.nan
            continue

        # Convert spike_times to a sorted NumPy array (assume already sorted)
        times = np.asarray(spike_times)
        # 1a) firing‐rate = 1 / ISI
        inst_rates = 1.0 / np.diff(times)  # (Hz)
        firing_rates_MN["mean"][mn_id] = np.mean(inst_rates)
        firing_rates_MN["std"][mn_id]  = np.std(inst_rates)
        firing_rates_MN["max"][mn_id]  = np.max(inst_rates)
        firing_rates_MN["min"][mn_id]  = np.min(inst_rates)

        # 1b) ISI values (in same time‐units as spike_times, e.g. seconds)
        isis = np.diff(times)
        mean_isi = np.mean(isis)
        std_isi  = np.std(isis)
        raw_cov = (std_isi / mean_isi) if mean_isi > 0 else np.nan
        cov_MN[mn_id] = raw_cov if (not np.isnan(raw_cov) and raw_cov >= 0.01) else 0.0
        # prevent extremely low values of cov

    # --- 1′) Compute firing‐rate stats and ISI‐COV for RCs (if provided) ---
    firing_rates_RC = {"mean": {}, "std": {}, "max": {}, "min": {}}
    cov_RC          = {}
    if spike_trains_RC is not None:
        for rc_id, spike_times in spike_trains_RC.items():
            if len(spike_times) < 2:
                firing_rates_RC["mean"][rc_id] = 0.0
                firing_rates_RC["std"][rc_id]  = 0.0
                firing_rates_RC["max"][rc_id]  = 0.0
                firing_rates_RC["min"][rc_id]  = 0.0
                cov_RC[rc_id] = np.nan
                continue

            times = np.asarray(spike_times)
            inst_rates = 1.0 / np.diff(times)
            firing_rates_RC["mean"][rc_id] = np.mean(inst_rates)
            firing_rates_RC["std"][rc_id]  = np.std(inst_rates)
            firing_rates_RC["max"][rc_id]  = np.max(inst_rates)
            firing_rates_RC["min"][rc_id]  = np.min(inst_rates)

            isis = np.diff(times)
            mean_isi = np.mean(isis)
            std_isi  = np.std(isis)
            raw_cov = std_isi / mean_isi if mean_isi > 0 else np.nan
            cov_RC[rc_id] = raw_cov if (not np.isnan(raw_cov) and raw_cov >= 0.01) else 0.0
    else:
        # Ensure cov_RC is at least an empty dict
        cov_RC = {}

    # --- 2) Optional plotting ---
    if generate_figure:
        # 2a) Extract arrays in neuron‐index order for MN firing rates
        mn_ids    = list(firing_rates_MN["mean"].keys())
        fr_means  = np.array([firing_rates_MN["mean"][i] for i in mn_ids])
        fr_stds   = np.array([firing_rates_MN["std"][i]  for i in mn_ids])
        fr_max    = np.array([firing_rates_MN["max"][i]  for i in mn_ids])
        fr_min    = np.array([firing_rates_MN["min"][i]  for i in mn_ids])
        idx_MN    = np.arange(len(mn_ids))

        # 2b) Extract arrays for MN ISI
        isi_vals  = {}  # store per‐MN array of ISIs if needed for plotting mean/std
        isi_means = []
        isi_stds  = []
        isi_max   = []
        isi_min   = []
        for mn_id in mn_ids:
            times = np.asarray(spike_trains_MN[mn_id])
            if len(times) < 2:
                isi_means.append(0.0)
                isi_stds.append(0.0)
                isi_max.append(0.0)
                isi_min.append(0.0)
            else:
                isis = np.diff(times)
                isi_vals[mn_id] = isis
                isi_means.append(np.mean(isis))
                isi_stds.append(np.std(isis))
                isi_max.append(np.max(isis))
                isi_min.append(np.min(isis))
        isi_means = np.array(isi_means)
        isi_stds  = np.array(isi_stds)
        isi_max   = np.array(isi_max)
        isi_min   = np.array(isi_min)

        # 2c) If RCs exist, extract their cov distribution
        if spike_trains_RC is not None and len(cov_RC)>0:
            rc_ids    = list(firing_rates_RC["mean"].keys())
            cov_vals_RC = np.array([cov_RC[r] for r in rc_ids])
        else:
            cov_vals_RC = np.array([])

        # --- Create a 2×3 grid: widths 3:1:1, heights (1,1) ---
        fig = plt.figure(figsize=(16, 8))
        gs  = fig.add_gridspec(
            nrows=2, ncols=3,
            width_ratios=[3,1,1],
            height_ratios=[1,1],
            hspace=0.4, wspace=0.3
        )

        # ───────── Row 1: Firing‐Rate Plots ─────────
        # (a) Mean±STD or max/min across MN index
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.plot(idx_MN, fr_means, color='C1', label='Mean FR')
        ax0.fill_between(
            idx_MN,
            fr_means - fr_stds,
            fr_means + fr_stds,
            color='C1', alpha=0.3,
            label='± STD'
        )
        ax0.plot(idx_MN, fr_max, color='C1', linestyle='--', label='Max FR')
        ax0.plot(idx_MN, fr_min, color='C1', linestyle=':',  label='Min FR')
        ax0.set_xlabel("Motoneuron index")
        ax0.set_ylabel("Firing rate (Hz)")
        ax0.set_title("Firing rates across MNs")
        ax0.legend(loc='upper right')

        # (b) Histogram of mean firing rates (MNs)
        finite_mask = np.isfinite(fr_means)
        fr_means = fr_means[finite_mask]
        # logger.info(f"          fr_means_MN (size = {fr_means.size})= {fr_means}")
        if fr_means.size > 0:
            ax1 = fig.add_subplot(gs[0, 1])
            ax1.hist(fr_means, bins='auto', color='C1', edgecolor='k', alpha=0.7)
            ax1.set_xlabel("Mean firing rate (Hz)")
            ax1.set_ylabel("Count")
            ax1.set_title("MN mean‐FR histogram")

        # (c) Histogram of mean firing rates (RCs), if available
        if spike_trains_RC is not None:
            fr_means_RC = np.array([firing_rates_RC["mean"][r] for r in rc_ids])
            finite_mask = np.isfinite(fr_means_RC)
            fr_means_RC = fr_means_RC[finite_mask]
            # logger.info(f"          fr_means_RC (size = {fr_means_RC.size})= {fr_means_RC}")
            if (fr_means_RC.size > 0) and (fr_means_RC[fr_means_RC > 0.1].size > 0): # Make sure at least some of them are firing
                ax2 = fig.add_subplot(gs[0, 2])
                ax2.hist(fr_means_RC, bins='auto', color='C4', edgecolor='k', alpha=0.7)
                ax2.set_xlabel("Mean firing rate (Hz)")
                ax2.set_ylabel("Count")
                ax2.set_title("RC mean‐FR histogram")

        # ───────── Row 2: ISI Plots ─────────
        # (d) Mean±STD or max/min ISI across MN index
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(idx_MN, isi_means, color='C2', label='Mean ISI')
        ax3.fill_between(
            idx_MN,
            isi_means - isi_stds,
            isi_means + isi_stds,
            color='C2', alpha=0.3,
            label='± STD'
        )
        ax3.plot(idx_MN, isi_max, color='C2', linestyle='--', label='Max ISI')
        ax3.plot(idx_MN, isi_min, color='C2', linestyle=':',  label='Min ISI')
        ax3.set_xlabel("Motoneuron index")
        ax3.set_ylabel("Inter‐spike interval (s)")
        ax3.set_title("ISIs across MNs")
        ax3.legend(loc='upper right')
        ax3.set_ylim(bottom=0, top=np.min(np.array([1,np.max(isi_max)])))

       # (e) Histogram of MN ISI‐COV
        cov_vals_MN = np.array([cov_MN[m] for m in mn_ids])
        finite_mask = np.isfinite(cov_vals_MN)
        cov_vals_MN = cov_vals_MN[finite_mask]
        # logger.info(f"          cov_vals_MN (size = {cov_vals_MN.size})= {cov_vals_MN}")
        if cov_vals_MN.size > 0:
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.hist(cov_vals_MN,
                     bins="auto", color="C2", edgecolor="k", alpha=0.7)
            ax4.set_xlabel("ISI CV (std/mean)")
            ax4.set_ylabel("Count")
            ax4.set_title("MN ISI‐CV histogram")

        # (f) Histogram of RC ISI‐COV (if any)
        finite_mask = np.isfinite(cov_vals_RC)
        cov_vals_RC = cov_vals_RC[finite_mask]
        # logger.info(f"          cov_vals_RC (size = {cov_vals_RC.size})= {cov_vals_RC}")
        if cov_vals_RC.size > 0:
            ax5 = fig.add_subplot(gs[1, 2])
            ax5.hist(cov_vals_RC,
                     bins="auto", color="C4", edgecolor="k", alpha=0.7)
            ax5.set_xlabel("ISI CV (std/mean)")
            ax5.set_ylabel("Count")
            ax5.set_title("RC ISI‐CV histogram")

        # 2d) Save if requested
        if savepath is not None:
            figure_name = os.path.join(savepath, "ANALYSIS_Firing_rate_and_ISI.png")
            plt.savefig(figure_name, bbox_inches='tight')
        plt.show()

    return firing_rates_MN, firing_rates_RC, cov_MN, cov_RC

def get_ground_truth_RI_connectivity(
    MN_idx_kept, total_nb_motoneurons, mean_firing_rates, MN_to_MN_connectivity_matrix,
    RC_to_MN_IPSP, MN_idx_by_pool,
    generate_figure=False, savepath=None ):
    """
    Compute the *true* recurrent‐inhibition strength between every pair of motoneurons,
    and also summarize for each pool‐to‐pool pairing.

    Returns a dict with keys:
      • 'MN_to_MN_eff':  N×N matrix E_full[i,j] = j→i inhibition
      • 'MN_received_total':  length‐N array of sum_j E_full[i,j]
      • 'MN_delivered_total': length‐N array of sum_i E_full[i,j]
      • 'pool_pair': dict mapping "poolA<->poolB" → {
            'received': length‐N array (only MNs in poolA get non‐zero from poolB),
            'delivered': length‐N array (only MNs in poolA deliver to poolB)
     • "per_pool_received": # How much each MN is inhibited by pool N
     • "per_pool_delivered": # How much each MN is inhibitig pool N
        }
    """
    Ntot = total_nb_motoneurons
    # convert to microamps (from nanoAmps)
    RC_to_MN_IPSP = RC_to_MN_IPSP * 1e3  

    # 1) build the full MN→MN matrix
    E_full = np.full((Ntot, Ntot), np.nan, dtype=float)
    received_total  = np.zeros(Ntot, dtype=float)
    delivered_total = np.zeros(Ntot, dtype=float)

    for i in MN_idx_kept:       # i = source MN (inhibitor) = rows of the matrix
        for j in MN_idx_kept:   # j = target MN (inhibited) = colums of the matrix
            eff = (
                MN_to_MN_connectivity_matrix[i, j]
                * RC_to_MN_IPSP
                * mean_firing_rates[f"MN_{i}"] # Multiply by firing rate?
            )
            E_full[i, j] = eff
            delivered_total[i] += eff
            received_total[j]  += eff
    max_across_delivered_received = np.nanmax(np.concatenate((delivered_total, received_total)))

    # 2) now build per‐pool‐pair summaries
    pool_pair = {}
    pool_names = list(MN_idx_by_pool.keys())
    for pool_i in pool_names:           
        idx_i = MN_idx_by_pool[pool_i]  # the MNs in pool_i
        for pool_j in pool_names:
            idx_j = MN_idx_by_pool[pool_j] # the MNs in pool_j
            key = f"{pool_i}<->{pool_j}"
            # received by pool_i *from* pool_j:
            rec_ij = np.zeros(Ntot, dtype=float)
            # for each MN in pool_i, sum over sources in pool_j
            rec_ij[idx_i] = np.nansum(E_full[np.ix_(idx_j, idx_i)], axis=0)

            # delivered by pool_i *to* pool_j:
            del_ij = np.zeros(Ntot, dtype=float)
            # for each MN in pool_i, sum over targets in pool_j
            del_ij[idx_i] = np.nansum(E_full[np.ix_(idx_j, idx_i)], axis=1)

            pool_pair[key] = {
                "received":  rec_ij,
                "delivered": del_ij
            }

    # 2a) per‐source‐pool totals across ALL MNs
    per_pool_received = {}
    per_pool_delivered = {}
    for pool_j, idx_j in MN_idx_by_pool.items():
        rec_from_j = np.nansum(E_full[idx_j, :], axis=0)
        deliv_to_j = np.nansum(E_full[:, idx_j], axis=1)
        per_pool_received[pool_j]  = rec_from_j
        per_pool_delivered[pool_j] = deliv_to_j

    out = {
        "MN_to_MN_eff":            E_full,
        "MN_received_total":       received_total,
        "MN_delivered_total":      delivered_total,
        "pool_pair_connectivity":  pool_pair,
        # the two new dicts:
        "per_pool_received":       per_pool_received,
        "per_pool_delivered":      per_pool_delivered,
    }

    # 3) optional figure
    if generate_figure:
        fig = plt.figure(figsize=(26, 16 + 2*len(MN_idx_by_pool)))
        # Build a GridSpec with 2 top‐rows plus one per source‐pool
        P = len(MN_idx_by_pool)
        height_ratios = [3, 2, 2] + [2]*P
        gs = gridspec.GridSpec(nrows=3+P, ncols=3,
                       width_ratios=[3, 2, 2],
                       height_ratios=height_ratios,
                       wspace=0.3, hspace=0.4)

        # ——————————————
        # Top‐left: MN→MN heatmap
        ax0 = fig.add_subplot(gs[0:2,0:2])
        cmap = cmr.bubblegum.copy()
        cmap.set_bad(color='lightgray')   # nan → gray
        im = ax0.imshow(E_full, origin='lower', aspect='equal', cmap=cmap)
        ax0.set_title("Effective MN→MN Inhibition")
        ax0.set_ylabel("Source MN index: inhibiting / delivering")
        ax0.set_xlabel("Target MN index: inhibited / receiving")
        cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
        cbar.set_label("Inhibition strength (mean µA)\n(mean firing rate × IPSP × nb of disynaptic connections)")
        # cbar.set_label("Inhibition strength (mean µA)\n(IPSP × nb of disynaptic connections)")

        # ——————————————
        # Top‐right: delivered_total
        ax1 = fig.add_subplot(gs[0,2])
        kept = ~np.isnan(delivered_total)
        # gray out the NaNs
        for i in np.where(~kept)[0]:
            ax1.axvspan(i-0.5, i+0.5, color='lightgray', zorder=0)
        ax1.bar(np.where(kept)[0], delivered_total[kept], color=[0.8,0.6,1], alpha=1)
        ax1.set_title("Total Inhibition Delivered\nHow much each MN is inhibiting all other MNs\n(mean firing rate × IPSP × nb of disynaptic connections)")
        # ax1.set_title("Total Inhibition Delivered\n(IPSP × nb of disynaptic connections)")
        ax1.set_ylabel("Sum over all targets (mean µA)")
        ax1.set_xlabel("MN index")
        ax1.set_xlim(-0.5, Ntot-0.5)
        ax1.set_ylim(0, max_across_delivered_received)
        ax1_ylim = ax1.get_ylim()

        # ——————————————
        # Top‐right 2nd row: received_total
        ax2 = fig.add_subplot(gs[1,2])
        kept = ~np.isnan(received_total)
        for i in np.where(~kept)[0]:
            ax2.axvspan(i-0.5, i+0.5, color='lightgray', zorder=0)
        ax2.bar(np.where(kept)[0], received_total[kept], color=[0.6,0.8,1], alpha=1)
        ax2.set_title("Total Inhibition Received\nHow much each MN is inhibited by all other MNs\n(mean firing rate × IPSP × nb of disynaptic connections)")
        # ax2.set_title("Total Inhibition Received\n(IPSP × nb of disynaptic connections)")
        ax2.set_ylabel("Sum over all sources (mean µA)")
        ax2.set_xlabel("MN index")
        ax2.set_xlim(-0.5, Ntot-0.5)
        ax2.set_ylim(0, max_across_delivered_received)
        ax2_ylim = ax2.get_ylim()

        # ——————————————
        # Bottom: one row per source‐pool
        pool_names = list(MN_idx_by_pool.keys())
        for row, pool in enumerate(pool_names, start=2):
            rec = per_pool_received[pool]
            deliv = per_pool_delivered[pool]

            # left: received *from* this pool
            ax = fig.add_subplot(gs[row, 1])
            mask = rec > 0
            ax.bar(np.arange(Ntot)[mask], rec[mask], color=[0.6,0.8,1], alpha=1)
            for i in np.where(~mask)[0]:
                ax.axvspan(i-0.5, i+0.5, color='lightgray', zorder=0)
            ax.set_title(f"Received from {pool}\nHow much each MN is inhibited by all MNs of {pool}")
            ax.set_xlim(-0.5, Ntot-0.5)
            ax.set_ylim(ax2_ylim)          # match the “received_total” scale
            if row < 2+P-1:
                ax.set_xticklabels([])
            ax.set_ylabel("µA")

            # right: delivered *to* this pool
            ax = fig.add_subplot(gs[row, 2])
            mask = deliv > 0
            ax.bar(np.arange(Ntot)[mask], deliv[mask], color=[0.8,0.6,1], alpha=1)
            for i in np.where(~mask)[0]:
                ax.axvspan(i-0.5, i+0.5, color='lightgray', zorder=0)
            ax.set_title(f"Delivered to {pool}\nHow much each MN is inhibiting all MNs of {pool}")
            ax.set_xlim(-0.5, Ntot-0.5)
            ax.set_ylim(ax1_ylim)         # match the “delivered_total” scale
            if row < 2+P-1:
                ax.set_xticklabels([])
            ax.set_ylabel("µA")

        # ——————————————
        # finally save or show
        if savepath is not None:
            fn = os.path.join(savepath, "ANALYSIS_connectivity_ground_truth.png")
            plt.savefig(fn, bbox_inches='tight')
        plt.show()

    return out

def get_graph_theory_connectivity_measures(MN_to_MN_connectivity_matrix, idx_kept,
                                           generate_figure=False, savepath=None):
    import networkx as nx
    graph_theory_measures = {"density": np.nan,
                            "mean_nb_disynaptic_inhib": np.nan,
                            "std_nb_disynaptic_inhib": np.nan,
                            "global_efficiency": np.nan,
                            "mean_shortest_path_length": np.nan,
                            "directed_clustering_coeff": np.nan,
                            "null_clustering_coeff_mean": np.nan,
                            "null_clustering_coeff_std": np.nan,
                            "clusering_coeff_Z_score": np.nan,
                            "ratio_empirical_vs_null": np.nan}
    # Set networkx graph
    MN_to_MN_connectivity_matrix = MN_to_MN_connectivity_matrix[idx_kept][:, idx_kept]
    total_nb_motoneurons = MN_to_MN_connectivity_matrix.shape[0]
    nb_edges = np.count_nonzero(MN_to_MN_connectivity_matrix)
    MN_to_MN_connectivity_matrix_clamped = np.minimum(
        MN_to_MN_connectivity_matrix, 1.0)
    graph_MN_to_MN = nx.DiGraph()
    graph_MN_to_MN_unweighted = nx.DiGraph()
    for i in range(total_nb_motoneurons):
        for j in range(total_nb_motoneurons):
            if MN_to_MN_connectivity_matrix[i,j]>0:
                graph_MN_to_MN.add_edge(i, j,
                        distance=1.0/MN_to_MN_connectivity_matrix[i,j],
                        weight=MN_to_MN_connectivity_matrix[i,j])
                graph_MN_to_MN_unweighted.add_edge(i, j,
                        distance=1.0/MN_to_MN_connectivity_matrix_clamped[i,j],
                        weight=MN_to_MN_connectivity_matrix_clamped[i,j])
    # # # # # Density (entire graph, so without considering specific pool pairs)
    mn_mn_density = nb_edges / (total_nb_motoneurons*(total_nb_motoneurons-1))
    # # # # # mean and std of disynaptic weights (entire graph, so without considering specific pool pairs)
    mean_nb_MN_to_MN_disynaptic_inhib = MN_to_MN_connectivity_matrix.mean()
    std_nb_MN_to_MN_disynaptic_inhib  = MN_to_MN_connectivity_matrix.std()
    # # # # # Average shortest path length, global efficiency
    # Convert the weighted adjacency into a graph where "distance" is the inverse disynaptic strength (1/Wij)
    # global efficiency (inverse of average shortest distance)
    # High efficiency means most MNs can rapidly influence any other via a small number of strong disynapses.
    lengths = dict(nx.all_pairs_dijkstra_path_length(
        graph_MN_to_MN_unweighted, weight='distance'))
    MN_to_MN_network_efficiency = np.mean([
        1/lengths[i][j] # clamp weight to 1
        for i in graph_MN_to_MN_unweighted
        for j in graph_MN_to_MN_unweighted
        if i!=j and j in lengths[i]
    ])
    # Compute characteristic path lengh (mean shortest distance between all pairs of nodes)
    shortest_path_lengths = []
    for i in graph_MN_to_MN.nodes():
        d = nx.single_source_dijkstra_path_length(graph_MN_to_MN, i, weight='distance')
        shortest_path_lengths += [d[j] for j in d if j!=i]
    mean_MN_MN_shortest_path_length = np.mean(shortest_path_lengths)
    #
    # # # # # Clustering coefficient (entire graph, so without considering specific pool pairs)
    # High clustering means your MN network has modules or cliques—small groups of MNs that share common Renshaw pathways and strongly inter‐inhibit one another.
    # Low clustering means a more “egalitarian” or random web of connections, with few tightly‐knit triplets.
    directed_clustering_results = directed_clustering_with_nulls(
        MN_to_MN_connectivity_matrix,
        num_random=100,
        null_model="ER",
        random_seed=42
    )
    # Whether recurrent‐inhibition architecture is geared toward broad, fast‐reach inhibition (high efficiency) VS tight, localized inhibition (high clustering).

    # Assgin values to the dictionary to be returned
    graph_theory_measures["density"] = mn_mn_density
    graph_theory_measures["mean_nb_disynaptic_inhib"] = mean_nb_MN_to_MN_disynaptic_inhib
    graph_theory_measures["std_nb_disynaptic_inhib"] = std_nb_MN_to_MN_disynaptic_inhib
    graph_theory_measures["global_efficiency"] = MN_to_MN_network_efficiency
    graph_theory_measures["mean_shortest_path_length"] = mean_MN_MN_shortest_path_length
    graph_theory_measures["directed_clustering_coeff"] = directed_clustering_results["C_emp"]
    graph_theory_measures["null_clustering_coeff_mean"] = directed_clustering_results["C_null_mean"]
    graph_theory_measures["null_clustering_coeff_std"] = directed_clustering_results["C_null_std"]
    graph_theory_measures["clusering_coeff_Z_score"] = directed_clustering_results["Z"]
    graph_theory_measures["ratio_empirical_vs_null"] = directed_clustering_results["ratio_emp_null"]

    # # # # # Optional plotting
    if generate_figure:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        # row 0: weighted
        ax00, ax01 = axes[0]
        # row 1: unweighted
        ax10, ax11 = axes[1]

        # (0,0) weighted adjacency heatmap
        im = ax00.imshow(MN_to_MN_connectivity_matrix, cmap='viridis', origin='lower', aspect='equal')
        ax00.set_title("Weighted adjacency matrix")
        ax00.set_xlabel("Source MN")
        ax00.set_ylabel("Target MN")
        fig.colorbar(im, ax=ax00, fraction=0.046, pad=0.04, label="weight")

        # (0,1) weighted network graph
        pos = nx.spring_layout(graph_MN_to_MN, seed=42)  # fixed layout for repeatability
        # draw nodes
        nx.draw_networkx_nodes(graph_MN_to_MN, pos, ax=ax01, node_size=50, node_color='k')
        # draw edges with width ∝ weight
        weights = np.array([graph_MN_to_MN[u][v]['weight'] for u, v in graph_MN_to_MN.edges()])
        # normalize widths to [0.5, 5]
        wmin, wmax = weights.min(), weights.max()
        widths = 0.5 + 4.5 * (weights - wmin) / (wmax - wmin)
        nx.draw_networkx_edges(
            graph_MN_to_MN, pos, ax=ax01,
            edgelist=list(graph_MN_to_MN.edges()),
            width=widths,
            arrowstyle='-|>',
            arrowsize=10,
            edge_color='C1'
        )
        ax01.set_title("Weighted directed graph")
        ax01.axis('off')

        # (1,0) unweighted adjacency heatmap
        im2 = ax10.imshow(MN_to_MN_connectivity_matrix_clamped, cmap='Greys', origin='lower', aspect='equal')
        ax10.set_title("Binarized adjacency matrix")
        ax10.set_xlabel("Source MN")
        ax10.set_ylabel("Target MN")
        fig.colorbar(im2, ax=ax10, fraction=0.046, pad=0.04, label="0 / 1")

        # (1,1) unweighted network graph
        pos2 = nx.spring_layout(graph_MN_to_MN_unweighted, seed=42)  # same layout for easy comparison
        nx.draw_networkx_nodes(graph_MN_to_MN_unweighted, pos2, ax=ax11, node_size=50, node_color='k')
        nx.draw_networkx_edges(
            graph_MN_to_MN_unweighted, pos2, ax=ax11,
            edgelist=list(graph_MN_to_MN_unweighted.edges()),
            width=1.0,
            arrowstyle='-|>',
            arrowsize=8,
            edge_color='0.5'
        )
        ax11.set_title("Unweighted directed graph")
        ax11.axis('off')

        plt.tight_layout()
        if savepath is not None:
            plt.savefig(os.path.join(savepath, "ANALYSIS_graph_theory_connectivity.png"),
                        bbox_inches='tight')
        plt.show()

    return graph_theory_measures

def get_cross_histogram_measures(spike_trains_MN, corresponding_MN_idx, fsamp,
    MU_corresponding_pool_list, list_of_MUs_by_pool,
    histogram_kind='normalized', lowpass_filter_prob_dist=60, minimum_spike_nb = 1000, minimum_r2 = 0.85, minimum_plateau_duration = 0.05, null_distrib_nb_iter = 100,
    ignore_homonymous_pool = False, ignore_heteronymous_pool = False,
    generate_figure=False, save_histograms=False, savepath=None, logger=None):
    """
    Compute per‐MU recurrent inhibition & synchrony metrics from MU‐vs‐pool cross‐histograms.

    Returns a nested dict:
      results[ 'muscleA<->muscleB' ][ mu_idx ][ 'inhibited' / 'inhibiting' / 'combined'] = { ...metrics... }
    """
    logger = logger or logging.getLogger(__name__)
    sim_foldername = get_last_folder(savepath)
    logger.info(f"       ({sim_foldername}) Building cross‐histograms…")
    cross_histograms_dict, psth_bins = compute_cross_histograms(
        spike_times=spike_trains_MN,
        corresponding_MN_idx=corresponding_MN_idx,
        fsamp=fsamp,
        MU_corresponding_pool_list=MU_corresponding_pool_list,
        list_of_MUs_by_pool=list_of_MUs_by_pool,
        ignore_homonymous_pool=ignore_homonymous_pool, ignore_heteronymous_pool=ignore_heteronymous_pool,
    )
    hMU = cross_histograms_dict['histograms_per_MU']
    hist_key = f"MU_hist_{histogram_kind}"

    results = {}
    if save_histograms:
        results['cross_histograms'] = {}

    for muscle_pair_key, mu_hist_collections in hMU.items():
        # only handle each undirected pair once - the directionality happens later, by running the same loop twice
        if '->inhibited_by->' in muscle_pair_key:
            continue
        base = muscle_pair_key.replace('->inhibiting->','<->')
        inhib_key     = muscle_pair_key
        inhibited_key = muscle_pair_key.replace('->inhibiting->','->inhibited_by->')
        results.setdefault(base, {})
        if save_histograms:
            results['cross_histograms'][base] = {}
        # get the list of MUs in this collection (both directions share same mu_idx keys)
        mu_indices = list(mu_hist_collections[hist_key].keys())
        if generate_figure:
            # Get highest probability value across all across-histograms (for plotting everything on the same y-scale)
            all_crosshist_prob_flattened = np.concatenate([
                # for each array v in each dict d...
                (v / v.sum()).ravel()    # 1) normalize v so it sums to 1, 2) then flatten
                for d in (cross_histograms_dict['histograms_per_MU'][inhibited_key][hist_key],
                        cross_histograms_dict['histograms_per_MU'][inhib_key][hist_key]) 
                for v in d.values()])
            np.nan_to_num(all_crosshist_prob_flattened, nan=0.0, posinf=0.0, neginf=0.0, copy=False) # Make sure there are no invalid values
            max_ylim = np.min([0.01,np.max(all_crosshist_prob_flattened)*1.1])

        mu_iter_within_muscle_pair = 0 # Just for display in the log
        for mu_idx in mu_indices:
            mu_iter_within_muscle_pair += 1
            # prepare storage
            results[base].setdefault(mu_idx, {})
            metrics = {}
            # Prepare plot if requested
            if generate_figure:
                fig = plt.figure(figsize=(26,16))
                fig_grid  = gridspec.GridSpec(2,4, figure=fig)
                cross_hist_colors = {"inhibited_prob_distrib": [0.5,0.7,0.8],
                    "inhibiting_prob_distrib": [0.7,0.5,0.8],
                    "trough_inhibited": [0.6,0.8,1],
                    "trough_inhibiting": [0.8,0.6,1],
                    "txt_inhibited": [0.0,0.2,0.5],
                    "txt_inhibiting": [0.2,0.0,0.85]}
                axes_bounding_boxes = []
            # Get histograms (both directions)
            hist_inh   = cross_histograms_dict['histograms_per_MU'][inhibited_key][hist_key][mu_idx]
            raw_inh    = cross_histograms_dict['histograms_per_MU'][inhibited_key]['MU_hist_raw'][mu_idx]
            Nspikes_inh = raw_inh.sum()
            hist_ing   = cross_histograms_dict['histograms_per_MU'][inhib_key][hist_key][mu_idx]
            raw_ing    = cross_histograms_dict['histograms_per_MU'][inhib_key]['MU_hist_raw'][mu_idx]
            Nspikes_ing = raw_ing.sum()
            if save_histograms:
                results['cross_histograms'][base][mu_idx]={}
            # do both directions in one go
            for direction in ['inhibited','inhibiting','combined']:
                # logger.info(f"          ({sim_foldername}) Processing {base} MU {mu_iter_within_muscle_pair} (idx {mu_idx}) / {len(mu_indices)} ({direction})…")
                if direction == 'inhibited':
                    hist = hist_inh
                    Nspikes = Nspikes_inh.astype(int)
                elif direction == 'inhibiting':
                    hist = hist_ing
                    Nspikes = Nspikes_ing.astype(int)
                elif direction == 'combined':
                    hist = hist_ing + hist_inh[::-1]
                    Nspikes = Nspikes_ing.astype(int) + Nspikes_inh.astype(int)
                # Initialize results with nans
                metrics[direction] = {
                    'forward':   dict(raw_area=np.nan, corrected_area=np.nan, z_score=np.nan, p_val=np.nan, null_mean=np.nan, null_std=np.nan),
                    'backward':  dict(raw_area=np.nan, corrected_area=np.nan, z_score=np.nan, p_val=np.nan, null_mean=np.nan, null_std=np.nan),
                    'asymmetry': dict(raw_area_asym_ratio=np.nan, corrected_area_asym_ratio=np.nan, raw_area_asym_diff=np.nan, corrected_area_asym_diff=np.nan),
                    'sync_height':    np.nan,
                    'sync_time':      np.nan,
                    'delay_forward_IPSP': np.nan,
                    'delay_backward_IPSP': np.nan,
                    'hist_plateau_duration': np.nan, # correspond to the time duration of the w_fwd + w_bwd
                    'proportion_of_prob_within_plateau_duration': np.nan, # correspond to the area of the probability distribution within the plateau duration
                    'r2_full':        np.nan,   # rest = [popt, r2_full, r2_base, ...]
                    'r2_base':        np.nan,
                    'n_spikes':       Nspikes}
                if Nspikes < minimum_spike_nb: # at least N spikes requested
                    # logger.info(f"              Not enough spikes: {Nspikes} < minumum of {minimum_spike_nb}. Skipping.")
                    continue
                # 1) smooth→prob
                prob = convert_histogram_into_smoothed_probability_distribution(hist,
                    lowpass_filter_prob_dist, bin_fsamp=fsamp)
                if save_histograms:
                    results['cross_histograms'][base][mu_idx][direction]=prob
                # 2) fit + windows
                (f_base, f_full, residuals_base, residuals_full,
                 w_fwd, w_bwd, trap_curve,
                 _, r2_full, r2_baseline) = fit_null_curve_and_extract_windows_from_probability_distrib(
                    psth_bins[:-1], prob, fsamp, min_plateau_duration = minimum_plateau_duration)
                # logger.info(f"x=[{psth_bins[0]};{psth_bins[-1]}]\nR² baseline = {r2_baseline}; R² full = {r2_full}")
                metrics[direction]['hist_plateau_duration'] = (len(w_fwd) + len(w_bwd))/ fsamp
                metrics[direction]['proportion_of_prob_within_plateau_duration'] = np.sum( # already normalized to 1 (because it is a probability distribution), so just take the area
                    prob[np.floor(len(psth_bins)/2).astype(int):np.floor(len(psth_bins)/2).astype(int)+len(w_fwd)]) # forward window area
                metrics[direction]['proportion_of_prob_within_plateau_duration'] += np.sum(
                    prob[np.ceil(len(psth_bins)/2).astype(int)-len(w_bwd):np.ceil(len(psth_bins)/2).astype(int)]) # backward window area
                metrics[direction]['r2_full'] = r2_full
                metrics[direction]['r2_base'] = r2_baseline
                if r2_full < minimum_r2: # Do not go further if the fit is not good enough
                    # logger.info(f"              Fit is not good enough (R² < {minimum_r2}). Skipping.")
                    continue
                # 3) trough areas
                area_fwd, area_bwd = compute_trough_areas(residuals_base, w_fwd, w_bwd)
                # 4) null distribution
                nm_fwd, ns_fwd, nm_bwd, ns_bwd = estimate_null_trough_distribution(
                    f_base, psth_bins, w_fwd, w_bwd, N_spikes=Nspikes,
                    lowpass_cutoff=lowpass_filter_prob_dist, n_iter = null_distrib_nb_iter)
                # 5) bias‐correct + z
                raw_f, corr_f, z_f, p_val_f = compute_corrected_spiking_probability_trough(
                    area_fwd, nm_fwd, ns_fwd)
                raw_b, corr_b, z_b, p_val_b = compute_corrected_spiking_probability_trough(
                    area_bwd, nm_bwd, ns_bwd)
                # 6) synchrony & delay
                sync_h, sync_t = compute_synchrony_peak_height(psth_bins, f_full, f_base)
                # forward IPSP timing
                delay_fwd, delay_idx_fwd, _, _   = estimate_trough_delay_from_autocorr(residuals_base, w_fwd, fsamp)
                # backward IPSP timing
                w_bwd_mirrored = np.arange(len(w_bwd))+w_bwd[-1]+1
                delay_bwd, delay_idx_bwd, _, _   = estimate_trough_delay_from_autocorr(residuals_base[::-1], w_bwd_mirrored, fsamp)
                delay_idx_bwd *= -1
                # 7) asymmetry (forward VS backward) (change of sign can happen for corrected values)
                def asym_ratio_fx(f, b):
                    if b == 0:
                        return 1 if f == 0 else np.inf # both zero → 1, otherwise → inf
                    else:
                        return f / b
                asym_raw_ratio = asym_ratio_fx(raw_f, raw_b)
                asym_corrected_ratio = asym_ratio_fx(corr_f, corr_b)
                asym_raw_diff = raw_f - raw_b
                asym_corrected_diff = corr_f - corr_b
                # Assign results
                metrics[direction]['forward'] = dict(raw_area=raw_f, corrected_area=corr_f, z_score=z_f, p_val=p_val_f, null_mean=nm_fwd, null_std=ns_fwd)
                metrics[direction]['backward'] = dict(raw_area=raw_b, corrected_area=corr_b, z_score=z_b,  p_val=p_val_b, null_mean=nm_bwd, null_std=ns_bwd)
                metrics[direction]['asymmetry'] = dict(raw_area_asym_ratio=asym_raw_ratio, corrected_area_asym_ratio=asym_corrected_ratio,
                                                       raw_area_asym_diff=asym_raw_diff, corrected_area_asym_diff=asym_corrected_diff)
                metrics[direction]['sync_height'] = sync_h
                metrics[direction]['sync_time'] = sync_t
                metrics[direction]['delay_forward_IPSP'] = delay_fwd
                metrics[direction]['delay_backward_IPSP'] = delay_bwd
                ######## FIGURE ############
                # Plot in the corresponding subplot if requested
                if generate_figure:
                    entry=metrics[direction]
                    if direction == 'inhibited':
                        ax = fig.add_subplot(fig_grid[0,0:2])
                        title_txt = f"MU #{mu_idx} {direction} by other MUs\nCross-hist built with MU #{mu_idx} used for test spikes, and other MUs used for ref spikes\nN={Nspikes} spikes, fit R² = {entry['r2_full']:.3g} (R² base = {entry['r2_base']:.3g})"
                        direction_for_txt = direction
                    elif direction == 'inhibiting':
                        ax = fig.add_subplot(fig_grid[0,2:4])
                        title_txt = f"MU #{mu_idx} {direction} other MUs\nCross-hist built with MU #{mu_idx} used for ref spikes, and other MUs used for test spikes\nN={Nspikes} spikes, fit R² = {entry['r2_full']:.3g} (R² base = {entry['r2_base']:.3g})"
                        direction_for_txt = direction
                    elif direction == 'combined':
                        ax = fig.add_subplot(fig_grid[1,1:3])
                        title_txt = f"MU #{mu_idx} inhibiting other MUs (forward window) or being inhibited by toher MUs (backward window)\nCross-hist as the sum of the cross-hist built in both directions\nN={Nspikes} spikes, fit R² = {entry['r2_full']:.3g} (R² base = {entry['r2_base']:.3g})"
                        direction_for_txt = "inhibiting"
                    direction_reversed = 'inhibiting' if direction_for_txt=='inhibited' else 'inhibited'
                    baseline_line_height = np.max(f_base)
                    t_fwd = psth_bins[np.floor(len(psth_bins)/2).astype(int):-1]
                    idx_fwd_start = np.floor(len(psth_bins)/2).astype(int)-1
                    t_bwd = psth_bins[:np.ceil(len(psth_bins)/2).astype(int)]
                    idx_bwd_end = np.ceil(len(psth_bins)/2).astype(int)
                    # expected null trough (plotted first because behind everything else)
                    null_mean_base_fwd = np.mean(f_base[w_fwd])
                    null_mean_base_bwd = np.mean(f_base[w_bwd])
                    ax.plot(t_fwd, np.full_like(t_fwd, null_mean_base_fwd),
                            color=cross_hist_colors[f"trough_{direction_for_txt}"],
                            linestyle='--', zorder=1)
                    ax.plot(t_bwd, np.full_like(t_bwd, null_mean_base_bwd),
                            color=cross_hist_colors[f"trough_{direction_reversed}"],
                            linestyle='--', zorder=1)
                    ax.text(x=0.125, y=null_mean_base_fwd,
                            s=f"Expected trough area\nover the forward window\nfrom null-sampling\n={metrics[direction]['forward']['null_mean']*100:.2g}±{metrics[direction]['forward']['null_std']*100:.2g}%",
                            color=cross_hist_colors[f'txt_{direction_for_txt}'], alpha=0.5,
                            horizontalalignment='left', verticalalignment='top')
                    ax.text(x=-0.125, y=null_mean_base_bwd,
                            s=f"Expected trough area\nover the backward window\nfrom null-sampling\n={metrics[direction]['backward']['null_mean']*100:.2g}±{metrics[direction]['backward']['null_std']*100:.2g}%",
                            color=cross_hist_colors[f'txt_{direction_reversed}'], alpha=0.5,
                            horizontalalignment='right', verticalalignment='top')
                    # probability distribution
                    ax.bar(t_fwd, prob[idx_fwd_start:-1],
                        width=psth_bins[1]-psth_bins[0],
                        facecolor=cross_hist_colors[f"{direction_for_txt}_prob_distrib"], alpha=1,
                        linewidth=0, label=f'Spiking probability ({direction_for_txt})')
                    ax.bar(t_bwd, prob[:idx_bwd_end],
                        width=psth_bins[1]-psth_bins[0],
                        facecolor=cross_hist_colors[f"{direction_reversed}_prob_distrib"], alpha=1,
                        linewidth=0, label=f'Spiking probability ({direction_reversed})')
                    # shade troughs
                    # logger.info(f"forward window = {psth_bins[w_fwd[0]]:.3f}s to {psth_bins[w_fwd[-1]]:.3f}s")
                    # logger.info(f"backward window = {psth_bins[w_bwd[0]]:.3f}s to {psth_bins[w_bwd[-1]]:.3f}s")
                    ax.fill_between(psth_bins[w_fwd], prob[w_fwd], f_base[w_fwd],
                                    where=residuals_base[w_fwd]<0, alpha=1,
                                    color=cross_hist_colors[f"trough_{direction_for_txt}"],
                                    label=f"Spiking probability trough ({direction_for_txt})")
                    ax.fill_between(psth_bins[w_bwd], prob[w_bwd], f_base[w_bwd],
                                    where=residuals_base[w_bwd]<0, alpha=1,
                                    color=cross_hist_colors[f"trough_{direction_reversed}"],
                                    label=f"Spiking probability trough ({direction_reversed})")
                    # Lines
                    ax.plot(psth_bins[:-1], prob, '-', color='white', alpha=1.0)
                    ax.plot(psth_bins[:-1], f_full, '--', label='fitted curve (/w mexican hat)', color='red', alpha=1.0)
                    ax.plot(psth_bins[:-1], f_base, '--', label='fitted curve (baseline)', color='black', alpha=1.0)
                    ax.plot(psth_bins[:-1], trap_curve*np.max(f_base), ':', label=f'Trapezoid fitted to baseline\n(used for window selection)', color='black', alpha=0.5)
                    # zero & windows
                    ax.axvline(0, color='white', alpha=1.0, linestyle='--')
                    ax.axvspan(psth_bins[0], psth_bins[w_bwd][0], color='black', alpha=0.1)
                    ax.axvspan(psth_bins[w_fwd[-1]], psth_bins[-1], color='black', alpha=0.1)
                    # window duration and % of area covered
                    ax.annotate(text="", xy=(psth_bins[np.ceil(len(psth_bins)/2).astype(int)-len(w_bwd)], (baseline_line_height+max_ylim)/2),
                                xytext=(psth_bins[np.floor(len(psth_bins)/2).astype(int)+len(w_fwd)], (baseline_line_height+max_ylim)/2),
                                arrowprops=dict(arrowstyle="<->", color='grey'))
                    ax.text(x=psth_bins[np.floor(len(psth_bins)/2).astype(int)], y = (baseline_line_height+max_ylim)/2 + 0.0002,
                            s=f"Histogram plateau duration = {metrics[direction]['hist_plateau_duration']*1000:.1f}ms\n{metrics[direction]['proportion_of_prob_within_plateau_duration']*100:.2g}% of probability distribution covered",
                            horizontalalignment='center', verticalalignment='bottom', color='grey')
                    # sync peak
                    sc = f_full - f_base
                    pk = np.argmax(sc) # index of peak height
                    ax.annotate(text="", xy=(psth_bins[pk]-0.025, baseline_line_height),
                                xytext=(psth_bins[pk]-0.025, baseline_line_height+sc[pk]),
                                arrowprops=dict(arrowstyle="<->", color='red')) # Get an arrow with no text # xy is the point to annoate ; xytext is the text position
                    ax.text(x=psth_bins[pk]-0.028, y=(baseline_line_height*2+sc[pk])/2,
                            s=f"Synchrony peak height\n= {entry['sync_height']*100:.2g}%\nDelay = {entry['sync_time']*1000:.1f}ms",
                            horizontalalignment='right', verticalalignment='bottom', color='red')
                    # Estimated inhibition text
                    txt_fwd_inhb = 'delivered' if direction_for_txt=='inhibiting' else 'received'
                    txt_bwd_inhb = 'received' if direction_for_txt=='inhibiting' else 'delivered'
                    # IPSP delay
                    #   IPSP delay forward
                    ax.annotate(text=f"",
                                xy=(0, prob[np.round(len(psth_bins)/2).astype(int)+delay_idx_fwd]),
                                xytext=(entry['delay_forward_IPSP'], prob[np.round(len(psth_bins)/2).astype(int)+delay_idx_fwd]),
                                arrowprops=dict(arrowstyle="<->", color=[0.1,0,0.7]))
                    ax.text(x=entry['delay_forward_IPSP']/2, y=prob[np.round(len(psth_bins)/2).astype(int)+delay_idx_fwd]-0.0002,
                            s=f"IPSP fwd timing ({direction_for_txt})\n= {entry['delay_forward_IPSP']*1000:.1f}ms\n(mix delay & duration)",
                            horizontalalignment='left', verticalalignment='top', color=[0.1,0,0.7])
                    #   IPSP delay backward
                    ax.annotate(text=f"",
                                xy=(0, prob[np.round(len(psth_bins)/2).astype(int)+delay_idx_bwd]),
                                xytext=(entry['delay_backward_IPSP']*(-1), prob[np.round(len(psth_bins)/2).astype(int)+delay_idx_bwd]),
                                arrowprops=dict(arrowstyle="<->", color=[0.1,0,0.7]))
                    ax.text(x=entry['delay_backward_IPSP']*(-1/2), y=prob[np.round(len(psth_bins)/2).astype(int)+delay_idx_bwd]-0.0002,
                            s=f"IPSP bwd timing ({direction_reversed})\n= {entry['delay_backward_IPSP']*1000:.1f}ms\n(mix delay & duration)",
                            horizontalalignment='right', verticalalignment='top', color=[0.1,0,0.7])
                    #   Forward direction
                    ax.text(x=0.1, y = max_ylim*0.99,
                        s=f"Estimated {txt_fwd_inhb} inhibition =\n{entry['forward']['raw_area']*100:.3g}% (raw)\n{entry['forward']['corrected_area']*100:.3g}% (noise-bias removal)\nP = {entry['forward']['p_val']:.4f}",
                        horizontalalignment='left', verticalalignment='top', color=cross_hist_colors[f'txt_{direction_for_txt}'])
                    #   Backward direction
                    ax.text(x=-0.1, y = max_ylim*0.99,
                        s=f"Estimated {txt_bwd_inhb} inhibition =\n{entry['backward']['raw_area']*100:.3g}% (raw)\n{entry['backward']['corrected_area']*100:.3g}% (noise-bias removal)\nP = {entry['backward']['p_val']:.4f}",
                        horizontalalignment='right', verticalalignment='top', color=cross_hist_colors[f'txt_{direction_reversed}'])
                    # Asymmetry
                    asym_str_suffix = ""
                    if direction_for_txt=="inhibited":
                        if (asym_raw_diff < 0) and (asym_corrected_diff < 0) and (asym_raw_ratio > 1.2) and (asym_corrected_ratio > 1.2):
                            asym_str_suffix = "Inhibition received > inhibition delivered"
                        elif (asym_raw_diff > 0) and (asym_corrected_diff > 0) and (asym_raw_ratio < (1/1.2)) and (asym_corrected_ratio < (1/1.2)):
                            asym_str_suffix = "Inhibition received < inhibition delivered"
                        else:
                            asym_str_suffix = "No clear diff between inhibition received & delivered"
                    elif direction_for_txt=="inhibiting":
                        if (asym_raw_diff < 0) and (asym_corrected_diff < 0) and (asym_raw_ratio > 1.2) and (asym_corrected_ratio > 1.2):
                            asym_str_suffix = "Inhibition received < inhibition delivered"
                        elif (asym_raw_diff > 0) and (asym_corrected_diff > 0) and (asym_raw_ratio < (1/1.2)) and (asym_corrected_ratio < (1/1.2)):
                            asym_str_suffix = "Inhibition received > inhibition delivered"
                        else:
                            asym_str_suffix = "No clear diff between inhibition received & delivered"
                    ax.text(x=0, y=max_ylim*0.99,
                            s=f"Forward ({direction_for_txt}) to backward ({direction_reversed}) ratio\n= {asym_raw_ratio:.2f} (raw), {asym_corrected_ratio:.2f} (noise bias-corrected)\nForward ({direction_for_txt}) to backward ({direction_reversed}) difference\n= {asym_raw_diff*100:.3g}% (raw), {asym_corrected_diff*100:.3g}% (noise bias-corrected)\n{asym_str_suffix}",
                            horizontalalignment='center', verticalalignment='top', color='k')
                    # Labels
                    ax.set_title(title_txt)
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel(f"Spiking probability\nDuring one sample = 1/{fsamp} s")
                    ax.set_ylim(0,max_ylim)
                    # display legend in a separate subplot (at the end)
                    if direction=='combined':
                        legend_handles, legend_labels = ax.get_legend_handles_labels()
                        ax = fig.add_subplot(fig_grid[1,3])
                        ax.legend(handles=legend_handles, labels=legend_labels,
                                  fontsize=10, loc='center')
                        ax.set_axis_off()
                    # get the bounding boxes of the axes in figure coordinates
                    axes_bounding_boxes.append(ax.get_position())
            # store everything in results (variable to be returned at the end of the function)
            results[base][mu_idx] = metrics
            # Perform last operations o the generated figure and save it (if requested)
            if generate_figure:
                # # Create arrows
                # # pick a point in the middle bottom edge of each top‐axes
                # start0 = ((axes_bounding_boxes[0].x0 + axes_bounding_boxes[0].x1)/2, axes_bounding_boxes[0].y0)  # halfway across top‐left plot, bottom edge
                # start1 = ((axes_bounding_boxes[1].x0 + axes_bounding_boxes[1].x1)/2, axes_bounding_boxes[1].y0)  # halfway across top‐right
                # end   = ((axes_bounding_boxes[2].x0 + axes_bounding_boxes[2].x1)/2, axes_bounding_boxes[2].y1)    # halfway across bottom plot, top edge
                # # now annotate on *any* one of your axes, e.g. axes[0,0]:
                # for start in (start0, start1):
                #     ax.annotate("", xy=end, xycoords="figure fraction",
                #         xytext=start, textcoords="figure fraction",
                #         arrowprops=dict( arrowstyle="->", lw=2, color="k"))
                # Suptitle and save
                plt.suptitle(f"{base}\nMU{mu_idx}")
                # check if savepath alrady has "cross_hist" in its name, if not, create a new subfolder
                if "cross_hist" not in savepath:
                    outdir = os.path.join(savepath, "Cross_histogram_metrics")
                    os.makedirs(outdir, exist_ok=True)
                else:
                    outdir = savepath 
                safe_base = base.replace('<->','_').replace('->','_')
                fname     = f"MU_{mu_idx}_{safe_base}.png"
                fig.savefig(os.path.join(outdir, fname))
                plt.close(fig)
            # End of "for each MU"

    return results

def get_coherence(
    spike_trains_MN,
    corresponding_MN_idx,
    fsamp,
    MU_corresponding_pool_list,
    list_of_MUs_by_pool,
    nb_of_samples,
    coherence_calc_max_iteration_nb_per_group_size,
    coh_window_length=1,
    coh_windows_overlap=0.5,
    max_freq=100,
    upsampling_frequency_resolution=1,
    mean_DR=None,
    ref_cst_group_size=5,
    common_input=None, # if an array corresponding to the received common input, will calculate coherence relative to common input
    generate_figure=False,
    savepath=None,
    logger=None
):
    logger = logger or logging.getLogger(__name__)
    # --- prepare outputs ---
    coherence_results                   = {"frequencies": None}
    coherence_delta_results             = {"frequencies": None}
    coherence_with_input_results        = {"frequencies": None, "pool0-input0": {}}
    coherence_with_input_delta_results  = {"frequencies": None, "pool0-input0": {}} 

    # filter your pools
    list_of_MUs_by_pool = {
        pool: np.intersect1d(mus, corresponding_MN_idx, assume_unique=True)
        for pool, mus in list_of_MUs_by_pool.items()
    }
    muscles     = list(list_of_MUs_by_pool.keys())
    muscle_pairs = list(combinations_with_replacement(muscles, 2))

    # figure setup
    if generate_figure:
        n_pairs = len(muscle_pairs)
        fig, axes = plt.subplots(
            n_pairs, 2,
            figsize=(16, 4*n_pairs),
            sharex='col',
            gridspec_kw={'width_ratios':[1,1]}
        )
        if n_pairs==1:
            axes = axes[np.newaxis, :]
        cmap_cst_size = cmr.cosmic
        pool_color   = {m: f"C{i}" for i,m in enumerate(muscles)}
        if common_input is not None:
            smoothed_CSTs_example = []

    # --- loop over muscle pairs ---
    for idx, (pool0, pool1) in enumerate(muscle_pairs):
        key = f"{pool0}-{pool1}"
        coherence_results[key]       = {}
        coherence_delta_results[key] = {}

        # figure axes
        if generate_figure:
            ax_coh  = axes[idx, 0]
            ax_dcoh = axes[idx, 1]

        # how big can your CST get?
        n0 = len(list_of_MUs_by_pool[pool0]) # n0 = nb of MUs in pool0
        n1 = len(list_of_MUs_by_pool[pool1]) # n1 = nb of MUs in pool1
        if pool0 == pool1:
            max_size = int(np.floor(max(n0,n1)/2))
        else:
            max_size = int(np.floor((n0+n1)/2))

        freq_bins = None

        # 1) compute mean‐coherence for each CST size
        for size_minus1 in range(n0): #n0 is the nb of MUs in the first pool #range(max_size):
            group_size = size_minus1 + 1
            n_iters    = int(np.round(coherence_calc_max_iteration_nb_per_group_size / group_size))
            all_iters  = []
            all_iters_with_common_input = []

            for iter in range(n_iters):
                if size_minus1 < max_size: # Half of the size of the MN pool
                    # pool0
                    mus0 = list_of_MUs_by_pool[pool0].tolist()
                    random.shuffle(mus0)
                    picks0 = mus0[:group_size]

                    # pool1
                    if pool0 != pool1:
                        mus1 = list_of_MUs_by_pool[pool1].tolist()
                        random.shuffle(mus1)
                        picks1 = mus1[:group_size]
                    else:
                        # same pool: take the *same* shuffle, but the *last* group_size
                        picks1 = mus0[-group_size:]

                    # build the two CSTs
                    def build_cst(picks):
                        mat = np.zeros((group_size, nb_of_samples), dtype=int)
                        for row_i, mn_i in enumerate(picks):
                            for t in spike_trains_MN[f"MN_{mn_i}"]:
                                idx_t = int(t * fsamp)
                                if idx_t < nb_of_samples:
                                    mat[row_i, idx_t] = 1
                        return mat.sum(axis=0)

                    cst0 = build_cst(picks0)
                    cst1 = build_cst(picks1)

                    # compute coherence
                    fb, coh = compute_coherence_between_signals(
                        cst0, cst1,
                        fsamp, coh_window_length, coh_windows_overlap,
                        max_freq, upsampling_frequency_resolution
                    )
                    if freq_bins is None:
                        freq_bins = fb

                    all_iters.append(coh.real)
                # end of 'if size_minus1 < max_size'

                # If desired (if common input is provided), compute coherence between cst and common input
                
                if common_input is not None: # only consider the first pool, and the first common input
                    # pool0
                    mus0 = list_of_MUs_by_pool[pool0].tolist()
                    random.shuffle(mus0)
                    picks0 = mus0[:group_size]
                    cst0 = build_cst(picks0)
                    if generate_figure and iter==0: # keep only the first iter cst as example
                        smoothed_CSTs_example.append(cst0)
                        # logger.info(f"Length of CST: {len(cst0_smooth)} samples\nLength of common input: {len(common_input[0,fsamp:-fsamp])} samples")
                    # compute coherence
                    fb, coh = compute_coherence_between_signals(
                        cst0, common_input[0,fsamp:-fsamp],
                        fsamp, coh_window_length, coh_windows_overlap,
                        max_freq, upsampling_frequency_resolution
                    )
                    all_iters_with_common_input.append(coh.real)
            # end of 'for _ in range(n_iters)'

            if size_minus1 < max_size:
                # average over all iters for this group size
                mean_coh = np.nanmean(np.stack(all_iters, axis=0), axis=0)
                coherence_results[key][group_size] = mean_coh
            if common_input is not None:
                # average over all iters for this group size
                mean_coh = np.nanmean(np.stack(all_iters_with_common_input, axis=0), axis=0)
                coherence_with_input_results["pool0-input0"][group_size] = mean_coh

        # store frequencies once
        coherence_results["frequencies"] = freq_bins
        coherence_with_input_results["frequencies"] = freq_bins

        # 2) compute Δ‐coherence from those *mean* curves
        sizes = sorted(coherence_results[key].keys())
        C_stack = np.stack([coherence_results[key][s] for s in sizes], axis=0)

        if C_stack.shape[0] > 1:
            dC = C_stack[1:] - C_stack[:-1]
            mean_dC    = np.nanmean(dC,   axis=0)
            median_dC  = np.nanmedian(dC, axis=0)
            max_min_dC = np.nanmax(dC,    axis=0) + np.nanmin(dC, axis=0)
        else:
            mean_dC    = np.zeros_like(freq_bins)
            median_dC  = np.zeros_like(freq_bins)
            max_min_dC = np.zeros_like(freq_bins)

        coherence_delta_results[key] = {
            "Mean":    mean_dC,
            "Median":  median_dC,
            "Max+Min": max_min_dC
        }
        coherence_delta_results["frequencies"] = freq_bins

        # 2.1) do the same, but this time for the coherence between input and CST
        if common_input is not None:
            sizes_common_input_cst_coh = sorted(coherence_with_input_results["pool0-input0"].keys())
            C_stack = np.stack([coherence_with_input_results["pool0-input0"][s] for s in sizes_common_input_cst_coh], axis=0)

            if C_stack.shape[0] > 1:
                dC = C_stack[1:] - C_stack[:-1]
                mean_dC    = np.nanmean(dC,   axis=0)
                median_dC  = np.nanmedian(dC, axis=0)
                max_min_dC = np.nanmax(dC,    axis=0) + np.nanmin(dC, axis=0)
            else:
                mean_dC    = np.zeros_like(freq_bins)
                median_dC  = np.zeros_like(freq_bins)
                max_min_dC = np.zeros_like(freq_bins)

            coherence_with_input_delta_results["pool0-input0"] = {
                "Mean":    mean_dC,
                "Median":  median_dC,
                "Max+Min": max_min_dC
            }
            coherence_with_input_delta_results["frequencies"] = freq_bins

        # --- optional plotting ---
        if generate_figure:
            # left: coherence curves
            for s in sizes:
                alpha = 0.75 if s == ref_cst_group_size else 0.25
                color = 'red' if s == ref_cst_group_size else cmap_cst_size((s-1)/(max_size-1))
                ax_coh.plot(freq_bins, coherence_results[key][s],
                            color=color, alpha=alpha, lw=2,
                            label=(f"{s} MNs" if s==ref_cst_group_size else None))
            ax_coh.set(title=f"{key}  CST coherence\nCSTs with up to {max_size} MNs",
                       xlabel="Frequency (Hz)", ylabel="Coherence",
                       xlim=(0,max_freq), ylim=(0,1))
            if mean_DR is not None:
                dr0 = np.nanmean([mean_DR[f"MN_{i}"] for i in list_of_MUs_by_pool[pool0]])
                ax_coh.axvline(dr0, color=pool_color[pool0], ls='--', label=f"Mean DR {pool0}")
                if pool0!=pool1:
                    dr1 = np.nanmean([mean_DR[f"MN_{i}"] for i in list_of_MUs_by_pool[pool1]])
                    ax_coh.axvline(dr1, color=pool_color[pool1], ls='--', label=f"Mean DR {pool1}")
                ax_coh.legend(loc="upper right")

            # right: Δ‐coherence curves
            for label, color in (("Mean","#6F00FFE1"), ("Median","#0077FFFF"), ("Max+Min","#8886ECB8")):
                y = coherence_delta_results[key][label]
                ax_dcoh.plot(freq_bins, y, color=color, lw=2, label=label, alpha=0.7)
            ax_dcoh.axhline(0, color='k', linestyle=':')
            ax_dcoh.set(title=f"{key}  Δ Coherence per extra MN",
                        xlabel="Frequency (Hz)", ylabel="ΔCoherence",
                        xlim=(0,max_freq))
            ax_dcoh.legend(loc="upper right")

    # end muscle-pairs

    if generate_figure:
        plt.tight_layout()
        if savepath:
            plt.savefig(os.path.join(savepath, "ANALYSIS_coherence.png"),
                        bbox_inches='tight')
        plt.show()

    ### Plotting (optional) for the coherence between first pool CST and first pool common input
    if generate_figure and (common_input is not None):
        # Diplay CSTs and common input
        fig, axes = plt.subplots(
            2, 1,
            figsize=(20, 10),
            gridspec_kw={'height_ratios':[1,1]}
        )
        time = np.arange(len(smoothed_CSTs_example[0])) / fsamp
        ax_cst  = axes[0]
        ax_CI = axes[1]
        for cst_i in range(len(smoothed_CSTs_example)):
            alpha = ((cst_i+1+len(smoothed_CSTs_example))/(len(smoothed_CSTs_example)*2))*0.5
            ax_cst.plot(time, smoothed_CSTs_example[cst_i], color='blue', alpha=alpha, lw=0.5)
        ax_cst.set_title(f"CST examples with 1 up to {len(smoothed_CSTs_example)} MNs")
        ax_cst.set_xlabel("Time (s)")
        ax_cst.set_ylabel("CST amplitude (spikes per sample)")

        ax_CI.plot(time, common_input[0,fsamp:-fsamp], color="#B700FF", lw=0.5)
        ax_CI.set_title("Common input")
        ax_CI.set_xlabel("Time (s)")
        ax_CI.set_ylabel("Common input amplitude")

        plt.tight_layout()
        if savepath:
            plt.savefig(os.path.join(savepath, "ANALYSIS_CSTs_vs_common_input.png"),
                        bbox_inches='tight')
        plt.show()

        # Coherence between CST and common input
        fig, axes = plt.subplots(
            1, 2,
            figsize=(16, 4),
            sharex='col',
            gridspec_kw={'width_ratios':[1,1]}
        )
        ax_coh  = axes[0]
        ax_dcoh = axes[1]
        # left: coherence curves
        for s in sizes_common_input_cst_coh:
            alpha = 0.75 if s == ref_cst_group_size else 0.25
            color = 'red' if s == ref_cst_group_size else cmap_cst_size((s-1)/(n0-1))
            ax_coh.plot(freq_bins, coherence_with_input_results["pool0-input0"][s],
                        color=color, alpha=alpha, lw=2,
                        label=(f"{s} MNs" if s==ref_cst_group_size else None))
        ax_coh.set(title=f"Pool0-Input0  Coherence between CSTs and common input\nCSTs with up to {n0} MNs",
                    xlabel="Frequency (Hz)", ylabel="Coherence",
                    xlim=(0,max_freq), ylim=(0,1))
        # right: Δ‐coherence curves
        for label, color in (("Mean","#6F00FFE1"), ("Median","#0077FFFF"), ("Max+Min","#8886ECB8")):
            y = coherence_with_input_delta_results["pool0-input0"][label]
            ax_dcoh.plot(freq_bins, y, color=color, lw=2, label=label, alpha=0.7)
        ax_dcoh.axhline(0, color='k', linestyle=':')
        ax_dcoh.set(title=f"Pool0-Input0  Δ Coherence per extra MN",
                    xlabel="Frequency (Hz)", ylabel="ΔCoherence",
                    xlim=(0,max_freq))
        ax_dcoh.legend(loc="upper right")
        plt.tight_layout()
        if savepath:
            plt.savefig(os.path.join(savepath, "ANALYSIS_coherence_with_common_input.png"),
                        bbox_inches='tight')
        plt.show()

    return coherence_results, coherence_delta_results, coherence_with_input_results, coherence_with_input_delta_results

###################
# FULL ANALYSIS FUNCTION WITH CLASS TO DECIDE WHICH ANALYSES TO RUN
###################
def analyze_data(file, analysis_params):
    """
    Main function to analyze either experimental or simulated data.
    It decides which analysis to run based on the type of data and the parameters provided.
    """
    # Initialize
    _ensure_logging()  # ensure logging is configured for _this_ process
    logger = logging.getLogger(__name__)
    # Find file
    if not os.path.exists(file):
        logger.error(f"File {file} does not exist.")
        return   
    if analysis_params.is_simulation:
        analyze_simulated_data(file, analysis_params)
    else:
        analyze_experimental_data(file, analysis_params)

# Experimental data analysis (less things to analyze, because a lot of it is done in another script)
def analyze_experimental_data(file, analyses_params):
    # Initialize
    _ensure_logging() # ensure logging is configured for _this_ process
    logger = logging.getLogger(__name__)
    # Load and start analyzes
    analysis_output = {} # initializes the output dictionary
    with h5py.File(file,'r') as f:
        # LOAD FILE AND GET NECESSARY VALUES ###########################################################
        sim_foldername = os.path.basename(os.path.dirname(file))
        logger.info(f"Opened file ({os.path.basename(file)}) in {sim_foldername}...")
        all_spikes = h5_to_dict(f['spike_trains'])
        # all_spikes is { 'MN': { 'MN_0': array(...), ... },
        mn_spikes = all_spikes['MN']
        # Convert them to second (right now they are is sample times)
        for mn_id, spike_times in mn_spikes.items():
            mn_spikes[mn_id] = spike_times / 2048
        idx_kept = np.arange(len(mn_spikes)) # indices of the kept motor neurons (all of them in this case))
        # Get indices and pools of the motor neurons
        mn_and_pool_group = h5_to_dict(f["motoneurons_and_pools_indices"])
        nb_pools = len(mn_and_pool_group['idx_of_MN_by_pool'])
        pool_by_MN_list = mn_and_pool_group['pool_list_by_MN'][()] # this is an (N,)-shaped array of bytes
        pool_by_MN_list = [s.decode('utf-8') for s in pool_by_MN_list] # decode to Python str and pack into a list
        MN_idx_by_pool = {}
        for poolname in mn_and_pool_group['idx_of_MN_by_pool'].keys():
            MN_idx_by_pool[poolname] = mn_and_pool_group['idx_of_MN_by_pool'][poolname][()]
            # Get cross histograms if specified #################################################
        if analyses_params.get_cross_histogram_measures:
            logger.info(f"  ({os.path.basename(file)}) Computing cross histograms metrics... (this takes a while)")
            # if analyses_params.cross_histogram_output_figures: # If figures are requested, create the output directory
            crosshist_figs_outdir = os.path.join(os.path.dirname(file), f"{os.path.basename(file)[:-3]}_cross_hist_figures")
            os.makedirs(crosshist_figs_outdir, exist_ok=True)
            cross_histogram_analysis_results = get_cross_histogram_measures(
                # Necessary values
                spike_trains_MN=mn_spikes,
                corresponding_MN_idx=idx_kept,
                fsamp=2048,
                MU_corresponding_pool_list=pool_by_MN_list,
                list_of_MUs_by_pool=MN_idx_by_pool,
                # Analysis params
                histogram_kind=analyses_params.cross_histogram_measures_histogram_kind,
                lowpass_filter_prob_dist=analyses_params.cross_histogram_measures_lowpass_filter_prob_dist,
                minimum_spike_nb = analyses_params.cross_histogram_measures_min_spikes,
                minimum_r2 = analyses_params.cross_histogram_measures_min_r2,
                minimum_plateau_duration = analyses_params.cross_histogram_measures_min_plateau,
                null_distrib_nb_iter = analyses_params.cross_histogram_measures_null_distrib_nb_iter,
                ignore_homonymous_pool = analyses_params.cross_histogram_ignore_homonymous_pool, ignore_heteronymous_pool = analyses_params.cross_histogram_ignore_heteronymous_pool,
                # Figure/output params
                generate_figure=analyses_params.cross_histogram_output_figures,
                save_histograms=analyses_params.cross_histogram_save_cross_hists,
                savepath=crosshist_figs_outdir)
            analysis_output['Cross_histograms'] = cross_histogram_analysis_results
            logger.info(f"  ({os.path.basename(file)}) ...Cross histograms metrics calculated")

        # Save everything in a single .pkl format for easy reading back in Python.
        analysis_output_savepath = os.path.join(os.path.dirname(file), f'{os.path.basename(file)}_cross_histogram_analysis_output.pkl')
        with open(analysis_output_savepath, 'wb') as f:
            pickle.dump(analysis_output, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"ANALYSIS RESULTS OF {os.path.basename(file)} HAVE SUCCESFULLY BEEN SAVED")

# Simulated data analysis
def analyze_simulated_data(file, analyses_params):
    # Initialize
    _ensure_logging() # ensure logging is configured for _this_ process
    logger = logging.getLogger(__name__)
    # Load and start analyzes
    analysis_output = {} # initializes the output dictionary
    analysis_output['Analysis_parameters'] = asdict(analyses_params)
    with h5py.File(file,'r') as f:
        # LOAD FILE AND GET NECESSARY VALUES ###########################################################
        sim_foldername = os.path.basename(os.path.dirname(file))
        logger.info(f"Opened file ({os.path.basename(file)}) in {sim_foldername}...")
        all_spikes = h5_to_dict(f['spike_trains'])
        # all_spikes is { 'MN': { 'MN_0': array(...), ... },
        #                'RC': { 'RC_0': array(...), ... } }
        mn_spikes = all_spikes['MN']
        rc_spikes = all_spikes['RC']
        common_input_from_sim = None
        if analyses_params.coherence_between_CST_and_common_input:
            common_input_from_sim = f['input/common_input'][:] # or [0,:] to grab only the first pool
            logger.info(f"Grabbed common input array of shape {common_input_from_sim.shape}")
        # Sort the spike dictionaries by index (to keep the order by size for the motor neurons)
        def sort_by_index(d):
            return dict(sorted(d.items(), key=lambda kv: int(kv[0].split('_', 1)[1])))
        mn_spikes = sort_by_index(mn_spikes)
        rc_spikes = sort_by_index(rc_spikes)
        # Get indices and pools of the motor neurons
        mn_and_pool_group = h5_to_dict(f["motoneurons_and_pools_indices"])
        nb_pools = f['simulation_parameters'].attrs["nb_pools"]
        pool_by_MN_list = mn_and_pool_group['pool_list_by_MN'][()] # this is an (N,)-shaped array of bytes
        pool_by_MN_list = [s.decode('utf-8') for s in pool_by_MN_list] # decode to Python str and pack into a list
        MN_idx_by_pool = {}
        for i in range(nb_pools):
            MN_idx_by_pool[f'pool_{i}'] = mn_and_pool_group['idx_of_MN_by_pool'][f'pool_{i}'][()]
        # Get duration of the simulation
        duration = f['simulation_parameters'].attrs["duration"]
        # Remove spikes in windows to ignore
        window_to_ignore_duration = f['simulation_parameters'].attrs["edges_ignore_duration"]
        for mn_id, spike_times in mn_spikes.items():
            mn_spikes[mn_id] = remove_spikes_in_windows_to_ignore(spike_times, duration, window_to_ignore_duration)
        for rc_id, spike_times in rc_spikes.items():
            rc_spikes[rc_id] = remove_spikes_in_windows_to_ignore(spike_times, duration, window_to_ignore_duration)
        
        # Remove discontinuous MNs if specified #################################################
        if analyses_params.remove_discontinuous_MNs:
            mn_spikes, idx_kept = remove_discontinuous_MNs(mn_spikes, sim_duration=duration,
                                                 max_ISI=analyses_params.ISI_above_which_MNs_are_considered_discontinuous)
            logger.info(f"...file in {sim_foldername} successfully loaded and ready to analyze.\n     Removing discontinuous MNs (max ISI > {analyses_params.ISI_above_which_MNs_are_considered_discontinuous}s): {len(mn_spikes)} MNs out of {f['simulation_parameters'].attrs['total_nb_motoneurons']} will be analyzed.")
        else:
            logger.info(f"...file in {sim_foldername} successfully loaded and ready to analyze.\n     Keeping all MNs - {len(mn_spikes)} MNs will be analyzed.")
            idx_kept = np.arange(f['simulation_parameters'].attrs["total_nb_motoneurons"])
        
        # Remove random subset of MNs if specified #############################################
        if analyses_params.select_random_subset_of_MUs_per_pool_for_analyses >= 1:
            idx_kept = []
            spike_times_subsampled = {}
            for pooli in range(nb_pools):
                nb_mn_in_pool_temp = len(MN_idx_by_pool[f"pool_{pooli}"])
                nb_mns_to_select = np.min([nb_mn_in_pool_temp, analyses_params.select_random_subset_of_MUs_per_pool_for_analyses])
                MNs_idx_subsampled_temp = np.random.choice(MN_idx_by_pool[f"pool_{pooli}"],
                                                                    nb_mns_to_select, replace=False)
                for mni in MNs_idx_subsampled_temp:
                    idx_kept.append(mni)
                    spike_times_subsampled[f"MN_{mni}"] = mn_spikes[f"MN_{mni}"]
            mn_spikes = spike_times_subsampled
            logger.info(f"      subsampling pools : {len(mn_spikes)} MNs will be analyzed.")

        # Make sure the list of MNs analyzed is available
        analysis_output['MN_idx_kept'] = idx_kept                                                          

        # Get mean firing rates if specified #################################################
        if analyses_params.get_firing_rates:
            firing_rates_MN, firing_rates_RC, isi_cov_MN, isi_cov_RC = get_firing_rate(mn_spikes, rc_spikes, 
                                            generate_figure=analyses_params.firing_rates_output_figures,
                                            savepath=os.path.dirname(file))
            analysis_output['Firing_rates'] = {}
            analysis_output['Firing_rates']['MN'] = firing_rates_MN
            analysis_output['Firing_rates']['RC'] = firing_rates_RC
            analysis_output['Firing_rates']['isi_cov_MN'] = isi_cov_MN
            analysis_output['Firing_rates']['isi_cov_RC'] = isi_cov_RC
            logger.info(f"  ({sim_foldername}) Firing rates values (mean, std, min, max) calculated")
        
        # Get ground truth RI connectivity if specified #################################################
        if analyses_params.get_ground_truth_RI_connectivity:
            if not analyses_params.get_firing_rates:
                logger.warning("Ground truth RI connectivity cannot be calculated without firing rates. Skipping...")
                pass
            ground_truth_RI_connectivity = get_ground_truth_RI_connectivity(MN_idx_kept=idx_kept, total_nb_motoneurons=f['simulation_parameters'].attrs["total_nb_motoneurons"],
                                                                            mean_firing_rates=firing_rates_MN['mean'],
                                                                            MN_to_MN_connectivity_matrix=f['connectivity']['MN_to_MN'][:],
                                                                            RC_to_MN_IPSP=f['simulation_parameters']["Renshaw_to_MN_IPSP"][()],
                                                                            # ^ This syntax specifically recovers the value only (irrespective of the Brian2 unit)
                                                                            MN_idx_by_pool=MN_idx_by_pool,
                                                                            generate_figure=analyses_params.ground_truth_RI_connectivity_output_figures,
                                                                           savepath=os.path.dirname(file))
            analysis_output['Ground_truth_RI_connectivity'] = ground_truth_RI_connectivity
            logger.info(f"  ({sim_foldername}) Ground truth RI connectivity calculated")

        # Get graph theory connectivity measures if specified #################################################
        if analyses_params.get_graph_theory_connectivity_measures:
            graph_theory_measures = get_graph_theory_connectivity_measures(
                MN_to_MN_connectivity_matrix=f['connectivity']['MN_to_MN'][:],
                idx_kept=idx_kept,
                generate_figure=analyses_params.graph_theory_output_figures,
                savepath=os.path.dirname(file))
            analysis_output['Graph_theory_connectivity_measures'] = graph_theory_measures
            logger.info(f"  ({sim_foldername}) Graph theory connectivity measures calculated")
        
        # Get cross histograms if specified #################################################
        if analyses_params.get_cross_histogram_measures:
            logger.info(f"  ({sim_foldername}) Computing cross histograms metrics... (this takes a while)")
            cross_histogram_analysis_results = get_cross_histogram_measures(
                # Necessary values
                spike_trains_MN=mn_spikes,
                corresponding_MN_idx=idx_kept,
                fsamp=f['simulation_parameters'].attrs["fsamp"],
                MU_corresponding_pool_list=pool_by_MN_list,
                list_of_MUs_by_pool=MN_idx_by_pool,
                # Analysis params
                histogram_kind=analyses_params.cross_histogram_measures_histogram_kind,
                lowpass_filter_prob_dist=analyses_params.cross_histogram_measures_lowpass_filter_prob_dist,
                minimum_spike_nb = analyses_params.cross_histogram_measures_min_spikes,
                minimum_r2 = analyses_params.cross_histogram_measures_min_r2,
                minimum_plateau_duration = analyses_params.cross_histogram_measures_min_plateau,
                null_distrib_nb_iter = analyses_params.cross_histogram_measures_null_distrib_nb_iter,
                ignore_homonymous_pool = analyses_params.cross_histogram_ignore_homonymous_pool, ignore_heteronymous_pool = analyses_params.cross_histogram_ignore_heteronymous_pool,
                # Figure/output params
                generate_figure=analyses_params.cross_histogram_output_figures,
                save_histograms=analyses_params.cross_histogram_save_cross_hists,
                savepath=os.path.dirname(file))
            analysis_output['Cross_histograms'] = cross_histogram_analysis_results
            logger.info(f"  ({sim_foldername}) ...Cross histograms metrics calculated")

        # Get coherence if specified #################################################
        if analyses_params.get_coherence:
            logger.info(f"  ({sim_foldername}) Computing coherence... (this takes a while)")
            coherence_results, coherence_delta_results, coherence_with_input_results, coherence_with_input_delta_results = get_coherence(
                spike_trains_MN=mn_spikes,
                corresponding_MN_idx=idx_kept,
                fsamp=f['simulation_parameters'].attrs["fsamp"],
                MU_corresponding_pool_list=pool_by_MN_list,
                list_of_MUs_by_pool=MN_idx_by_pool,
                nb_of_samples=np.round(f['simulation_parameters'].attrs["duration"]*f['simulation_parameters'].attrs["fsamp"]).astype(int),
                # Analysis params
                coherence_calc_max_iteration_nb_per_group_size=analyses_params.coherence_calc_max_iteration_nb_per_group_size,
                coh_window_length=analyses_params.coherence_window_length,
                coh_windows_overlap=analyses_params.coherence_windows_overlap,
                max_freq=analyses_params.coherence_max_freq,
                upsampling_frequency_resolution=analyses_params.coherence_upsampling_frequency_resolution,
                mean_DR=analysis_output['Firing_rates']['MN']['mean'] if analyses_params.get_firing_rates else None,
                # Coherence of CST with common input
                common_input=common_input_from_sim,
                # Figure params
                generate_figure=analyses_params.coherence_output_figures,
                savepath=os.path.dirname(file),
                logger=logger)
            analysis_output['Coherence'] = {}
            analysis_output['Coherence']['Coherence_total'] = coherence_results
            analysis_output['Coherence']['Coherence_delta'] = coherence_delta_results
            analysis_output['Coherence']['Coherence_with_input_total'] = coherence_with_input_results
            analysis_output['Coherence']['Coherence_with_input_delta'] = coherence_with_input_delta_results
            logger.info(f"  ({sim_foldername}) ...Coherence calculated!")

        # Save everything in a single .pkl format for easy reading back in Python.
        # Give a different name to each analysis output file if analysis_output_name is not None (the name is set in the script 'analyze_from_batch_simulations.ipynb')
        # However, the other analysis outputs (figures) will be overwritten if the same file is analyzed several times, with different analyzis parameters
        output_savename = 'analysis_output'
        if analyses_params.analysis_output_name is not None:
            output_savename = f"{output_savename}_{analyses_params.analysis_output_name}"
        analysis_output_savepath = os.path.join(os.path.dirname(file), f'{output_savename}.pkl')
        with open(analysis_output_savepath, 'wb') as f:
            pickle.dump(analysis_output, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"ANALYSIS RESULTS OF {sim_foldername} HAVE SUCCESFULLY BEEN SAVED")

    return analysis_output
