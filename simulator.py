import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import sys
import json
import pandas as pd
from brian2 import *
from scipy.signal import windows, butter, filtfilt, sosfiltfilt
from dataclasses import dataclass, field, asdict
from typing import Dict
from scipy.signal import welch
import time
import Cython
import h5py
from h5py import string_dtype
from typing import List
import traceback
from threading import local
import logging
import warnings

# # # IGNORE WARNINGS # # # 
# # # Ignore warnings from Brian2 which do not affect simulation behavior
# Silence the “TimedArray uses dt … not aligned” warning
logging.getLogger('brian2.input.timedarray').setLevel(logging.ERROR)
# Silence the “internal variable … exists in the namespace” warning
logging.getLogger('brian2.groups.group').setLevel(logging.ERROR)
# # #
# ignore only the “not compatible with tight_layout” UserWarning
warnings.filterwarnings(
    "ignore",
    message=r".*not compatible with tight_layout.*",
    category=UserWarning,
)


# # # DATA CLASS FOR SIMULATION's PARAMETERS
@dataclass
class SimulationParameters:
    # # # OUTPUT PARAMETERS
    output_folder_name: str = "simulation_batch_"
    make_unique_output_folder: bool = True
    output_plots: bool = False

    # # # RANDOM SEED
    pre_specify_random_seed: bool = True
    random_seed: int = field(init=False)

    # # # TIME PARAMETERS
    fsamp: int = 2048 # in samples per second
    # note that this is used to determine the time bins for input and output, but the actual integration time step is determined by 'defaultclock.dt' from Brian2 (0.1ms by default)
    duration: float = 5 # in seconds
    duration_with_ignored_window: float = field(init=False) # in seconds
    edges_ignore_duration: float = 1 # in seconds

    # # # NEURON NUMBERS AND SIZE
    nb_pools: int = 1
    full_pool_sim: bool = True
    # ^ If True, will simulate the chosen number of motor neurons as a population going from min soma size to max soma size, with a distribution determined by an exponent
    # ^ If True, min_soma_diameter, max_soma_diameter and size_distribution_exponent will be used
    # ^ If false, will simulate N motor neurons by sampling from a gaussian distribution specified by mean_soma_diameter and sd_prctl_soma_diameter
    # In both cases, the normalized size will depend on min_soma_diameter & max_soma_diameter
    nb_motoneurons_per_pool: int = 300
    nb_RCs_per_pool_pair: int = 60  # if 10 and 1 pool, only 10 RCs. If 2 pools, then there are 4 pairs of pools, so 40 RCs. Generally, total nb of RCs = nb_pools**2 * nb_RCs_per_pool_pair
    # size parameters if full_pool_sim == True ; if
    min_soma_diameter: float = 50 # in micrometers, for smallest motor neuron # Manuel et al. 2019 "Scaling of motor output, from Mouse to Humans"
    max_soma_diameter: float = 100 # in micrometers, for largest motor neuron # Manuel et al. 2019 "Scaling of motor output, from Mouse to Humans"
    size_distribution_exponent: float = 2 # between 0-1 => more large MN than small MN; 1 => uniform distribution (linear relationship between MN index and soma diameter); >1 => more small motoneurons than large MNs
    # size parameters if full_pool_sim == False
    mean_soma_diameter: float = 60 # in micrometers, for average motor neuron size of the simulated population
    sd_prct_soma_diameter: float = 5 # as a % of mean_soma_diameter. If mean_soma_diameter=50 and sd_prct_soma_diameter=10 for example, the distribution will be a gaussian with mean 50 and sd 5 micrometers
    # To calculate at initialization
    total_nb_motoneurons: int = field(init=False)
    total_nb_renshaw_cells: int = field(init=False)
    RC_pair_indices: Dict[tuple, np.ndarray] = field(init=False)

    # # # NEURON THRESHOLD AND EQUILIBRIUM POTENTIALS
    voltage_rest: Quantity = field(default_factory=lambda: 0 * mvolt) # arbitrary; 0 at rest
    voltage_thresh: Quantity = field(default_factory=lambda: 10 * mvolt) # arbitrary; 10 for generating a spike
    voltage_AHP: Quantity = field(default_factory=lambda: -13.3 * mvolt) # resting potential is typically -70mvolt, spike threshold is typically -55mvolt, and potassium reversal potential is typically -90mvolt.
    # To keep the relationships the same despite the arbitrary 0mvolt (resting voltage) and 10mvolt (spike generation threshold), voltage_AHP is set to -13.3 mvolt

    # # # COMMON INPUT PARAMETERS
    nb_of_common_inputs: int = 2 # Each comon input is distributed to the whole pool, but each has its own frequency content
    frequency_range_to_set_input_power: List[float] = field(
       default_factory=lambda: [0,5]) # in hz
    # Will set the frequency range over which the std scaling (both for excitatory common input and independent input) is done.
    # Thus, total power of the input signal will be the one selected by the user within the 'frequency_range_to_set_input_power',
    # but the total power increases with the frequency range of the input signal
    excitatory_input_baseline: List[float] = field( # in nA (nanoAmperes) # List of length at least equal to nb_pools
        default_factory=lambda: [25*1e3, 25*1e3])
    common_input_std: np.ndarray = field( # in nA (nanoAmperes) or in % of excitatory_input_baseline (check common_input_std_as_amp_or_prct) => it has to be a numpy array of size =  nb_pools (or more) x nb_of_inputs
        default_factory=lambda: np.array([
            [3.0*1e3, 0.0], # common_input_std for pool 0 [input 0, input 1]
            [3.0*1e3, 0.0] # common_input_std for pool 1 [input 0, input 1]
            ]))
    common_input_std_as_amp_or_prct: List[str] = field( # list of size nb_of_common_inputs => the unit will be the same regardless of the pool
        default_factory=lambda: ['nA', 'nA']) # 'percent' or 'nA'
    frequency_range_of_common_input: np.ndarray = field( # in Hz. This has to be a numpy array of size = nb_pools (or more) x nb_of_inputs x 2 (lower and upper bounds of frequency bandwidth)
        default_factory=lambda: np.array([
            [[0.0,5.0],[30.0,40.0]], # frequency range for pool 0 inputs [[low, high] for input 0], [[low, high] for input 1]
            [[0.0,5.0],[30.0,40.0]] # frequency range for pool 1 inputs [[low, high] for input 0], [[low, high] for input 1]
            ]))
    max_frequency_of_any_input: float = 80 # This can be necessary when setting the frequency content through another script
    common_input_characteristics: dict = field(init=False) # Those values are set AFTER initialization - they are just nicer to work with for SBI later
    
    # ──── FREQUENCY FILTERING "MASTER PARAMETERS" ────
    scale_filter_order_to_frequency: bool = True
    filter_order_scaling_coeff: float = 0.5 # Multiplying the cutoff frequency to determine filter order. Used only if scale_filter_order_to_frequency == True
    default_freq_filter_order: int = 5 # Used if scale_filter_order_to_frequency = False
    lowest_freq_filter_order: int = 5 # # Used if scale_filter_order_to_frequency = True
    max_freq_filter_order: int = 100
    # ──── INPUTS DISTRIBUTION AND CORRELATION ACROSS POOLS
    set_same_excitatory_input_for_all_pools: bool = False # If true, this will override all the generated excitatory inputs to be the same for all pools
    set_arbitrary_correlation_between_excitatory_inputs: bool = True
    between_pool_excitatory_input_correlation: float = 0.5 # > 0 and < 1 # used only if set_arbitrary_correlation_between_excitatory_inputs == True
    # negative correlations are accepted as inputs but the procedure to set arbitrary correlations doesn't work for negative correlations (it results in correlations around 0)
    # a rough method has been implemented to deal with this problem, but it can induce spurious reversal of the sign of correlations

    # # # INDEPENDENT (NOISY) INPUT PARAMETERS
    low_pass_filter_of_MN_independent_input: float = 50 # in Hz
    # Remember that the power will be scaled according to the frequency band set by frequency_range_to_set_input_power, so for instance,
    # if (MN_independent_input_std = common_input_std) and if max freq of independent input = 50 and max freq of common input = 5,
    # then the total power of the independent noise will be 10 times larger than the common input power (because the frequency range is 10 times larger)
    # So in order to get the independent input total power to be 3 times larger than the total common input power when
    # THE COMMON INPUT FREQUENCY RANGE IS 0-5Hz, and when THE INDEPENDENT INPUT FREQUENCY RANGE IS 0-50Hz,
    # we need to set the independent input std to be be sqrt(0.3)
    independent_input_absolute_or_ratio: str = 'ratio' # 'absolute' or 'ratio'
    independent_input_power: float = 3
    # If independent_input_absolute_or_ratio == 'ratio', then independent_input_power is the ratio of the independent input std to the common input std
    #       Assuming that the independent input frequency range is 10 times larger than the common input frequency range
    #       ^ This ratio is set relative to the FIRST common input only
    # If independent_input_absolute_or_ratio == 'absolute', then independent_input_power is the absolute value of the independent input std
    low_pass_filter_of_RC_independent_input: float = 50 # in Hz

    # # # MOTOR NEURONS ELECTROPHYSIOLOGICAL PROPERTIES CONSTANTS (Caillet et al 2022)
    # Resistance constants, used to generate the motor neuron resistance (in Ohms) according to their size
    resistance_constant: float = 9.6*(10**5)
    resistance_exponent: float = 2.4*(-1)
    # Rheobase constants, used to generate the motor neuron input current offset (in nA) according to their size
    rheobase_constant: float = 9.0*(10**-4)
    rheobase_exponent: float = 2.5
    rheobase_scaling: float = 6.0*1e2 # manually-tuned scaling
    # Capacitance constants, used to generate the motor neuron capacitance (in Farads) according to their size
    capacitance_constant: float = 1.2
    capacitance_exponent: float = 1
    # Afterhyperpolarization constants, used to generate the motor neuron AHP duration (in ms) according to their size + refractory period (in ms)
    AHP_duration_constant: float = 2.5 * (10**4)
    AHP_duration_exponent: float = 1.5 * (-1)
    # ^ these variables are used to create the variable 'motoneurons_AHP_conductance_decay_time_constant'
    # Caillet et al describe the relationship for the DURATION of the AHP, but for the equations I am using a time constant to control for the AHP conductance decay
    # I consider that the duration of the AHP correspond to the time it takes for the peak input at time 0 (x0) to decay to a tenth of its value x(t)<=X0/10
    # Thus, motoneurons_AHP_conductance_decay_time_constant = AHP_duration / ln(10)
    AHP_conductance_delta_after_spiking: Quantity = field(default_factory=lambda: 1.0 * msiemens) # Hyperpolarizing conductance change after a spike
    refractory_period_absolute: float = 5 # in ms
    # Axonal conduction velocity constants, used to generate the MN-to-fiber velocity (in m/s) according to their size
    # Then, the delay (ms) is calculated from the axonal conduction velocity, assuming a 0.5m axon length => so correspond to the conduction speed from MN to muscle fiber (speed in m/s, so multiply speed by 2)
    axonal_conduction_velocity_constant: float = 4.0*2
    axonal_conduction_velocity_exponent: float = 0.7

    # # # RENSHAW CELLS ELECTROPHYSIOLOGICAL PROPERTIES
    tau_Renshaw: Quantity = field(default_factory=lambda: 8*ms)
    refractory_period_RC: Quantity = field(default_factory=lambda: 5*ms)

    # # # MOTOR NEURONS <=> RENSHAW CELLS CONNECTIVITY
    binary_connectivity: bool = False # If true, MN to RC and RC to MN weights are either 0 or 1. 
    # This makes the distribution of disynaptic connections between MNs (especially the dsitribution's std) less controllable
    # 1) Full MN→MN target matrix (size: nb_pools × nb_pools)
    #    disynpatic_inhib_connections_desired_MN_MN[i,j] is the *desired* mean number of disynaptic synapses from MN pool i to MN pool j
    #    This is NOT a probability and can thus be > 1
    disynpatic_inhib_connections_desired_MN_MN: np.ndarray = field(
        default_factory=lambda: np.array([ # Increase the size of the array if nb_pools > 2
            [1.0, 0.0],   # pool 0→pool 0, pool 0→pool 1
            [0.0, 0.0],   # pool 1→pool 0, pool 1→pool 1
    ])
    )
    # 2) Split ratio of excitation vs inhibition:
    #    alpha = fraction of weight allocated to MN→RC vs RC→MN
    #    (so MN→RC uses p = (disynpatic_inhib_connections_desired_MN_MN)**alpha, RC→MN uses p = (disynpatic_inhib_connections_desired_MN_MN)**(1-alpha) )
    split_MN_RC_ratio: float = 0.5
    # 3) Distribution rule for sampling each bipartite graph:
    #    'binarize'      → Bernoulli(p), and max MN->MN inhibition of 1
    #    'gaussian'     → each pre: k∼N(mean*n_post, std*n_post)
    #    'size_gaussian'→ same but mean interpolates by MN size rank
    distribution_type: str   = 'gaussian'
    # 4) Distribution rule parameters:
    #    - binarize:       {}  # no extra params
    #    - gaussian:      {'std': 0.1}
    #    - size_gaussian: {'std': 0.1,
    #                      'ratio_large_small': 2.0}
    distribution_params: Dict[str, float] = field(default_factory=lambda: {
        'std': 0.2,     # net std of the MN->MN connectivity is sqrt(2*std**2 + std**4) / sqrt(nb_RCs_per_pool_pair). So with std=0.4 and nb_RCs_per_pool_pair=10, the std of MN->MN connectivity is ~0.20
        'std_is_prct': True, # if 'std_is_prct == True', then std will be interpreted as a % relative to disynpatic_inhib_connections_desired_MN_MN (with std=0.1 being 10% of mean connectivity for instance)
        # if 'std_is_prct == False', the value will be interpreted directly as the std of the number of disynaptic connections (weights)
        # if 'std=0.2" and if 'std_is_prct=True', the actual std will be 10% and not 20%
        'ratio_large_small': 3.0
    })
    disynaptic_inhib_received_arbitrary_adjustment: float = 0, # Correspond to the std of connection weights from all RCs to a given motorneuron that will be added. If 0, correspond to the specified connectivity, if 1 for, will be the specified connectivity on average +/- 1 (with a cutoff at 0)
    # 5) Enforce heteronymous‐pool non‐overlap?
    prevent_heteronymous_pool_overlap: bool = False

    # # # MOTOR NEURONS <=> RENSHAW CELLS POST-SYNAPTIC EFFECTS
    MN_to_Renshaw_EPSP: Quantity = field(default_factory=lambda: 6.7*mvolt) # 6.7*mvolt # when > 10*mvolt, ensures that 1 MN spike = 1 RC spike # increase in V in renshaw cell when receiving spike from MN - From Moore et al 2015 = MN-RC pair recordings, with 1 MN spike on average resulting in a probability of 0.3 of RC spike
    Renshaw_to_MN_IPSP: Quantity = field(default_factory=lambda: 3.0*1e3*nA)
    # ^ later in the code, this is turned into Coulomb (Total charge of an IPSP = current per second)
    # If scale_initial_IPSP_to_be_same_integral_regardless_of_synaptic_tau == True:
    #   Then the value (in nA) defined in Renshaw_to_MN_IPSP will be the total charge (in Coulomb)
    # If scale_initial_IPSP_to_be_same_integral_regardless_of_synaptic_tau == False:
    #   Then the value (in nA) defined in Renshaw_to_MN_IPSP will be the initial IPSP current
    scale_initial_IPSP_to_be_same_integral_regardless_of_synaptic_tau: bool = False # If True, Renshaw_to_MN_IPSP will be used not as initial current when an IPSP is received, but as a target total current received regardless of the chosen synaptic time constant
    synaptic_IPSP_membrane_or_user_defined_time_constant: str = 'membrane' # 'user_defined' or 'membrane'
    synaptic_IPSP_decay_time_constant: Quantity = field(default_factory=lambda: 10*ms) # Used only if 'synaptic_IPSP_membrane_or_user_defined_time_constant' == 'user_defined'
    # Time constant of the synaptic IPSP decay in the MN # Check figures from Williams & Baker 2009 (simulation); between ~5 and ~10ms membrane tau for Uchiyama & Windhorst 2007 (simulation); ~15-30ms inhibition duration in Ozyurt et al 2019 (experimental data)
    # If 10ms for example, the IPSP will decay to ~37% of its peak value after 10ms and to ~14% after 20ms (0.14 is ~0.37²)
    # William & Baker 2009 J Neuroscience: RC's IPSP rise time of 5.5 ms and half-width of 18.5 ms
    MN_RC_synpatic_delay: Quantity = field(default_factory=lambda: 5*ms)
    RC_independent_input_std: float = 3.333 # in mvolt # if 3.33, on average, RC membrane potential will fluctuate at 1/3 of the spike threshold (10mvolt)

    # # # POST INIT FUNCTION, EXECUTING AFTER INITIALIZATION
    def __post_init__(self):
        # Setting variables that are calculated post-initialization
        if self.pre_specify_random_seed:
            self.random_seed = 42
        else:
            self.random_seed = np.random.randint(2**10)
        self.duration_with_ignored_window = self.duration + (2 * self.edges_ignore_duration)
        self.total_nb_motoneurons = self.nb_pools * self.nb_motoneurons_per_pool
        self.total_nb_renshaw_cells = (self.nb_pools**2)*self.nb_RCs_per_pool_pair
        self.RC_pair_indices = {(i, j): np.arange(
                (i*self.nb_pools + j)*self.nb_RCs_per_pool_pair,
                (i*self.nb_pools + j + 1)*self.nb_RCs_per_pool_pair)
            for i in range(self.nb_pools) for j in range(self.nb_pools)}
        #### PARAMETERS CHECK ####
        if self.nb_pools < 1:
            raise ValueError("nb_pools must be ≥1")
        if self.distribution_type not in ("binarize","gaussian","size_gaussian"):
            raise ValueError(f"Unknown distribution_type={self.distribution_type!r}")
        # — Check excitatory_input_baseline —
        if len(self.excitatory_input_baseline) < self.nb_pools:
            raise ValueError(
                f"excitatory_input_baseline must be a list of length >= nb_pools"
            )
        # — Check frequency_range_of_common_input —
        if not isinstance(self.frequency_range_of_common_input, np.ndarray):
            raise TypeError(
                f"frequency_range_of_common_input must be of type np.ndarray, not {type(self.frequency_range_of_common_input).__name__}"
            )
        array_shape_temp = np.shape(self.frequency_range_of_common_input)
        if (array_shape_temp[0] < self.nb_pools) and (array_shape_temp[1] < self.nb_of_common_inputs) and (array_shape_temp[2] < 2):
            raise ValueError(
                f"frequency_range_of_common_input must be np.ndarray of size [nb_pools (or more) x nb_of_inputs x 2], but got shape {np.shape(self.frequency_range_of_common_input)} instead"
            )
        # — Check limit of frequency_range_of_common_input —
        highs = self.frequency_range_of_common_input[..., 1] # grab a view of all the “high” edges
        M    = self.max_frequency_of_any_input
        # Clamp & warn
        if np.any(highs > M):
            warnings.warn(f"Clamping {np.sum(highs > M)} upper‐bounds down to {M} Hz.")
            # write back into the dataclass array
            self.frequency_range_of_common_input[..., 1] = np.where(
                highs > M,
                M,        # if above the max, set to M
                highs     # otherwise leave as-is
            )
        # — Check common_input_std —
        if not isinstance(self.common_input_std, np.ndarray):
            raise TypeError(
                f"common_input_std must be np.ndarray of size [nb_pools (or more) x nb_of_inputs], not {type(self.common_input_std).__name__}"
            )
        array_shape_temp = np.shape(self.common_input_std)
        if (array_shape_temp[0] < self.nb_pools) and (array_shape_temp[1] < self.nb_of_common_inputs):
            raise ValueError(
                f"common_input_std must be np.ndarray of size [nb_pools (or more) x nb_of_inputs], but got shape {np.shape(self.common_input_std)} instead"
            )
        # - Check independent_input_absolute_or_ratio -
        if self.independent_input_absolute_or_ratio not in ('absolute', 'ratio'):
            raise ValueError(
                f"independent_input_absolute_or_ratio must be 'absolute' or 'ratio', not {self.independent_input_absolute_or_ratio!r}"
            )
        # Set common input std to nA if defined as a % of baseline
        for pooli in range(self.nb_pools):
            for inputi in range(self.nb_of_common_inputs):
                if self.common_input_std_as_amp_or_prct[inputi] == 'percent':
                    self.common_input_std[pooli][inputi] = (self.common_input_std[pooli][inputi]*self.excitatory_input_baseline[pooli])/100 # common_input_std should be in %
                elif self.common_input_std_as_amp_or_prct[inputi] == 'nA':
                    self.common_input_std[pooli][inputi] = self.common_input_std[pooli][inputi] # no change
                else:
                    raise ValueError(f"Unknown common_input_std_as_amp_or_prct={self.common_input_std_as_amp_or_prct[inputi]!r}\n   Should be 'percent' or 'nA'")
        # Reframe some of the parameter values to make simulation-based inference easier
        self.common_input_characteristics = {"Frequency_middle_of_range": {},
                                             "Frequency_half_width_of_range": {}}
        for pooli in range(self.nb_pools):
            key_pool = f"pool_{pooli}"
            self.common_input_characteristics["Frequency_middle_of_range"][f"{key_pool}"] = {}
            self.common_input_characteristics["Frequency_half_width_of_range"][f"{key_pool}"] = {}
            for inputi in range(self.nb_of_common_inputs):
                key_input = f"input_{inputi}"
                self.common_input_characteristics["Frequency_middle_of_range"][f"{key_pool}"][f"{key_input}"] = np.mean(
                    self.frequency_range_of_common_input[pooli][inputi])
                self.common_input_characteristics["Frequency_half_width_of_range"][f"{key_pool}"][f"{key_input}"] = (
                    self.frequency_range_of_common_input[pooli][inputi][1] - self.frequency_range_of_common_input[pooli][inputi][0]) / 2
            # frequency_range_of_common_input = np.array([
            #     [ [low, high] (input 0), [low, high] (input 1) ], # pool 0
            #     [ [low, high] (input 0), [low, high] (input 1) ] # pool 1
            # ])
######################## END OF SimulationParameters CLASS

######################################
### FETCH THREAD-LOCAL STORAGE FOR "CURRENT" PARAMETERS
### So that they can be used in the helper functions without explicitely passing them as arguments
######################################
_params_context = local()
def _set_filter_params(params: SimulationParameters):
    """Call this at the top of run_simulation to make params visible to the filters."""
    _params_context.params = params
def _get_filter_params() -> SimulationParameters:
    try:
        return _params_context.params
    except AttributeError:
        raise RuntimeError("No filter params set; forgot to call _set_filter_params?")

######################################
### HELPER FUNCTIONS
######################################
def lerp(a, b, t):
    return a + t * (b - a)
# ───────────────────
# Filtering functions
def butter_lowpass(cutoff, fs, order):
    """
    Return an SOS filter for a lowpass Butterworth of the given order.
    """
    nyq   = 0.5 * fs
    Wn    = cutoff / nyq
    # design in SOS form
    sos   = butter(order, Wn,
                   btype='low',
                   analog=False,
                   output='sos')
    return sos
def butter_highpass(cutoff, fs, order):
    """
    Return an SOS filter for a highpass Butterworth of the given order.
    """
    nyq   = 0.5 * fs
    Wn    = cutoff / nyq
    sos   = butter(order, Wn,
                   btype='high',
                   analog=False,
                   output='sos')
    return sos
# public filter‐wrappers
def lowpass_filter(data, cutoff, fs):
    p = _get_filter_params()
    order = p.default_freq_filter_order
    if p.scale_filter_order_to_frequency:
        if cutoff > p.lowest_freq_filter_order:
            order = cutoff * p.filter_order_scaling_coeff
    if ~np.isfinite(order):
        order = p.default_freq_filter_order
    elif order > p.max_freq_filter_order:
        order = p.max_freq_filter_order
    elif order < p.lowest_freq_filter_order:
        order = p.lowest_freq_filter_order
    order = np.floor(order).astype(int)
    sos = butter_lowpass(cutoff, fs, order=order)
    return sosfiltfilt(sos, data)
def highpass_filter(data, cutoff, fs, order=None):
    p = _get_filter_params()
    order = p.default_freq_filter_order
    if p.scale_filter_order_to_frequency:
        if cutoff > p.lowest_freq_filter_order:
            order = cutoff * p.filter_order_scaling_coeff
    if ~np.isfinite(order):
        order = p.default_freq_filter_order
    elif order > p.max_freq_filter_order:
        order = p.max_freq_filter_order
    elif order < p.lowest_freq_filter_order:
        order = p.lowest_freq_filter_order
    order = np.floor(order).astype(int)
    sos = butter_highpass(cutoff, fs, order=order)
    return sosfiltfilt(sos, data)
# ───────────────────────
def filter_artifact_removal(fsamp, edges_ignore_duration):
    duration_to_remove = 1/edges_ignore_duration # in second
    Wind_s = duration_to_remove * 2
    artifact_removal_window = windows.hann(round(fsamp * Wind_s))
    artifact_removal_window = artifact_removal_window[:int(np.round(len(artifact_removal_window)/2))]
    nb_samples_artifact_removal_window = len(artifact_removal_window)
    return artifact_removal_window, nb_samples_artifact_removal_window
def scale_to_band_std(signal, fs, band, desired_std):
    """
    Scale `signal` so that its *power* (variance) in [fmin,fmax] ⟶ (desired_std)^2.
    Uses Welch's PSD + frequency‐bin masking.  Avoids filtfilt entirely.

    Parameters
    ----------
    signal : 1D array
    fs : float
        Sampling rate (Hz)
    band : (fmin, fmax)
    desired_std : float
        Target standard deviation within [fmin,fmax]
    nperseg : int
        segment length for Welch.  If signal_length < nperseg, welch auto‐truncates.

    Returns
    -------
    scaled_signal : ndarray, same shape as `signal`
    """
    fmin, fmax = band
    if fmin < 0 or fmax >= fs/2 or fmin >= fmax:
        raise ValueError("band must be [fmin, fmax] with 0≤fmin<fmax<fs/2")

    # 1) full‐PSD
    freqs, Pxx = welch(signal, fs=fs, window='hann', nperseg=fs, scaling='spectrum')

    # 2) mask to only [fmin, fmax]
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        raise ValueError(f"No PSD bins in [{fmin},{fmax}] Hz; choose a smaller nperseg or adjust band.")

    current_power = np.trapz(Pxx[mask], freqs[mask])    # = ∫_{fmin}^{fmax} PSD(f) df
    if current_power <= 0:
        raise ValueError(f"No power in the {fmin}–{fmax} Hz band to scale.")

    # 3) we want variance in [fmin,fmax] = (desired_std)^2
    desired_power = desired_std**2
    scale_factor = np.sqrt(desired_power / current_power)

    # 4) apply to entire time-series
    return signal * scale_factor
def Generate_filtered_gaussian_noise_input(fsamp, 
        duration_with_ignored_window, edges_ignore_duration, artifact_removal_window,
        low_pass_filter_cutoff, input_mean, scaling_std, high_pass_filter_cutoff=0):
    # Generate random input
    temp_input = np.random.normal(0, 1, int(duration_with_ignored_window * fsamp))
    # Apply artifact removal window to the end of the signal
    end_ignore_start = int(duration_with_ignored_window * fsamp) - int(np.round(edges_ignore_duration * fsamp))
    temp_input[end_ignore_start:] = temp_input[end_ignore_start:] * np.flip(artifact_removal_window)
    # Apply artifact removal window to the beginning of the signal
    beginning_ignore_end = int(np.round(edges_ignore_duration * fsamp))
    temp_input[:beginning_ignore_end] = temp_input[:beginning_ignore_end] * artifact_removal_window
    # Apply low-pass filter
    temp_input = lowpass_filter(temp_input, low_pass_filter_cutoff, fsamp)
    # Apply high-pass filter
    if high_pass_filter_cutoff >= 1:
        temp_input = highpass_filter(temp_input, high_pass_filter_cutoff, fsamp)
    # Normalize the signal
    temp_input = temp_input - np.mean(temp_input)
    temp_input = temp_input / np.std(temp_input)
    # Scale and add mean
    temp_input = temp_input * scaling_std
    temp_input = temp_input + input_mean
    
    return temp_input
def is_positive_definite(X):
    try:
        np.linalg.cholesky(X)
        return True
    except np.linalg.LinAlgError:
        return False
def nearest_positive_definite(A):
    B = (A + A.T) / 2
    U, s, Vt = np.linalg.svd(B)
    H = np.dot(Vt.T * s, Vt)
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if is_positive_definite(A3):
        return A3
    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not is_positive_definite(A3):
        A3 += I * spacing * k
        k += 1
    return A3
def adjust_correlation(data, R_desired, threshold_for_adjusting_correl=0.05):
    """
    Adjust correlation of 'data' so its correlation matrix approximates R_desired.
    Handles negative correlations by a post-processing sign-flip heuristic.
    """
    data = np.asarray(data, dtype=float)
    n_samples, n_vars = data.shape

    if R_desired.shape != (n_vars, n_vars):
        raise ValueError("R_desired must be an n_vars x n_vars matrix.")
    if not np.allclose(R_desired, R_desired.T):
        raise ValueError("R_desired must be symmetric.")
    if not np.all(np.diag(R_desired) == 1.0):
        raise ValueError("Diagonal elements of R_desired must be 1.")

    # Store the sign pattern of desired correlations
    sign_pattern = np.sign(R_desired)

    # Use absolute values to form a positive definite target
    R_abs = np.abs(R_desired)

    # Make sure R_abs is positive definite
    R_abs_pd = R_abs if is_positive_definite(R_abs) else nearest_positive_definite(R_abs)

    # Standardize data
    orig_means = np.mean(data, axis=0)
    orig_stds = np.std(data, axis=0, ddof=1)
    if np.any(orig_stds == 0):
        raise ValueError("One or more input variables are constant; cannot adjust correlation.")
    data_std = (data - orig_means) / orig_stds

    # Compute current correlation
    R_current = np.corrcoef(data_std, rowvar=False)
    # Make sure R_current is PD
    R_current_pd = R_current if is_positive_definite(R_current) else nearest_positive_definite(R_current)

    # Cholesky decompositions
    L_current = np.linalg.cholesky(R_current_pd)
    L_desired = np.linalg.cholesky(R_abs_pd)

    # Whiten data
    inv_L_current = np.linalg.inv(L_current)
    data_white = data_std @ inv_L_current

    # Apply desired correlation (absolute values)
    data_transformed_std = data_white @ L_desired

    # Rescale back
    data_transformed = data_transformed_std * orig_stds + orig_means

    # Now, adjust signs if needed
    # Check the resulting correlations
    R_final = np.corrcoef(data_transformed, rowvar=False)

    # Check the resulting correlation and impose correlation relative to the first (first column of data) input if necessary
    for inputi in range(len(data_transformed[0])):
        if inputi==0:
            continue # skipping the first input (correlation with itself)
        else:
            temp_correl = np.corrcoef(data_transformed[:,0],data_transformed[:,inputi])[0, 1]
            temp_desired_correl = R_desired[0, inputi]
            diff_actual_VS_desired = temp_correl - temp_desired_correl # negative if the correlation is too low, positive if the correlation is too high
            if (abs(diff_actual_VS_desired) > threshold_for_adjusting_correl) and (diff_actual_VS_desired < 0): # if the correlation is too low
                data_transformed[:, inputi] = data_transformed[:,inputi]*(1-abs(diff_actual_VS_desired)) + (data_transformed[:,0]*abs(diff_actual_VS_desired))
            if (abs(diff_actual_VS_desired) > threshold_for_adjusting_correl) and (diff_actual_VS_desired > 0): # if the correlation is too high
                data_transformed[:, inputi] = data_transformed[:,inputi]*(1-abs(diff_actual_VS_desired)) - (data_transformed[:,0]*abs(diff_actual_VS_desired))

    # If a desired correlation is negative but we got a positive one, flip one column
    # This is a heuristic. We try flipping the second column in the pair.
    min_R_final_for_flip = 0.2  # adjust this threshold as needed
    min_R_desired_for_flip = -0.2  # adjust this threshold as needed
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            if R_desired[i, j] < min_R_desired_for_flip:  # desired negative correlation exceeding a threshold
                if R_final[i, j] > min_R_final_for_flip:  # got a positive correlation exceeding a threshold instead
                    # Flip the sign of column j
                    data_transformed[:, j] = -data_transformed[:, j]
                    # Recalculate R_final after the flip
                    # R_final = np.corrcoef(data_transformed, rowvar=False)
    # Example usage:
    # A = np.random.randn(1000, 3)
    # R_desired = np.array([[1.0, -0.5, 0.3],
    #                       [-0.5, 1.0, -0.2],
    #                       [0.3, -0.2, 1.0]])
    # data_new = adjust_correlation(A, R_desired)
    # np.corrcoef(data_new, rowvar=False) should reflect the sign pattern of R_desired.
    return data_transformed
def sample_block(n_pre, n_post, p_block, dist, params, ranks=None, binary=False):
    

    # If we want real‐valued weights, create A as float.  If binary, keep it integer.
    if binary:
        A = np.zeros((n_pre, n_post), dtype=int)
    else:
        A = np.zeros((n_pre, n_post), dtype=float)
    if dist == "binarize":
        if binary:
            # pure 0/1 Bernoulli
            A[:] = (np.random.rand(n_pre, n_post) < p_block).astype(int)
        else:
            # real‐valued in [0, p_block)
            A[:] = np.random.rand(n_pre, n_post) * p_block

    elif dist == "gaussian":
        for u in range(n_pre):
            for v in range(n_post):
                μ = p_block
                if params['std_is_prct']:
                    σ = p_block * params.get("std", 0.0)
                else:
                    σ = params.get("std", 0.0)
                if binary:
                    # sample a probability p_samp ∼ TruncNormal(μ, σ), then Bernoulli
                    p_samp = np.clip(np.random.randn() * σ + μ, 0, 1)
                    A[u, v] = (np.random.rand() < p_samp).astype(int)
                else:
                    # sample a truncated normal weight; keep it as float
                    raw = np.random.randn() * σ + μ
                    A[u, v] = max(raw, 0.0)

    elif dist == "size_gaussian" and (ranks is not None):
        R = params.get("ratio_large_small", 1.0)
        for u in range(n_pre):
            r = ranks[u]
            # interpolate mean between μ_s and μ_l
            μ0 = p_block
            if params['std_is_prct']:
                σ = p_block * params.get("std", 0.0)
            else:
                σ = params.get("std", 0.0)
            μ_s = 2 * μ0 / (1 + R)
            μ_l = R * μ_s
            μ_ij = μ_s + r * (μ_l - μ_s)
            for v in range(n_post):
                if binary:
                    p_samp = np.clip(np.random.randn() * σ + μ_ij, 0, 1)
                    A[u, v] = (np.random.rand() < p_samp).astype(int)
                else:
                    raw = np.random.randn() * σ + μ_ij
                    A[u, v] = max(raw, 0.0)
    else:
        # fallback to Gaussian logic
        for u in range(n_pre):
            for v in range(n_post):
                μ = p_block
                if params['std_is_prct']:
                    σ = p_block * params.get("std", 0.0)
                else:
                    σ = params.get("std", 0.0)
                if binary:
                    p_samp = np.clip(np.random.randn() * σ + μ, 0, 1)
                    A[u, v] = (np.random.rand() < p_samp).astype(int)
                else:
                    raw = np.random.randn() * σ + μ
                    A[u, v] = max(raw, 0.0)
    return A
def plot_connectivity_matrix(
            mat, title,
            pool_labels_pre, pool_labels_post,
            n_pre_pool, n_post_pool,
            cmap='viridis',
            add_colorbar=False,
            savepath=None
        ):
            n_pre, n_post = mat.shape
            n_pre_blocks  = len(pool_labels_pre)
            n_post_blocks = len(pool_labels_post)

            # compute block means & stds (unchanged) …
            means = np.zeros((n_pre_blocks, n_post_blocks))
            stds  = np.zeros((n_pre_blocks, n_post_blocks))
            for i in range(n_pre_blocks):
                for j in range(n_post_blocks):
                    r = slice(i*n_pre_pool, (i+1)*n_pre_pool)
                    c = slice(j*n_post_pool, (j+1)*n_post_pool)
                    blk = mat[r,c]
                    means[i,j] = blk.mean()
                    stds[i,j]  = blk.std()

            fig, ax = plt.subplots(figsize=(10,10))
            im = ax.imshow(mat, cmap=cmap, aspect='equal')

            # grid lines at pool boundaries
            for y in np.arange(n_pre_pool, n_pre, n_pre_pool):
                ax.axhline(y-0.5, color='white', lw=1)
            for x in np.arange(n_post_pool, n_post, n_post_pool):
                ax.axvline(x-0.5, color='white', lw=1)

            # ### 1) show every cell index tick ###
            ax.set_xticks(np.arange(n_post))
            ax.set_xticklabels([str(i) for i in range(n_post)], rotation=90, fontsize=6)
            ax.set_yticks(np.arange(n_pre))
            ax.set_yticklabels([str(i) for i in range(n_pre)], fontsize=6)
            ax.invert_yaxis()

            # ### 2) overlay pool labels at group centers ###
            # x‐axis (pool‐pair labels)
            post_centers = [j*n_post_pool + n_post_pool/2 for j in range(n_post_blocks)]
            for j, lbl in enumerate(pool_labels_post):
                ax.text(
                    post_centers[j], -0.7,   # x at center, y just above top row
                    lbl,
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold',
                    rotation=90,
                    clip_on=False
                )

            # y‐axis (pre‐pool labels)
            pre_centers = [i*n_pre_pool + n_pre_pool/2 for i in range(n_pre_blocks)]
            for i, lbl in enumerate(pool_labels_pre):
                ax.text(
                    -0.7, pre_centers[i],   # x just before first column, y at center
                    lbl,
                    ha='right', va='center',
                    fontsize=10, fontweight='bold',
                    clip_on=False
                )

            # title + subtitle
            μ, σ = mat.mean(), mat.std()
            subtitle = "\n".join(f"{pool_labels_pre[i]}→{pool_labels_post[j]}: {means[i,j]:.2f}±{stds[i,j]:.2f}"
                                for i in range(n_pre_blocks)
                                for j in range(n_post_blocks))
            ax.set_title(f"{title}\nOverall μ={μ:.2f}, σ={σ:.2f}\n{subtitle}", pad=50)

            ax.set_xlabel(" ".join(pool_labels_post[0].split()[:1]).capitalize() + " cell index (receiving)")
            ax.set_ylabel(" ".join(pool_labels_pre[0].split()[:1]).capitalize() + " cell index (delivering)")

            if add_colorbar:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("# of disynaptic connections", rotation=90, labelpad=20)

            plt.tight_layout()
            if savepath is not None:
                plt.savefig(savepath)
            plt.show()
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
def make_unique_output_dir(parent_folder, prefix="simulation_output_"):
    i = 0
    while True:
        dirname = f"{parent_folder}//{prefix}{i}"
        try:
            # this call is atomic: if two processes hit the same i,
            # only one will succeed, the other will get FileExistsError
            os.mkdir(dirname)
            return dirname, i
        except FileExistsError:
            i += 1
def wrap_motoneuron_properties(motoneuron_soma_diameter,
                               motoneurons_resistance, motoneurons_input_weight,
                               motoneurons_capacitance, motoneurons_membrane_conductance,
                               motoneurons_membrane_time_constant,
                               motoneurons_AHP_duration, motoneurons_AHP_conductance_decay_time_constant,
                               motoneurons_refractory_periods, motoneurons_rheobases,
                               motoneurons_spike_transmission_delays):
    return {
        'soma_diameter': motoneuron_soma_diameter,
        'resistance': motoneurons_resistance,
        'input_weight': motoneurons_input_weight,
        'capacitance': motoneurons_capacitance,
        'membrane_conductance': motoneurons_membrane_conductance,
        'membrane_time_constant': motoneurons_membrane_time_constant,
        'AHP_duration': motoneurons_AHP_duration,
        'AHP_conductance_decay_time_constant': motoneurons_AHP_conductance_decay_time_constant,
        'refractory_period': motoneurons_refractory_periods,
        'rheobase': motoneurons_rheobases,
        'spike_transmission_delay': motoneurons_spike_transmission_delays
    }

######################################
### FUNCTIONS TO SET UP THE SIMULATION
######################################

# # # SET BRIAN2 EQUATIONS
def set_brian2_equations():
    MN_equations = Equations('''
        dv/dt = (
            - g_leak*(v - voltage_rest)           # leak current from membrane conductance
            - g_ahp*(v - voltage_AHP)             # AHP current from AHP conductance (based on potassium reversal potential)
            + input_weight*(input_MN_timedarray_amp(t,i) + I_syn)  # excitatory + inhibitory synaptic currents (weighted by input_weight, i.e. normalized resistance)
        )/C_m : volt (unless refractory)

        dI_syn/dt = -I_syn/tau_syn  : amp  # Input from RC decays exponentially
        dg_ahp/dt = -g_ahp/tau_ahp       : siemens  # AHP conductance decays exponentially

        g_leak            : siemens
        C_m               : farad                    
        input_weight      : 1
        voltage_rest      : volt
        voltage_AHP       : volt
        refractory_period : second
        tau_syn           : second  # synaptic input from RC time constant (can be either the membrane time constant or an arbitrarily defined time constant)
        tau_ahp           : second   # AHP time constant
    ''')
    RC_equations = Equations('''
        dv/dt = (input_RC_timedarray_volt(t,i)-v)/tau: volt (unless refractory)
        tau : second
        ''')
    return MN_equations, RC_equations

# # # CREATE MOTOR NEURON SIZES AND ASSIGN THEM TO THEIR POOLS
def generate_motor_neurons(full_pool_sim,
        min_soma_diameter, max_soma_diameter,
        nb_pools, nb_motoneurons_per_pool, total_nb_motoneurons,
        size_distribution_exponent,
        mean_soma_diameter, sd_prct_soma_diameter,
        generate_figure=False, savepath=None):
    motoneuron_soma_diameters = np.zeros(total_nb_motoneurons)
    motoneuron_normalized_soma_diameters = np.zeros(total_nb_motoneurons)
    pool_list_by_MN, idx_of_MN_by_pool = [], {}
    for pooli in range(nb_pools):
        poolname_temp = f"pool_{pooli}"
        idx_of_MN_by_pool[poolname_temp] = np.arange(pooli*nb_motoneurons_per_pool, (pooli+1)*nb_motoneurons_per_pool)
        mni_offset = pooli*nb_motoneurons_per_pool
        if full_pool_sim == True:
            for mni in range(nb_motoneurons_per_pool):
                pool_list_by_MN.append(poolname_temp)
                motoneuron_soma_diameters[mni_offset+mni] = lerp(min_soma_diameter, max_soma_diameter, (mni/(nb_motoneurons_per_pool-1))**size_distribution_exponent )
                motoneuron_normalized_soma_diameters[mni_offset+mni] = lerp(0, 1, (mni/(nb_motoneurons_per_pool-1))**size_distribution_exponent )
        else:
            motoneuron_sizes_sampled_from_gaussian = np.random.normal(
                loc=mean_soma_diameter, scale=(sd_prct_soma_diameter/100)*mean_soma_diameter, size=nb_motoneurons_per_pool)
            motoneuron_sizes_sampled_from_gaussian = np.clip(motoneuron_sizes_sampled_from_gaussian, min_soma_diameter, max_soma_diameter)
            motoneuron_sizes_sampled_from_gaussian = np.sort(motoneuron_sizes_sampled_from_gaussian)
            for mni in range(nb_motoneurons_per_pool):
                pool_list_by_MN.append(poolname_temp)
                motoneuron_soma_diameters[mni_offset+mni] = motoneuron_sizes_sampled_from_gaussian[mni]
                motoneuron_normalized_soma_diameters[mni_offset+mni] = (motoneuron_sizes_sampled_from_gaussian[mni] - min_soma_diameter) / (max_soma_diameter - min_soma_diameter)
    
    if generate_figure:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        # 1) Histogram of raw soma diameters
        ax = axes[0]
        weights = np.ones_like(motoneuron_soma_diameters) * (100 / len(motoneuron_soma_diameters))
        ax.hist(
            motoneuron_soma_diameters,
            density=False,
            weights=weights,
            edgecolor='white',
            color='gray',
            alpha=1
        )
        ymin, ymax = ax.get_ylim()
        ax.vlines(min_soma_diameter, ymin, ymax, color='C1', label='Min soma diameter', linewidth=2)
        ax.vlines(max_soma_diameter, ymin, ymax, color='C3', label='Max soma diameter', linewidth=2)
        ax.set_xlabel("Soma diameter (μm)")
        ax.set_ylabel("Proportion (% of MNs)")
        ax.set_title("Distribution of MN soma diameters")
        ax.legend(loc='upper right')
        # 2) Histogram of normalized soma diameters
        ax = axes[1]
        weights_norm = np.ones_like(motoneuron_normalized_soma_diameters) * (100 / len(motoneuron_normalized_soma_diameters))
        ax.hist(
            motoneuron_normalized_soma_diameters,
            density=False,
            weights=weights_norm,
            edgecolor='white',
            color='gray',
            alpha=0.5
        )
        ymin, ymax = ax.get_ylim()
        ax.vlines(0, ymin, ymax, color='C1', label='Min normalized', linewidth=2)
        ax.vlines(1, ymin, ymax, color='C3', label='Max normalized', linewidth=2)
        ax.set_xlabel("Normalized soma diameter")
        ax.set_ylabel("Proportion (% of MNs)")
        ax.set_title("Normalized MN size distribution")
        ax.legend(loc='upper right')
        # 3) Soma diameter vs index
        ax = axes[2]
        ax.plot(motoneuron_soma_diameters, color='gray')
        mean_diameter = np.mean(motoneuron_soma_diameters)
        ax.hlines(mean_diameter, 0, len(motoneuron_soma_diameters)-1, color='C2', linestyle='--',
                label=f"Mean = {mean_diameter:.1f} μm")
        ax.set_xlabel("MN index")
        ax.set_ylabel("Soma diameter (μm)")
        ax.set_title("MN soma diameter vs. index")
        ax.legend(loc='upper right')
        # Set layout and save
        plt.tight_layout()
        if savepath is not None:
            save_file = os.path.join(savepath, 'MN_sizes.png')
            fig.savefig(save_file)
        plt.show()

    return (motoneuron_soma_diameters, motoneuron_normalized_soma_diameters,
            pool_list_by_MN, idx_of_MN_by_pool)

# # # CREATE MOTOR NEURON ELECTROPHYSIOLOGICAL PROPERTIES
def generate_motor_neuron_electrophysiological_properties(
        total_nb_motoneurons, motoneuron_soma_diameters,
        resistance_constant, resistance_exponent,
        capacitance_constant, capacitance_exponent,
        AHP_duration_constant, AHP_duration_exponent,
        rheobase_constant, rheobase_exponent, rheobase_scaling,
        refractory_period_absolute,
        axonal_conduction_velocity_constant, axonal_conduction_velocity_exponent,
        generate_figure=False, savepath=None):
    motoneurons_resistance = np.zeros(total_nb_motoneurons)
    motoneurons_input_weight = np.zeros(total_nb_motoneurons)
    motoneurons_capacitance = np.zeros(total_nb_motoneurons)
    motoneurons_membrane_conductance = np.zeros(total_nb_motoneurons)
    motoneurons_membrane_time_constant = np.zeros(total_nb_motoneurons)
    motoneurons_AHP_duration = np.zeros(total_nb_motoneurons)
    motoneurons_AHP_conductance_decay_time_constant = np.zeros(total_nb_motoneurons)
    motoneurons_refractory_periods = np.zeros(total_nb_motoneurons)
    motoneurons_rheobases = np.zeros(total_nb_motoneurons)
    motoneurons_spike_transmission_delays = np.zeros(total_nb_motoneurons)
    for mni in range(total_nb_motoneurons):
        motoneurons_resistance[mni] = resistance_constant*(motoneuron_soma_diameters[mni]**resistance_exponent)
        motoneurons_input_weight[mni] = motoneurons_resistance[mni] / motoneurons_resistance[0] # normalized value so that smallest MN has weight of 1
        motoneurons_capacitance[mni] = capacitance_constant*(motoneuron_soma_diameters[mni]**capacitance_exponent)
        motoneurons_membrane_conductance[mni] = 1/motoneurons_resistance[mni]
        motoneurons_membrane_time_constant[mni] = (motoneurons_resistance[mni]*ohm) * (motoneurons_capacitance[mni]*farad) / 100 # using the right units should make it be in seconds already. Diving by 100 because some mismatch of scale somewhere
        motoneurons_AHP_duration[mni] = AHP_duration_constant*(motoneuron_soma_diameters[mni]**AHP_duration_exponent)
        motoneurons_AHP_conductance_decay_time_constant[mni] = motoneurons_AHP_duration[mni] / np.log(10) # Time constant defined so that the AHP duration corresponds to the duration to reach 1/10th of the hyperpolarizing increase in conductance caused by a spike
        motoneurons_refractory_periods[mni] = refractory_period_absolute
        motoneurons_rheobases[mni] = rheobase_constant*(motoneuron_soma_diameters[mni]**rheobase_exponent)*rheobase_scaling
        motoneurons_spike_transmission_delays[mni] = 0.5/(axonal_conduction_velocity_constant*(motoneuron_soma_diameters[mni]**axonal_conduction_velocity_exponent))  # in s # The delay (s) is calculated from the axonal conduction velocity, assuming a 0.5m axon length => so correspond to the conduction speed from MN to muscle fiber (speed in m/s, so multiply speed by 2 -> numerator is 0.5 meter)
    
    motoneuron_properties_dict = wrap_motoneuron_properties(motoneuron_soma_diameters,
        motoneurons_resistance, motoneurons_input_weight,
        motoneurons_capacitance, motoneurons_membrane_conductance,
        motoneurons_membrane_time_constant,
        motoneurons_AHP_duration, motoneurons_AHP_conductance_decay_time_constant,
        motoneurons_refractory_periods, motoneurons_rheobases,
        motoneurons_spike_transmission_delays)
    
    if generate_figure:
        # Create a 5×1 grid of subplots
        fig, axes = plt.subplots(6, 1, figsize=(8, 22), sharex=True)
        # 1) Resistance & Input weight (dual axis)
        ax1 = axes[0]
        curve1, = ax1.plot(motoneurons_resistance, label="Resistance (Ω)", color='C1', linewidth=2)
        ax2 = ax1.twinx()
        curve2, = ax2.plot(
            motoneurons_input_weight, 
            label="Normalized input resistance", 
            color='C5', 
            linewidth=2, 
            linestyle=":"
        )
        ax1.set_ylabel("Resistance (Ω)", color='C1')
        ax2.set_ylabel("Normalized input resistance", color='C5')
        ax1.tick_params(axis='y', labelcolor='C1')
        ax2.tick_params(axis='y', labelcolor='C5')
        ax1.legend([curve1, curve2], [curve1.get_label(), curve2.get_label()], loc='best')
        ax1.set_title("Motoneuron Resistance & Input Weight")
        # 2) Capacitance
        ax3 = axes[1]
        ax3.plot(motoneurons_capacitance, label="Capacitance (F)", color='C2')
        ax3.set_ylabel("Capacitance (F)")
        ax3.legend(loc='best')
        ax3.set_title("Motoneuron Capacitance")
        # 3) Membrane conductance
        ax4 = axes[2]
        ax4.plot(motoneurons_membrane_conductance, label="Membrane conductance (mS)", color='C3')
        ax4.set_ylabel("Conductance (mS)")
        ax4.legend(loc='best')
        ax4.set_title("Motoneuron Membrane Conductance")
        # 5) Membrane time constant (determines RC IPSP effect)
        ax5 = axes[3]
        ln1, = ax5.plot(motoneurons_membrane_time_constant, label="Membrane time constant (ms)", color='black', linestyle='--')
        ax5.set_ylabel("Time (ms)")
        ax5.set_title("Membrane time constant\n(can be used for RC's IPSPs decay rate)")
        # 6) AHP & Refractory
        ax6 = axes[4]
        ln1, = ax6.plot(motoneurons_AHP_duration, label="AHP duration (ms)", color='blue', linestyle='--')
        ln2, = ax6.plot(motoneurons_AHP_conductance_decay_time_constant, label="AHP time constant (ms)", color='blue')
        ln3, = ax6.plot(motoneurons_refractory_periods, label="Refractory period (ms)", color='C6')
        ax6.set_ylabel("Time (ms)")
        ax6.legend([ln1, ln2, ln3], [ln1.get_label(), ln2.get_label(), ln3.get_label()], loc='best')
        ax6.set_title("AHP & Refractory Properties")
        # 7) Rheobase
        ax7 = axes[5]
        ax7.plot(motoneurons_rheobases, label="Rheobase (nA)", color='C7')
        ax7.set_xlabel("MN index")
        ax7.set_ylabel("Rheobase (nA)")
        ax7.legend(loc='best')
        ax7.set_title("Motoneuron Rheobase")
        # Adjust layout and save
        plt.tight_layout()
        if savepath is not None:
            new_filename = f'MN_electrophysiological_properties.png'
            save_file_path = os.path.join(savepath, new_filename)
            plt.savefig(save_file_path)
        plt.show()

    return (motoneurons_resistance, motoneurons_input_weight, motoneurons_capacitance, motoneurons_membrane_conductance, motoneurons_membrane_time_constant,
            motoneurons_AHP_conductance_decay_time_constant, motoneurons_refractory_periods, motoneurons_rheobases, motoneurons_spike_transmission_delays,
            motoneuron_properties_dict)

# # # GENERATE AND DISTRIBUTE COMMON INPUT(S)
def generate_and_distribute_common_inputs(fsamp, nb_of_common_inputs,
    nb_pools, nb_motoneurons_per_pool,
    duration_with_ignored_window, edges_ignore_duration,
    frequency_range_of_common_input, # array of shape [nb of pools x nb of common inpts x 2] (each cell contains the low and high frequency boundaries)
    common_input_std, # array of size [nb of pools x nb of common inpts] (at this point, should already be input as nA)
    frequency_range_to_set_input_power,
    set_arbitrary_correlation_between_excitatory_inputs,
    between_pool_excitatory_input_correlation,
    set_same_excitatory_input_for_all_pools,
    generate_figure=False, savepath=None
):
    """
    Generate nb_of_common_inputs Gaussian noise inputs per pool,
    filtered in the range specified by common_input_std.
    Optionally impose a target correlation structure,
    scale to desired band std, and plot:
      1) time series
      2) power spectrum
      3) correlation matrix
    """
    # logger = logging.getLogger(__name__)
    # 1) Generate raw inputs
    MN_excit_input = {}
    for pooli in range(nb_pools):
        MN_excit_input[pooli] = {}
        # Generate_filtered_gaussian_noise_input returns array shape (n_samples,)
        artifact_removal_window, _ = filter_artifact_removal(fsamp, edges_ignore_duration)
        for inputi in range(nb_of_common_inputs):
            MN_excit_input[pooli][inputi] = Generate_filtered_gaussian_noise_input(fsamp, 
                duration_with_ignored_window, edges_ignore_duration, artifact_removal_window,
                low_pass_filter_cutoff=frequency_range_of_common_input[pooli][inputi][1], # the low pass is the second boundary
                input_mean=0, scaling_std=1,
                high_pass_filter_cutoff=frequency_range_of_common_input[pooli][inputi][0]) # the high pas is the first boundary
    # 2) Impose correlation if requested - correlations are adjusted with the following mapping: input 0 of pool 0 is adjusted to input 0 of pool 1, input 1 of pool 0 is adjusted to input 1 of pool 1, etc. ...
    if set_arbitrary_correlation_between_excitatory_inputs and nb_pools > 1:
        # stack into shape (n_pools, n_samples)
        for inputi in range(nb_of_common_inputs):
            data = np.vstack([MN_excit_input[pooli][inputi] for pooli in range(nb_pools)])
            data = data.T  # shape (n_samples, n_pools)
            # build target correlation matrix
            C_target = np.zeros((nb_pools, nb_pools))
            for i in range(nb_pools):
                for j in range(nb_pools):
                    if i == j:
                        C_target[i, j] = 1.0
                    else:
                        C_target[i, j] = between_pool_excitatory_input_correlation
            # adjust
            transformed = adjust_correlation(data, C_target)
            # unpack
            for pooli in range(nb_pools):
                MN_excit_input[pooli][inputi] = transformed[:, pooli]
    # 3) Scale to desired band std - only for the first (likely low-freq) input
    for pooli in range(nb_pools):
        for inputi in range(nb_of_common_inputs):
            if inputi == 0: # The first input (which should be low freq) is scaled relative to the desired frequency band. 
                MN_excit_input[pooli][inputi] = scale_to_band_std(
                    MN_excit_input[pooli][inputi],
                    fsamp,
                    frequency_range_to_set_input_power,
                    common_input_std[pooli][inputi])
            else: # The other inputs are scaled directly to their requested stds, relative to the power of the first input in the frequency_range_to_set_input_power
                freq_range_of_input_temp_diff = frequency_range_of_common_input[pooli][inputi][1] - frequency_range_of_common_input[pooli][inputi][0]
                frequency_range_to_set_input_power_diff = frequency_range_to_set_input_power[1] - frequency_range_to_set_input_power[0]
                MN_excit_input[pooli][inputi] *= (
                    common_input_std[pooli][inputi] * np.sqrt(freq_range_of_input_temp_diff/frequency_range_to_set_input_power_diff))
    # 4) Optionally make all pools share the same input
    if set_same_excitatory_input_for_all_pools:
        for inputi in range(nb_of_common_inputs):
            base = MN_excit_input[0][inputi]
            for pooli in range(nb_pools):
                MN_excit_input[pooli][inputi] = base
    # 5) Collapse (merge) all the inputs from a given pool together
    for pooli in range(nb_pools):
        # Get all the arrays for this pool as a real list
        arrays = list(MN_excit_input[pooli].values())
        # Start with zeros of the same shape as the first array
        total = np.zeros_like(arrays[0])
        # Sum them all
        for arr in arrays:
            total += arr
        # Replace the dict with the collapsed array
        MN_excit_input[pooli] = total
    # 6) Compute pairwise correlation matrix
    corr_mat = np.eye(nb_pools)
    for i in range(nb_pools):
        for j in range(i+1, nb_pools):
            r = np.corrcoef(MN_excit_input[i], MN_excit_input[j])[0,1]
            corr_mat[i,j] = corr_mat[j,i] = r
    # 7) Build weight matrix: each MN receives 100% of its pool's input
    total_nb_mn = nb_pools * nb_motoneurons_per_pool
    inputs_to_mn_weight_matrix = np.zeros((total_nb_mn, nb_pools))
    for pooli in range(nb_pools):
        row_start = pooli * nb_motoneurons_per_pool
        row_end   = row_start + nb_motoneurons_per_pool
        inputs_to_mn_weight_matrix[row_start:row_end, pooli] = 1.0
    # 8) Compute power spectrum of common input
    total_power = {}
    power_within_band_of_interest = {}
    psd = {}
    # psd_within_band = {}
    nperseg = min(fsamp, len(MN_excit_input[0]))  # only first pool
    for pooli in range(nb_pools):
        freqs, psd[pooli] = welch( # freqs stay the same each time so no need to have a dict containing them for each pool
            MN_excit_input[pooli],
            fs=fsamp,
            window='hann',
            nperseg=nperseg,
            noverlap=nperseg // 2,
            scaling='density',
            detrend='constant')
        total_power[pooli] = np.trapz(psd[pooli], freqs)
        mask = (freqs >= frequency_range_to_set_input_power[0]) & (freqs <= frequency_range_to_set_input_power[1])
        power_within_band_of_interest[pooli] = np.trapz(psd[pooli][mask], freqs[mask])
    # logger.info(f"Time domain total power of common input = {np.mean(MN_excit_input[0]**2):.2f}")
    # logger.info(f"Frequency domain total power of common input = {total_power[0]:.2f}")
    # logger.info(f"Time domain within-band power of common input = undefined yet")
    # logger.info(f"Frequency domain within-band power of common input = {power_within_band_of_interest[0]:.2f}")
    # 8) Plotting: time series, power spectrum, correlation matrix, distribution of inputs to MNs
    max_freq_lim = 150
    power_per_frequency_band = {"frequencies": freqs[np.array(range(max_freq_lim)).astype(int)],
                                "power": psd[0][np.array(range(max_freq_lim)).astype(int)]}  # only first pool
    # Save power_per_frequency_band as csv file
    if savepath is not None:
        power_per_frequency_band_df = pd.DataFrame(power_per_frequency_band)
        power_per_frequency_band_df.to_csv(os.path.join(savepath, 'power_per_frequency_band_common_input.csv'), index=False) 
    if generate_figure:
        pool_colors = {0: "blue", 1: "red", 2: "orange", 3: "green"}
        # Build figure with GridSpec
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1], hspace=0.4, wspace=0.3)
        custom_input_labels = [f'Pool {p}' for p in range(nb_pools)]
        # 1) Time series - full (row 0, all columns)
        ax_ts = fig.add_subplot(gs[0, 0:2])
        t = np.linspace(0, duration_with_ignored_window, len(MN_excit_input[0]))
        for pooli in range(nb_pools):
            ax_ts.plot(t, MN_excit_input[pooli]/1000, label=f'Pool {pooli}',
                       color=pool_colors[pooli], alpha = 0.5)
        ax_ts.set_xlabel("Time (s)")
        ax_ts.set_ylabel("Input amplitude (microAmperes)")
        ax_ts.set_title("Common Input(s) Time Series (full)")
        ax_ts.legend(loc='upper right')
        # 1) Time series - zoomed-in (first 3 seconds) (row 0, all columns)
        ax_ts_zommed = fig.add_subplot(gs[0, 2:])
        time_to_display = np.min([edges_ignore_duration+3, duration_with_ignored_window])
        for pooli in range(nb_pools):
            ax_ts_zommed.plot(t, MN_excit_input[pooli]/1000, label=f'Pool {pooli}',
                              color=pool_colors[pooli], alpha = 0.5)
        ax_ts_zommed.set_xlabel("Time (s)")
        ax_ts_zommed.set_ylabel("Input amplitude (microAmperes)")
        ax_ts_zommed.set_title("Common Input(s) Time Series (Zoomed-in)")
        ax_ts_zommed.set_xlim(left=edges_ignore_duration, right=time_to_display)
        ax_ts_zommed.legend(loc='upper right')
        # 2) Power spectrum (row 1, columns 0-1)
        ax_spec = fig.add_subplot(gs[1, 0:2])
        plt.axvspan(frequency_range_to_set_input_power[0], frequency_range_to_set_input_power[1],
            color='red',      # the fill color
            alpha=0.2,        # transparency
            label=f"Scaling band: {frequency_range_to_set_input_power[0]}–{frequency_range_to_set_input_power[1]} Hz")
        for pooli in range(nb_pools):
            # first, fill under the PSD curve:
            ax_spec.fill_between(
                freqs, psd[pooli],
                y2=0, color=pool_colors[pooli], alpha=0.3,
                label=f'Pool {pooli}\n - Power={total_power[pooli]:.2f})\n - Power within band to set input={power_within_band_of_interest[pooli]:.2f}')
            # then draw the line on top
            ax_spec.plot(freqs, psd[pooli], color=pool_colors[pooli])
        ax_spec.set_xlabel("Frequency (Hz)")
        ax_spec.set_ylabel("Power spectrum")
        ax_spec.set_xlim([0, max_freq_lim])
        ax_spec.set_title("Common input(s) power Spectrum (Welch)")
        ax_spec.legend(fontsize=8, loc='upper right')
        # 3) Correlation matrix (row 1, column 2)
        ax_corr = fig.add_subplot(gs[1, 2])
        sns.heatmap(
            corr_mat,
            ax=ax_corr,
            annot=True,
            cmap="Spectral_r", # "RdYlBu_r",
            vmin=-1, vmax=1,
            xticklabels=custom_input_labels,
            yticklabels=custom_input_labels
        )
        ax_corr.set_title("Pairwise Correlation Matrix")
        ax_corr.set_xlabel("Pool")
        ax_corr.set_ylabel("Pool")
        # 4) Weight distribution matrix (row 1, column 3)
        ax_wgt = fig.add_subplot(gs[1, 3])
        sns.heatmap(
            inputs_to_mn_weight_matrix,
            ax=ax_wgt,
            cmap= "viridis", # "RdYlBu_r",
            vmin=0, vmax=1,
            xticklabels=custom_input_labels,
            yticklabels=False     # turn off Seaborn’s default labels
        )
        # now add just every Nth MN index:
        max_labels = 10
        step = max(1, total_nb_mn // max_labels)
        yticks = np.arange(0, total_nb_mn, step)
        ax_wgt.set_yticks(yticks)
        ax_wgt.set_yticklabels(yticks)
        ax_wgt.set_title("Input distribution to MNs")
        ax_wgt.set_xlabel("Pool Input")
        ax_wgt.set_ylabel("MN Index")
        # Adjust layout and save
        plt.tight_layout()
        if savepath is not None:
            new_filename = f'Common_inputs.png'
            save_file_path = os.path.join(savepath, new_filename)
            plt.savefig(save_file_path)
        plt.show()

    return (MN_excit_input, corr_mat, inputs_to_mn_weight_matrix,
            total_power, power_within_band_of_interest, power_per_frequency_band)

# # # GENERATE INDEPENDENT INPUTS AND CREATE BRIAN2 TIME ARRAYS
def generate_independent_inputs(fsamp,
        nb_pools, total_nb_motoneurons, total_nb_renshaw_cells,
        low_pass_filter_of_MN_independent_input,
        low_pass_filter_of_RC_independent_input,
        ref_common_input_power,
        MN_independent_input_absolute_or_ratio,
        MN_independent_input_power,
        RC_independent_input_std,
        MN_excit_input,
        inputs_to_mn_weight_matrix,
        excitatory_input_baseline, # list of size <= nb_pools
        duration_with_ignored_window,
        edges_ignore_duration,
        motoneurons_rheobases,
        generate_figure=False, savepath=None
    ):
    # logger = logging.getLogger(__name__)
    # Motor neuron - generate independent input
    artifact_removal_window, _ = filter_artifact_removal(fsamp, edges_ignore_duration) # Get artifact removal window
    MN_independent_input = []
    nperseg = min(fsamp, len(MN_excit_input[0]))  # only first pool
    for mni in range(total_nb_motoneurons):
        temp_input = Generate_filtered_gaussian_noise_input(fsamp, 
            duration_with_ignored_window, edges_ignore_duration, artifact_removal_window,
            low_pass_filter_cutoff=low_pass_filter_of_MN_independent_input,
            input_mean=0, scaling_std=1)
        pooli = np.round(mni // (total_nb_motoneurons/nb_pools)).astype(int)
        # Scale it to the desired power
        temp_input_power = np.mean(temp_input**2)
        if MN_independent_input_absolute_or_ratio == 'absolute':
            independent_input_scaling_factor = MN_independent_input_power
        elif MN_independent_input_absolute_or_ratio == 'ratio':
            desired_independent_power = MN_independent_input_power * ref_common_input_power[pooli]
            independent_input_scaling_factor = np.sqrt(desired_independent_power / temp_input_power)
        temp_input *= independent_input_scaling_factor
        # Keep the first independent input as a separate variable for plotting purpose:
        MN_independent_input.append(temp_input)
        if generate_figure and mni==0:
            independent_input_for_plotting = temp_input.copy()
        # # Some logs for debugging
        #     logger.info(f"Pool # = {pooli}")
        #     logger.info(f"inital temp_input_power = {temp_input_power:.2f}")
        #     logger.info(f"ref_common_input_power = {ref_common_input_power[0]:.2f}")
        #     logger.info(f"desired_independent_power = {desired_independent_power:.2f}")
        #     logger.info(f"Scaling factor = {independent_input_scaling_factor:.2f}")
        #     logger.info(f"Transformed temp input power = {np.mean(temp_input**2):.2f}")
        #     logger.info(f"independent_input_for_plotting power = {np.mean(independent_input_for_plotting**2):.2f}")
        #     logger.info(f"Excitatory input shape = {MN_excit_input[pooli].shape}")
        #     # logger.info(f"Excitatory input power = {np.mean((np.array(MN_excit_input[pooli]).flatten() * inputs_to_mn_weight_matrix[mni, pooli])**2)}")
        #     logger.info(f"Excitatory input power = {np.mean(np.array(MN_excit_input[pooli])**2)}")
        #     logger.info(f"Excitatory input power multiplied by weight matrix = {np.mean((np.array(MN_excit_input[pooli]) * inputs_to_mn_weight_matrix[mni, pooli])**2)}")
    # Get mean power of the independent input
    max_freq_lim = 150
    psd_independent_input = []
    for mni in range(len(MN_independent_input)):
        freqs, psd_temp = welch( # freqs stay the same each time so no need to have a dict containing them for each pool
            MN_independent_input[mni],
            fs=fsamp,
            window='hann',
            nperseg=nperseg,
            noverlap=nperseg // 2,
            scaling='density',
            detrend='constant')
        psd_independent_input.append(psd_temp)
    power_per_frequency_band_independent = {"frequencies": freqs,
                                            "power": np.mean(np.array(psd_independent_input), axis = 0)}
    total_power_independent = np.trapz(power_per_frequency_band_independent["power"], freqs)
    power_per_frequency_band_independent = {"frequencies": freqs[np.array(range(max_freq_lim)).astype(int)],
                                            "power": power_per_frequency_band_independent["power"][np.array(range(max_freq_lim)).astype(int)]} 
    # Save power_per_frequency_band as csv file
    if savepath is not None:
        power_per_frequency_band_df = pd.DataFrame(power_per_frequency_band_independent)
        power_per_frequency_band_df.to_csv(os.path.join(savepath, 'power_per_frequency_band_independent_input.csv'), index=False) 
    temp_timed_array = np.zeros((int(np.round(duration_with_ignored_window/second*fsamp)), total_nb_motoneurons))
    for mni in range(total_nb_motoneurons):
        pooli = np.round(mni // (total_nb_motoneurons/nb_pools)).astype(int)
        temp_timed_array[:, mni] += np.array(
                MN_excit_input[pooli]) * inputs_to_mn_weight_matrix[mni, pooli] # the weights are relative to the distribution of common input to the different pools, not the size of the motor neurons!
        # add excitatory baseline and independent input
        temp_timed_array[:, mni] += excitatory_input_baseline[pooli]
        temp_timed_array[:, mni] += MN_independent_input[mni]
        # Rheobase = clip value to 0 if it is below a given value (in nA)
        temp_timed_array[:, mni] = np.clip(
            temp_timed_array[:, mni]-motoneurons_rheobases[mni],
            a_min=0, a_max=np.inf)
    input_MN_timedarray_amp = TimedArray(temp_timed_array * nA, dt=(1/fsamp)*second) # in nano Ampere

    # Renshaw cell - generate independent input and fill time array
    RC_independent_input = []
    temp_timed_array = np.zeros((int(np.round(duration_with_ignored_window/second*fsamp)), total_nb_renshaw_cells))
    for renshawi in range(total_nb_renshaw_cells):
        RC_independent_input.append(Generate_filtered_gaussian_noise_input(fsamp, 
            duration_with_ignored_window, edges_ignore_duration, artifact_removal_window,
            low_pass_filter_cutoff=low_pass_filter_of_RC_independent_input,
            input_mean=0, scaling_std=RC_independent_input_std))
    input_RC_timedarray_volt = TimedArray(temp_timed_array * mvolt, dt=(1/fsamp)*second)

    # Sanity check plot
    if generate_figure:
        plt.figure(figsize=(30,5))
        plt.plot(input_MN_timedarray_amp.values[:,
            np.linspace(0, total_nb_motoneurons-1, 10, dtype=int)],
            alpha=0.2, color = 'C0')
        test = np.mean(input_MN_timedarray_amp.values[:,np.arange(1,total_nb_motoneurons)],axis=1)
        plt.plot(test, color='darkblue')
        plt.xlabel("Time (samples)")
        plt.ylabel("Input (Amperes)")
        if savepath is not None:
            new_filename = f'Input_TimedArray_SanityCheck.png'
            save_file_path = os.path.join(savepath, new_filename)
            plt.savefig(save_file_path)
        plt.show()

        plt.figure()
        duration_to_plot = 3 # in seconds
        nb_samples_to_plot = int(np.round(duration_to_plot*fsamp))
        independent_input_for_plotting_power = np.mean(independent_input_for_plotting**2)
        common_input_for_plotting_power = np.mean(MN_excit_input[0]**2)
        if nb_samples_to_plot > len(independent_input_for_plotting):
            nb_samples_to_plot = len(independent_input_for_plotting)
        plt.plot(np.arange(nb_samples_to_plot)/fsamp,
            MN_excit_input[0][:nb_samples_to_plot] * inputs_to_mn_weight_matrix[0, 0],
            label=f"Common input delivered to MN#0\nPower={common_input_for_plotting_power:.2f}",
            color = 'red')
        plt.plot(np.arange(nb_samples_to_plot)/fsamp,
            independent_input_for_plotting[:nb_samples_to_plot],
            label=f"Independent input delivered to MN#0\nPower={independent_input_for_plotting_power:.2f}",
            color='blue', alpha=0.5, linewidth=0.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Input (nA)")
        plt.title(f"Recalculated ratio of independent VS common input = {independent_input_for_plotting_power/common_input_for_plotting_power:.2f}")
        plt.legend()
        if savepath is not None:
            new_filename = f'Input_common_VS_independent_MN0.png'
            save_file_path = os.path.join(savepath, new_filename)
            plt.savefig(save_file_path)
        plt.show()

    return input_MN_timedarray_amp, input_RC_timedarray_volt, total_power_independent, power_per_frequency_band_independent

# # # CREATE CONNECTIVITY BETWEEN MOTOR NEURONS AND RENSHAW CELLS
def create_connectivity(total_nb_motoneurons, total_nb_renshaw_cells,
        nb_pools, nb_motoneurons_per_pool, RC_pair_indices, nb_RCs_per_pool_pair,
        disynpatic_inhib_connections_desired_MN_MN,
        split_MN_RC_ratio, motoneuron_normalized_soma_diameters,
        distribution_type, distribution_params, disynaptic_inhib_received_arbitrary_adjustment,
        distribution_binary_weights,
        generate_figure=False, savepath=None):
    # --- Allocate global adjacency matrices ---
    MN_to_Renshaw_connectivity_matrix = np.zeros(
        (total_nb_motoneurons, total_nb_renshaw_cells), dtype=float
    )
    Renshaw_to_MNs_connectivity_matrix = np.zeros(
        (total_nb_renshaw_cells, total_nb_motoneurons), dtype=float
    )
    # --- Fill in each pool‐pair block independently ---
    motoneuron_size_ranks   = np.asarray(motoneuron_normalized_soma_diameters) # need this if using size_gaussian
    for i in range(nb_pools):
        mn_pre = np.arange(i*nb_motoneurons_per_pool,
                        (i+1)*nb_motoneurons_per_pool)
        for j in range(nb_pools):
            mn_post = np.arange(j*nb_motoneurons_per_pool,
                                (j+1)*nb_motoneurons_per_pool)
            rc_inds  = RC_pair_indices[(i, j)]
            nRC      = len(rc_inds)
            # desired *mean disynaptic count* for this block
            mean_disyn = disynpatic_inhib_connections_desired_MN_MN[i, j]
            # solve p_mn_rc * p_rc_mn = mean_disyn / nRC
            base_conn = mean_disyn / nRC
            p_mn_rc   = base_conn ** split_MN_RC_ratio
            p_rc_mn   = base_conn ** (1.0 - split_MN_RC_ratio)
            # sample MN→RC sub‐matrix and write into global
            subA = sample_block(
                n_pre=len(mn_pre),
                n_post=nRC,
                p_block=p_mn_rc,
                dist=distribution_type,
                params=distribution_params,
                ranks=(motoneuron_size_ranks[mn_pre]
                    if distribution_type=='size_gaussian' else None),
                binary=distribution_binary_weights
            )
            MN_to_Renshaw_connectivity_matrix[np.ix_(mn_pre, rc_inds)] = subA
            # sample RC→MN sub‐matrix and write into global
            subB = sample_block(
                n_pre=nRC,
                n_post=len(mn_post),
                p_block=p_rc_mn,
                dist=distribution_type,
                params=distribution_params,
                ranks=None,
                binary=distribution_binary_weights
            )
            Renshaw_to_MNs_connectivity_matrix[np.ix_(rc_inds, mn_post)] = subB
    # --- Modify the connectivity from RCs to MNs from the point of view of each MN
    if disynaptic_inhib_received_arbitrary_adjustment > 0:
        for mni in range(total_nb_motoneurons):
            Renshaw_to_MNs_connectivity_matrix[:, mni] += np.random.normal(loc=0, scale=disynaptic_inhib_received_arbitrary_adjustment, size=total_nb_renshaw_cells)
        # Make sure the weights are non-negative (no excitation from Renshaw cells!)
        Renshaw_to_MNs_connectivity_matrix = np.maximum(Renshaw_to_MNs_connectivity_matrix, 0)
    # --- Compute disynaptic MN→MN counts (or binary for 'binarize') ---
    MN_to_MN_counts = MN_to_Renshaw_connectivity_matrix.dot(
        Renshaw_to_MNs_connectivity_matrix)
    # np.fill_diagonal(MN_to_MN_counts, 0)
    if distribution_type == 'binarize':
        # interpret exactly as mean *probability* of ≥1 path
        MN_to_MN_connectivity_matrix = (MN_to_MN_counts > 0).astype(int)
    else:
        # keep raw counts as your “weights”
        MN_to_MN_connectivity_matrix = MN_to_MN_counts
    
    ### Plotting, if requested
    if generate_figure:
        mn_pool_labels = [f"MN pool {i}" for i in range(nb_pools)]
        rc_pair_list   = [(i,j) for i in range(nb_pools) for j in range(nb_pools)]
        rc_pool_pair_labels = [f"RC pool pair {i}-{j}" for (i,j) in rc_pair_list]
        plot_connectivity_matrix(
            MN_to_Renshaw_connectivity_matrix,
            "Monosynaptic MN → RC",
            mn_pool_labels,
            rc_pool_pair_labels,
            n_pre_pool=nb_motoneurons_per_pool,
            n_post_pool=nb_RCs_per_pool_pair,
            cmap='autumn',
            add_colorbar=False,
            savepath=f'{savepath}/Connectivity_MN_to_RC.png'
        )
        plot_connectivity_matrix(
            Renshaw_to_MNs_connectivity_matrix,
            "Monosynaptic RC → MN",
            rc_pool_pair_labels,
            mn_pool_labels,
            n_pre_pool=nb_RCs_per_pool_pair,
            n_post_pool=nb_motoneurons_per_pool,
            cmap='winter',
            add_colorbar=False,
            savepath=f'{savepath}/Connectivity_RC_to_MN.png'
        )
        plot_connectivity_matrix(
            MN_to_MN_connectivity_matrix,
            "Disynaptic MN → MN via RCs",
            mn_pool_labels,
            mn_pool_labels,
            n_pre_pool=nb_motoneurons_per_pool,
            n_post_pool=nb_motoneurons_per_pool,
            cmap='viridis',
            add_colorbar=True,
            savepath=f'{savepath}/Connectivity_MN_to_MN.png'
        )
        # ─── Histograms per pool→pool ─────────────────── #
        nP = nb_pools
        counts_MN_RC = {}
        counts_RC_MN = {}
        counts_MN_MN = {}
        for i in range(nP):
            mn_pre  = slice(i*nb_motoneurons_per_pool, (i+1)*nb_motoneurons_per_pool)
            for j in range(nP):
                mn_post = slice(j*nb_motoneurons_per_pool, (j+1)*nb_motoneurons_per_pool)
                rc_inds = RC_pair_indices[(i,j)]
                counts_MN_RC[(i,j)] = (
                    MN_to_Renshaw_connectivity_matrix[np.ix_(range(*mn_pre.indices(total_nb_motoneurons)), rc_inds)]
                    .sum(axis=1)
                )
                counts_RC_MN[(i,j)] = (
                    Renshaw_to_MNs_connectivity_matrix[np.ix_(rc_inds, range(*mn_post.indices(total_nb_motoneurons)))]
                    .sum(axis=0)
                )
                counts_MN_MN[(i,j)] = (
                    MN_to_MN_connectivity_matrix[mn_pre, mn_post]
                    .sum(axis=1)
                )
        fig = plt.figure(figsize=(4*nP, 4*nP))
        outer = gridspec.GridSpec(nP, nP, wspace=0.4, hspace=0.6)
        for i in range(nP):
            for j in range(nP):
                cell = outer[i,j]
                inner = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec=cell,
                                                        height_ratios=[1,1], hspace=0.2)
                # top
                ax1 = fig.add_subplot(inner[0])
                ax1.hist(counts_MN_RC[(i,j)], bins='auto', alpha=0.6, label='MN→RC', color='C1')
                ax1.hist(counts_RC_MN[(i,j)], bins='auto', alpha=0.6, label='RC→MN', color='C0')
                ax1.set_ylabel("Count")
                ax1.set_title(f"Pools {i}→{j}")
                ax1.legend(fontsize=8, loc="upper left")
                ax1.text(0.95,0.7,
                        f"μ₁={counts_MN_RC[(i,j)].mean():.1f}±{counts_MN_RC[(i,j)].std():.1f}\n"
                        f"μ₂={counts_RC_MN[(i,j)].mean():.1f}±{counts_RC_MN[(i,j)].std():.1f}",
                        transform=ax1.transAxes, ha='right', va='top', fontsize=7)
                # bottom
                ax2 = fig.add_subplot(inner[1])
                ax2.hist(counts_MN_MN[(i,j)], bins='auto', alpha=0.6, color='green', label='MN→MN')
                ax2.set_xlabel("Number of synapses")
                ax2.set_ylabel("Count")
                ax2.legend(fontsize=8, loc="upper left")
                ax2.text(0.95,0.7,
                        f"μ₃={counts_MN_MN[(i,j)].mean():.1f}±{counts_MN_MN[(i,j)].std():.1f}",
                        transform=ax2.transAxes, ha='right', va='top', fontsize=7)
        plt.suptitle("Number of synapses per pool, per cell type")
        plt.tight_layout()
        if savepath is not None:
            new_filename = f'Connectivity_histogram.png'
            save_file_path = os.path.join(savepath, new_filename)
            plt.savefig(save_file_path)
        plt.show()
    
    return MN_to_MN_connectivity_matrix, MN_to_Renshaw_connectivity_matrix, Renshaw_to_MNs_connectivity_matrix

# # # CREATE Brian2 NEURONGROUPS AND SYNAPSES OBJECTS
def create_neurongroups_and_synapses_objects(
        total_nb_motoneurons, total_nb_renshaw_cells,
        MN_equations, RC_equations, voltage_rest, voltage_thresh,
        motoneurons_membrane_conductance, motoneurons_capacitance,
        motoneurons_input_weight, synaptic_IPSP_decay_time_constant_per_MN,
        motoneurons_AHP_conductance_decay_time_constant, motoneurons_refractory_periods,
        AHP_conductance_delta_after_spiking, MN_to_Renshaw_excit, Renshaw_to_MN_inhib,
        scale_initial_IPSP_to_be_same_integral_regardless_of_synaptic_tau,
        tau_Renshaw, MN_RC_synpatic_delay, refractory_period_RC,
        MN_to_Renshaw_connectivity_matrix, Renshaw_to_MNs_connectivity_matrix):
    logger = logging.getLogger(__name__)
    # # ----------------- BRIAN2 COMMON NAMESPACE FOR VARIABLES
    common_brian2_namespace = {
        'voltage_thresh': voltage_thresh,
        'voltage_rest':   voltage_rest,
        'refractory_period_RC': refractory_period_RC,
        'AHP_conductance_delta_after_spiking': AHP_conductance_delta_after_spiking,
        'MN_to_Renshaw_excit': MN_to_Renshaw_excit,
        'Renshaw_to_MN_IPSP_integral': Renshaw_to_MN_inhib * second # Should be in amp * s = Coulomb (total charge). Renshaw_to_MN_inhib is already in amp. 
    }
    # ----------------- NEURON GROUPS ----------- 
    # MOTOR NEURONS
    motoneurons = NeuronGroup(
        total_nb_motoneurons, 
        MN_equations, 
        threshold='v > voltage_thresh', 
        reset='''
        v = voltage_rest
        g_ahp += AHP_conductance_delta_after_spiking
        ''',
        refractory='refractory_period',
        method='euler',
        namespace=common_brian2_namespace
    )
    motoneurons.v = voltage_rest # in mV #
    motoneurons.g_leak = motoneurons_membrane_conductance * msiemens # in milisiemens
    motoneurons.C_m = motoneurons_capacitance * ufarad # in microfarads
    motoneurons.refractory_period = motoneurons_refractory_periods * ms  # in milliseconds
    motoneurons.input_weight = motoneurons_input_weight # dimensionless unit
    motoneurons.tau_syn = synaptic_IPSP_decay_time_constant_per_MN # already in millisecond
    motoneurons.tau_ahp = motoneurons_AHP_conductance_decay_time_constant * ms # in millisecond
    motoneurons.g_ahp = 0*siemens # Initialize AHP with a current of 0
    # logger.info(f"motoneurons tau syn = {motoneurons.tau_syn}")
    # logger.info(f"motoneurons input weights = {motoneurons.input_weight}")
    # logger.info(f"IPSP integrals = {common_brian2_namespace['Renshaw_to_MN_IPSP_integral']}")
    # RENSHAW CELLS
    renshaw_cells = NeuronGroup(
        total_nb_renshaw_cells,
        RC_equations,
        threshold='v > voltage_thresh', 
        reset='v = voltage_rest',
        refractory='refractory_period_RC',
        method='euler',
        namespace=common_brian2_namespace
    )
    renshaw_cells.v = voltage_rest  # Initialize membrane potential
    renshaw_cells.tau = tau_Renshaw
    # ----------------- SYNAPSES ----------- 
    # Connect motor neurons to Renshaw cells
    synapses_MN_to_Renshaw = Synapses(motoneurons, renshaw_cells, 'w : 1',
                            on_pre='v += MN_to_Renshaw_excit*w',
                            delay = MN_RC_synpatic_delay,
                            namespace=common_brian2_namespace)
    pre_indices, post_indices = np.nonzero(MN_to_Renshaw_connectivity_matrix)
    weights_to_assign = MN_to_Renshaw_connectivity_matrix[pre_indices,post_indices]
    if len(pre_indices)>0 and len(post_indices)>0:
        synapses_MN_to_Renshaw.connect(i=pre_indices, j=post_indices)
        synapses_MN_to_Renshaw.w = weights_to_assign
    else:
        synapses_MN_to_Renshaw.active = False
    # Connect Renshaw cells to motor neurons
    # # ----------------- CHANGE Renshaw_to_MN_inhib IF IT IS USED AS A TARGET RELATIVE TO THE SYNAPTIC TIME CONSTANT
    if scale_initial_IPSP_to_be_same_integral_regardless_of_synaptic_tau:
        on_pre_action = 'I_syn -= (Renshaw_to_MN_IPSP_integral * w) / tau_syn'
        logger.info(f"      Defined IPSP is an integral over hyperpolarizing current (hyperpolarizing charge, in Coulomb) = {common_brian2_namespace['Renshaw_to_MN_IPSP_integral']}")
    else:
        on_pre_action = 'I_syn -= (Renshaw_to_MN_IPSP_integral * w) / (1*second)'   # Re-interpret the initial IPSP amplitude as an area over 1s
        logger.info(f"      Defined IPSP is the initial hyperpolarizing current induced by the Renshaw cell's IPSP (hyperpolarizing current, in Amp) = {common_brian2_namespace['Renshaw_to_MN_IPSP_integral']/second}")
    synapses_Renshaw_to_MN = Synapses(renshaw_cells, motoneurons, 
                                '''
                                w: 1
                                Renshaw_to_MN_IPSP_integral: amp*second     # the total ∫I(t)dt you want = total charge (Coulomb)
                                ''', 
                                on_pre=on_pre_action, # pick up each post‐cell's tau_syn
                                delay = MN_RC_synpatic_delay,
                                namespace=common_brian2_namespace)
    pre_indices, post_indices = np.nonzero(Renshaw_to_MNs_connectivity_matrix)
    weights_to_assign = Renshaw_to_MNs_connectivity_matrix[pre_indices, post_indices]
    if len(pre_indices)>0 and len(post_indices)>0:
        synapses_Renshaw_to_MN.connect(i=pre_indices, j=post_indices)
        synapses_Renshaw_to_MN.w = weights_to_assign
        synapses_Renshaw_to_MN.Renshaw_to_MN_IPSP_integral = common_brian2_namespace['Renshaw_to_MN_IPSP_integral']
    else:
        synapses_Renshaw_to_MN.active = False
        
    return motoneurons, renshaw_cells, synapses_MN_to_Renshaw, synapses_Renshaw_to_MN

# # # GET SPIKE TRAINS
def get_spike_trains(spike_monitor_MN, spike_monitor_RC,
    spike_transmission_delay, total_nb_motoneurons, total_nb_renshaws,
    edges_ignore_duration, duration_with_ignored_window, motoneurons_soma_diameters,
    generate_figure=False, savepath=None):
    """
    Pull out and (optionally) plot the post‐delay spike trains from your monitors.

    Parameters
    ----------
    spike_monitor_MN : Brian2 SpikeMonitor recording the motoneurons
    spike_monitor_RC : Brian2 SpikeMonitor recording the Renshaw cells
    spike_transmission_delay : array_like, length total_nb_motoneurons
        delay (in seconds) to add to each MN spike train before returning.
    total_nb_motoneurons : int
        How many MNs in total you expect (so that even silent ones get an empty list).
    generate_figure : bool
    savepath : str or None
    """
    # 1) Extract raw trains from the monitors
    raw_MN = spike_monitor_MN.spike_trains()   # { neuron_index: array_of_times_in_s, ... }
    raw_RC = spike_monitor_RC.spike_trains()

    # helper to strip units only if needed
    def to_seconds(qt):
        # If it's a Brian Quantity, divide by second
        if isinstance(qt, Quantity):
            return np.asarray(qt/second, dtype=float)
        # Otherwise assume it's already a float array in seconds
        return np.asarray(qt, dtype=float)

    # build MN list
    spike_trains_MN = []
    for m in range(total_nb_motoneurons):
        qt = raw_MN.get(m, np.array([],float))
        times_s = to_seconds(qt)
        # add your float delays (in seconds)
        times_s = times_s + float(spike_transmission_delay[m])
        spike_trains_MN.append(times_s)

    # build RC list
    # max_r = max(raw_RC.keys())+1 if raw_RC else 0 # Get nb of Renshaw cells directly from the monitor
    spike_trains_RC = []
    for r in range(total_nb_renshaws):
        qt = raw_RC.get(r, np.array([],float))
        times_s = to_seconds(qt)
        spike_trains_RC.append(times_s)

    # (Optional) Plot them
    if generate_figure:
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(20,10), sharex=True)
        t_min, t_max = 0.0, duration_with_ignored_window

        # shade + dashed lines for ignored windows
        for ax in (ax1, ax2):
            ax.axvspan(t_min, edges_ignore_duration, color='grey', alpha=0.3)
            ax.axvspan(t_max - edges_ignore_duration, t_max, color='grey', alpha=0.3)
            ax.axvline(edges_ignore_duration, color='black', linestyle='--', linewidth=1)
            ax.axvline(t_max - edges_ignore_duration, color='black', linestyle='--', linewidth=1)

        # prepare colormap for MN sizes
        cmap = plt.get_cmap('viridis')
        norm = mpl.colors.Normalize(
            vmin=motoneurons_soma_diameters.min(),
            vmax=motoneurons_soma_diameters.max()
        )
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # for the colorbar

        # panel 1: MN rasters, colored by soma diameter
        ax1.set_title("Motoneuron Spike Trains")
        for mni, times in enumerate(spike_trains_MN):
            c = cmap(norm(motoneurons_soma_diameters[mni]))
            ax1.eventplot(times, lineoffsets=mni, colors=[c], alpha=0.6)
        ax1.set_ylabel("MN index")
        cbar = fig.colorbar(sm, ax=ax1, pad=0.02)
        cbar.set_label("MN soma diameter (µm)")

        # panel 2: RC rasters (unchanged)
        ax2.set_title("Renshaw Cell Spike Trains")
        for rci, times in enumerate(spike_trains_RC):
            ax2.eventplot(times, lineoffsets=rci, colors='purple', alpha=0.2)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("RC index")

        plt.tight_layout()
        if savepath is not None:
            fn = os.path.join(savepath, "Spike_Trains.png")
            plt.savefig(fn)
        plt.show()

    return spike_trains_MN, spike_trains_RC

# # # SAVE OUTPUT AS HDF5 FILE WITH H5PY
def save_output_hdf5(directory_name,
                     params,                    # SimulationParameters dataclass
                     motoneurons_and_pools_idx, # dict of dicts, see generate_motor_neurons() (saving {pool_list_by_MN, idx_of_MN_by_pool})
                     motoneurons_properties,    # dict of np.ndarrays
                     connectivity_matrix_MN_to_RC,
                     connectivity_matrix_RC_to_MN,
                     connectivity_matrix_MN_to_MN,
                     spike_trains_MN,           # list of 1D np.ndarrays
                     spike_trains_RC,           # list of 1D np.ndarrays
                     common_input_MN,           # dict of 1D np.arrays (one per pool)
                     common_input_power_total, common_input_power_spectrum, # floats
                     independent_input_power_total, independent_input_power_spectrum): # dict with keys "frequencies" and "power"             
    
    output_file = os.path.join(directory_name, "simulation_output.h5")
    with h5py.File(output_file, "w") as f:
        # 1) simulation parameters
        sim_grp = f.create_group("simulation_parameters")
        param_dict = asdict(params)
        param_dict.pop('RC_pair_indices', None) # Remove this value from the parameter list (it's a tuple and not readily writable)
        for k, v in param_dict.items():
            # numpy arrays → datasets
            if isinstance(v, np.ndarray):
                sim_grp.create_dataset(k, data=v)
            # Brian2 quantities → two entries: numeric + unit
            elif isinstance(v, Quantity):
                sim_grp.create_dataset(k + "_value", data=v.magnitude)
                sim_grp.attrs[k + "_unit"] = str(v.units)
            # “simple” scalars OK as attrs
            elif isinstance(v, (int, float, bool, str)):
                sim_grp.attrs[k] = v
            # anything else (lists, tuples, dicts, etc.) → JSON‐dumped string
            else:
                sim_grp.attrs[k] = json.dumps(v)
        
        # 2) motoneuron and pool indices
        mn_pool_grp = f.create_group("motoneurons_and_pools_indices")
        # (a) pool_list_by_MN -- a string array
        pool_list = motoneurons_and_pools_idx["pool_list_by_MN"]
        # make a numpy array of dtype "variable‐length UTF‐8 string"
        str_dt = string_dtype(encoding="utf-8")
        mn_pool_grp.create_dataset(
            "pool_list_by_MN",
            data=np.array(pool_list, dtype=str_dt),
            dtype=str_dt)
        # (b) idx_of_MN_by_pool -- subgroup of integer arrays
        by_pool_grp = mn_pool_grp.create_group("idx_of_MN_by_pool")
        for poolname, idx_array in motoneurons_and_pools_idx["idx_of_MN_by_pool"].items():
            # poolname is something like "pool_0", "pool_1", ...
            by_pool_grp.create_dataset(poolname, data=idx_array.astype(int))
        # # Example to read back:
        # with h5py.File(..., "r") as f:
        #     grp = f["motoneurons_and_pools_indices"]
        #     pool_list = grp["pool_list_by_MN"][()]        # array of bytes→ decode to str if you like
        #     by_pool   = grp["idx_of_MN_by_pool"]
        #     idx0      = by_pool["pool_0"][()]             # MN indices in pool_0

        # 3) motoneuron properties
        mn_grp = f.create_group("motoneurons_properties")
        for k, arr in motoneurons_properties.items():
            mn_grp.create_dataset(k, data=arr)

        # 2) connectivity
        conn = f.create_group("connectivity")
        conn.create_dataset("MN_to_RC", data=connectivity_matrix_MN_to_RC)
        conn.create_dataset("RC_to_MN", data=connectivity_matrix_RC_to_MN)
        conn.create_dataset("MN_to_MN", data=connectivity_matrix_MN_to_MN)

        # 5) spike trains
        spikes = f.create_group("spike_trains")
        mn_spikes = spikes.create_group("MN")
        for i, tr in enumerate(spike_trains_MN):
            mn_spikes.create_dataset(f"MN_{i}", data=tr)
        rc_spikes = spikes.create_group("RC")
        for i, tr in enumerate(spike_trains_RC):
            rc_spikes.create_dataset(f"RC_{i}", data=tr)
        
        # 6) Synaptic input
        input_grp = f.create_group("input")
        # Common input
        common_input_list = []
        for pooli in sorted(common_input_MN):
            common_input_list.append(common_input_MN[pooli]) # now this is a 2D numeric array: (n_pools, n_timepoints)
        common_input_array = np.stack(common_input_list, axis=0)
        input_grp.create_dataset("common_input", data=common_input_array)
        # Power spectrums
        #       # Total
        input_grp.attrs["total_power_common_input"] = common_input_power_total[0] # only from first pool
        input_grp.attrs["total_power_independent_input"] = independent_input_power_total
        #       # Power spectrum (per frequency)
        for input_type_i in ["common_input", "independent_input"]:
            if input_type_i == "common_input":
                power_spectrum_to_use = common_input_power_spectrum
                input_grp.create_dataset("frequencies", data=power_spectrum_to_use["frequencies"])
            else: # if input_type_i == "independent_input"
                power_spectrum_to_use = independent_input_power_spectrum
            input_grp.create_dataset(f"power_spectrum_{input_type_i}", data=power_spectrum_to_use["power"]) # only the power spectrum of the first common input is saved (for the independent input, it is the average over all MNs)

    return output_file

######################################
### RUN SIMULATION
######################################

def run_simulation(params=None):
    """
    Runs one simulation.  `params` may be:
      • None                             → use every default
      • a SimulationParameters object    → used directly
    """
    # Create new folder and get simulation index number
    if params.make_unique_output_folder:
        directory_name, sim_index = make_unique_output_dir(parent_folder=params.output_folder_name)
    else:
        directory_name = params.output_folder_name
        sim_index = os.path.basename(os.path.normpath(directory_name))
    # Initialize
    _ensure_logging() # ensure logging is configured for _this_ process
    logger = logging.getLogger(__name__)
    logger.info(f"Initializing simulation {sim_index}...")
    _set_filter_params(params)
    start_scope()  # Re-initialize Brian
    start_time = time.time()
    # # # SET PARAMETERS AND CREATE OUTPUT FOLDER
    # Get parameters
    if params is None:
        params = SimulationParameters()
    elif isinstance(params, SimulationParameters):
        params = params
    else:
        raise ValueError("run_simulation() expects a SimulationParameters object, or None")
    np.random.seed(params.random_seed)
    # write JSON with all parameters
    param_dict = asdict(params)
    param_dict.pop('RC_pair_indices', None) # Remove this value from the parameter list (it's a tuple and not readily writable in a json file)
    with open(f"{directory_name}/sim_parameters.json","w") as fp:
        json.dump(param_dict, fp, indent=2, default=str)

    # # # SET EQUATIONS
    (MN_equations, RC_equations) = set_brian2_equations()

    # # # GENERATE MOTOR NEURONS
    (motoneuron_soma_diameters, motoneuron_normalized_soma_diameters,
     pool_list_by_MN, idx_of_MN_by_pool) = generate_motor_neurons(full_pool_sim=params.full_pool_sim,
            min_soma_diameter=params.min_soma_diameter, max_soma_diameter=params.max_soma_diameter,
            nb_pools=params.nb_pools, nb_motoneurons_per_pool=params.nb_motoneurons_per_pool, total_nb_motoneurons=params.total_nb_motoneurons,
            size_distribution_exponent=params.size_distribution_exponent,
            mean_soma_diameter=params.mean_soma_diameter, sd_prct_soma_diameter=params.sd_prct_soma_diameter,
            generate_figure=params.output_plots, savepath=directory_name)
    
    # # # GENERATE MOTOR NEURONS ELECTROPHYSIOLOGICAL PROPERTIES
    (motoneurons_resistance, motoneurons_input_weight,
     motoneurons_capacitance, motoneurons_membrane_conductance,
     motoneurons_membrane_time_constant,
     motoneurons_AHP_conductance_decay_time_constant, motoneurons_refractory_periods,
     motoneurons_rheobases, motoneurons_spike_transmission_delays,
     motoneurons_properties_dict) = generate_motor_neuron_electrophysiological_properties(
        total_nb_motoneurons=params.total_nb_motoneurons, motoneuron_soma_diameters=motoneuron_soma_diameters,
        resistance_constant=params.resistance_constant, resistance_exponent=params.resistance_exponent,
        capacitance_constant=params.capacitance_constant, capacitance_exponent=params.capacitance_exponent,
        AHP_duration_constant=params.AHP_duration_constant, AHP_duration_exponent=params.AHP_duration_exponent,
        rheobase_constant=params.rheobase_constant, rheobase_exponent=params.rheobase_exponent, rheobase_scaling=params.rheobase_scaling,
        refractory_period_absolute=params.refractory_period_absolute,
        axonal_conduction_velocity_constant=params.axonal_conduction_velocity_constant, axonal_conduction_velocity_exponent=params.axonal_conduction_velocity_exponent,
        generate_figure=params.output_plots, savepath=directory_name)
    
    # # # GENERATE AND DISTRIBUTE COMMON INPUT
    (MN_excit_input, corr_mat, inputs_to_mn_weight_matrix, total_power, power_within_band_of_interest,
     power_per_frequency_band) = generate_and_distribute_common_inputs(
        fsamp=params.fsamp, nb_of_common_inputs=params.nb_of_common_inputs,
        nb_pools=params.nb_pools, nb_motoneurons_per_pool=params.nb_motoneurons_per_pool,
        duration_with_ignored_window=params.duration_with_ignored_window, edges_ignore_duration=params.edges_ignore_duration,
        frequency_range_of_common_input=params.frequency_range_of_common_input,
        common_input_std=params.common_input_std,
        frequency_range_to_set_input_power=params.frequency_range_to_set_input_power,
        set_arbitrary_correlation_between_excitatory_inputs=params.set_arbitrary_correlation_between_excitatory_inputs,
        between_pool_excitatory_input_correlation=params.between_pool_excitatory_input_correlation,
        set_same_excitatory_input_for_all_pools=params.set_same_excitatory_input_for_all_pools,
        generate_figure=params.output_plots, savepath=directory_name)
    
    # # # ADD INDEPENDENT INPUTS AND CREATE Brian2 TIMED ARRAY OBJECTS
    (input_MN_timedarray_amp, input_RC_timedarray_volt,
     total_power_independent, power_per_frequency_band_independent) = generate_independent_inputs(
        fsamp=params.fsamp,
        nb_pools=params.nb_pools, total_nb_motoneurons=params.total_nb_motoneurons, total_nb_renshaw_cells=params.total_nb_renshaw_cells,
        low_pass_filter_of_MN_independent_input=params.low_pass_filter_of_MN_independent_input,
        low_pass_filter_of_RC_independent_input=params.low_pass_filter_of_RC_independent_input,
        ref_common_input_power = power_within_band_of_interest,
        MN_independent_input_absolute_or_ratio=params.independent_input_absolute_or_ratio,
        MN_independent_input_power=params.independent_input_power,
        RC_independent_input_std=params.RC_independent_input_std,
        MN_excit_input=MN_excit_input,
        inputs_to_mn_weight_matrix=inputs_to_mn_weight_matrix,
        excitatory_input_baseline=params.excitatory_input_baseline,
        duration_with_ignored_window=params.duration_with_ignored_window,
        edges_ignore_duration=params.edges_ignore_duration,
        motoneurons_rheobases=motoneurons_rheobases,
        generate_figure=params.output_plots, savepath=directory_name)
    
    # # # IF DESIRED, DISPLAY POWER OF COMMON AND INDEPENDENT INPUTS
    if params.output_plots:
        plt.figure(figsize=(10,6))
        scaling_power = 1/1e6
        plt.fill_between(
                power_per_frequency_band_independent['frequencies'],
                power_per_frequency_band_independent['power']*scaling_power,
                y2=0, color='blue', alpha=0.3)
        plt.fill_between(
                power_per_frequency_band['frequencies'],
                power_per_frequency_band['power']*scaling_power,
                y2=0, color='red', alpha=0.3)
        plt.plot(power_per_frequency_band['frequencies'],
                 power_per_frequency_band['power']*scaling_power,
                 color='red', label=f'Common input power\ntotal={total_power[0]*scaling_power:.2f} a.u.')
        plt.plot(power_per_frequency_band_independent['frequencies'],
            power_per_frequency_band_independent['power']*scaling_power,
            color='blue', label=f'Independent input power\ntotal={total_power_independent*scaling_power:.2f} a.u.')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (a.u.)")
        plt.legend(loc="upper right")
        plt.savefig(f"{directory_name}/Power_of_common_and_independent_inputs.png")
    
    # # # CREATE CONNECTIVITY BETWEEN MOTOR NEURONS AND RENSHAW CELLS
    (MN_to_MN_connectivity_matrix,
     MN_to_Renshaw_connectivity_matrix,
     Renshaw_to_MNs_connectivity_matrix) = create_connectivity(
        total_nb_motoneurons = params.total_nb_motoneurons, total_nb_renshaw_cells = params.total_nb_renshaw_cells,
        nb_pools = params.nb_pools, nb_motoneurons_per_pool = params.nb_motoneurons_per_pool, RC_pair_indices = params.RC_pair_indices, nb_RCs_per_pool_pair = params.nb_RCs_per_pool_pair,
        disynpatic_inhib_connections_desired_MN_MN = params.disynpatic_inhib_connections_desired_MN_MN,
        split_MN_RC_ratio = params.split_MN_RC_ratio, motoneuron_normalized_soma_diameters = motoneuron_normalized_soma_diameters,
        distribution_type = params.distribution_type, distribution_params = params.distribution_params,
        disynaptic_inhib_received_arbitrary_adjustment = params.disynaptic_inhib_received_arbitrary_adjustment,
        distribution_binary_weights = params.binary_connectivity,
        generate_figure=params.output_plots, savepath=directory_name)
    # logger = logging.getLogger(__name__)
    # # # CREATE Brian2 NEURONGROUPS AND SYNAPSES OBJECTS
    # set synaptic_IPSP_decay_time_constant_per_MN
    if params.synaptic_IPSP_membrane_or_user_defined_time_constant == "user_defined":
        synaptic_IPSP_decay_time_constant_per_MN = []
        for mni in range(params.total_nb_motoneurons):
            synaptic_IPSP_decay_time_constant_per_MN.append(params.synaptic_IPSP_decay_time_constant) # params.synaptic_IPSP_decay_time_constant is already in milliseconds
    elif params.synaptic_IPSP_membrane_or_user_defined_time_constant == "membrane":
        synaptic_IPSP_decay_time_constant_per_MN = motoneurons_membrane_time_constant * ms  # motoneurons_membrane_time_constant is a list of unitless values
    else:
        raise ValueError("Please enter a valid value for synaptic_IPSP_decay_time_constant_per_MN: 'user_defined' or 'membrane'")
    # logger.info(f"Synaptic tau: {synaptic_IPSP_decay_time_constant_per_MN}")
    # actually create the objects
    (motoneurons, renshaw_cells, synapses_MN_to_Renshaw, synapses_Renshaw_to_MN) = create_neurongroups_and_synapses_objects(
            total_nb_motoneurons=params.total_nb_motoneurons, total_nb_renshaw_cells=params.total_nb_renshaw_cells,
            MN_equations=MN_equations, RC_equations=RC_equations, voltage_rest=params.voltage_rest, voltage_thresh=params.voltage_thresh,
            motoneurons_membrane_conductance=motoneurons_membrane_conductance, motoneurons_capacitance=motoneurons_capacitance,
            motoneurons_input_weight=motoneurons_input_weight, synaptic_IPSP_decay_time_constant_per_MN=synaptic_IPSP_decay_time_constant_per_MN,
            motoneurons_AHP_conductance_decay_time_constant=motoneurons_AHP_conductance_decay_time_constant, motoneurons_refractory_periods=motoneurons_refractory_periods,
            AHP_conductance_delta_after_spiking=params.AHP_conductance_delta_after_spiking, MN_to_Renshaw_excit=params.MN_to_Renshaw_EPSP, Renshaw_to_MN_inhib=params.Renshaw_to_MN_IPSP,
            scale_initial_IPSP_to_be_same_integral_regardless_of_synaptic_tau = params.scale_initial_IPSP_to_be_same_integral_regardless_of_synaptic_tau,
            tau_Renshaw=params.tau_Renshaw, MN_RC_synpatic_delay=params.MN_RC_synpatic_delay, refractory_period_RC=params.refractory_period_RC,
            MN_to_Renshaw_connectivity_matrix=MN_to_Renshaw_connectivity_matrix, Renshaw_to_MNs_connectivity_matrix=Renshaw_to_MNs_connectivity_matrix)

    # # # RUN SIMULATION
    # Initialize values
    motoneurons.v = params.voltage_rest # in mV
    renshaw_cells.v = params.voltage_rest  # Initialize membrane potential
    # Set monitors
    monitor_spikes_motoneurons = SpikeMonitor(motoneurons, record=True)
    # monitor_voltage_motoneurons = StateMonitor(motoneurons, 'v', record=True) # get the voltage trace
    monitor_spikes_renshaw_cells = SpikeMonitor(renshaw_cells, record=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    # logger.info(f"...Initialization of sim {sim_index} finished ({elapsed_time:.2f} seconds)")
    logger.info(f"Starting simulation run for sim {sim_index} (total simulated time = {params.duration_with_ignored_window:.1f} seconds)...")
    start_time = time.time()
    # RUN THE SIMULATION
    try:
        run(params.duration_with_ignored_window * second)
    except Exception as e:
        # log the full Python traceback to your simulations_progress_log.log
        logger.error("Exception during Brian2 run():\n" + traceback.format_exc())
        # re-raise so Joblib will propagate (or at least you’ll see something)
        raise
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"...simulation {sim_index} finished! ({elapsed_time:.2f} seconds)")

    spike_trains_MN, spike_trains_RC = get_spike_trains(
        spike_monitor_MN=monitor_spikes_motoneurons, spike_monitor_RC=monitor_spikes_renshaw_cells, 
        spike_transmission_delay=motoneurons_spike_transmission_delays,
        total_nb_motoneurons=params.total_nb_motoneurons, total_nb_renshaws=params.total_nb_renshaw_cells,
        edges_ignore_duration=params.edges_ignore_duration, duration_with_ignored_window=params.duration_with_ignored_window,
        motoneurons_soma_diameters=motoneuron_soma_diameters,
        generate_figure=params.output_plots, savepath=directory_name
    )
    # Saving output
    output_savefile = save_output_hdf5(directory_name=directory_name,
                    params=params,
                    motoneurons_and_pools_idx={"pool_list_by_MN": pool_list_by_MN,
                                                "idx_of_MN_by_pool": idx_of_MN_by_pool},
                    motoneurons_properties=motoneurons_properties_dict,
                    connectivity_matrix_MN_to_RC=MN_to_Renshaw_connectivity_matrix, connectivity_matrix_RC_to_MN=Renshaw_to_MNs_connectivity_matrix,
                    connectivity_matrix_MN_to_MN=MN_to_MN_connectivity_matrix,
                    spike_trains_MN=spike_trains_MN, spike_trains_RC=spike_trains_RC,
                    common_input_MN=MN_excit_input,
                    common_input_power_total=total_power, common_input_power_spectrum=power_per_frequency_band,
                    independent_input_power_total=total_power_independent, independent_input_power_spectrum=power_per_frequency_band_independent)
    logger.info(f"Data of simulation {sim_index} saved successfully to '{output_savefile}'.")

    return output_savefile