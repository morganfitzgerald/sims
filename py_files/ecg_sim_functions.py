# ECG Simulation Functions

import numpy as np
from neurodsp.utils import set_random_seed



def gaussian_function(xs, *params):
    """Gaussian fitting function.
    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters that define gaussian function.
    Returns
    -------
    ys : 1d array
        Output values for gaussian function.
    """

    ys = np.zeros_like(xs)

    for ii in range(0, len(params), 3):

        ctr, hgt, wid = params[ii : ii + 3]

        ys = ys + hgt * np.exp(-((xs - ctr) ** 2) / (2 * wid**2))

    return ys



def generate_ecg_cycle(xs, gaussian_params):
    """
    Generate ONE ECG (electrocardiogram) cycle by summing multiple Gaussian functions.

    Parameters:
    - xs (array-like): A sequence of points at which the ECG signal is to be evaluated. Typically represents time.
    - gaussian_params (list): A flat list of parameters for the Gaussian functions used to generate the ECG signal. The list should contain sets of three values (center, height, width) for each Gaussian component, thus its length should be a multiple of 3.

    Returns:
    - one_sig (array-like): The generated ECG signal, obtained by summing the values of the Gaussian functions at each point in xs. This simulates the composite waveform of an ECG signal.
    """
    waveforms = np.array(
        [
            gaussian_function(xs, *gaussian_params[i : i + 3])
            for i in range(0, len(gaussian_params), 3)
        ]
    )
    one_sig = np.sum(waveforms, axis=0)
    return one_sig


def generate_rhythmic_process(time_range, min_spike_interval):
    """
    Generate a rhythmic process signal that marks the occurrence of an event (spike) at regular intervals.

    Parameters:
    - time_range (array-like): A sequence of time points at which the rhythmic process is evaluated. Should be evenly spaced.
    - min_spike_interval (float): The minimum interval between consecutive spikes in the process. This determines the 'rhythm' of the process.

    Returns:
    - process (array-like): A binary signal representing the rhythmic process, where 1 indicates the occurrence of a spike and 0 indicates no spike. The length of 'process' matches that of 'time_range'.
    """
    rhythmic_process = np.zeros(
        len(time_range)
    )  # Initialize the process signal with zeros
    last_spike_time = -min_spike_interval  # Initialize the last spike time
    for index in range(len(time_range)):
        if time_range[index] - last_spike_time >= min_spike_interval:
            rhythmic_process[index] = 1  # Mark a spike
            last_spike_time = time_range[index]
    return rhythmic_process


def generate_rhythmic_signal(time_range, xs, min_spike_interval, gaussian_params):
    """
    Generate a composite signal that overlays an ECG cycle signal onto a rhythmic process, simulating an ECG signal triggered at regular intervals.

    Parameters:
    - time_range (array-like): A sequence of time points for the entire signal, representing the overall duration.
    - xs (array-like): A sequence of points at which the ECG cycle is evaluated, typically a subset of 'time_range' or a pattern repeated in the signal.
    - min_spike_interval (float): The minimum interval between consecutive spikes in the rhythmic process, determining the triggering of an ECG cycle.
    - gaussian_params (list): Parameters for the Gaussian functions used to generate the ECG cycle.

    Returns:
    - process (array-like): The rhythmic process signal, indicating the points in time where an ECG cycle is triggered.
    - sig (array-like): The composite signal, combining the rhythmic process with the ECG cycles. The length of 'sig' matches that of 'time_range'.
    """
    rhythmic_process = generate_rhythmic_process(
        time_range, min_spike_interval
    )  # Generate the rhythmic process
    one_sig = generate_ecg_cycle(xs, gaussian_params)  # Generate one ECG cycle
    signal_length = len(time_range)
    ecg_cycle_length = len(one_sig)
    rhythmic_sig = np.zeros(signal_length)  # Initialize the composite signal
    trigger_indices = np.where(rhythmic_process == 1)[
        0
    ]  # Find indices where ECG cycles are triggered
    for index in trigger_indices:
        if (
            index + ecg_cycle_length <= signal_length
        ):  # Ensure the ECG cycle fits within the signal length
            rhythmic_sig[index : index + ecg_cycle_length] = (
                one_sig  # Overlay the ECG cycle onto the composite signal
            )

    return rhythmic_process, rhythmic_sig


def generate_poisson_process(time_range, rate_parameter, min_spike_interval, seed):
    """
    Generate a Poisson process with an optional refractory period.

    Parameters:
    - time_range: array-like, the time points at which to generate the Poisson process.
    - rate_parameter: float, the average rate (events per unit time) of the process.
    - min_spike_interval: float, the minimum interval between events to simulate a refractory period.
    - seed: int, the seed for the random number generator to ensure reproducibility.

    Returns:
    - poisson_process: array-like, indicating the occurrence of events at each time point in time_range.
    """
    # Set the random seed for reproducibility
    set_random_seed(seed)

    # Initialize an array to store the Poisson process
    poisson_process = np.zeros(len(time_range))

    # Generate Poisson process with refractory period
    last_spike_time = -min_spike_interval
    for i in range(len(time_range)):
        if (
            np.random.poisson(rate_parameter) > 0
            and (time_range[i] - last_spike_time) >= min_spike_interval
        ):
            poisson_process[i] = 1
            last_spike_time = time_range[i]

    return poisson_process


def generate_poisson_signal(
    time_range, rate_parameter, min_spike_interval, seed, xs, gaussian_params
):
    """
    Generate a composite signal that overlays an ECG cycle signal onto a Poisson process, simulating an ECG signal triggered at random intervals.

    Parameters:
    - time_range: array-like, the time points for the entire signal.
    - rate_parameter: float, the average rate (events per unit time) of the Poisson process.
    - min_spike_interval: float, the minimum interval between events to simulate a refractory period.
    - seed: int, the seed for random number generation to ensure reproducibility.
    - xs: array-like, the sequence of points at which the ECG cycle is evaluated.
    - gaussian_params: list, parameters for the Gaussian functions used to generate the ECG cycle.

    Returns:
    - poisson_process: array-like, indicating the occurrence of events at each time point in time_range.
    - poisson_sig: array-like, the composite signal combining the Poisson process with the ECG cycles.
    """
    # Generate the Poisson process
    poisson_process = generate_poisson_process(
        time_range, rate_parameter, min_spike_interval, seed
    )
    # Generate one ECG cycle
    one_sig = generate_ecg_cycle(xs, gaussian_params)
    signal_length = len(time_range)
    ecg_cycle_length = len(one_sig)
    poisson_sig = np.zeros(signal_length)  # Initialize the composite signal
    trigger_indices = np.where(poisson_process == 1)[
        0
    ]  # Find indices where ECG cycles are triggered
    for index in trigger_indices:
        if (
            index + ecg_cycle_length <= signal_length
        ):  # Ensure the ECG cycle fits within the signal length
            poisson_sig[index : index + ecg_cycle_length] = (
                one_sig  # Overlay the ECG cycle onto the composite signal
            )

    return poisson_process, poisson_sig
