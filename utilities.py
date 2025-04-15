# Import the OS module for interacting with the operating system.
# This is commonly used for file and directory manipulations, as well as environment variable access.
import os

# Import the JSON module to handle JSON serialization and deserialization.
# Useful for encoding Python objects to JSON strings and decoding JSON strings back to Python objects.
import json

# Import the threading module which provides tools to run code concurrently in separate threads.
# This is important for multi-threaded applications where tasks can run in parallel.
import threading

# Import NumPy, a powerful library for numerical computations, particularly when working with arrays and matrices.
import numpy as np

# Import the main matplotlib library for creating static, animated, and interactive visualizations.
import matplotlib
# Set matplotlib to use a non-GUI backend ('Agg') which is useful for generating plots in environments 
# without a display (e.g., servers or automated scripts). This ensures that plots are rendered without a graphical interface.
matplotlib.use('Agg')  # Use a non-GUI backend

# Import the pyplot module from matplotlib which provides a MATLAB-like interface for plotting.
import matplotlib.pyplot as plt
# Import the Axes class from matplotlib.axes for type annotations and advanced axis handling.
from matplotlib.axes import Axes

# Import the pywt module for performing wavelet transforms.
# PyWavelets is used for signal processing tasks such as denoising and feature extraction using wavelets.
import pywt

# Import Seaborn, a statistical data visualization library built on top of matplotlib.
# Seaborn simplifies the creation of attractive visualizations with default styles and color palettes.
import seaborn as sns

# Import various signal processing functions from scipy.signal.
# These functions include:
# - hilbert: For computing the analytic signal via the Hilbert transform.
# - welch: For estimating a signal’s power spectral density using Welch's method.
# - butter, cheby1, cheby2, ellip: For designing different types of digital filters.
# - filtfilt: For applying a filter forward and backward to prevent phase distortion.
# - firwin: For designing Finite Impulse Response (FIR) filters.
# - find_peaks: For identifying local peaks in a signal.
# - spectrogram: For computing a spectrogram of a signal.
from scipy.signal import hilbert, welch, butter, cheby1, cheby2, ellip, filtfilt, firwin, find_peaks, spectrogram

# Import statistical functions from scipy.stats:
# - kurtosis: To measure the tailedness of the probability distribution of a signal.
# - skew: To measure the asymmetry of the probability distribution of a signal.
from scipy.stats import kurtosis, skew

# Import type hints from the typing module.
# These are used to provide type annotations that help with code readability and static type checking.
from typing import Any, Optional, List, Dict, Union

# Import the OpenAI module which provides a client to interact with OpenAI's API.
from openai import OpenAI

# Create an instance of the OpenAI client.
# The client is initialized with an API key, which is retrieved from the environment variable "OPENAI_API_KEY".
# This API key is required to authenticate requests to the OpenAI service.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Define a function to generate content using the OpenAI chat completions API.
def generate_content_openai(user_prompt: str,
                            model: str = "gpt-4o",  # Default model set to "gpt-4o".
                            system_prompt: str = "You are a helpful assistant, replying in json",  # Default system instruction.
                            response_format: Optional[str] = None,  # Optionally define the response format.
                            web_search_options: Optional[Dict] = None,  # Optionally provide web search options.
                            temperature: float = 0.1) -> Any:  # Default temperature set for API output randomness.
    """
    Generate content using the OpenAI chat completions API.

    This function constructs a chat conversation with a system prompt and a user prompt,
    and sends a request to OpenAI's chat completions API through a global `client` object.
    It conditionally includes web search options if provided and returns the parsed JSON 
    output from the response.

    Parameters:
        user_prompt (str): The user-provided prompt.
        model (str): The model identifier for generating content (default "gpt-4o").
        system_prompt (str): System-level message setting the assistant's behavior.
        response_format (Optional[str]): The expected format of the response.
        web_search_options (Optional[Dict]): Additional options for web search integration.
        temperature (float): Sampling temperature that influences randomness (default 0.1).

    Returns:
        Any: The parsed JSON output from the API response.
    """
    # Create a list called 'message' containing dictionaries for system and user messages.
    message = [
        {"role": "system", "content": system_prompt},  # Add the system prompt with key "role" set to "system".
        {"role": "user", "content": user_prompt}         # Add the user prompt with key "role" set to "user".
    ]

    # Check if web_search_options is provided (i.e., not None or empty).
    if web_search_options:
        # If web search options exist, call the OpenAI API with web_search_options included.
        response = client.chat.completions.create(
            model=model,  # Specify the model to use.
            web_search_options=web_search_options,  # Pass the web search options parameter.
            messages=message,  # Pass the constructed messages list.
            response_format=response_format  # Pass the desired response format.
        )
    else:
        # If no web search options are provided, call the API without them.
        response = client.chat.completions.create(
            model=model,  # Specify the model to use.
            messages=message,  # Pass the constructed messages list.
            response_format=response_format,  # Pass the desired response format.
            temperature=temperature  # Include the temperature parameter for output randomness.
        )
    
    # Parse the JSON string from the API response's first message and return it as a Python object.
    return json.loads(response.choices[0].message.content)

# Define a function to generate a JSON schema for the expected response format.
def generate_response_format(keys: List[str], values: List[str]) -> Dict[str, Any]:
    """
    Generate a response format specification using a JSON schema.

    This function creates a dictionary specifying a response format for verifying
    and validating API responses according to a predefined schema. The schema includes
    properties defined by the provided keys and their corresponding descriptions, while
    ensuring that at least one or two keys are required.

    Parameters:
        keys (List[str]): A list of keys for the response properties.
        values (List[str]): A list of descriptions corresponding to each key.

    Returns:
        Dict[str, Any]: A dictionary representing the response format with a JSON schema.
    """
    response_format = {}  # Initialize an empty dictionary for the response format.
    response_format["type"] = "json_schema"  # Define the type of response format as "json_schema".

    json_schema = {}  # Initialize an empty dictionary for the JSON schema.
    json_schema["name"] = "subheadings_schema"  # Set the name of the schema.

    schema = {}  # Initialize an empty dictionary for the schema structure.
    schema["type"] = "object"  # Specify that the schema will validate an object (dictionary).

    properties = {}  # Initialize an empty dictionary to hold the properties for the schema.

    # Loop over each pair of key and value from the provided lists using zip.
    for key, value in zip(keys, values):
        # For each key, create a property with its description and type.
        properties[key] = {"description": value, "type": "string"}  # Each property is expected to be a string.

    schema["properties"] = properties  # Assign the properties dictionary to the schema.

    # Determine the required keys: use the first two keys if at least two keys are provided; otherwise, use the first key.
    schema["required"] = [keys[0], keys[1]] if len(keys) >= 2 else [keys[0]]
    schema["additionalProperties"] = False  # Do not allow properties other than those defined.

    json_schema["schema"] = schema  # Embed the schema inside the json_schema dictionary.
    response_format["json_schema"] = json_schema  # Set the json_schema key in the response_format dictionary.

    return response_format  # Return the completed response format dictionary.

# Define a function to retrieve values from a nested dictionary or JSON string.
def get_values(df: Union[str, Dict[Any, Any]]) -> Any:
    """
    Retrieve values from a nested dictionary or JSON string.

    This function attempts to parse a JSON string into a dictionary, and recursively extracts
    values. If the dictionary contains only one key, it drills down; otherwise, it returns a list
    of values for all keys.

    Parameters:
        df (Union[str, Dict[Any, Any]]): A dictionary or its JSON string representation.

    Returns:
        Any: A single value or a list of values extracted from the nested dictionary.
    """
    # Check if df is a string.
    if isinstance(df, str):
        try:
            df = json.loads(df)  # Attempt to parse the string as JSON.
        except Exception:
            pass  # If parsing fails, silently continue with the original string.
    # Check if df is now a dictionary.
    if isinstance(df, dict):
        keys = list(df.keys())  # Get the list of keys from the dictionary.
        # If the dictionary has only one key, recursively extract its value.
        if len(keys) == 1:
            return get_values(df[keys[0]])
        else:
            # Otherwise, return a list of values corresponding to all keys in the dictionary.
            return [df[key] for key in keys]
    else:
        # If df is neither a string nor a dictionary, return it as is.
        return df


# Signals Functions & Classes (20250414)
def safe_signal(signal: np.ndarray) -> np.ndarray:
    """
    Replace problematic numerical values in a signal array with safe defaults.
    
    This function converts any NaN (Not a Number), positive infinity, or negative infinity
    values in the input numpy array into 0.0. This is useful for ensuring that subsequent
    computations are not affected by invalid numerical entries.
    
    Parameters
    ----------
    signal : np.ndarray
        A numpy array representing the signal data which may contain NaNs or infinities.
    
    Returns
    -------
    np.ndarray
        A numpy array with all NaNs and infinities replaced by 0.0.
    """
    # Use np.nan_to_num to replace NaN with 0.0, positive infinity with 0.0, and negative infinity with 0.0.
    return np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

def is_all_zeros_or_has_nan_or_large(signal: np.ndarray, atol: float = 1e-12, max_val: Optional[float] = None) -> bool:
    """
    Determine if the signal array is either entirely zero, contains NaNs, or exceeds a maximum value.
    
    The function first checks for any NaN values in the signal. If any are found, it returns True.
    If a 'max_val' is provided, it then checks whether the maximum absolute value of the signal
    exceeds this value. If no 'max_val' is provided, it finally checks whether every element 
    in the signal is close to zero within a specified absolute tolerance.
    
    Parameters
    ----------
    signal : np.ndarray
        A numpy array representing the signal data.
    atol : float, optional
        The absolute tolerance to consider the signal as being all zeros (default is 1e-12).
    max_val : Optional[float], optional
        A threshold value for the maximum absolute value allowed in the signal. If any value 
        exceeds this, the function returns True. Defaults to None, in which case the all-zeros 
        check is performed.
    
    Returns
    -------
    bool
        True if the signal either contains NaNs, (if max_val is provided) any element's absolute 
        value exceeds max_val, or (if max_val is not provided) the signal is approximately zero.
    """
    # Check if there are any NaN values in the signal array.
    if np.isnan(signal).any():
        # If a NaN is found, immediately return True.
        return True
    
    # If a maximum allowable value is specified...
    if max_val is not None:
        # Compute the maximum of the absolute values in the signal.
        # If this maximum exceeds the given max_val, return True; otherwise, return False.
        return max(np.abs(signal)) > max_val

    # If no maximum value constraint is given, check if all elements in the signal are approximately zero.
    # np.allclose compares each element in the signal to 0 within the specified tolerance (atol).
    return np.allclose(signal, 0, atol=atol)

def mag_to_db(signal: np.ndarray) -> np.ndarray:
    """
    Convert a magnitude signal to decibel (dB) scale.
    
    This function applies a decibel conversion to each element in the input magnitude array using
    the formula: dB = 20 * log10(magnitude). It uses np.maximum to ensure that the input to the logarithm
    is never less than 1e-12 in order to avoid taking the log of zero or negative numbers.
    
    Parameters
    ----------
    signal : np.ndarray
        A numpy array containing the magnitude values.
    
    Returns
    -------
    np.ndarray
        A numpy array with the magnitudes converted to decibels.
    """
    # Ensure that every element in the signal is at least 1e-12 to prevent issues with logarithm
    # then apply the logarithmic conversion formula to convert magnitude values to decibels.
    return 20 * np.log10(np.maximum(signal, 1e-12))

def idx_to_freq(idx: int, fs: int, n: int) -> float:
    """
    Convert an index from the frequency domain to its corresponding frequency in Hertz.
    
    The conversion is based on the sampling frequency and the number of points in a Fourier transform.
    
    Parameters
    ----------
    idx : int
        The index (or bin number) from the frequency domain (e.g., output from an FFT).
    fs : int
        The sampling frequency in Hertz of the original signal.
    n : int
        The total number of points in the Fourier transform (FFT).
    
    Returns
    -------
    float
        The frequency in Hertz corresponding to the provided index.
    """
    # Multiply the index by the sampling frequency and divide by the total number of FFT points
    # to obtain the corresponding frequency.
    # The result is cast to float for consistency.
    return float(idx * fs / n)

def freq_to_idx(freq: int, fs: int, n: int) -> int:
    """
    Convert a frequency in Hertz to its corresponding index in the frequency domain.
    
    This function computes the index (or bin number) in a Fourier transform that corresponds to 
    a given frequency, using the sampling frequency and the total number of FFT points.
    
    Parameters
    ----------
    freq : int
        The frequency in Hertz to be converted.
    fs : int
        The sampling frequency in Hertz of the original signal.
    n : int
        The total number of points in the Fourier transform (FFT).
    
    Returns
    -------
    int
        The corresponding index in the frequency domain for the given frequency.
    """
    # Multiply the given frequency by the total number of FFT points and divide by the sampling frequency.
    # Convert the result to an integer to obtain the bin index.
    return int(freq * n / fs)

class RollingDefectAnalysis:
    """
    A class for performing rolling defect analysis on a signal.

    This class implements various filtering, frequency domain analysis, envelope extraction, 
    and plotting functions to analyze rolling element bearing defects based on a signal and 
    provided defect frequency parameters. The analysis includes options for downsampling, 
    filtering using different methods, computing the signal envelope, frequency analysis 
    via FFT or Welch, and visualization (time domain, frequency domain, spectrogram).
    """

    # Class-level default plotting styles and parameters.
    _linestyles = ['-', '-', '-', '-']  # List of line styles for plotting defect markers.
    _colors     = ['r', 'b', 'm', 'g']    # List of colors for defect plots.
    _alphas     = [0.6, 0.6, 0.7, 0.7]    # List of alpha (transparency) values for defect plots.
    _figsize = (15, 20)                   # Default figure size for frequency domain plots.
    _lock = threading.Lock()              # Threading lock to ensure thread-safe plotting.

    def __init__(self,
                 signal: np.ndarray,
                 fs: int,
                 rpm: int,
                 bpfi: float,
                 bpfo: float,
                 ftf: float,
                 bsf: float,
                 title: str,
                 filter_type: str = "butterworth",
                 use_walch: bool = False,
                 peak_dividant: int = 25,
                 downsample_factor: int = 1,
                 band: tuple | None = None) -> None:
        """
        Initialize the RollingDefectAnalysis object with the signal and analysis parameters.

        Parameters
        ----------
        signal : np.ndarray
            The input signal to be analyzed.
        fs : int
            The sampling frequency of the input signal.
        rpm : int
            The rotations per minute (RPM) of the machinery.
        bpfi : float
            Ball Pass Frequency of the Inner race defect (scaling factor).
        bpfo : float
            Ball Pass Frequency of the Outer race defect (scaling factor).
        ftf : float
            Fundamental Train Frequency defect factor.
        bsf : float
            Ball Spin Frequency defect factor.
        title : str
            Title for the analysis and plots.
        filter_type : str, optional
            The type of filter to use; default is "butterworth".
        use_walch : bool, optional
            Flag to use Welch's method for frequency analysis if True; default is False.
        peak_dividant : int, optional
            Dividing factor to determine threshold for frequency peak detection; default is 25.
        downsample_factor : int, optional
            Factor by which to downsample the signal; default is 1 (no downsampling).
        band : tuple or None, optional
            Frequency band to be used for filtering; if None, the default band is computed.
        """
        # Replace invalid values in the signal (NaNs, inf) with safe numbers (0.0)
        self.signal = safe_signal(signal)
        # Store the downsample factor.
        self.downsample_factor = downsample_factor
        # Downsample the signal if factor is not 1.
        if downsample_factor != 1:
            # Update the signal to its downsampled version using the _downsample_signal method.
            self.signal = safe_signal(self._downsample_signal())
        # Determine the number of samples in the processed signal.
        self.n = len(self.signal)
        # Adjust the sampling frequency according to the downsampling factor.
        self.fs = int(fs / downsample_factor)
        # Create a time array based on the number of samples and the adjusted sampling frequency.
        self.time = np.arange(self.n) / self.fs
        # Calculate the rotation frequency from RPM.
        self.fr = rpm / 60.0
        # Calculate the defect frequencies scaled by the rotation frequency.
        self.defects = dict(bpfi=bpfi * self.fr,
                            bpfo=bpfo * self.fr,
                            ftf=ftf * self.fr,
                            bsf=bsf * self.fr)
        # Set up the harmonic frequency scaling factors for each defect category.
        self.harmonics = dict(bpfo=ftf * self.fr,
                              bpfi=self.fr,
                              bsf=ftf * self.fr,
                              ftf=self.fr)  # Note: For bsf it could be ftf, fr, or bpfo.
        # Store the peak dividant value for frequency peak detection.
        self.peak_dividant = peak_dividant
        # Store the flag indicating whether to use Welch's method.
        self.use_walch = use_walch
        # Store the title for the analysis.
        self.title = title
        # Store the chosen filter type.
        self.filter_type = filter_type
        # Set the frequency band; if not provided (None), compute a default band.
        self.band = band or self._default_band()

        # Apply the filtering method to the signal.
        self.signal_filtered = self._filter_signal(self.signal)

        # If initial filtering produces a None result, try wavelet denoising using 'db4'.
        if self.signal_filtered is None:
            self.filter_type = 'db4'
            self.signal_filtered = self._filter_signal(self.signal)
        # If filtering still fails, raise an error to indicate the signal cannot be processed.
        if self.signal_filtered is None:
            raise NotImplementedError("The signal cannot be processed with this app (finer downsampling?)")

        # Compute the envelope of the filtered signal using the Hilbert transform.
        self.envelope = self._envelope_signal()
        # Compute the frequency and magnitude arrays for the raw signal.
        self.signal_freq, self.signal_magn = self._get_freq(self.signal, use_welch=self.use_walch)
        # Compute the frequency and magnitude arrays for the filtered signal.
        self.signal_filtered_freq, self.signal_filtered_magn = self._get_freq(self.signal_filtered, use_welch=self.use_walch)
        # Compute the frequency and magnitude arrays for the envelope of the signal.
        self.envelope_freq, self.envelope_magn = self._get_freq(self.envelope, use_welch=self.use_walch)

        # Identify frequency peaks from the magnitude spectra for the raw, filtered, and envelope signals.
        self.freq_peaks = dict(raw=self._get_freq_picks(self.signal_magn),
                               filtered=self._get_freq_picks(self.signal_filtered_magn),
                               envelope=self._get_freq_picks(self.envelope_magn))

        # Construct a title string and a filename string for saving plots, including filtering and FFT method info.
        self._title = f" | {'WALCH' if self.use_walch else 'RFFT'} | {self.filter_type}".upper()
        self._filename = f"_{'walch' if self.use_walch else 'rfft'}_{self.filter_type}".lower()

    def _downsample_signal(self, signal: np.ndarray | None = None) -> np.ndarray:
        """
        Downsample the input signal using a decimation method.

        Parameters
        ----------
        signal : np.ndarray or None, optional
            The signal to downsample. If None, the object's current signal is used.

        Returns
        -------
        np.ndarray
            The downsampled signal.
        """
        # If no signal is provided, default to using the object's current signal.
        if signal is None:
            signal = self.signal
        # Downsample the signal by the given downsample factor using the decimate function.
        return decimate(signal, self.downsample_factor)

    def _change_and_filter(self, signal: np.ndarray | None = None) -> np.ndarray | None:
        """
        Try alternative filter types to filter the signal if the current filter fails.

        This method cycles through a list of alternative filter types (excluding the current one)
        and applies them until a successful filtered signal is produced.

        Parameters
        ----------
        signal : np.ndarray or None, optional
            The signal to filter; if None, the object's current signal is used.

        Returns
        -------
        np.ndarray or None
            The filtered signal if a working filter is found; otherwise, None.
        """
        # Use the object's signal if none is provided.
        if signal is None:
            signal = self.signal
        # Define a list of alternative filter types.
        other_filters = ["butterworth", "chebyshev1", "chebyshev2", "elliptic", "fir", "db4"]
        # Remove the current filter type from the list.
        other_filters.remove(self.filter_type)
        # Iterate through the alternative filters.
        for other_filter in other_filters:
            # Set the filter type to the current candidate.
            self.filter_type = other_filter
            # Try to filter the signal using the new filter type.
            filtered_signal = self._filter_signal(signal)
            # If successful, return the filtered signal.
            if filtered_signal is not None:
                return filtered_signal
        # If none of the alternative filters produce a valid result, return None.
        return None

        # The following line is unreachable because it comes after a return statement.
        filters = ["chebyshev1", "chebyshev2", "elliptic", 'fir', "db4"]

    def _default_band(self) -> tuple:
        """
        Compute a default frequency band based on defect frequency values.

        The lower bound is set to half the minimum defect frequency, and the upper bound
        to twice the maximum defect frequency.

        Returns
        -------
        tuple
            A tuple representing the default (low, high) frequency band.
        """
        # Calculate lower bound: half of the smallest defect frequency.
        # Calculate upper bound: twice the largest defect frequency.
        return (0.5 * min(self.defects.values()),
                2 * max(self.defects.values()))

    def _envelope_signal(self) -> np.ndarray:
        """
        Compute the envelope of the filtered signal using the Hilbert transform.

        Returns
        -------
        np.ndarray
            The absolute value of the Hilbert transform applied to the filtered signal.
        """
        # Use the Hilbert transform to obtain the analytic signal and take its absolute value.
        return np.abs(hilbert(self.signal_filtered))

    def _get_freq(self, signal: np.ndarray, use_welch: bool = False) -> tuple:
        """
        Compute the frequency bins and corresponding magnitudes of a signal.

        This method offers two approaches: either by applying Welch's method for spectral estimation
        or by using a standard FFT (rfft).

        Parameters
        ----------
        signal : np.ndarray
            The input signal for which to compute frequency information.
        use_welch : bool, optional
            If True, Welch's method is used; otherwise, FFT is employed.

        Returns
        -------
        tuple
            A tuple (freq, magnitude) where freq is an array of frequency bins and 
            magnitude is an array of spectral magnitudes.
        """
        if use_welch:
            # Use Welch's method for spectral density estimation.
            freq, p = welch(signal, fs=self.fs, window='hann',
                            nperseg=2048, noverlap=1024, scaling='spectrum')
            # Return frequency bins and the safe square-root of the power spectral density.
            return freq, safe_signal(np.sqrt(p))
        else:
            # Compute the FFT frequencies corresponding to a real FFT.
            freq = np.fft.rfftfreq(self.n, d=1 / self.fs)
            # Return the frequency bins and the absolute values of the FFT result.
            return freq, safe_signal(np.abs(np.fft.rfft(signal)))

    def _get_freq_picks(self, signal_magn: np.ndarray) -> dict:
        """
        Identify prominent frequency peaks in a magnitude spectrum.

        This method limits the frequency range based on the defect frequencies, computes a threshold,
        and then finds peaks that exceed a fraction of the maximum magnitude.

        Parameters
        ----------
        signal_magn : np.ndarray
            The magnitude spectrum from which to pick peaks.

        Returns
        -------
        dict
            A dictionary containing 'frequency' (array of frequencies) and 'magnitudes'
            (peak heights) for the identified peaks.
        """
        # Define an upper frequency limit as 2.05 times the maximum defect frequency.
        fmax = 2.05 * max(self.defects.values())
        # Convert the frequency limit to an index using fs and signal length.
        idx = freq_to_idx(fmax, self.fs, self.n)
        # Limit the magnitude spectrum to the computed frequency range.
        signal_magn_limited = signal_magn[:idx]
        # Set a threshold: any magnitude higher than max divided by peak_dividant is of interest.
        mmax = signal_magn_limited.max() / self.peak_dividant
        # Find peaks that exceed the threshold using find_peaks.
        idxs, magns = find_peaks(signal_magn_limited, height=mmax)
        # Convert the indices of the peaks to actual frequency values.
        freqs = np.array([idx_to_freq(idx, self.fs, self.n) for idx in idxs])
        # Return a dictionary containing the detected frequencies and their peak magnitudes.
        return dict(frequency=freqs, magnitudes=magns['peak_heights'])

    def _legend(self, ax: plt.gca) -> None:
        """
        Create and apply a sorted legend to the given plot axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis on which to place the legend.
        """
        # Retrieve current legend handles and labels from the axis.
        handles, labels = ax.get_legend_handles_labels()
        # Sort the (label, handle) pairs alphabetically based on the label.
        sorted_handles_labels = sorted(zip(labels, handles), key=lambda pair: pair[0])
        # Unzip the sorted pairs into separate lists for labels and handles.
        sorted_labels, sorted_handles = zip(*sorted_handles_labels)
        # Create the legend on the axis with the sorted handles and labels.
        plt.legend(sorted_handles, sorted_labels,
                   ncol=len(self.defects),
                   loc="upper center",
                   bbox_to_anchor=(0.5, -0.2),
                   fontsize="small",
                   handlelength=2,
                   columnspacing=1.2,
                   frameon=True)

    def _plot_defects(self, ax: plt.gca, direction: str = 'vertical', alpha: float | None = 0.9, defect: str = None) -> None:
        """
        Plot defect lines on the provided axis.

        For each defect in the defects dictionary, this method plots a vertical or horizontal line 
        at the defect frequency using the default style parameters. Optionally, a specific defect
        can be highlighted.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis on which to plot the defect lines.
        direction : str, optional
            Direction to draw the line; 'vertical' for axvline and other for axhline (default is 'vertical').
        alpha : float or None, optional
            Transparency value for the defect lines; if None, default alpha from class attributes is used.
        defect : str or None, optional
            A specific defect type to plot. If None, all defects are plotted.
        """
        # Iterate over each defect in the defects dictionary along with styling parameters.
        for (label, freq), c, a, ls in zip(self.defects.items(), self._colors, self._alphas, self._linestyles):
            # Check if we are plotting a specific defect or all defects.
            if label == defect or defect is None:
                if direction == 'vertical':
                    # Plot a vertical line at the defect frequency.
                    ax.axvline(freq, ls=ls, c=c, lw=9, alpha=a if alpha is None else alpha, label=label.upper())
                else:
                    # Plot a horizontal line at the defect frequency.
                    ax.axhline(freq, ls=ls, c=c, lw=9, alpha=a if alpha is None else alpha, label=label.upper())

    def _plot_harmonics(self, ax: plt.gca, direction: str = 'vertical', alpha: float | None = 0.9, defect: str = None) -> None:
        """
        Plot harmonic lines for each defect on the provided axis.

        For each defect, this method plots the 2nd to 4th harmonics as additional lines with adjusted
        styles and transparency.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis on which to plot the harmonic lines.
        direction : str, optional
            Direction to draw the lines; 'vertical' for axvline and otherwise for axhline (default is 'vertical').
        alpha : float or None, optional
            Base transparency value for the harmonic lines; adjusted for each harmonic order.
        defect : str or None, optional
            Specific defect type to plot; if None, all defects are plotted.
        """
        # Iterate over each defect and associated style parameters.
        for (label, freq), c, a, ls in zip(self.defects.items(), self._colors, self._alphas, self._linestyles):
            if label == defect or defect is None:
                # Plot harmonics from the 2nd to 4th (i.e., n = 2, 3, 4).
                for n in range(2, 5):
                    if direction == 'vertical':
                        # For vertical direction, plot a line at n times the defect frequency.
                        ax.axvline(n * freq, ls="-", c=c, lw=9 - n, alpha=a if alpha is None else alpha - (n - 1) / 10,
                                   label=label.upper() + f"_har_{n}".upper())
                    else:
                        # For horizontal direction, plot the line accordingly.
                        ax.axhline(n * freq, ls="-", c=c, lw=9 - n, alpha=a if alpha is None else alpha - (n - 1) / 10,
                                   label=label.upper() + f"_har_{n}".upper())

    def _plot_modulations(self, ax: plt.gca, direction: str = 'vertical', alpha: float | None = 0.9, defect: str = None) -> None:
        """
        Plot modulation lines for each defect on the provided axis.

        This method calculates modulated frequencies by adding and subtracting harmonics to/from the
        defect frequency and plots them as either vertical or horizontal lines.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis on which to plot the modulation lines.
        direction : str, optional
            Direction to draw the lines; 'vertical' for vertical lines, otherwise horizontal (default is 'vertical').
        alpha : float or None, optional
            Base transparency for the modulation lines; adjusted for each modulation order.
        defect : str or None, optional
            Specific defect type to target; if None, modulations for all defects are plotted.
        """
        # Iterate through each defect using its styling parameters.
        for (label, _), c, a, ls in zip(self.defects.items(), self._colors, self._alphas, self._linestyles):
            if label == defect or defect is None:
                # For each defect, plot modulation lines for n = 1 to 4.
                for n in range(1, 5):
                    # Calculate the positive modulated frequency.
                    freq_p = max(self.defects[label] + n * self.harmonics[label], 0)
                    # Calculate the negative modulated frequency.
                    freq_n = max(self.defects[label] - n * self.harmonics[label], 0)
                    if direction == 'vertical':
                        # Plot positive modulation line if frequency is non-zero.
                        if freq_p:
                            ax.axvline(freq_p, ls='-', c=c, lw=5 - n,
                                       alpha=a if alpha is None else alpha - (n - 1) / 10,
                                       label=label.upper() + f"_mod_pos_{n}".upper())
                        # Plot negative modulation line if frequency is non-zero.
                        if freq_n:
                            ax.axvline(freq_n, ls='-', c=c, lw=5 - n,
                                       alpha=a if alpha is None else alpha - (n - 1) / 10,
                                       label=label.upper() + f"_mod_neg_{n}".upper())
                    else:
                        # For horizontal plotting: similar logic for positive frequency modulation.
                        if freq_p:
                            ax.axhline(freq_p, ls='-', c=c, lw=5 - n,
                                       alpha=a if alpha is None else alpha - (n - 1) / 10,
                                       label=label.upper() + f"_mod_pos_{n}".upper())
                        # For horizontal plotting: negative frequency modulation.
                        if freq_n:
                            ax.axhline(freq_n, ls='-', c=c, lw=5 - n,
                                       alpha=a if alpha is None else alpha - (n - 1) / 10,
                                       label=label.upper() + f"_mod_neg_{n}".upper())

    def _filter_signal(self, signal: np.ndarray) -> np.ndarray | None:
        """
        Filter the input signal using the specified filter type and frequency band.

        Depending on self.filter_type, this method applies different filtering algorithms including:
        Butterworth, Chebyshev type I & II, Elliptic, FIR, and wavelet denoising with 'db' filters.
        A helper function is used to decide between zero-phase and causal filtering approaches.

        Parameters
        ----------
        signal : np.ndarray
            The input signal to be filtered.

        Returns
        -------
        np.ndarray or None
            The filtered signal if successful; otherwise, None.
        """
        # Ensure the input is a numpy array.
        signal = np.asarray(signal)
        # If the signal array is empty, return it directly.
        if signal.size == 0:
            return signal

        # Retrieve the band limits (low, high) for the filter.
        low, high = self.band
        # Compute the Nyquist frequency based on the sampling frequency.
        nyquist = 0.5 * self.fs

        # Clamp the low cutoff frequency to avoid values <= 0.
        if low <= 0:
            low = 1e-6
        # Clamp the high cutoff frequency to avoid reaching the Nyquist frequency.
        if high >= nyquist:
            high = nyquist - 1e-6

        # Ensure that the low cutoff frequency is less than the high cutoff.
        if low >= high:
            raise ValueError("Invalid frequency band: low cutoff must be less than high cutoff.")

        # Normalize the cutoff frequencies with respect to the Nyquist frequency.
        wn = [low / nyquist, high / nyquist]

        # Define a helper function to apply filtering and choose between zero-phase and causal approaches.
        def apply_filter(b, a, sig):
            # First, attempt zero-phase filtering using filtfilt.
            filtered1 = filtfilt(b, a, sig)
            # Also perform causal filtering using lfilter as a fallback.
            filtered2 = lfilter(b, a, sig)
            
            # If filtered1 returns a near-zero, NaN, or large value but filtered2 is acceptable, use filtered2.
            if is_all_zeros_or_has_nan_or_large(filtered1, max_val=self.signal.max()) and not is_all_zeros_or_has_nan_or_large(filtered2, max_val=self.signal.max()):
                return filtered2
            # If both filtering methods result in near-zero, NaN, or large values, return None.
            elif is_all_zeros_or_has_nan_or_large(filtered1, max_val=self.signal.max()) and is_all_zeros_or_has_nan_or_large(filtered2, max_val=self.signal.max()):
                return None
            else:
                # Otherwise, use the zero-phase filtered output.
                return filtered1

        # Apply filtering based on the current filter type.
        if self.filter_type == "butterworth":
            # Design a 4th order Butterworth bandpass filter.
            b, a = butter(4, wn, btype='bandpass')
            # Apply the filter using the helper function.
            return apply_filter(b, a, signal)

        elif self.filter_type == "chebyshev1":
            # Design a Chebyshev Type I bandpass filter with 1 dB ripple.
            b, a = cheby1(N=4, rp=1, Wn=wn, btype='bandpass')
            return apply_filter(b, a, signal)

        elif self.filter_type == "chebyshev2":
            # Design a Chebyshev Type II bandpass filter with 40 dB stopband attenuation.
            b, a = cheby2(N=4, rs=40, Wn=wn, btype='bandpass')
            return apply_filter(b, a, signal)

        elif self.filter_type == "elliptic":
            # Design an Elliptic bandpass filter with 1 dB ripple and 40 dB stopband attenuation.
            b, a = ellip(N=4, rp=1, rs=40, Wn=wn, btype='bandpass')
            return apply_filter(b, a, signal)

        elif self.filter_type == 'fir':
            # Determine the number of taps for the FIR filter based on fs.
            numtaps = int(max(101, self.fs // 5))
            # Create the FIR filter coefficients using firwin.
            taps = firwin(numtaps, wn, pass_zero=False)
            return apply_filter(taps, [1.0], signal)

        elif self.filter_type.startswith('db'):
            # Wavelet denoising: perform wavelet decomposition and apply soft thresholding.
            coefficients = pywt.wavedec(signal, self.filter_type, level=None)
            # Estimate noise level using the median absolute deviation on the last coefficient.
            sigma = np.median(np.abs(coefficients[-1])) / 0.6745
            # Calculate the threshold for soft thresholding.
            threshold = sigma * np.sqrt(2 * np.log(len(signal)))
            # Apply soft thresholding to the detail coefficients.
            new_coeffs = [coefficients[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coefficients[1:]]
            # Reconstruct the signal from the thresholded coefficients.
            y = pywt.waverec(new_coeffs, self.filter_type)
            # Truncate the reconstructed signal to match the original length.
            filtered = y[:len(signal)]
            # If the filtered signal is near-zero, NaN, or unreasonably large, return None.
            if is_all_zeros_or_has_nan_or_large(filtered, max_val=self.signal.max()):
                return None
            else:
                # Otherwise, return the wavelet-denoised signal.
                return filtered

        else:
            # Raise an error if the specified filter type is not supported.
            raise ValueError(f"Filter is not supported - {self.filter_type}")

    def extract_features(self, signal: np.ndarray | None = None) -> dict:
        """
        Extract key statistical features from the signal.

        Computes several metrics including Root Mean Square (RMS), kurtosis, crest factor,
        skewness, and impulse factor.

        Parameters
        ----------
        signal : np.ndarray or None, optional
            The signal for which to extract features. If None, uses the object's raw signal.

        Returns
        -------
        dict
            A dictionary containing the computed features.
        """
        # Use the object's raw signal if no signal is provided.
        signal = self.signal if signal is None else signal
        # Compute the RMS value; ensure it is not below a small threshold to avoid division by zero.
        rms = max(np.sqrt(np.mean(signal**2)), 1e-12)
        # Compute the kurtosis of the signal.
        kurt = kurtosis(signal)
        # Calculate the crest factor (maximum absolute value divided by RMS).
        crest = np.max(np.abs(signal)) / rms
        # Compute the skewness of the signal.
        sk = skew(signal)
        # Compute the impulse factor (max absolute value divided by mean absolute value).
        impulse = np.max(np.abs(signal)) / (np.mean(np.abs(signal)) + 1e-12)
        # Return a dictionary containing all extracted features.
        return dict(rms=rms, kurtosis=kurt, crest_factor=crest, skew=sk, impulse=impulse)

    def plot_time(self, seconds: float | None = None, draw_filtered: bool = True, close: bool = False) -> None:
        """
        Plot the time-domain signal.

        Plots the raw (and optionally filtered) signal over a specified number of seconds.
        The plot is saved to a file.

        Parameters
        ----------
        seconds : float or None, optional
            The number of seconds to display; if None, the entire signal is shown.
        draw_filtered : bool, optional
            Whether to plot the filtered signal along with the raw signal (default is True).
        close : bool, optional
            Whether to close the plot after saving (default is False).
        """
        # Create a slice object for selecting the portion of the signal based on the specified seconds.
        idx = slice(None) if seconds is None else slice(0, int(seconds * self.fs))
        # Use a lock to ensure thread-safe plotting.
        with self._lock:
            # Create a new figure with a specific size.
            plt.figure(figsize=(15, 5))
            # Plot the raw signal in black with a thin line.
            plt.plot(self.time[idx], self.signal[idx], c='k', lw=0.5, label="Raw Signal")
            # Set the x and y axis tick density.
            plt.locator_params(axis='x', nbins=20)
            plt.locator_params(axis='y', nbins=20)
            # If requested, plot the filtered signal in green with a thin line.
            if draw_filtered:
                plt.plot(self.time[idx], self.signal_filtered[idx], c='g', lw=0.5, label="Filtered Signal")
            # Set the title using the analysis title and filtering info.
            plt.title(f"{self.title} | Time Domain".upper() + self._title)
            # Label the x-axis.
            plt.xlabel("Time (sec)")
            # Label the y-axis.
            plt.ylabel("Amplitude")
            # Enable grid lines.
            plt.grid(True)
            # Retrieve the current axis to fetch legend handles.
            ax = plt.gca()
            handles, labels = ax.get_legend_handles_labels()
            # Create and place the legend below the plot.
            plt.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=len(handles))
            # Adjust layout so that the plot fits into the figure area.
            plt.tight_layout()
            # Draw the plot to update the figure.
            plt.draw()
            # Save the figure to a file with a filename based on the title and filter method.
            plt.savefig(f"{self.title}_time".lower() + self._filename + '.png')
            # Close the figure if requested.
            if close:
                plt.close()

    def plot_frequency(self, signal_type: str = 'raw', db: bool = False, close: bool = False) -> None:
        """
        Plot the frequency domain representation of a signal.

        This method creates frequency domain plots for one of the signal types: raw, filtered, 
        or envelope. It overlays defect frequencies, their harmonics, and modulation lines.

        Parameters
        ----------
        signal_type : str, optional
            One of 'raw', 'filtered', or 'envelope' to specify which signal's frequency domain to plot.
        db : bool, optional
            If True, the magnitudes are converted to decibels; otherwise, they are plotted as-is.
        close : bool, optional
            Whether to close the plot after saving (default is False).
        """
        # Select the corresponding frequency and magnitude arrays for the specified signal type.
        freq, magn = dict(raw=(self.signal_freq, self.signal_magn),
                          filtered=(self.signal_filtered_freq, self.signal_filtered_magn),
                          envelope=(self.envelope_freq, self.envelope_magn))[signal_type]
        # Convert magnitudes to decibels if requested.
        magn = mag_to_db(magn) if db else magn
        # Check if the magnitude array contains any finite values; if not, skip plotting.
        if not np.isfinite(magn).any():
            print(f"Warning! NaN frequency - skip {signal_type} frequency plots")
            return 
        # Use the lock to ensure thread-safe plotting.
        with self._lock:
            # Create a subplot layout with 5 rows and 1 column using the default figure size.
            fig, axs = plt.subplots(5, 1, figsize=self._figsize)
            
            # Iterate over the first four subplots corresponding to each defect.
            for i, defect in enumerate(self.defects.keys()):
                # Plot defect markers, harmonics, and modulations for the current defect.
                self._plot_defects(axs[i], alpha=0.9, defect=defect)
                self._plot_harmonics(axs[i], alpha=0.8, defect=defect)
                self._plot_modulations(axs[i], alpha=0.5, defect=defect)

                # Plot the signal magnitude spectrum.
                axs[i].plot(freq, magn, lw=2, c='k', label=signal_type.title() + ' Signal Frequency')
                # Adjust tick density on x and y axes.
                axs[i].locator_params(axis='x', nbins=40)
                axs[i].locator_params(axis='y', nbins=20)
                # Set the subplot title with defect and filtering information.
                axs[i].set_title(f"{self.title} | FREQUENCY DOMAIN".upper() + self._title + f" | {signal_type.upper()} | {defect.upper()}")
                # Label the x-axis.
                axs[i].set_xlabel("Frequency (Hz)")
                # Label the y-axis based on whether decibel scaling is used.
                axs[i].set_ylabel("Magnitude (db)" if db else "Magnitude")
                # Set the y-axis limits based on the magnitude values.
                axs[i].set_ylim(magn.min() if db else 0.0, 1.05 * mag_to_db(self.freq_peaks[signal_type]['magnitudes']).max() if db else 1.05 * self.freq_peaks[signal_type]['magnitudes'].max())
                # Set the x-axis limits based on defect frequency range.
                axs[i].set_xlim(0, 2.05 * max(self.defects.values()))
                # Enable grid lines with a dotted linestyle.
                axs[i].grid(True, which='major', linestyle=':', linewidth=0.5)
                # Optionally, a custom legend can be added here.
                # self._legend(axs[i])
            
            # For the last subplot, plot all defects together.
            self._plot_defects(axs[-1], alpha=0.9)
            self._plot_harmonics(axs[-1], alpha=0.8)
            self._plot_modulations(axs[-1], alpha=0.5)
            axs[-1].plot(freq, magn, lw=2, c='k', label=signal_type.title() + ' Signal Frequency')
            axs[-1].locator_params(axis='x', nbins=40)
            axs[-1].locator_params(axis='y', nbins=20)
            axs[-1].set_title(f"{self.title} | FREQUENCY DOMAIN".upper() + self._title + f" | {signal_type.upper()} | ALL DEFECTS")
            axs[-1].set_xlabel("Frequency (Hz)")
            axs[-1].set_ylabel("Magnitude (db)" if db else "Magnitude")
            axs[-1].set_ylim(magn.min() if db else 0.0, 1.05 * mag_to_db(self.freq_peaks[signal_type]['magnitudes']).max() if db else 1.05 * self.freq_peaks[signal_type]['magnitudes'].max())
            axs[-1].set_xlim(0, 2.05 * max(self.defects.values()))
            axs[-1].grid(True, which='major', linestyle=':', linewidth=0.5)
            # Add a legend using the custom legend function.
            self._legend(axs[-1])
            
            # Adjust the layout to ensure the legend and plots are properly contained.
            plt.tight_layout(rect=[0, 0, 1, 1])
            plt.draw()
            # Save the figure as a PNG file based on the title, filter, and signal type.
            plt.savefig(f"{self.title}_freq".lower() + self._filename + f"_{signal_type.lower()}.png")
            # Close the plot if requested.
            if close:
                plt.close()

    def plot_spectrogram(self, signal_type: str = "raw", fmax: int = 200, close: bool = False) -> None:
        """
        Plot the spectrogram of a signal.

        This method computes the spectrogram using a Hann window, then plots the spectrogram
        along with overlayed defect marker lines. The plot is saved to a file.

        Parameters
        ----------
        signal_type : str, optional
            One of 'raw', 'filtered', or 'envelope' to specify which signal to analyze.
        fmax : int, optional
            The maximum frequency to display on the spectrogram plot (default is 200 Hz).
        close : bool, optional
            Whether to close the plot after saving (default is False).
        """
        # Choose the appropriate signal (raw, filtered, or envelope) based on the signal_type.
        signal = dict(raw=self.signal,
                      filtered=self.signal_filtered,
                      envelope=self.envelope)[signal_type]
        # Compute the spectrogram using a Hann window with specified parameters.
        f, t, s = spectrogram(signal,
                              self.fs,
                              window='hann',
                              nperseg=1024,
                              noverlap=512,
                              scaling='spectrum')
        # Use the lock for thread-safe plotting.
        with self._lock:
            # Create a new figure with specified size.
            plt.figure(figsize=(12, 7))
            ax = plt.gca()
            
            # Overlay defect lines horizontally on the spectrogram.
            self._plot_defects(ax, direction='horizontal')
            # The modulation and harmonic overlays are commented out but can be enabled if needed.
            # self._plot_modulations(ax, direction='horizontal')
            # self._plot_harmonics(ax, direction='horizontal')
            
            # Display the spectrogram; convert the spectral magnitude to dB scale safely.
            plt.pcolormesh(t, f, safe_signal(mag_to_db(s)), shading='gouraud', cmap='grey')
            # Set the tick density for the x and y axes.
            plt.locator_params(axis='x', nbins=20)
            plt.locator_params(axis='y', nbins=20)
            # Limit the y-axis to the maximum frequency specified.
            plt.ylim(0, fmax)
            # Add a colorbar with a label.
            plt.colorbar(label='dB')
            # Label the x-axis.
            plt.xlabel('Time (sec)')
            # Label the y-axis.
            plt.ylabel('Frequency (Hz)')
            # Set the plot title with spectrogram and signal information.
            plt.title(f"{self.title} | SPECTROGRAM " + self._title + f" | {signal_type.upper()}")
            
            # Add a custom legend to the plot.
            self._legend(ax)
            
            # Adjust the layout to ensure all elements are contained properly.
            plt.tight_layout(rect=[0, 0, 1, 1])
            plt.draw()
            # Save the spectrogram as a PNG file.
            plt.savefig(f"{self.title}_spectrogram".lower() + self._filename + f"_{signal_type.lower()}.png")
            # Close the plot if requested.
            if close:
                plt.close()

