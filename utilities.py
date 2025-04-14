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

# Define a function to clean a signal by replacing NaN and infinite values.
def safe_signal(signal: np.ndarray) -> np.ndarray:
    """
    Replace NaN and infinite values in a signal with finite numbers.

    This function utilizes numpy's nan_to_num to convert NaNs to 0.0 and replace
    positive and negative infinite values with 0.0.

    Parameters:
        signal (np.ndarray): The input signal array.

    Returns:
        np.ndarray: The cleaned signal with all NaNs and infinities replaced.
    """
    # Use numpy's nan_to_num to replace NaN, positive infinity, and negative infinity.
    return np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

# Define a function to convert a signal's magnitude to decibels.
def mag_to_db(signal: np.ndarray) -> np.ndarray:
    """
    Convert a signal's magnitude to decibels (dB).

    This function calculates the decibel representation of the signal by applying
    20 * log10 on the signal values. It ensures numerical stability by using a 
    minimum threshold of 1e-12.

    Parameters:
        signal (np.ndarray): The input magnitude signal.

    Returns:
        np.ndarray: The signal converted to decibel units.
    """
    # Calculate the decibel value using log10, ensuring the signal is at least 1e-12 to avoid log(0).
    return 20 * np.log10(np.maximum(signal, 1e-12))

# Define a function to convert an index in the frequency domain to its corresponding frequency.
def idx_to_freq(idx: int, fs: int, n: int) -> float:
    """
    Convert an index to its corresponding frequency.

    Given an index in the spectrum, the sampling frequency, and the total number of
    points, this function computes the frequency represented by that index.

    Parameters:
        idx (int): The index in the frequency domain.
        fs (int): The sampling frequency in Hz.
        n (int): The total number of samples.

    Returns:
        float: The frequency in Hz corresponding to the given index.
    """
    # Calculate the frequency using the formula frequency = (index * sampling frequency) / total samples.
    return float(idx * fs / n)

# Define a function to convert a given frequency to its corresponding index in the FFT spectrum.
def freq_to_idx(freq: int, fs: int, n: int) -> int:
    """
    Convert a frequency to its corresponding index in the spectrum.

    Given a frequency value, the sampling frequency, and the total number of samples,
    this function computes the index in the frequency domain that corresponds to the
    given frequency.

    Parameters:
        freq (int): The frequency in Hz.
        fs (int): The sampling frequency in Hz.
        n (int): The total number of samples.

    Returns:
        int: The index corresponding to the provided frequency.
    """
    # Calculate the index using the formula index = (frequency * total samples) / sampling frequency.
    return int(freq * n / fs)

# Define a function to round the frequency values in a dictionary to two decimal places.
def round_freqs(df: Dict[str, Union[float, int]]) -> Dict[str, float]:
    """
    Round numerical frequency values in a dictionary to two decimal places.

    This function iterates over the dictionary's keys and rounds each corresponding
    numerical value to two decimal places using numpy's round function.

    Parameters:
        df (Dict[str, Union[float, int]]): A dictionary with numeric values.

    Returns:
        Dict[str, float]: A new dictionary with each value rounded to two decimal places.
    """
    # Use dictionary comprehension to create a new dictionary with values rounded using numpy's round.
    return {key: np.round(df[key], 2) for key in df.keys()}

# Define the RollingDefectAnalysis class to perform analysis on vibration signals.
class RollingDefectAnalysis:
    """
    A class to perform rolling defect analysis on vibration signals.

    This class provides methods to process a vibration signal, filter it using various
    filters, compute its envelope and frequency spectrum, extract defect-related frequency
    peaks and features, and plot the results in the time domain, frequency domain, and
    as spectrograms.

    Class Attributes:
        _linestyles (list[str]): List of line styles for plotting.
        _colors (list[str]): List of colors for plotting.
        _alphas (list[float]): List of transparency values for plotting.
        _figsize (tuple[int, int]): Default figure size for frequency domain plots.
        _lock (threading.Lock): A lock to coordinate plot generation in multithreaded contexts.
    """
    # Define a list of line styles to be used during plotting.
    _linestyles = ['-', '-', '-', '-']
    # Define a list of colors corresponding to each defect for plotting.
    _colors = ['r', 'b', 'm', 'c']
    # Define a list of alpha (transparency) values for plotting.
    _alphas = [0.6, 0.6, 0.7, 0.7]
    # Set a default figure size for frequency domain plots.
    _figsize = (15, 20)
    # Create a lock to ensure that plot generation in multi-threaded environments does not conflict.
    _lock = threading.Lock()

    # Apply a global seaborn style for plotting to achieve a uniform darkgrid appearance.
    sns.set(style="darkgrid")
    
    # Define the constructor for RollingDefectAnalysis.
    def __init__(self,
                 signal: np.ndarray,        # Input vibration signal.
                 fs: int,                   # Sampling frequency (Hz).
                 rpm: int,                  # Rotations per minute.
                 bpfi: float,               # Ball Pass Frequency Inward.
                 bpfo: float,               # Ball Pass Frequency Outward.
                 ftf: float,                # Fundamental Train Frequency.
                 bsf: float,                # Ball Spin Frequency.
                 title: str,                # Title for plots and analysis.
                 root_folder: str,          # Folder path to save generated images.
                 filter_type: str = "butterworth",  # Type of filter to be applied; default set to "butterworth".
                 use_walch: bool = False,   # Flag indicating if Welch's method should be used for frequency estimation.
                 peak_dividant: int = 25,   # Divider factor to decide threshold for significant frequency peaks.
                 band: tuple | None = None) -> None:  # Frequency band for filtering; if None, compute a default band.
        """
        Initialize the RollingDefectAnalysis instance.

        Parameters:
            signal (np.ndarray): The input vibration signal array.
            fs (int): Sampling frequency of the signal in Hz.
            rpm (int): Rotations per minute of the machine.
            bpfi (float): Ball Pass Frequency Inward.
            bpfo (float): Ball Pass Frequency Outward.
            ftf (float): Fundamental Train Frequency.
            bsf (float): Ball Spin Frequency.
            title (str): Title for the analysis and plots.
            root_folder (str): Folder where plot images will be saved.
            filter_type (str, optional): Type of filter to apply (default "butterworth").
            use_walch (bool, optional): Flag to use Welch's method for frequency estimation (default False).
            peak_dividant (int, optional): Divider factor to determine significant frequency peaks (default 25).
            band (tuple | None, optional): Tuple defining the lower and upper frequency bounds for filtering;
                                            if None, a default band is computed.

        The instance pre-computes filtered signals, envelopes, frequency spectra, and defect frequency peaks.
        """
        # Save the root folder path for later use in saving images.
        self.root_folder = root_folder
        # Save the raw signal. (Optionally one could apply safe_signal here.)
        self.signal = signal
        # Determine the number of samples in the signal.
        self.n = len(signal)
        # Save the sampling frequency.
        self.fs = fs
        # Construct a time vector for the signal in seconds.
        self.time = np.arange(self.n) / fs
        # Calculate the fundamental rotation frequency in Hz from the RPM.
        self.fr = rpm / 60.0
        
        # Compute defect frequencies by multiplying the provided multipliers with the fundamental frequency.
        self.defects = dict(bpfi=bpfi * self.fr, 
                            bpfo=bpfo * self.fr, 
                            ftf=ftf * self.fr, 
                            bsf=bsf * self.fr)
        # Define harmonics for each defect frequency. For BSF, note that it could have different considerations.
        self.harmonics = dict(bpfo=ftf * self.fr,
                              bpfi=self.fr,
                              bsf=ftf * self.fr,
                              ftf=self.fr)
        # Save the peak divider value.
        self.peak_dividant = peak_dividant
        # Save the flag indicating whether to use Welch's method.
        self.use_walch = use_walch
        # Save the title for the analysis.
        self.title = title
        # Save the filter type.
        self.filter_type = filter_type
        # Determine the frequency band: use the provided band if given, otherwise calculate a default band.
        self.band = band or self._default_band()
        # Apply the filtering operation on the raw signal.
        self.signal_filtered = self._filter_signal(self.signal)
        # Compute the envelope of the filtered signal using the Hilbert transform.
        self.envelope = self._envelope_signal()

        # Compute the frequency spectrum (frequency bins and magnitudes) for the raw signal.
        self.signal_freq, self.signal_magn = self._get_freq(self.signal, use_welch=self.use_walch)
        # Compute the frequency spectrum for the filtered signal.
        self.signal_filtered_freq, self.signal_filtered_magn = self._get_freq(self.signal_filtered, use_welch=self.use_walch)
        # Compute the frequency spectrum for the envelope signal.
        self.envelope_freq, self.envelope_magn = self._get_freq(self.envelope, use_welch=self.use_walch)

        # Detect significant frequency peaks for each of the three signal representations.
        self.freq_peaks = dict(raw=self._get_freq_picks(self.signal_magn),
                               filtered=self._get_freq_picks(self.signal_filtered_magn),
                               envelope=self._get_freq_picks(self.envelope_magn))

        # Define a title suffix for plots based on the method used (WALCH vs RFFT) and the filter type.
        self._title = f" | {'WALCH' if self.use_walch else 'RFFT'} | {self.filter_type}".upper()
        # Define a filename suffix for saving images.
        self._filename = f"_{'walch' if self.use_walch else 'rfft'}_{self.filter_type}".lower()

    # Private method to compute a default filtering band based on defect frequencies.
    def _default_band(self) -> tuple[float, float]:
        """
        Compute the default frequency band based on defect frequencies.

        Returns:
            tuple[float, float]: A tuple representing (lower_bound, upper_bound) for the filter.
        """
        # The lower bound is set to half of the minimum defect frequency.
        # The upper bound is set to twice the maximum defect frequency.
        return (0.5 * min(self.defects.values()),
                2 * max(self.defects.values()))
        
    # Private method to compute the envelope of the filtered signal.
    def _envelope_signal(self) -> np.ndarray:
        """
        Compute the envelope of the filtered signal using the Hilbert transform.

        Returns:
            np.ndarray: The envelope of the filtered signal.
        """
        # Apply the Hilbert transform and then take the absolute value to compute the envelope.
        return np.abs(hilbert(self.signal_filtered))
    
    # Private method to compute the frequency spectrum of a given signal.
    def _get_freq(self, signal: np.ndarray, use_welch: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the frequency spectrum of a signal.

        Parameters:
            signal (np.ndarray): The input signal.
            use_welch (bool, optional): Whether to use Welch's method (default False). 
                                        If False, uses FFT.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the frequency bins and their corresponding magnitudes.
        """
        # Check if Welch's method is to be used.
        if use_welch:
            # Compute frequency bins and power spectrum using Welch's method.
            freq, p = welch(signal, fs=self.fs, window='hann',
                              nperseg=2048, noverlap=1024, scaling='spectrum')
            # Return the frequency bins and the square root of the power spectrum (magnitude),
            # while applying safe_signal to remove any NaN or infinite values.
            return freq, safe_signal(np.sqrt(p))
        else:
            # Compute frequency bins using FFT (specifically, rfft which computes FFT for real inputs).
            freq = np.fft.rfftfreq(self.n, d=1/self.fs)
            # Return the frequency bins and the magnitude spectrum using FFT,
            # with safe_signal applied to ensure numerical stability.
            return freq, safe_signal(np.abs(np.fft.rfft(signal)))

    # Private method to detect significant frequency peaks in a given magnitude spectrum.
    def _get_freq_picks(self, signal_magn: np.ndarray) -> dict[str, np.ndarray]:
        """
        Identify frequency peaks in the magnitude spectrum that exceed a certain threshold.

        Parameters:
            signal_magn (np.ndarray): Magnitude spectrum of a signal.

        Returns:
            dict[str, np.ndarray]: A dictionary with keys 'frequency' and 'magnitudes' containing
                                   the detected peak frequencies and their magnitudes.
        """
        # Determine a maximum frequency limit for searching peaks.
        fmax = 2.05 * max(self.defects.values())
        # Convert this maximum frequency to an index in the FFT result.
        idx = freq_to_idx(fmax, self.fs, self.n)
        # Limit the magnitude spectrum to frequencies up to the calculated index.
        signal_magn_limited = signal_magn[:idx]
        # Define a threshold for peak detection: any magnitude greater than (max magnitude divided by peak_dividant).
        mmax = signal_magn_limited.max() / self.peak_dividant
        # Use the find_peaks function to detect peaks with heights exceeding the threshold.
        idxs, magns = find_peaks(signal_magn_limited, height=mmax)
        # Convert the indices of detected peaks into corresponding frequencies.
        freqs = np.array([idx_to_freq(idx, self.fs, self.n) for idx in idxs])
        # Return the detected peak frequencies and their corresponding magnitudes.
        return dict(frequency=freqs, magnitudes=magns['peak_heights'])

    # Private method to create a sorted legend for a matplotlib axes.
    def _legend(self, ax: Axes) -> None:
        """
        Create and apply a sorted legend to the specified plot axes.

        Parameters:
            ax (Axes): The matplotlib Axes object on which to place the legend.
        """
        # Retrieve the existing legend handles and labels from the axes.
        handles, labels = ax.get_legend_handles_labels()
        # Zip the labels and handles together, sort them by label name.
        sorted_handles_labels = sorted(zip(labels, handles), key=lambda pair: pair[0])
        # Unpack the sorted pairs back into labels and handles.
        sorted_labels, sorted_handles = zip(*sorted_handles_labels)
        # Create a legend with sorted entries, set in the upper center with specified layout parameters.
        plt.legend(sorted_handles, sorted_labels,
                   ncol=len(self.defects),
                   loc="upper center",
                   bbox_to_anchor=(0.5, -0.2),
                   fontsize="small",
                   handlelength=2,
                   columnspacing=1.2,
                   frameon=True)

    # Private method to plot defect frequency lines on an axes.
    def _plot_defects(self, ax: Axes, direction: str = 'vertical', alpha: float | None = 0.9, defect: str = None) -> None:
        """
        Plot defect frequency lines on the specified axes.

        Parameters:
            ax (Axes): The matplotlib Axes object to plot on.
            direction (str, optional): 'vertical' for vertical lines or 'horizontal' for horizontal lines (default 'vertical').
            alpha (float | None, optional): Transparency for the plotted lines (default 0.9).
            defect (str, optional): If specified, only plot for that defect; otherwise, plot all defects.
        """
        # Iterate over defects and their corresponding plotting attributes.
        for (label, freq), c, a, ls in zip(self.defects.items(), self._colors, self._alphas, self._linestyles):
            # If a specific defect is requested or if no filtering is needed...
            if label == defect or defect is None:
                # Check the direction for plotting.
                if direction == 'vertical':
                    # Plot a vertical line at the defect frequency with defined style and label.
                    ax.axvline(freq, ls=ls, c=c, lw=9, alpha=a if alpha is None else alpha, label=label.upper())
                else:
                    # Plot a horizontal line at the defect frequency with defined style and label.
                    ax.axhline(freq, ls=ls, c=c, lw=9, alpha=a if alpha is None else alpha, label=label.upper())

    # Private method to plot harmonic frequency lines for defects.
    def _plot_harmonics(self, ax: Axes, direction: str = 'vertical', alpha: float | None = 0.9, defect: str = None) -> None:
        """
        Plot harmonic frequency lines for each defect on the specified axes.

        Parameters:
            ax (Axes): The matplotlib Axes object to plot on.
            direction (str, optional): 'vertical' for vertical lines or 'horizontal' for horizontal lines (default 'vertical').
            alpha (float | None, optional): Base transparency for the plotted lines (default 0.9).
            defect (str, optional): If specified, only plot for that defect; otherwise, plot all defects.
        """
        # Iterate over defects and their corresponding plotting attributes.
        for (label, freq), c, a, ls in zip(self.defects.items(), self._colors, self._alphas, self._linestyles):
            if label == defect or defect is None:
                # Loop over the 2nd to 4th harmonics.
                for n in range(2, 5):
                    if direction == 'vertical':
                        # Plot a vertical line for the nth harmonic.
                        ax.axvline(n * freq, ls="-", c=c, lw=9 - n,
                                   alpha=a if alpha is None else alpha - (n - 1) / 10,
                                   label=label.upper() + f"_har_{n}".upper())
                    else:
                        # Plot a horizontal line for the nth harmonic.
                        ax.axhline(n * freq, ls="-", c=c, lw=9 - n,
                                   alpha=a if alpha is None else alpha - (n - 1) / 10,
                                   label=label.upper() + f"_har_{n}".upper())

    # Private method to plot modulation frequency lines around each defect frequency.
    def _plot_modulations(self, ax: Axes, direction: str = 'vertical', alpha: float | None = 0.9, defect: str = None) -> None:
        """
        Plot modulation frequency lines around each defect frequency.

        Parameters:
            ax (Axes): The matplotlib Axes object to plot on.
            direction (str, optional): 'vertical' for vertical lines or 'horizontal' for horizontal lines (default 'vertical').
            alpha (float | None, optional): Base transparency for the plotted lines (default 0.9).
            defect (str, optional): If specified, only plot for that defect; otherwise, plot for all defects.
        """
        # Iterate over each defect along with its plotting attributes.
        for (label, _), c, a, ls in zip(self.defects.items(), self._colors, self._alphas, self._linestyles):
            if label == defect or defect is None:
                # For modulation, consider modulation orders 1 to 4.
                for n in range(1, 5):
                    # Calculate the positive modulation frequency.
                    freq_p = max(self.defects[label] + n * self.harmonics[label], 0)
                    # Calculate the negative modulation frequency.
                    freq_n = max(self.defects[label] - n * self.harmonics[label], 0)
                    if direction == 'vertical':
                        # If the positive modulation frequency exists, plot a vertical line.
                        if freq_p:
                            ax.axvline(freq_p, ls='-', c=c, lw=5 - n,
                                       alpha=a if alpha is None else alpha - (n - 1) / 10,
                                       label=label.upper() + f"_mod_pos_{n}".upper())
                        # If the negative modulation frequency exists, plot a vertical line.
                        if freq_n:
                            ax.axvline(freq_n, ls='-', c=c, lw=5 - n,
                                       alpha=a if alpha is None else alpha - (n - 1) / 10,
                                       label=label.upper() + f"_mod_neg_{n}".upper())
                    else:
                        # For horizontal direction, apply similar logic.
                        if freq_p:
                            ax.axhline(freq_p, ls='-', c=c, lw=5 - n,
                                       alpha=a if alpha is None else alpha - (n - 1) / 10,
                                       label=label.upper() + f"_mod_pos_{n}".upper())
                        if freq_n:
                            ax.axhline(freq_n, ls='-', c=c, lw=5 - n,
                                       alpha=a if alpha is None else alpha - (n - 1) / 10,
                                       label=label.upper() + f"_mod_neg_{n}".upper())
    
    # Private method to apply a specified filter to the signal.
    def _filter_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply the specified filter to the signal.

        Parameters:
            signal (np.ndarray): The input signal to filter.

        Returns:
            np.ndarray: The filtered signal.

        Raises:
            ValueError: If the filter type is not supported.
        """
        # Unpack the lower and upper bounds of the frequency band.
        low, high = self.band
        # Normalize the frequency band with respect to the Nyquist frequency.
        wn = [low / (0.5 * self.fs), high / (0.5 * self.fs)]
        
        # Check if the filter type is "butterworth".
        if self.filter_type == "butterworth":
            # Design a 4th order Butterworth bandpass filter.
            b, a = butter(4, wn, btype='bandpass')
            # Apply zero-phase filtering to the signal.
            return filtfilt(b, a, signal)
        
        # Check if the filter type is "chebyshev1".
        elif self.filter_type == "chebyshev1":
            # Design a 4th order Chebyshev Type I bandpass filter with 1 dB ripple.
            b, a = cheby1(N=4, rp=1, Wn=wn, btype='bandpass')
            # Apply the filter to the signal.
            return filtfilt(b, a, signal)
        
        # Check if the filter type is "chebyshev2".
        elif self.filter_type == "chebyshev2":
            # Design a 4th order Chebyshev Type II bandpass filter with 40 dB stopband attenuation.
            b, a = cheby2(N=4, rs=40, Wn=wn, btype='bandpass')
            return filtfilt(b, a, signal)
        
        # Check if the filter type is "elliptic".
        elif self.filter_type == "elliptic":
            # Design a 4th order elliptic bandpass filter with 1 dB passband ripple and 40 dB stopband attenuation.
            b, a = ellip(N=4, rp=1, rs=40, Wn=wn, btype='bandpass')
            return filtfilt(b, a, signal)
        
        # Check if the filter type is 'fir'.
        elif self.filter_type == 'fir':
            # Determine the number of filter taps, ensuring a minimum of 101.
            numtaps = int(max(101, self.fs // 5))
            # Design a Finite Impulse Response (FIR) filter using the window method.
            taps = firwin(numtaps, wn, pass_zero=False)
            # Apply the FIR filter to the signal.
            return filtfilt(taps, [1.0], signal)
        
        # Check if the filter type indicates wavelet denoising (starts with 'db').
        elif self.filter_type.startswith('db'):
            # Decompose the signal using wavelet transformation.
            coefficients = pywt.wavedec(signal, self.filter_type, level=None)
            # Estimate the noise sigma using the median absolute deviation of the last coefficient.
            sigma = np.median(np.abs(coefficients[-1])) / 0.6745
            # Calculate the threshold for wavelet denoising.
            threshold = sigma * np.sqrt(2 * np.log(len(signal)))
            # Threshold the wavelet coefficients (soft thresholding) except the first approximation coefficient.
            newc = [coefficients[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coefficients[1:]]
            # Reconstruct the signal from the thresholded coefficients.
            y = pywt.waverec(newc, self.filter_type)
            # Return the reconstructed signal truncated to the original signal length.
            return y[:len(signal)]
        
        else:
            # If the filter type is not recognized, raise an error.
            raise ValueError(f"Filter is not supported - {self.filter_type}")

    # Public method to extract statistical features from the signal.
    def extract_features(self, signal: np.ndarray | None = None) -> dict[str, float]:
        """
        Extract statistical features from the given signal.

        The features include the root mean square (rms), kurtosis, crest factor,
        skewness, and impulse factor.

        Parameters:
            signal (np.ndarray | None, optional): The signal from which to extract features.
                                                    If None, uses the original signal.
        
        Returns:
            dict[str, float]: A dictionary containing the extracted features.
        """
        # Use the provided signal if available; otherwise, use the instance's signal.
        signal = self.signal if signal is None else signal
        # Compute the RMS value; avoid division by zero by using a minimum threshold.
        rms = max(np.sqrt(np.mean(signal ** 2)), 1e-12)
        # Compute the kurtosis of the signal.
        kurt = kurtosis(signal)
        # Calculate the crest factor as the maximum absolute value divided by RMS.
        crest = np.max(np.abs(signal)) / rms
        # Compute the skewness of the signal.
        sk = skew(signal)
        # Compute the impulse factor as the maximum absolute value divided by the mean absolute value.
        impulse = np.max(np.abs(signal)) / (np.mean(np.abs(signal)) + 1e-12)
        # Return a dictionary containing all extracted statistical features.
        return dict(rms=rms, kurtosis=kurt, crest_factor=crest, skew=sk, impulse=impulse)

    # Public method to plot the signal in the time domain.
    def plot_time(self, seconds: float | None = None, draw_filtered: bool = True, close: bool = False) -> None:
        """
        Plot the signal in the time domain.

        Parameters:
            seconds (float | None, optional): Duration in seconds to plot (plots full signal if None).
            draw_filtered (bool, optional): Whether to overlay the filtered signal (default True).
            close (bool, optional): Whether to close the plot after saving (default False).
        """
        # Determine slice index for plotting based on the provided duration.
        idx = slice(None) if seconds is None else slice(0, int(seconds * self.fs))
        # Acquire the lock to ensure thread-safe plotting.
        with self._lock:
            # Create a new figure with a specified size.
            plt.figure(figsize=(15, 5))
            # Plot the raw signal with a black line.
            plt.plot(self.time[idx], self.signal[idx], c='k', lw=0.5, label="Raw Signal")
            # Set the number of bins for x-axis and y-axis tick locator parameters.
            plt.locator_params(axis='x', nbins=20)
            plt.locator_params(axis='y', nbins=20)
            # If required, overlay the filtered signal with a green line.
            if draw_filtered:
                plt.plot(self.time[idx], self.signal_filtered[idx], c='g', lw=0.5, label="Filtered Signal")
            # Set the title, combining the user-provided title with analysis-specific suffixes.
            plt.title(f"{self.title} | TIME DOMAIN".upper() + self._title)
            # Label the x-axis and y-axis.
            plt.xlabel("Time (sec)")
            plt.ylabel("Amplitude")
            # Enable grid for better readability.
            plt.grid(True)
            # Get the current axes to handle the legend.
            ax = plt.gca()
            # Retrieve current legend handles and labels.
            handles, labels = ax.get_legend_handles_labels()
            # Create a legend positioned above the plot.
            plt.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=len(handles))
            # Adjust the layout to prevent clipping.
            plt.tight_layout()
            plt.draw()
            # Define the filename for saving the time domain plot.
            filename = os.path.join(self.root_folder, f"{self.title}_time".lower() + self._filename + '.png')
            # Save the figure to the specified filename.
            plt.savefig(filename)
            # If the close flag is set, close the plot.
            if close:
                plt.close()

    # Public method to plot the frequency domain representation of the signal.
    def plot_frequency(self,
                       signal_type: str = 'raw',  # Type of spectrum: 'raw', 'filtered', or 'envelope'.
                       db: bool = False,          # Flag to convert magnitude to decibels.
                       close: bool = False) -> None:
        """
        Plot the frequency spectrum of the signal.

        Parameters:
            signal_type (str, optional): Specifies which spectrum to plot: 'raw', 'filtered', or 'envelope' (default 'raw').
            db (bool, optional): Convert magnitudes to decibels if True (default False).
            close (bool, optional): Whether to close the plot after saving (default False).
        """
        # Select the frequency and magnitude arrays based on the specified signal type.
        freq, magn = dict(
            raw=(self.signal_freq, self.signal_magn),
            filtered=(self.signal_filtered_freq, self.signal_filtered_magn),
            envelope=(self.envelope_freq, self.envelope_magn)
        )[signal_type]
        # If dB conversion is requested, convert the magnitude.
        magn = mag_to_db(magn) if db else magn
        # If no finite values exist in the magnitude, print a warning and skip plotting.
        if not np.isfinite(magn).any():
            print(f"Warning! NaN frequency - skip {signal_type} frequency plots")
            return 
        # Acquire the lock for thread-safe plotting.
        with self._lock:
            # Create subplots with 5 rows and 1 column, using the preset figure size.
            fig, axs = plt.subplots(5, 1, figsize=self._figsize)
            # Loop over each defect and plot its corresponding features in separate subplots.
            for i, defect in enumerate(self.defects.keys()):
                # Plot defect lines, harmonic lines, and modulation lines for the current defect.
                self._plot_defects(axs[i], alpha=0.9, defect=defect)
                self._plot_harmonics(axs[i], alpha=0.8, defect=defect)
                self._plot_modulations(axs[i], alpha=0.5, defect=defect)

                # Plot the frequency spectrum on the subplot.
                axs[i].plot(freq, magn, lw=2, c='k', label=signal_type.title() + ' Signal Frequency')
                # Set tick locator parameters.
                axs[i].locator_params(axis='x', nbins=40)
                axs[i].locator_params(axis='y', nbins=20)
                # Set the title to include defect information.
                axs[i].set_title(f"{self.title} | FREQUENCY DOMAIN".upper() + self._title + f" | {signal_type.upper()} | {defect.upper()}")
                # Label the x-axis and y-axis.
                axs[i].set_xlabel("Frequency (Hz)")
                axs[i].set_ylabel("Magnitude (db)" if db else "Magnitude")
                # Set the y-axis limits.
                axs[i].set_ylim(magn.min() if db else 0.0,
                                1.05 * mag_to_db(self.freq_peaks[signal_type]['magnitudes']).max() if db else 1.05 * self.freq_peaks[signal_type]['magnitudes'].max())
                # Set the x-axis limits.
                axs[i].set_xlim(0, 2.05 * max(self.defects.values()))
                # Enable grid with dotted lines.
                axs[i].grid(True, which='major', linestyle=':', linewidth=0.5)
            
            # On the last subplot, plot all defect information without filtering by defect.
            self._plot_defects(axs[-1], alpha=0.9)
            self._plot_harmonics(axs[-1], alpha=0.8)
            self._plot_modulations(axs[-1], alpha=0.5)

            # Plot the frequency spectrum for all defects on the last subplot.
            axs[-1].plot(freq, magn, lw=2, c='k', label=signal_type.title() + ' Signal Frequency')
            axs[-1].locator_params(axis='x', nbins=40)
            axs[-1].locator_params(axis='y', nbins=20)
            axs[-1].set_title(f"{self.title} | FREQUENCY DOMAIN".upper() + self._title + f" | {signal_type.upper()} | ALL DEFECTS")
            axs[-1].set_xlabel("Frequency (Hz)")
            axs[-1].set_ylabel("Magnitude (db)" if db else "Magnitude")
            axs[-1].set_ylim(magn.min() if db else 0.0,
                             1.05 * mag_to_db(self.freq_peaks[signal_type]['magnitudes']).max() if db else 1.05 * self.freq_peaks[signal_type]['magnitudes'].max())
            axs[-1].set_xlim(0, 2.05 * max(self.defects.values()))
            axs[-1].grid(True, which='major', linestyle=':', linewidth=0.5)
            # Add a sorted legend to the last subplot.
            self._legend(axs[-1])
            
            # Adjust the layout to fit all subplots without overlapping.
            plt.tight_layout(rect=[0, 0, 1, 1])
            plt.draw()
            # Define the filename for saving the frequency domain plot.
            filename = os.path.join(self.root_folder, f"{self.title}_freq".lower() + self._filename + f"_{signal_type.lower()}.png")
            plt.savefig(filename)
            # Close the plot if specified.
            if close:
                plt.close()

    # Public method to plot a spectrogram of the signal.
    def plot_spectrogram(self,
                         signal_type: str = "raw",  # Choose the type of signal to display: 'raw', 'filtered', or 'envelope'.
                         fmax: int = 200,           # Maximum frequency (Hz) to display.
                         close: bool = False) -> None:
        """
        Plot a spectrogram of the signal.

        Parameters:
            signal_type (str, optional): Type of signal to use ('raw', 'filtered', or 'envelope', default "raw").
            fmax (int, optional): Maximum frequency (Hz) to display in the spectrogram (default 200).
            close (bool, optional): Whether to close the plot after saving (default False).
        """
        # Select the appropriate signal based on the provided signal_type.
        signal = dict(raw=self.signal,
                      filtered=self.signal_filtered,
                      envelope=self.envelope)[signal_type]
        # Compute the spectrogram using a Hann window, 1024 samples per segment and 512 overlap.
        f, t, s = spectrogram(signal,
                              self.fs,
                              window='hann',
                              nperseg=1024,
                              noverlap=512,
                              scaling='spectrum')
        # Acquire the lock for thread-safe plotting.
        with self._lock:
            # Create a new figure for the spectrogram.
            plt.figure(figsize=(12, 7))
            # Get the current axes.
            ax = plt.gca()
            
            # Plot defect frequency lines as horizontal lines on the spectrogram.
            self._plot_defects(ax, direction='horizontal')
            # Display the spectrogram using a pseudocolor plot; convert magnitude to dB and apply safe_signal.
            plt.pcolormesh(t, f, safe_signal(mag_to_db(s)), shading='gouraud', cmap='viridis')
            # Set tick locator parameters for both axes.
            plt.locator_params(axis='x', nbins=20)
            plt.locator_params(axis='y', nbins=20)
            # Limit the y-axis to the specified maximum frequency.
            plt.ylim(0, fmax)
            # Add a colorbar with a label.
            plt.colorbar(label='dB')
            # Label the x-axis and y-axis.
            plt.xlabel('Time (sec)')
            plt.ylabel('Frequency (Hz)')
            # Set the title to include analysis details.
            plt.title(f"{self.title} | SPECTROGRAM " + self._title + f" | {signal_type.upper()}")
            
            # Add a sorted legend to the plot.
            self._legend(ax)
            
            # Adjust the layout to prevent overlap.
            plt.tight_layout(rect=[0, 0, 1, 1])
            plt.draw()
            # Define the filename for saving the spectrogram plot.
            filename = os.path.join(self.root_folder, f"{self.title}_spectrogram".lower() + self._filename + f"_{signal_type.lower()}.png")
            plt.savefig(filename)
            # Close the plot if specified.
            if close:
                plt.close()

