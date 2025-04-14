import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

import pywt
import os
import time
import threading
import concurrent.futures
import argparse

from typing import * 
from scipy.stats import * 
from scipy.signal import * 

import seaborn as sns

import os
import json
import requests
import nbformat as nbf
import datetime
import argparse
from pathlib import Path
from openai import OpenAI
import base64
import pandas as pd
import glob
import concurrent.futures

client = OpenAI(api_key =  os.getenv("OPENAI_API_KEY"))

def generate_content_openai(user_prompt, model = "gpt-4o", 
                            system_prompt = "You are a helpful assistant, replying in json", 
                            response_format = None,
                            web_search_options = None,
                            temperature = 0.1):

    message = [{"role": "system", "content": system_prompt},
               {"role": "user", "content": user_prompt}]

    if web_search_options:
        response =  client.chat.completions.create(
                    model=model,
                    web_search_options=web_search_options,
                    messages=message,
                    response_format=response_format)
    else:
        response =  client.chat.completions.create(
                    model=model,
                    messages=message,
                    response_format=response_format,
                    temperature=temperature) 
    
    return json.loads(response.choices[0].message.content)

def generate_response_format(keys, values):
    response_format = {}
    response_format["type"]="json_schema"

    json_schema = {}
    json_schema["name"] = "subheadings_schema"

    schema = {}
    schema["type"] = "object"

    properties = {}

    for key, value in zip(keys, values):
        properties[key] = {"description":value, "type":"string"}

    schema["properties"] = properties

    schema["required"] =  [keys[0], keys[1]] if len(keys) >= 2 else [keys[0]] # 1 to 2 keys
    schema["additionalProperties"] =  False  # Disallow any other keys

    json_schema["schema"]  = schema

    response_format["json_schema"] = json_schema

    return response_format

def get_value(df):
    if isinstance(df, str):
        try:
            df = json.loads(df)
        except:
            pass
    if isinstance(df, dict):
        key = list(df.keys())[0]
        if isinstance(df[key], dict):
            return get_value(df[key])
        else:
            return df[key]
    else:
        return df

def get_values(df):
    # print(df)
    if isinstance(df, str):
        try:
            df = json.loads(df)
        except:
            pass
    if isinstance(df, dict):
        # print("HERE 1")
        keys = list(df.keys())
        if len(keys) == 1:
            # print("HERE 2")
            return get_values(df[keys[0]])
        else:
            # print("HERE 3")
            return [df[key] for key in keys]
    else:
        # print("HERE 4")
        # print(df)
        # print(type(df))
        # print(isinstance(df, dict))
        # print("Here 5")
        return df

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def image_content(image_path, user_prompt, system_prompt, response_format, temperature = 0.1):
    # Getting the Base64 string
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=temperature,
        response_format=response_format,
        messages=[{"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
    )

    return json.loads(response.choices[0].message.content)

def safe_signal(signal: np.ndarray) -> np.ndarray:
    return np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

def mag_to_db(signal: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.maximum(signal, 1e-12))

def idx_to_freq(idx: int, fs: int, n: int) -> float:
    return float(idx*fs/n)

def freq_to_idx(freq: int, fs: int, n: int) -> int:
    return int(freq * n/fs)

def round_freqs(df):
    return {key:np.round(df[key], 2) for key in df.keys()}


class RollingDefectAnalysis:
    _linestyles = ['-','-','-','-']
    _colors     = ['r', 'b', 'm', 'c']
    _alphas     = [0.6, 0.6, 0.7, 0.7]
    _figsize = (15, 20)
    _lock = threading.Lock()


# Use seaborn style
    sns.set(style="darkgrid")
    def __init__(self,
                 signal: np.ndarray,
                 fs: int,
                 rpm: int,
                 bpfi: float,
                 bpfo: float,
                 ftf: float,
                 bsf: float,
                 title: str,
                 root_folder: str,
                 filter_type: str = "butterworth",
                 use_walch: bool=False,
                 peak_dividant: int=25,
                 band: tuple | None = None):
        
        self.root_folder = root_folder
        self.signal = signal#safe_signal(signal)
        self.n = len(signal)
        self.fs = fs
        self.time = np.arange(self.n)/fs
        self.fr = rpm / 60.0
        self.defects = dict(bpfi=bpfi*self.fr, 
                            bpfo=bpfo*self.fr, 
                            ftf=ftf*self.fr, 
                            bsf=bsf*self.fr)
        self.harmonics = dict(bpfo=ftf*self.fr,
                              bpfi=self.fr,
                              bsf=ftf*self.fr,
                              ftf=self.fr) # for bsf it could be ftf, fr, and bpfo
        self.peak_dividant = peak_dividant
        self.use_walch = use_walch
        self.title = title
        self.filter_type = filter_type
        self.band = band or self._default_band() # if None, then it goes to default
        self.signal_filtered = self._filter_signal(self.signal)
        self.envelope = self._envelope_signal()

        self.signal_freq, self.signal_magn = self._get_freq(self.signal, use_welch=self.use_walch)
        self.signal_filtered_freq, self.signal_filtered_magn = self._get_freq(self.signal_filtered, use_welch=self.use_walch)
        self.envelope_freq, self.envelope_magn = self._get_freq(self.envelope, use_welch=self.use_walch)

        self.freq_peaks=dict(raw=self._get_freq_picks(self.signal_magn),
                             filtered=self._get_freq_picks(self.signal_filtered_magn),
                             envelope=self._get_freq_picks(self.envelope_magn))

        self._title = f" | {'WALCH' if self.use_walch else 'RFFT'} | {self.filter_type}".upper()
        self._filename = f"_{'walch' if self.use_walch else 'rfft'}_{self.filter_type}".lower()

    def _default_band(self) -> tuple:
        return (0.5 * min(self.defects.values()),
                2 * max(self.defects.values()))
        
    def _envelope_signal(self) -> np.ndarray:
        return np.abs(hilbert(self.signal_filtered))
    
    def _get_freq(self, signal: np.ndarray, use_welch: bool=False) -> tuple:
        if use_welch:
            freq, p = welch(signal, fs = self.fs, window='hann',
                         nperseg=2048, noverlap=1024, scaling= 'spectrum')
            return freq, safe_signal(np.sqrt(p))
        else:
            freq = np.fft.rfftfreq(self.n, d=1/self.fs)
            return freq, safe_signal(np.abs(np.fft.rfft(signal)))

    def _get_freq_picks(self, signal_magn: np.ndarray) -> dict:
        fmax = 2.05*max(self.defects.values())
        idx = freq_to_idx(fmax, self.fs, self.n)
        signal_magn_limited = signal_magn[:idx]
        mmax = signal_magn_limited.max() / self.peak_dividant # any magnitude greater than this is of interest
        idxs, magns = find_peaks(signal_magn_limited, height=mmax)
        freqs = np.array([idx_to_freq(idx, self.fs, self.n) for idx in idxs])
        return dict(frequency=freqs, magnitudes=magns['peak_heights'])

    def _legend(self, ax: plt.gca):
        handles, labels = ax.get_legend_handles_labels()
        sorted_handles_labels = sorted(zip(labels, handles), key=lambda pair: pair[0])
        sorted_labels, sorted_handles = zip(*sorted_handles_labels)
        plt.legend(sorted_handles, sorted_labels,
                    ncol=len(self.defects),
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.2),
                    fontsize="small",
                    handlelength=2,
                    columnspacing=1.2,
                    frameon=True)

    def _plot_defects(self, ax: plt.gca, direction: str='vertical', alpha: float | None=0.9, defect: str=None):
        for (label, freq), c, a, ls in zip(self.defects.items(), self._colors, self._alphas, self._linestyles):
            if label==defect or defect is None:
                if direction=='vertical':
                    ax.axvline(freq, ls=ls, c=c, lw=9, alpha=a if alpha is None else alpha, label=label.upper())
                else:
                    ax.axhline(freq, ls=ls, c=c, lw=9, alpha=a if alpha is None else alpha, label=label.upper())

    def _plot_harmonics(self, ax: plt.gca, direction: str='vertical', alpha: float | None=0.9, defect: str=None):
        for (label, freq), c, a, ls in zip(self.defects.items(), self._colors, self._alphas, self._linestyles):
            if label==defect or defect is None:
                for n in range(2,5): #2nd to 4th harmonics
                    if direction=='vertical':
                        ax.axvline(n * freq, ls="-", c=c, lw=9-n, alpha=a if alpha is None else alpha-(n-1)/10, label=label.upper()+f"_har_{n}".upper())
                    else:
                        ax.axhline(n * freq, ls="-", c=c, lw=9-n, alpha=a if alpha is None else alpha-(n-1)/10, label=label.upper()+f"_har_{n}".upper())

    def _plot_modulations(self, ax: plt.gca, direction: str='vertical', alpha: float | None=0.9, defect: str=None):
        for (label, _), c, a, ls in zip(self.defects.items(), self._colors, self._alphas, self._linestyles):
            if label==defect or defect is None:
                for n in range(1,5): # 2nd to 4th harmonics
                    freq_p = max(self.defects[label] + n * self.harmonics[label], 0)
                    freq_n = max(self.defects[label] - n * self.harmonics[label], 0)
                    if direction=='vertical':
                        if freq_p: ax.axvline(freq_p, ls='-', c=c, lw=5-n, alpha=a if alpha is None else alpha-(n-1)/10, label=label.upper()+f"_mod_pos_{n}".upper())
                        if freq_n: ax.axvline(freq_n, ls='-', c=c, lw=5-n, alpha=a if alpha is None else alpha-(n-1)/10, label=label.upper()+f"_mod_neg_{n}".upper())
                    else:
                        if freq_p: ax.axhline(freq_p, ls='-', c=c, lw=5-n, alpha=a if alpha is None else alpha-(n-1)/10, label=label.upper()+f"_mod_pos_{n}".upper())
                        if freq_n: ax.axhline(freq_n, ls='-', c=c, lw=5-n, alpha=a if alpha is None else alpha-(n-1)/10, label=label.upper()+f"_mod_neg_{n}".upper())
    
    def _filter_signal(self, signal):
        low, high = self.band
        wn = [low / (0.5 * self.fs), high / (0.5 * self.fs)]
        
        if self.filter_type == "butterworth":
            b, a = butter(4, wn, btype='bandpass')
            return filtfilt(b, a, signal)
        
        elif self.filter_type == "chebyshev1":
            b, a = cheby1(N=4, rp=1, Wn=wn, btype='bandpass')
            return filtfilt(b, a, signal)
        
        elif self.filter_type == "chebyshev2":
            b, a = cheby2(N=4, rs=40, Wn=wn, btype='bandpass')
            return filtfilt(b, a, signal)
        
        elif self.filter_type == "elliptic":
            b, a = ellip(N=4, rp=1, rs=40, Wn=wn, btype='bandpass')
            return filtfilt(b, a, signal)
        
        elif self.filter_type == 'fir':
            numtaps = int(max(101, self.fs // 5))
            taps = firwin(numtaps, wn, pass_zero=False)
            return filtfilt(taps, [1.0], signal)
        
        elif self.filter_type.startswith('db'):
            coefficients = pywt.wavedec(signal, self.filter_type, level=None)
            sigma = np.median(np.abs(coefficients[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(signal)))
            newc = [coefficients[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coefficients[1:]]
            y = pywt.waverec(newc, self.filter_type)
            return y[:len(signal)]
        
        else:
            raise ValueError(f"Filter is not supported - {self.filter_type}")

    def extract_features(self, signal: np.ndarray | None=None) -> dict:
        signal = self.signal if signal is None else signal
        rms = max(np.sqrt(np.mean(signal**2)), 1e-12)
        kurt = kurtosis(signal)
        crest = np.max(np.abs(signal)) / rms
        sk = skew(signal)
        impulse = np.max(np.abs(signal)) / (np.mean(np.abs(signal)) + 1e-12)
        return dict(rms=rms, kurtosis=kurt, crest_factor=crest, skew=sk, impulse=impulse)

    def plot_time(self, seconds: float | None=None, draw_filtered: bool=True, close: bool=False):
        idx = slice(None) if seconds is None else slice(0, int(seconds*self.fs))
        with self._lock:
            plt.figure(figsize=(15,5))
            plt.plot(self.time[idx], self.signal[idx], c='k', lw=0.5, label="Raw Signal")
            plt.locator_params(axis='x', nbins=20)
            plt.locator_params(axis='y', nbins=20)
            if draw_filtered:
                plt.plot(self.time[idx], self.signal_filtered[idx], c='g', lw=0.5, label="Filtered Signal")
            plt.title(f"{self.title} | Time Domain".upper() + self._title)
            plt.xlabel("Time (sec)")
            plt.ylabel("Amplitude")
            plt.grid(True)
            ax = plt.gca()
            handles, labels = ax.get_legend_handles_labels()
            plt.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=len(handles))
            plt.tight_layout() 
            plt.draw()
            filename = os.path.join(self.root_folder, f"{self.title}_time".lower() + self._filename + '.png')
            plt.savefig(filename)
            if close:
                plt.close()

    def plot_frequency(self, 
                       signal_type: str='raw', 
                       db: bool=False, 
                       close: bool=False):
        freq, magn = dict(raw=(self.signal_freq, self.signal_magn),
                          filtered=(self.signal_filtered_freq, self.signal_filtered_magn),
                          envelope=(self.envelope_freq, self.envelope_magn))[signal_type]
        magn = mag_to_db(magn) if db else magn
        if not np.isfinite(magn).any():
            print(f"Warning! NaN frequency - skip {signal_type} frequency plots")
            return 
        with self._lock:
            fig, axs = plt.subplots(5, 1, figsize=self._figsize)  # 2 rows, 1 column

            
            # plt.figure(figsize=self._figsize)
            for i, defect in enumerate(self.defects.keys()):
                # Plot defects, modulations, and harmonics.
                self._plot_defects(axs[i], alpha = 0.9, defect=defect)
                self._plot_harmonics(axs[i], alpha = 0.8, defect=defect)
                self._plot_modulations(axs[i], alpha = 0.5, defect=defect)

                # Plot the signal.
                axs[i].plot(freq, magn, lw=2, c='k', label=signal_type.title() + ' Signal Frequency')
                axs[i].locator_params(axis='x', nbins=40)
                axs[i].locator_params(axis='y', nbins=20)
                axs[i].set_title(f"{self.title} | FREQUENCY DOMAIN".upper() + self._title + f" | {signal_type.upper()} | {defect.upper()}")
                axs[i].set_xlabel("Frequency (Hz)")
                axs[i].set_ylabel("Magnitude (db)" if db else "Magnitude")
                axs[i].set_ylim(magn.min() if db else 0.0, 1.05 * mag_to_db(self.freq_peaks[signal_type]['magnitudes']).max() if db else 1.05 * self.freq_peaks[signal_type]['magnitudes'].max())
                axs[i].set_xlim(0, 2.05 * max(self.defects.values()))
                axs[i].grid(True, which='major', linestyle=':', linewidth=0.5)
                #self._legend(axs[i])
            
            # Plot defects, modulations, and harmonics.
            self._plot_defects(axs[-1], alpha = 0.9)
            self._plot_harmonics(axs[-1], alpha = 0.8)
            self._plot_modulations(axs[-1], alpha = 0.5)

            # Plot the signal.
            axs[-1].plot(freq, magn, lw=2, c='k', label=signal_type.title() + ' Signal Frequency')
            axs[-1].locator_params(axis='x', nbins=40)
            axs[-1].locator_params(axis='y', nbins=20)
            axs[-1].set_title(f"{self.title} | FREQUENCY DOMAIN".upper() + self._title + f" | {signal_type.upper()} | ALL DEFECTS")
            axs[-1].set_xlabel("Frequency (Hz)")
            axs[-1].set_ylabel("Magnitude (db)" if db else "Magnitude")
            axs[-1].set_ylim(magn.min() if db else 0.0, 1.05 * mag_to_db(self.freq_peaks[signal_type]['magnitudes']).max() if db else 1.05 * self.freq_peaks[signal_type]['magnitudes'].max())
            axs[-1].set_xlim(0, 2.05 * max(self.defects.values()))
            axs[-1].grid(True, which='major', linestyle=':', linewidth=0.5)
            self._legend(axs[-1])
            
            # Adjust layout to make room for the external legend.
            plt.tight_layout(rect=[0, 0, 1, 1])
            plt.draw()
            filename = os.path.join(self.root_folder, f"{self.title}_freq".lower() + self._filename + f"_{signal_type.lower()}.png")
            plt.savefig(filename)
            if close:
                plt.close()

    def plot_spectrogram(self, 
                         signal_type: str="raw",
                         fmax: int=200,
                         close: bool=False):
        signal = dict (raw=self.signal,
                  filtered=self.signal_filtered,
                  envelope=self.envelope)[signal_type]
        f, t, s = spectrogram(signal, 
                              self.fs, 
                              window='hann',
                              nperseg=1024, 
                              noverlap=512, 
                              scaling='spectrum')
        with self._lock:
            plt.figure(figsize=(12, 7))
            ax = plt.gca()
            
            # Plot defects, modulations, and harmonics horizontally.
            self._plot_defects(ax, direction='horizontal')
            #self._plot_modulations(ax, direction='horizontal')
            #self._plot_harmonics(ax, direction='horizontal')
            
            # Display the spectrogram.
            plt.pcolormesh(t, f, safe_signal(mag_to_db(s)), shading='gouraud', cmap = 'viridis')
            plt.locator_params(axis='x', nbins=20)
            plt.locator_params(axis='y', nbins=20)
            plt.ylim(0, fmax)
            plt.colorbar(label='dB')
            plt.xlabel('Time (sec)')
            plt.ylabel('Frequency (Hz)')
            plt.title(f"{self.title} | SPECTROGRAM " + self._title + f" | {signal_type.upper()}")
            
            # Use the custom legend function.
            self._legend(ax)
            
            plt.tight_layout(rect=[0, 0, 1, 1])
            plt.draw()
            filename = os.path.join(self.root_folder, f"{self.title}_spectrogram".lower() + self._filename + f"_{signal_type.lower()}.png")
            plt.savefig(filename)
            if close:
                plt.close()



# def get_args():
#     description = "Rolling Defect Analysis."
#     parser = argparse.ArgumentParser(description=description)
#     parser.add_argument('--signal', type=str, required=True, help='Path to the signal file')
#     parser.add_argument('--fs', type=int, required=True, help='Sample rate of the signal')
#     parser.add_argument('--rpm', type=int, required=True, help='RPM of the signal')
#     parser.add_argument('--bpfi', type=float, required=True, help='BPFI of the signal')
#     parser.add_argument('--bpfo', type=float, required=True, help='BPFO of the signal')
#     parser.add_argument('--ftf', type=float, required=True, help='FTF of the signal')
#     parser.add_argument('--bsf', type=float, required=True, help='BSF of the signal')
#     parser.add_argument('--title', type=str, required=True, help='Title of the project')
#     parser.add_argument('--filter_type', type=str, default='butterworth', help="Filter type: butterworth, chebyshev1, chebyshev2, elliptic, fir, db4")
#     parser.add_argument('--use_walch', type=bool, help='Use Welch method for frequency analysis')
#     parser.add_argument('--peak_dividant', type=int, default=50, help='Peak dividant for frequency analysis')

#     return parser.parse_args()
