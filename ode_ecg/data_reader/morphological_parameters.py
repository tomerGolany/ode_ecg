"""Find morphological values of ECG heatbeats."""
import numpy as np
import biosppy
import logging


def find_r_peaks(signal: np.ndarray):
    """Find R-peak locations of every heartbeat."""
    rpeaks, = biosppy.ecg.hamilton_segmenter(np.array(signal), sampling_rate=360)
    rpeaks, = biosppy.ecg.correct_rpeaks(signal=np.array(signal), rpeaks=rpeaks, sampling_rate=360, tol=0.05)
    return rpeaks


def find_peaks(signal: np.ndarray) -> np.ndarray:
    """Locate peaks based on derivative."""
    derivative = np.gradient(signal, 2)
    peaks = np.where(np.diff(np.sign(derivative)))[0]
    return peaks


def find_p_q_r_s_t_peaks(ecg_signal: np.ndarray, r_peaks_locations):
    """Find PQRST locations of each heart-beat within a signal."""
    p_peaks = []
    q_peaks = []
    s_peaks = []
    t_peaks = []

    for i_r_peak in range(1, len(r_peaks_locations) - 3):
        try:
            r_peak_location = r_peaks_locations[i_r_peak]
            epoch_before = ecg_signal[(r_peaks_locations[i_r_peak - 1]):r_peak_location]
            epoch_before = epoch_before[(len(epoch_before) // 2):len(epoch_before)]
            epoch_before = list(reversed(epoch_before))

            # Q wave:
            q_wave_index = np.min(find_peaks(epoch_before))
            q_wave = r_peak_location - q_wave_index

            # P wave:
            p_wave_index = q_wave_index + np.argmax(epoch_before[q_wave_index:])
            p_wave = r_peak_location - p_wave_index

            epoch_after = ecg_signal[r_peak_location:r_peaks_locations[i_r_peak + 1]]
            epoch_after = epoch_after[0:(len(epoch_after) // 2)]

            # S wave:
            s_wave_index = np.min(find_peaks(epoch_after))
            s_wave = r_peak_location + s_wave_index

            # T wave:
            t_wave_index = s_wave_index + np.argmax(epoch_after[s_wave_index:])
            t_wave = r_peak_location + t_wave_index

        except Exception as e:
            logging.info("Got error %s...", e)
            q_wave = -99999999
            p_wave = -99999999
            s_wave = 9999999999
            t_wave = 9999999999
        p_peaks.append(p_wave)
        q_peaks.append(q_wave)
        s_peaks.append(s_wave)
        t_peaks.append(t_wave)
    return p_peaks, q_peaks, s_peaks, t_peaks
