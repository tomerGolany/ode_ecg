import numpy as np
import logging
from ode_ecg import train_configs
from ode_ecg.data_reader import heartbeat_types
import wfdb
import pandas as pd
from ode_ecg.data_reader import morphological_parameters
import matplotlib.pyplot as plt

DATA_DIR = train_configs.base + 'ecg_pytorch/ecg_pytorch/data_reader/text_files/'

train_set = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220,
             223, 230]  # DS1
train_set = [str(x) for x in train_set]
test_set = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232,
            233, 234]  # DS2
test_set = [str(x) for x in test_set]


class Patient(object):
    """Patient object represents a patient from the MIT-BIH AR database.

    Attributes:
        patient_number: Unique patient number.
        signals: Raw ecg signals from two leads.
        additional_fields:
        mit_bih_labels_str:
        r_peaks_locations: Locations of R-peaks of each heartbeat.
        labels_descriptions:
        heartbeats: Sliced heartbeats.
    """

    def __init__(self, patient_number):
        """Init Patient object from corresponding text file.

        :param patient_number: string which represents the patient number.
        """
        logging.info("Creating patient {}...".format(patient_number))
        self.patient_number = patient_number
        self.signals, self.additional_fields = self.get_raw_signals()
        self.mit_bih_labels_str, self.r_peaks_annotations, self.labels_descriptions = self.get_annotations()

        self.each_lead_two, self.ecg_other_lead = self.separate_leads()
        self.r_peaks_locations = morphological_parameters.find_r_peaks(self.each_lead_two)
        self.p_peaks, self.q_peaks, self.s_peaks, self.t_peaks = morphological_parameters.find_p_q_r_s_t_peaks(
            self.each_lead_two, self.r_peaks_locations)

        self.heartbeats = self.slice_heartbeats()
        logging.info("Completed patient {}.\n\n".format(patient_number))

    def separate_leads(self):
        """Separate the signal to two leads."""
        lead_ii_pos = None
        other_lead_pos = None
        for i, lead in enumerate(self.additional_fields['sig_name']):
            if lead == 'MLII':
                lead_ii_pos = i
            else:
                other_lead_pos = i
        if lead_ii_pos is None:
            raise AssertionError("Didn't find lead 2 position. LEADS: {}".format(self.additional_fields['sig_name']))
        logging.info("LEAD 2 position: {}".format(lead_ii_pos))
        ecg_lead_two = self.signals[:, lead_ii_pos]
        ecg_signal_other_lead = self.signals[:, other_lead_pos]
        return ecg_lead_two, ecg_signal_other_lead

    def get_raw_signals(self):
        """Get raw signal using the wfdb package.

        :return: signals : numpy array
                    A 2d numpy array storing the physical signals from the record.
                fields : dict
                    A dictionary containing several key attributes of the read
                    record:
                      - fs: The sampling frequency of the record
                      - units: The units for each channel
                      - sig_name: The signal name for each channel
                      - comments: Any comments written in the header
        """
        signals, fields = wfdb.rdsamp(self.patient_number, pn_dir='mitdb', warn_empty=True)
        logging.info("Patient {} additional info: {}".format(self.patient_number, fields))
        return signals, fields

    def get_annotations(self):
        """Get signal annotation using the wfdb package."""
        ann = wfdb.rdann(self.patient_number, 'atr', pn_dir='mitdb',
                         return_label_elements=['symbol', 'label_store', 'description'], summarize_labels=True)
        mit_bih_labels_str = ann.symbol

        labels_locations = ann.sample

        labels_description = ann.description

        return mit_bih_labels_str, labels_locations, labels_description

    def slice_heartbeats(self):
        """Slice heartbeats from the raw signal.

        :return:
        """
        sampling_rate = self.additional_fields['fs']  # 360 samples per second
        logging.info("Sampling rate: {}".format(sampling_rate))
        assert sampling_rate == 360
        before = 0.2  # 0.2 seconds == 0.2 * 10^3 miliseconds == 200 ms
        after = 0.4  # --> 400 ms

        #
        # Find lead 2 position:
        #
        lead_ii_pos = None
        other_lead_pos = None
        other_lead_name = None
        for i, lead in enumerate(self.additional_fields['sig_name']):
            if lead == 'MLII':
                lead_ii_pos = i
            else:
                other_lead_pos = i
                other_lead_name = lead
        if lead_ii_pos is None:
            raise AssertionError("Didn't find lead 2 position. LEADS: {}".format(self.additional_fields['sig_name']))
        logging.info("LEAD 2 position: {}".format(lead_ii_pos))
        ecg_signal = self.signals[:, lead_ii_pos]
        ecg_signal_other_lead = self.signals[:, other_lead_pos]
        r_peak_locations = self.r_peaks_locations[1:]

        # convert seconds to samples
        before = int(before * sampling_rate)  # Number of samples per 200 ms.
        after = int(after * sampling_rate)  # number of samples per 400 ms.

        len_of_signal = len(ecg_signal)

        heart_beats = []
        logging.info("Number of r-peaks: %d", len(r_peak_locations))
        num_skip_heartbeats = 0
        for ind, r_peak in enumerate(r_peak_locations):
            start = r_peak - before
            if start < 0:
                logging.info("Skipping beat {}".format(ind))
                continue
            end = r_peak + after
            if end > len_of_signal - 1:
                logging.info("Skipping beat {}".format(ind))
                break
            if ind >= len(self.q_peaks):
                break

            heart_beats_dict = {}
            heart_beat = np.array(ecg_signal[start:end])
            heart_beat_other_lead = np.array(ecg_signal_other_lead[start:end])
            heart_beats_dict['patient_number'] = self.patient_number
            heart_beats_dict['cardiac_cycle'] = heart_beat
            heart_beats_dict['cardiac_cycle_other_lead'] = heart_beat_other_lead
            aami_label_str = heartbeat_types.convert_heartbeat_mit_bih_to_aami(self.mit_bih_labels_str[ind])
            aami_label_ind = heartbeat_types.convert_heartbeat_mit_bih_to_aami_index_class(self.mit_bih_labels_str[ind])
            heart_beats_dict['mit_bih_label_str'] = self.mit_bih_labels_str[ind]
            heart_beats_dict['aami_label_str'] = aami_label_str
            heart_beats_dict['aami_label_ind'] = aami_label_ind
            heart_beats_dict['aami_label_one_hot'] = heartbeat_types.convert_to_one_hot(aami_label_ind)
            heart_beats_dict['beat_ind'] = ind
            heart_beats_dict['lead'] = 'MLII'
            heart_beats_dict['other_lead_name'] = other_lead_name
            heart_beats_dict['r'] = r_peak - start
            heart_beats_dict['q'] = self.q_peaks[ind] - start
            heart_beats_dict['p'] = self.p_peaks[ind] - start
            heart_beats_dict['s'] = self.s_peaks[ind] - start
            heart_beats_dict['t'] = self.t_peaks[ind] - start

            # Validate values of PQRST:
            if heart_beats_dict['p'] < 0 or heart_beats_dict['t'] >= len(heart_beat):
                logging.info("Skipping heartbeat...")
                num_skip_heartbeats += 1
                continue

            heart_beats.append(heart_beats_dict)
        logging.info("Skipped %d heartbeats...", num_skip_heartbeats)
        return heart_beats

    def get_heartbeats_of_type(self, aami_label_str):
        """

        :param aami_label_str:
        :return:
        """
        return [hb for hb in self.heartbeats if hb['aami_label_str'] == aami_label_str]

    def num_heartbeats(self, aami_label_str):
        """

        :param aami_label_str:
        :return:
        """
        return len(self.get_heartbeats_of_type(aami_label_str))

    def heartbeats_summaries(self):
        """Create summaries:


        :return:
        """
        heartbeat_summaries = []
        for hb_aami in heartbeat_types.AAMIHeartBeatTypes:
            hb_summary = {}
            hb_summary['heartbeat_aami_str'] = hb_aami.name
            num_hb = self.num_heartbeats(hb_aami.name)
            hb_summary['number_of_beats'] = num_hb
            heartbeat_summaries.append(hb_summary)
        total_summary = {}
        total_summary['heartbeat_aami_str'] = 'ALL'
        total_summary['number_of_beats'] = len(self.heartbeats)
        heartbeat_summaries.append(total_summary)
        return pd.DataFrame(heartbeat_summaries)

    def get_patient_df(self):
        """Get data frame with patient details per heartbeat.

        :return: pandas dataframe.
        """
        df = pd.DataFrame(self.heartbeats)
        df.drop(columns=['cardiac_cycle'], inplace=True)
        df.drop(columns=['cardiac_cycle_other_lead'], inplace=True)
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p_100 = Patient('101')
    df = p_100.get_patient_df()
    print(df)
    print(p_100.heartbeats_summaries())
    print("Number of heartbeats: %d" % len(p_100.heartbeats))
    hb = p_100.heartbeats[100]

    plt.plot(hb['cardiac_cycle'])
    plt.plot(hb['p'], hb['cardiac_cycle'][hb['p']], 'o', color='red')
    plt.plot(hb['q'], hb['cardiac_cycle'][hb['q']], 'o', color='yellow')
    plt.plot(hb['r'], hb['cardiac_cycle'][hb['r']], 'o', color='black')
    plt.plot(hb['s'], hb['cardiac_cycle'][hb['s']], 'o', color='orange')
    plt.plot(hb['t'], hb['cardiac_cycle'][hb['t']], 'o', color='green')
    plt.show()
