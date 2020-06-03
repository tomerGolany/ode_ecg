import pickle
from ode_ecg import train_configs
from ode_ecg.data_reader import ecg_mit_bih
import logging
full_path = train_configs.base + 'ode_ecg/ode_ecg/data_reader'


def load_ecg_input_from_pickle():
    with open(full_path + '/train_beats.pickle', 'rb') as handle:
        train_beats = pickle.load(handle)
    with open(full_path + '/val_beats.pickle', 'rb') as handle:
        validation_beats = pickle.load(handle)
    with open(full_path + '/test_beats.pickle', 'rb') as handle:
        test_beats = pickle.load(handle)
    return train_beats, validation_beats, test_beats


def save_ecg_mit_bih_to_pickle():
    print("start pickling:")
    with open(full_path + '/ecg_mit_bih.pickle', 'wb') as output:
        ecg_ds = ecg_mit_bih.ECGMitBihDataset()
        pickle.dump(ecg_ds, output, pickle.HIGHEST_PROTOCOL)
        logging.info("Done pickling")


def load_ecg_mit_bih_from_pickle():
    logging.info("Loading ecg-mit-bih from pickle...")
    with open(full_path + '/ecg_mit_bih.pickle', 'rb') as handle:
        ecg_ds = pickle.load(handle)
        logging.info("Loaded successfully...")
        return ecg_ds


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    save_ecg_mit_bih_to_pickle()
