import logging
import unittest
from ecg_pytorch.data_reader import ecg_dataset_pytorch, dataset_configs, heartbeat_types


class TestEcgHearBeatsDatasetPytorch(unittest.TestCase):
    def test_one_vs_all_N_train(self):
        configs = dataset_configs.DatasetConfigs('train', 'N', one_vs_all=True, lstm_setting=False,
                                                 over_sample_minority_class=False,
                                                 under_sample_majority_class=False, only_take_heartbeat_of_type=None)
        ds = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(configs, transform=None)

        for sample in ds:
            heartbeat = sample['cardiac_cycle']
            label_str = sample['beat_type']
            label_one_hot = sample['label']
            self.assertEqual(len(label_one_hot), 2)
            self.assertIn(label_one_hot[0], [0, 1])
            self.assertIn(label_one_hot[1], [0, 1])

            self.assertIn(label_str, heartbeat_types.AAMIHeartBeatTypes.__members__)
            if label_str == heartbeat_types.AAMIHeartBeatTypes.N.name:
                self.assertEqual(list(label_one_hot), ([1, 0]))
            else:
                self.assertEqual(list(label_one_hot), ([0, 1]))

            self.assertEqual(len(heartbeat), 216)

    def test_one_vs_all_N_test(self):
        configs = dataset_configs.DatasetConfigs('test', 'N', one_vs_all=True, lstm_setting=False,
                                                 over_sample_minority_class=False,
                                                 under_sample_majority_class=False, only_take_heartbeat_of_type=None)
        ds = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(configs, transform=None)
        logging.info("Length dataset: {}".format(len(ds)))
        for sample in ds:
            heartbeat = sample['cardiac_cycle']
            label_str = sample['beat_type']
            label_one_hot = sample['label']
            self.assertEqual(len(label_one_hot), 2)
            self.assertIn(label_one_hot[0], [0, 1])
            self.assertIn(label_one_hot[1], [0, 1])

            self.assertIn(label_str, heartbeat_types.AAMIHeartBeatTypes.__members__)
            if label_str == heartbeat_types.AAMIHeartBeatTypes.N.name:
                self.assertEqual(list(label_one_hot), ([1, 0]))
            else:
                self.assertEqual(list(label_one_hot), ([0, 1]))

            self.assertEqual(len(heartbeat), 216)

    def test_one_vs_all_N_only(self):
        configs = dataset_configs.DatasetConfigs('train', 'N', one_vs_all=True, lstm_setting=False,
                                                 over_sample_minority_class=False,
                                                 under_sample_majority_class=False, only_take_heartbeat_of_type='N')
        ds = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(configs, transform=None)
        self.assertEqual(len(ds), heartbeat_types.NUMBER_OF_HEART_BEATS_TRAIN['N'])
        for sample in ds:
            heartbeat = sample['cardiac_cycle']
            label_str = sample['beat_type']
            label_one_hot = sample['label']
            self.assertEqual(len(label_one_hot), 2)
            self.assertEqual(list(label_one_hot), ([1, 0]))
            self.assertEqual(label_str, heartbeat_types.AAMIHeartBeatTypes.N.name)
            self.assertEqual(len(heartbeat), 216)

    def test_one_vs_all_N_only_others(self):
        configs = dataset_configs.DatasetConfigs('train', 'N', one_vs_all=True, lstm_setting=False,
                                                 over_sample_minority_class=False,
                                                 under_sample_majority_class=False, only_take_heartbeat_of_type=
                                                 heartbeat_types.OTHER_HEART_BEATS)
        ds = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(configs, transform=None)
        self.assertGreater(len(ds), 0)
        self.assertEqual(len(ds), sum(heartbeat_types.NUMBER_OF_HEART_BEATS_TRAIN.values()) -
                                      heartbeat_types.NUMBER_OF_HEART_BEATS_TRAIN['N'])
        for sample in ds:
            heartbeat = sample['cardiac_cycle']
            label_str = sample['beat_type']
            label_one_hot = sample['label']
            self.assertEqual(len(label_one_hot), 2)
            self.assertEqual(list(label_one_hot), ([0, 1]))
            self.assertNotEqual(label_str, heartbeat_types.AAMIHeartBeatTypes.N.name)
            self.assertEqual(len(heartbeat), 216)

    def test_one_vs_all_S(self):
        configs = dataset_configs.DatasetConfigs('train', 'S', one_vs_all=True, lstm_setting=False,
                                                 over_sample_minority_class=False,
                                                 under_sample_majority_class=False, only_take_heartbeat_of_type=None)
        ds = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(configs, transform=None)

        for sample in ds:
            heartbeat = sample['cardiac_cycle']
            label_str = sample['beat_type']
            label_one_hot = sample['label']
            self.assertEqual(len(label_one_hot), 2)
            self.assertIn(label_one_hot[0], [0, 1])
            self.assertIn(label_one_hot[1], [0, 1])

            self.assertIn(label_str, heartbeat_types.AAMIHeartBeatTypes.__members__)
            if label_str == heartbeat_types.AAMIHeartBeatTypes.S.name:
                self.assertEqual(list(label_one_hot), ([1, 0]))
            else:
                self.assertEqual(list(label_one_hot), ([0, 1]))

            self.assertEqual(len(heartbeat), 216)

    def test_one_vs_all_S_only(self):
        configs = dataset_configs.DatasetConfigs('train', 'S', one_vs_all=True, lstm_setting=False,
                                                 over_sample_minority_class=False,
                                                 under_sample_majority_class=False, only_take_heartbeat_of_type='S')
        ds = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(configs, transform=None)
        self.assertEqual(len(ds), heartbeat_types.NUMBER_OF_HEART_BEATS_TRAIN['S'])
        for sample in ds:
            heartbeat = sample['cardiac_cycle']
            label_str = sample['beat_type']
            label_one_hot = sample['label']
            self.assertEqual(len(label_one_hot), 2)
            self.assertEqual(list(label_one_hot), ([1, 0]))
            self.assertEqual(label_str, heartbeat_types.AAMIHeartBeatTypes.S.name)
            self.assertEqual(len(heartbeat), 216)

    def test_one_vs_all_S_only_others(self):
        configs = dataset_configs.DatasetConfigs('train', 'S', one_vs_all=True, lstm_setting=False,
                                                 over_sample_minority_class=False,
                                                 under_sample_majority_class=False, only_take_heartbeat_of_type=
                                                 heartbeat_types.OTHER_HEART_BEATS)
        ds = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(configs, transform=None)
        self.assertGreater(len(ds), 0)
        self.assertEqual(len(ds), sum(heartbeat_types.NUMBER_OF_HEART_BEATS_TRAIN.values()) -
                         heartbeat_types.NUMBER_OF_HEART_BEATS_TRAIN['S'])
        for sample in ds:
            heartbeat = sample['cardiac_cycle']
            label_str = sample['beat_type']
            label_one_hot = sample['label']
            self.assertEqual(len(label_one_hot), 2)
            self.assertEqual(list(label_one_hot), ([0, 1]))
            self.assertNotEqual(label_str, heartbeat_types.AAMIHeartBeatTypes.S.name)
            self.assertEqual(len(heartbeat), 216)

    def test_one_vs_all_V(self):
        configs = dataset_configs.DatasetConfigs('train', 'V', one_vs_all=True, lstm_setting=False,
                                                 over_sample_minority_class=False,
                                                 under_sample_majority_class=False, only_take_heartbeat_of_type=None)
        ds = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(configs, transform=None)

        for sample in ds:
            heartbeat = sample['cardiac_cycle']
            label_str = sample['beat_type']
            label_one_hot = sample['label']
            self.assertEqual(len(label_one_hot), 2)
            self.assertIn(label_one_hot[0], [0, 1])
            self.assertIn(label_one_hot[1], [0, 1])

            self.assertIn(label_str, heartbeat_types.AAMIHeartBeatTypes.__members__)
            if label_str == heartbeat_types.AAMIHeartBeatTypes.V.name:
                self.assertEqual(list(label_one_hot), ([1, 0]))
            else:
                self.assertEqual(list(label_one_hot), ([0, 1]))

            self.assertEqual(len(heartbeat), 216)

    def test_one_vs_all_V_only(self):
        configs = dataset_configs.DatasetConfigs('train', 'V', one_vs_all=True, lstm_setting=False,
                                                 over_sample_minority_class=False,
                                                 under_sample_majority_class=False, only_take_heartbeat_of_type='V')
        ds = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(configs, transform=None)
        self.assertEqual(len(ds), heartbeat_types.NUMBER_OF_HEART_BEATS_TRAIN['V'])
        for sample in ds:
            heartbeat = sample['cardiac_cycle']
            label_str = sample['beat_type']
            label_one_hot = sample['label']
            self.assertEqual(len(label_one_hot), 2)
            self.assertEqual(list(label_one_hot), ([1, 0]))
            self.assertEqual(label_str, heartbeat_types.AAMIHeartBeatTypes.V.name)
            self.assertEqual(len(heartbeat), 216)

    def test_one_vs_all_V_only_others(self):
        configs = dataset_configs.DatasetConfigs('train', 'V', one_vs_all=True, lstm_setting=False,
                                                 over_sample_minority_class=False,
                                                 under_sample_majority_class=False, only_take_heartbeat_of_type=
                                                 heartbeat_types.OTHER_HEART_BEATS)
        ds = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(configs, transform=None)
        self.assertGreater(len(ds), 0)
        self.assertEqual(len(ds), sum(heartbeat_types.NUMBER_OF_HEART_BEATS_TRAIN.values()) -
                         heartbeat_types.NUMBER_OF_HEART_BEATS_TRAIN['V'])
        for sample in ds:
            heartbeat = sample['cardiac_cycle']
            label_str = sample['beat_type']
            label_one_hot = sample['label']
            self.assertEqual(len(label_one_hot), 2)
            self.assertEqual(list(label_one_hot), ([0, 1]))
            self.assertNotEqual(label_str, heartbeat_types.AAMIHeartBeatTypes.V.name)
            self.assertEqual(len(heartbeat), 216)

    def test_one_vs_all_F(self):
        configs = dataset_configs.DatasetConfigs('train', 'F', one_vs_all=True, lstm_setting=False,
                                                 over_sample_minority_class=False,
                                                 under_sample_majority_class=False, only_take_heartbeat_of_type=None)
        ds = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(configs, transform=None)

        for sample in ds:
            heartbeat = sample['cardiac_cycle']
            label_str = sample['beat_type']
            label_one_hot = sample['label']
            self.assertEqual(len(label_one_hot), 2)
            self.assertIn(label_one_hot[0], [0, 1])
            self.assertIn(label_one_hot[1], [0, 1])

            self.assertIn(label_str, heartbeat_types.AAMIHeartBeatTypes.__members__)
            if label_str == heartbeat_types.AAMIHeartBeatTypes.F.name:
                self.assertEqual(list(label_one_hot), ([1, 0]))
            else:
                self.assertEqual(list(label_one_hot), ([0, 1]))

            self.assertEqual(len(heartbeat), 216)

    def test_one_vs_all_F_only(self):
        configs = dataset_configs.DatasetConfigs('train', 'F', one_vs_all=True, lstm_setting=False,
                                                 over_sample_minority_class=False,
                                                 under_sample_majority_class=False, only_take_heartbeat_of_type='F')
        ds = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(configs, transform=None)
        self.assertEqual(len(ds), heartbeat_types.NUMBER_OF_HEART_BEATS_TRAIN['F'])
        for sample in ds:
            heartbeat = sample['cardiac_cycle']
            label_str = sample['beat_type']
            label_one_hot = sample['label']
            self.assertEqual(len(label_one_hot), 2)
            self.assertEqual(list(label_one_hot), ([1, 0]))
            self.assertEqual(label_str, heartbeat_types.AAMIHeartBeatTypes.F.name)
            self.assertEqual(len(heartbeat), 216)


    def test_one_vs_all_F_only_others(self):
        configs = dataset_configs.DatasetConfigs('train', 'F', one_vs_all=True, lstm_setting=False,
                                                 over_sample_minority_class=False,
                                                 under_sample_majority_class=False, only_take_heartbeat_of_type=
                                                 heartbeat_types.OTHER_HEART_BEATS)
        ds = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(configs, transform=None)
        self.assertGreater(len(ds), 0)
        self.assertEqual(len(ds), sum(heartbeat_types.NUMBER_OF_HEART_BEATS_TRAIN.values()) -
                         heartbeat_types.NUMBER_OF_HEART_BEATS_TRAIN['F'])
        for sample in ds:
            heartbeat = sample['cardiac_cycle']
            label_str = sample['beat_type']
            label_one_hot = sample['label']
            self.assertEqual(len(label_one_hot), 2)
            self.assertEqual(list(label_one_hot), ([0, 1]))
            self.assertNotEqual(label_str, heartbeat_types.AAMIHeartBeatTypes.F.name)
            self.assertEqual(len(heartbeat), 216)

    def test_one_vs_all_Q(self):
        configs = dataset_configs.DatasetConfigs('train', 'Q', one_vs_all=True, lstm_setting=False,
                                                 over_sample_minority_class=False,
                                                 under_sample_majority_class=False, only_take_heartbeat_of_type=None)
        ds = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(configs, transform=None)

        for sample in ds:
            heartbeat = sample['cardiac_cycle']
            label_str = sample['beat_type']
            label_one_hot = sample['label']
            self.assertEqual(len(label_one_hot), 2)
            self.assertIn(label_one_hot[0], [0, 1])
            self.assertIn(label_one_hot[1], [0, 1])

            self.assertIn(label_str, heartbeat_types.AAMIHeartBeatTypes.__members__)
            if label_str == heartbeat_types.AAMIHeartBeatTypes.Q.name:
                self.assertEqual(list(label_one_hot), ([1, 0]))
            else:
                self.assertEqual(list(label_one_hot), ([0, 1]))

            self.assertEqual(len(heartbeat), 216)

    def test_one_vs_all_Q_only(self):
        configs = dataset_configs.DatasetConfigs('train', 'Q', one_vs_all=True, lstm_setting=False,
                                                 over_sample_minority_class=False,
                                                 under_sample_majority_class=False, only_take_heartbeat_of_type='Q')
        ds = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(configs, transform=None)
        self.assertEqual(len(ds), heartbeat_types.NUMBER_OF_HEART_BEATS_TRAIN['Q'])
        for sample in ds:
            heartbeat = sample['cardiac_cycle']
            label_str = sample['beat_type']
            label_one_hot = sample['label']
            self.assertEqual(len(label_one_hot), 2)
            self.assertEqual(list(label_one_hot), ([1, 0]))
            self.assertEqual(label_str, heartbeat_types.AAMIHeartBeatTypes.Q.name)
            self.assertEqual(len(heartbeat), 216)

    def test_one_vs_all_Q_only_others(self):
        configs = dataset_configs.DatasetConfigs('train', 'Q', one_vs_all=True, lstm_setting=False,
                                                 over_sample_minority_class=False,
                                                 under_sample_majority_class=False, only_take_heartbeat_of_type=
                                                 heartbeat_types.OTHER_HEART_BEATS)
        ds = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(configs, transform=None)
        self.assertGreater(len(ds), 0)
        self.assertEqual(len(ds), sum(heartbeat_types.NUMBER_OF_HEART_BEATS_TRAIN.values()) -
                         heartbeat_types.NUMBER_OF_HEART_BEATS_TRAIN['Q'])
        for sample in ds:
            heartbeat = sample['cardiac_cycle']
            label_str = sample['beat_type']
            label_one_hot = sample['label']
            self.assertEqual(len(label_one_hot), 2)
            self.assertEqual(list(label_one_hot), ([0, 1]))
            self.assertNotEqual(label_str, heartbeat_types.AAMIHeartBeatTypes.Q.name)
            self.assertEqual(len(heartbeat), 216)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
