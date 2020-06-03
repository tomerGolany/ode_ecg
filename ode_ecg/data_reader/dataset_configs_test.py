import unittest
from ecg_pytorch.data_reader import dataset_configs


class TestDatasetConfigs(unittest.TestCase):
    def test_inputs(self):
        with self.assertRaises(ValueError):
            dataset_configs.DatasetConfigs('validation', 'N', one_vs_all=True, lstm_setting=False,
                                           over_sample_minority_class=False,
                                           under_sample_majority_class=False, only_take_heartbeat_of_type='N')

        ds = dataset_configs.DatasetConfigs('train', 'N', one_vs_all=True, lstm_setting=False,
                                            over_sample_minority_class=False,
                                            under_sample_majority_class=False, only_take_heartbeat_of_type='N')
        self.assertEqual(ds.partition, 'train')
        self.assertEqual(ds.classified_heartbeat, 'N')
        self.assertEqual(ds.one_vs_all, True)
        self.assertEqual(ds.lstm_setting, False)
        self.assertEqual(ds.over_sample_minority_class, False)
        self.assertEqual(ds.under_sample_majority_class, False)

        ds = dataset_configs.DatasetConfigs('test', 'N', one_vs_all=True, lstm_setting=False,
                                            over_sample_minority_class=False,
                                            under_sample_majority_class=False, only_take_heartbeat_of_type='N')

        self.assertEqual(ds.partition, 'test')

        ds = dataset_configs.DatasetConfigs('test', 'S', one_vs_all=True, lstm_setting=False,
                                            over_sample_minority_class=False,
                                            under_sample_majority_class=False, only_take_heartbeat_of_type='N')
        self.assertEqual(ds.classified_heartbeat, 'S')

        ds = dataset_configs.DatasetConfigs('test', 'V', one_vs_all=True, lstm_setting=False,
                                            over_sample_minority_class=False,
                                            under_sample_majority_class=False, only_take_heartbeat_of_type='N')
        self.assertEqual(ds.classified_heartbeat, 'V')
        ds = dataset_configs.DatasetConfigs('test', 'F', one_vs_all=True, lstm_setting=False,
                                            over_sample_minority_class=False,
                                            under_sample_majority_class=False, only_take_heartbeat_of_type='N')
        self.assertEqual(ds.classified_heartbeat, 'F')
        ds = dataset_configs.DatasetConfigs('test', 'Q', one_vs_all=True, lstm_setting=False,
                                            over_sample_minority_class=False,
                                            under_sample_majority_class=False, only_take_heartbeat_of_type='N')
        self.assertEqual(ds.classified_heartbeat, 'Q')
        with self.assertRaises(ValueError):
            dataset_configs.DatasetConfigs('train', 'P', one_vs_all=True, lstm_setting=False,
                                           over_sample_minority_class=False,
                                           under_sample_majority_class=False, only_take_heartbeat_of_type='N')

    def test_only_take_hb_of_type(self):
        with self.assertRaises(ValueError):
            dataset_configs.DatasetConfigs('train', 'N', one_vs_all=True, lstm_setting=False,
                                           over_sample_minority_class=False,
                                           under_sample_majority_class=False, only_take_heartbeat_of_type='w')

        ds = dataset_configs.DatasetConfigs('train', 'N', one_vs_all=True, lstm_setting=False,
                                            over_sample_minority_class=False,
                                            under_sample_majority_class=False, only_take_heartbeat_of_type=None)
        self.assertEqual(ds.only_take_heartbeat_of_type, None)

        ds = dataset_configs.DatasetConfigs('train', 'N', one_vs_all=True, lstm_setting=False,
                                            over_sample_minority_class=False,
                                            under_sample_majority_class=False, only_take_heartbeat_of_type='S')
        self.assertEqual(ds.only_take_heartbeat_of_type, 'S')

        ds = dataset_configs.DatasetConfigs('train', 'N', one_vs_all=True, lstm_setting=False,
                                            over_sample_minority_class=False,
                                            under_sample_majority_class=False, only_take_heartbeat_of_type='N')
        self.assertEqual(ds.only_take_heartbeat_of_type, 'N')

        ds = dataset_configs.DatasetConfigs('train', 'N', one_vs_all=True, lstm_setting=False,
                                            over_sample_minority_class=False,
                                            under_sample_majority_class=False, only_take_heartbeat_of_type='V')
        self.assertEqual(ds.only_take_heartbeat_of_type, 'V')

        ds = dataset_configs.DatasetConfigs('train', 'N', one_vs_all=True, lstm_setting=False,
                                            over_sample_minority_class=False,
                                            under_sample_majority_class=False, only_take_heartbeat_of_type='F')
        self.assertEqual(ds.only_take_heartbeat_of_type, 'F')

        ds = dataset_configs.DatasetConfigs('train', 'N', one_vs_all=True, lstm_setting=False,
                                            over_sample_minority_class=False,
                                            under_sample_majority_class=False, only_take_heartbeat_of_type='Q')
        self.assertEqual(ds.only_take_heartbeat_of_type, 'Q')


if __name__ == '__main__':
    unittest.main()
