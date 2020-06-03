from ecg_pytorch.data_reader.ecg_dataset_pytorch import EcgHearBeatsDataset, ToTensor
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import logging


def TestEcgDatasetLSTM_1():
    ecg_dataset = EcgHearBeatsDataset()
    fig = plt.figure()

    for i in range(4):
        sample = ecg_dataset[i]

        print(i, sample['cardiac_cycle'].shape, sample['label'].shape, sample['beat_type'])

        ax = plt.subplot(2, 2, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        beat = sample['cardiac_cycle'].reshape((43, 5))
        beat = beat.flatten()
        plt.plot(beat)
    plt.show()


def TestIteration():
    composed = transforms.Compose([ToTensor()])
    ecg_dataset = EcgHearBeatsDataset(transform=composed)
    for i in range(4):
        sample = ecg_dataset[i]
        print(i, sample['cardiac_cycle'].size(), sample['label'].size())


def test_iterate_with_dataloader():
    composed = transforms.Compose([ToTensor()])
    transformed_dataset = EcgHearBeatsDataset(transform=composed)
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=4)

    # Helper function to show a batch
    def show_landmarks_batch(sample_batched):
        """Show image with landmarks for a batch of samples."""
        ecg_batch, label_batch = \
            sample_batched['cardiac_cycle'], sample_batched['label']
        batch_size = len(ecg_batch)

        for i in range(batch_size):
            ax = plt.subplot(2, 2, i + 1)
            plt.tight_layout()
            ax.set_title('Sample #{}'.format(i))
            # ax.axis('off')
            plt.plot(ecg_batch[i].numpy())
            print(label_batch[i])
            # plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
            #             landmarks_batch[i, :, 1].numpy(),
            #             s=10, marker='.', c='r')
            #
            # plt.title('Batch from dataloader')

    for i_batch, sample_batched in enumerate(dataloader):
        # print(sample_batched['cardiac_cycle'].shape)
        ecg_lstm_batch = sample_batched['cardiac_cycle'].permute(1, 0, 2)
        print(i_batch, ecg_lstm_batch.size(),
              sample_batched['label'].size())

        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            # show_landmarks_batch(sample_batched)
            # # plt.axis('off')
            beat = ecg_lstm_batch[:, 0, :]
            print(beat.shape)
            beat = beat.flatten()
            plt.plot(beat.numpy())
            plt.show()
            # plt.ioff()
            # plt.show()
            break


def test_one_vs_all():
    composed = transforms.Compose([ToTensor()])
    dataset = EcgHearBeatsDataset(transform=composed, beat_type='N', one_vs_all=True)
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        ecg_lstm_batch = sample_batched['cardiac_cycle'].view(43, -1, 5)
        print(i_batch, ecg_lstm_batch.size(),
              sample_batched['label'].size())
        print(sample_batched['label'].numpy())
        print(sample_batched['beat_type'])
        if i_batch == 50:
            break


def test_data_from_simulator():
    dataset = EcgHearBeatsDataset()
    sim_beats = dataset.add_beats_from_simulator(9, 'F')

    #
    # Debug: plot some beats:
    #
    print(sim_beats.shape)
    single_beat = sim_beats[5]
    plt.figure()
    plt.plot(single_beat)
    plt.show()


if __name__ == "__main__":
    # TestEcgDatasetLSTM_1()
    # TestIteration()
    # test_iterate_with_dataloader()
    # test_one_vs_all()
    logging.basicConfig(level=logging.INFO)
    test_data_from_simulator()