# Lint as: python3
"""Train a physical ODE GAN."""
from torchdiffeq import odeint
import torch
from ode_ecg.data_reader import dataset_configs
from ode_ecg.data_reader import ecg_dataset_pytorch
from ode_ecg.models.architectures import physical_ode_resnet
from torch import nn
from torch import optim
import logging
import matplotlib.pyplot as plt
import os
import numpy as np


def mit_bih_heartbeat_dataloader(heartbeat_type: str, batch_size: int) -> torch.utils.data.DataLoader:
    """Create Pytorch dataloader which loads heartbeats from MIT-BIH dataset.

    :param heartbeat_type: Type of heartbeat to hold in the dataset.
    :param batch_size: Size of batch per iteration.
    :return: ECG heartbeats dataset of a specific type of heartbeat.
    """
    positive_configs = dataset_configs.DatasetConfigs('train', heartbeat_type, one_vs_all=True, lstm_setting=False,
                                                      over_sample_minority_class=False,
                                                      under_sample_majority_class=False,
                                                      only_take_heartbeat_of_type=heartbeat_type,
                                                      add_data_from_gan=False,
                                                      gan_configs=None)
    dataset = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(positive_configs,
                                                             transform=ecg_dataset_pytorch.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=1)
    logging.info(f"Number of heartbeats of type {heartbeat_type} in training set: {len(dataset)}")
    return dataloader


def generate_fake_heartbeats(generator_obj, initial_value, device: torch.device, method):
    timestamps = torch.from_numpy(np.linspace(1, 216, num=216)).float().to(device)
    fake_heartbeats = odeint(generator_obj, initial_value, timestamps, method=method)
    fake_heartbeats = fake_heartbeats.permute(1, 0, 2).view(-1, 216)
    return fake_heartbeats


def discriminator_training_step(discriminator_obj, fake_heartbeats, real_heartbeats_batch, loss_function,
                                discriminator_optimizer, device: torch.device):
    b_size = real_heartbeats_batch.shape[0]
    labels = torch.full((b_size,), 1, device=device)

    discriminator_obj.zero_grad()

    # Input from real heartbeats:
    predictions_on_real_heartbeats = discriminator_obj(real_heartbeats_batch)
    loss_on_real_heartbeats = loss_function(predictions_on_real_heartbeats, labels)
    loss_on_real_heartbeats.backward()
    mean_prediction_on_real_heartbeats = predictions_on_real_heartbeats.mean().item()

    predictions_from_fake_heartbeats = discriminator_obj(fake_heartbeats.detach())
    labels.fill_(0)
    loss_on_fake_heartbeats = loss_function(predictions_from_fake_heartbeats, labels)
    mean_prediction_on_fake_heartbeats = predictions_from_fake_heartbeats.mean().item()
    loss_on_fake_heartbeats.backward()
    total_loss = loss_on_real_heartbeats + loss_on_fake_heartbeats
    discriminator_optimizer.step()
    return {
        "total_loss": total_loss,
        "loss_on_real_heartbeats": loss_on_real_heartbeats,
        "loss_on_fake_heartbeats": loss_on_fake_heartbeats,
        "mean_prediction_real_heartbeats": mean_prediction_on_real_heartbeats,
        "mean_prediction_fake_heartbeats": mean_prediction_on_fake_heartbeats,
    }


def generator_training_step(generator_obj, discriminator_obj, fake_heartbeats, loss_function, generator_optimizer,
                            device: torch.device):
    """Perform single Generator optimization step.

    :param generator_obj:
    :param discriminator_obj:
    :param fake_heartbeats:
    :param loss_function:
    :param generator_optimizer:
    :param device
    :return:
    """
    b_size = fake_heartbeats.shape[0]
    labels = torch.full((b_size,), 1, device=device)
    generator_obj.zero_grad()
    discriminator_prediction_on_fake_heartbeats = discriminator_obj(fake_heartbeats)
    loss = loss_function(discriminator_prediction_on_fake_heartbeats, labels)
    loss.backward()
    generator_optimizer.step()
    return {
        "total_loss": loss,
        "mean_prediction_on_fake": discriminator_prediction_on_fake_heartbeats.mean().item()
    }


def train_physical_ode_gan(batch_size: int, num_iterations: int, model_dir: str, beat_type: str, device: torch.device,
                           method: str):
    """Train a physical ODE-GAN.

    :param batch_size: Size of Batch.
    :param num_iterations: Number of training iterations.
    :param model_dir: Model directory to write checkpoints and summaries.
    :param beat_type: Type of heartbeat to generate.
    :param device: GPU device to train on or cpu.
    :param method: Integration method to solve to ODE.
    :return: None.
    """

    if os.path.isdir(model_dir):
        logging.info("Dir already exists... exiting...")
        exit(-1)
    else:
        os.mkdir(os.path.join(model_dir))

    dataloader = mit_bih_heartbeat_dataloader(beat_type, batch_size)

    num_heartbeats = len(dataloader.dataset)

    generator_network = physical_ode_resnet.ResGenerator(device, None).to(device)
    discriminator_network = physical_ode_resnet.DCDiscriminator(0).to(device)

    # Loss functions:
    cross_entropy_loss = nn.BCELoss()

    # Optimizers:
    lr = 0.0002
    beta1 = 0.5
    optimizer_d = optim.Adam(discriminator_network.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_g = optim.Adam(generator_network.parameters(), lr=lr, betas=(beta1, 0.999))

    # Noise for validation:
    loss_d_real_hist = []
    loss_d_fake_hist = []
    loss_g_fake_hist = []
    d_real_pred_hist = []
    d_fake_pred_hist = []
    epoch = 0
    iters = 0
    while True:
        epoch += 1
        num_of_beats_seen = 0
        if iters == num_iterations:
            break
        for i, data in enumerate(dataloader):
            if iters == num_iterations:
                break
            ecg_batch = data['cardiac_cycle'].float().to(device)
            physical_params = data['physical_params'].float().to(device)
            generator_network.physical_params = physical_params
            b_size = ecg_batch.shape[0]
            if b_size != batch_size:
                logging.info(f"End of epoch {epoch}...")
                continue

            v0 = ecg_batch[:, 0].view(-1, 1)  # For ODE solver initial step.
            num_of_beats_seen += ecg_batch.shape[0]

            fake_heartbeats = generate_fake_heartbeats(generator_network, v0, device, method)

            discriminator_outputs = discriminator_training_step(discriminator_network, fake_heartbeats,
                                                                ecg_batch, cross_entropy_loss, optimizer_d, device)
            loss_d_real_hist.append(discriminator_outputs['loss_on_real_heartbeats'].item())
            d_real_pred_hist.append(discriminator_outputs['mean_prediction_real_heartbeats'])
            loss_d_fake_hist.append(discriminator_outputs['loss_on_fake_heartbeats'].item())
            d_fake_pred_hist.append(discriminator_outputs['mean_prediction_fake_heartbeats'])

            generator_outputs = generator_training_step(generator_network, discriminator_network, fake_heartbeats,
                                                        cross_entropy_loss, optimizer_g, device)
            loss_g_fake_hist.append(generator_outputs['total_loss'].item())

            if iters % 25 == 0:
                logging.info(f"{num_of_beats_seen}/{num_heartbeats}: Epoch #{epoch}: Iteration #{iters}: "
                             f"Mean D(real_hb_batch) = {discriminator_outputs['mean_prediction_real_heartbeats']}, "
                             f"mean D(G(z)) = {discriminator_outputs['mean_prediction_fake_heartbeats']}.")
                logging.info(f"mean D(G(z)) = {generator_outputs['mean_prediction_on_fake']} After backprop of D")
                logging.info(f"Loss D from real beats = {discriminator_outputs['loss_on_real_heartbeats'].item()}. "
                             f"Loss D from Fake beats = {discriminator_outputs['loss_on_fake_heartbeats'].item()}. "
                             f"Total Loss D = {discriminator_outputs['total_loss'].item()}")
                logging.info(f"Loss G = {generator_outputs['total_loss']}")

                with torch.no_grad():
                    generator_output = generate_fake_heartbeats(generator_network, v0, device, method)
                    plt.figure()
                    plt.title("Fake beats from Generator. iteration {}".format(i))
                    physical_params = physical_params.detach().cpu().numpy()
                    for p in range(4):
                        plt.subplot(2, 2, p + 1)
                        plt.plot(generator_output[p].detach().cpu().numpy(), label="fake beat")
                        plt.plot(ecg_batch[p].detach().cpu().numpy(), label="real beat")
                        plt.plot(int(physical_params[p][0]), physical_params[p][1], 'o', color='red')
                        plt.plot(int(physical_params[p][2]), physical_params[p][3], 'o', color='yellow')
                        plt.plot(int(physical_params[p][4]), physical_params[p][5], 'o', color='black')
                        plt.plot(int(physical_params[p][6]), physical_params[p][7], 'o', color='orange')
                        plt.plot(int(physical_params[p][8]), physical_params[p][9], 'o', color='green')
                        plt.legend()
                    plt.savefig(os.path.join(model_dir, 'generator_output_{}.png'.format(iters)))
                    plt.show()
                    plt.close()
            iters += 1
        epoch += 1
    logging.info("Training complete...")
    # torch.save({
    #     'epoch': epoch,
    #     'generator_state_dict': netG.state_dict(),
    #     'discriminator_state_dict': netD.state_dict(),
    #     'optimizer_g_state_dict': optimizer_g.state_dict(),
    #     'optimizer_d_state_dict': optimizer_d.state_dict(),
    #     'loss': cross_entropy_loss,
    #
    # }, model_dir + '/checkpoint_epoch_{}_iters_{}'.format(epoch, iters))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", main_device)
    train_physical_ode_gan(batch_size=40, num_iterations=100000,
                           model_dir='/Users/tomer.golany/Desktop/ecg_tweleve_lead_research/ode_gan/model_summaries/physical_ode_gan/6',
                           beat_type='N', device=main_device, method='euler')
