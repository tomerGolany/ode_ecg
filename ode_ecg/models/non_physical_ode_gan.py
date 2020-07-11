"""Implementation of the non-physical ode gan."""
# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ode_ecg.data_reader import dataset_configs
from ode_ecg.data_reader import ecg_dataset_pytorch
from ode_ecg import train_configs
from ode_ecg.models.architectures import resnet
import os
import logging


#
# Train:
#
def train_ecg_gan(batch_size, num_train_steps, model_dir, beat_type):
    if os.path.isdir(os.path.join(train_configs.base, 'ode_ecg', 'ode_ecg', 'models', model_dir)):
        logging.info("Dir already exists... exiting...")
        exit(-1)
    else:
        os.mkdir(os.path.join(train_configs.base, 'ode_ecg', 'ode_ecg', 'models', model_dir))
    # Support for tensorboard:

    # 1. create the ECG dataset:
    positive_configs = dataset_configs.DatasetConfigs('train', beat_type, one_vs_all=True, lstm_setting=False,
                                                      over_sample_minority_class=False,
                                                      under_sample_majority_class=False,
                                                      only_take_heartbeat_of_type=beat_type, add_data_from_gan=False,
                                                      gan_configs=None)

    dataset = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(positive_configs,
                                                             transform=ecg_dataset_pytorch.ToTensor())

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=1)
    logging.info("Size of real dataset is {}".format(len(dataset)))

    # 2. Create the models:
    netG = resnet.ResGenerator(device).to(device)
    netD = resnet.DCDiscriminator(0).to(device)

    # Loss functions:
    cross_entropy_loss = nn.BCELoss()
    # mse_loss = nn.MSELoss()

    # Optimizers:
    lr = 0.0002
    beta1 = 0.5
    optimizer_d = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_g = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Noise for validation:
    loss_d_real_hist = []
    loss_d_fake_hist = []
    loss_g_fake_hist = []
    d_real_pred_hist = []
    d_fake_pred_hist = []
    epoch = 0
    iters = 0
    while True:
        num_of_beats_seen = 0
        if iters == num_train_steps:
            break
        for i, data in enumerate(dataloader):
            if iters == num_train_steps:
                break
            logging.info("Iter {}".format(i))
            ecg_batch = data['cardiac_cycle'].float().to(device)
            b_size = ecg_batch.shape[0]
            if b_size != batch_size:
                logging.info("End of epoch...")
                continue
            labels = torch.full((b_size,), 1, device=device)
            v0 = ecg_batch[:, 0].view(-1, 1)  # For ODE solver initial step.
            num_of_beats_seen += ecg_batch.shape[0]

            #
            # Discriminator:
            #
            netD.zero_grad()

            # Feed real input:
            output = netD(ecg_batch)
            ce_loss_d_real = cross_entropy_loss(output, labels)
            ce_loss_d_real.backward()
            loss_d_real_hist.append(ce_loss_d_real.item())
            mean_d_real_output = output.mean().item()
            d_real_pred_hist.append(mean_d_real_output)

            # Create fake input from generator:
            timestamsp = torch.from_numpy(np.linspace(1, 216, num=216)).float().to(device)
            output_g_fake = odeint(netG, v0, timestamsp, method='rk4')
            output_g_fake = output_g_fake.permute(1, 0, 2).view(-1, 216)
            output = netD(output_g_fake.detach())
            labels.fill_(0)
            ce_loss_d_fake = cross_entropy_loss(output, labels)
            loss_d_fake_hist.append(ce_loss_d_fake.item())
            mean_d_fake_output = output.mean().item()
            d_fake_pred_hist.append(mean_d_fake_output)
            total_loss_d = ce_loss_d_fake + ce_loss_d_real

            # Update discriminator every 5 iterations:
            # if i % 5 == 0:
            ce_loss_d_fake.backward()
            optimizer_d.step()

            #
            # Generator optimization:
            #
            netG.zero_grad()
            labels.fill_(1)
            output = netD(output_g_fake)
            # mse_loss_real_fake = mse_loss(output_g_fake, ecg_batch)

            ce_loss_g_fake = cross_entropy_loss(output, labels)
            # total_loss = mse_loss_real_fake + ce_loss_g_fake
            ce_loss_g_fake.backward()
            # total_loss.backward()
            loss_g_fake_hist.append(ce_loss_g_fake.item())
            mean_d_fake_output_2 = output.mean().item()
            optimizer_g.step()

            print("{}/{}: Epoch #{}: Iteration #{}: Mean D(real_hb_batch) = {}, mean D(G(z)) = {}."
                  .format(num_of_beats_seen, len(dataset), epoch, iters, mean_d_real_output, mean_d_fake_output),
                  end=" ")
            print("mean D(G(z)) = {} After backprop of D".format(mean_d_fake_output_2))

            print("Loss D from real beats = {}. Loss D from Fake beats = {}. Total Loss D = {}".
                  format(ce_loss_d_real, ce_loss_d_fake, total_loss_d), end=" ")
            print("Loss G = {}".format(ce_loss_g_fake))

            if iters % 25 == 0:
                with torch.no_grad():
                    output_g = odeint(netG, v0, timestamsp).permute(1, 0, 2).view(-1, 216)
                    fig = plt.figure()
                    plt.title("Fake beats from Generator. iteration {}".format(i))
                    for p in range(4):
                        plt.subplot(2, 2, p + 1)
                        plt.plot(output_g[p].detach().cpu().numpy(), label="fake beat")
                        plt.plot(ecg_batch[p].detach().cpu().numpy(), label="real beat")
                        plt.legend()
                    plt.savefig(os.path.join(train_configs.base, 'ode_ecg', 'ode_ecg', 'models', model_dir,
                                             'generator_output_{}.png'.format(iters)))
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(device)
    train_ecg_gan(batch_size=40, num_train_steps=100000, model_dir='logs_rk4_update_dic_equal_x', beat_type='N')