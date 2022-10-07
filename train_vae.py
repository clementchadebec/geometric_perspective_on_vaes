import torch
from utils import Digits
from models.training import train_vae
from models.evaluation import eval_vae
from copy import deepcopy
import torch.optim as optim
import numpy as np
import os
from models.vae import VAE, RHVAE
from imageio import imread
from scipy.io import loadmat
from config import *


def main(args):

    if args.path_to_train.split('/')[-2] == 'mnist':
        im_size_x, im_size_y = 32, 32
        im_channels = 1
        latent_dim = 16
        beta = .01
        patience = 5
        n_epochs = 100
        architecture = 'convnet'
        lr = 1e-3

        mnist_digits = np.load(args.path_to_train)

        train_data = torch.tensor(mnist_digits['x_train'][:-10000]).type(torch.float) / 255.
        train_targets = torch.tensor(mnist_digits['y_train'][:-10000])
        val_data = torch.tensor(mnist_digits['x_train'][-10000:]).type(torch.float) / 255.
        val_targets = torch.tensor(mnist_digits['y_train'][-10000:])


    elif args.path_to_train.split('/')[-2] == 'cifar':
        im_size_x, im_size_y = 32, 32
        im_channels = 3
        latent_dim = 32
        beta = 0.001
        lr=5e-4
        patience = 5
        n_epochs = 200
        architecture = 'convnet'
        n_components = 100

        cifar_digits = np.load(args.path_to_train)

        train_data = torch.tensor(cifar_digits['x_train'][:-10000]).type(torch.float).permute(0, 3, 1, 2) / 255.
        train_targets = torch.tensor(cifar_digits['y_train'][:-10000])
        val_data = torch.tensor(cifar_digits['x_train'][10000:]).type(torch.float).permute(0, 3, 1, 2) / 255.
        val_targets = torch.tensor(cifar_digits['y_train'][10000:])


    elif args.path_to_train.split('/')[-1] == 'celeba' or args.path_to_train.split('/')[-2] == 'celeba':
        im_size_x, im_size_y = 64, 64
        im_channels = 3
        latent_dim = 64
        beta = 0.05
        architecture = 'convnet'
        lr = 1e-3
        patience = 5
        n_epochs = 100

        train_data = torch.load(os.path.join(args.path_to_train, 'train_data.pt'))
        train_targets = torch.ones(len(train_data))
        val_data = torch.load(os.path.join(args.path_to_train, 'val_data.pt'))
        val_targets = torch.ones(len(val_data))


    elif args.path_to_train.split('/')[-2] == 'oasis':
        im_size_x, im_size_y = 208, 176
        im_channels = 1
        latent_dim = 16
        beta = 1
        architecture = 'convnet'
        lr = 1e-4
        patience = 20
        n_epochs = 1000

        oasis_data = np.load(args.path_to_train)

        train_targets = torch.tensor(oasis_data['y_train'][:])
        train_data = torch.tensor(oasis_data['x_train']).type(torch.float).permute(0, 3, 1, 2) / 255.
        val_targets = torch.tensor(oasis_data['y_train'][:])
        val_data = torch.tensor(oasis_data['x_train']).type(torch.float).permute(0, 3, 1, 2) / 255.


    elif args.path_to_train.split('/')[-2] == 'svhn':
        im_size_x, im_size_y = 32, 32
        im_channels = 3
        latent_dim = 16
        beta = 0.01
        lr=1e-3
        patience = 5
        n_epochs = 100
        architecture = 'mlp'
    
        if args.model == 'rhvae':
            lr = 5e-4
            temperature = 2.5

        svnh_digits = loadmat(args.path_to_train)['X']
        svnh_targets = loadmat(args.path_to_train)['y']

        svnh_digits = np.transpose(svnh_digits, (3, 0, 1, 2))

        train_data = torch.tensor(svnh_digits[:-10000]).type(torch.float).permute(0, 3, 1, 2) / 255.
        train_targets = torch.tensor(svnh_targets[:-10000])
        val_data = torch.tensor(svnh_digits[-10000:]).permute(0, 3, 1, 2) / 255.
        val_targets = torch.tensor(svnh_targets[-10000:])

    else:
        raise NotImplementedError()


    train = Digits(train_data.reshape(-1, im_channels, im_size_x, im_size_y), train_targets)
    val = Digits(val_data.reshape(-1, im_channels, im_size_x, im_size_y), val_targets)
        
    train_loader = torch.utils.data.DataLoader(
        dataset=train, batch_size=100, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val, batch_size=100, shuffle=True
    )


    print('---------------------------------------------------------------')
    print(f'Train size: {train_loader.dataset.data.shape, train_loader.dataset.data.min(), train_loader.dataset.data.max()}')
    print(f'Val size: {val_loader.dataset.data.shape, val_loader.dataset.data.min(), val_loader.dataset.data.max()}')
    print('---------------------------------------------------------------')


    if args.model == 'vae':
        path_to_save = os.path.join('trained_vae_models', 'vae', args.path_to_train.split('/')[-2])

    elif args.model == 'ae':
         path_to_save = os.path.join('trained_vae_models', 'ae', args.path_to_train.split('/')[-2])

    elif args.model == 'rhvae':
        path_to_save = os.path.join('trained_vae_models', 'rhvae', args.path_to_train.split('/')[-2])

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
        print(f"Created folder {path_to_save}. Best model is saved here")


    ##### Training #####

    if args.model == 'vae':

        train_args = VAE_config(
                input_dim=im_size_x*im_size_y,
                latent_dim=latent_dim,
                architecture=architecture,
                n_channels=im_channels,
                dataset=args.path_to_train.split('/')[-2],
                beta=beta
                )

        vae = VAE(train_args)

    elif args.model == 'ae':

        train_args = VAE_config(
                model_name="AE",
                input_dim=im_size_x*im_size_y,
                latent_dim=latent_dim,
                architecture=architecture,
                n_channels=im_channels,
                dataset=args.path_to_train.split('/')[-2],
                beta=beta,
                )

        vae = VAE(train_args)

    elif args.model == 'rhvae':

        train_args = RHVAE_config(
                model_name="RHVAE",
                input_dim=im_size_x*im_size_y,
                latent_dim=latent_dim,
                architecture=architecture,
                n_channels=im_channels,
                dataset=args.path_to_train.split('/')[-2],
                beta=beta
                )

        vae = RHVAE(train_args)

    print(train_args)

    if torch.cuda.is_available():
        print('Using cuda')
        vae.cuda()
    print("Model")
    print(vae)
    if train_args.architecture == 'convnet' and args.model == 'vae':
        print(f"Encoder num params: {sum(p.numel() for p in vae.fc21.parameters() if p.requires_grad) + sum(p.numel() for p in vae.conv.parameters() if p.requires_grad)} + log_var: {sum(p.numel() for p in vae.fc22.parameters() if p.requires_grad)}")
        print(f"Decoder num params: {sum(p.numel() for p in vae.fc3.parameters() if p.requires_grad) + sum(p.numel() for p in vae.deconv.parameters() if p.requires_grad)}")

        print("Nu params", sum(p.numel() for p in vae.parameters() if p.requires_grad) - sum(p.numel() for p in vae.fc21.parameters() if p.requires_grad) )

    elif train_args.architecture == 'convnet' and args.model=='rhvae':
        num_metric_param = sum(p.numel() for p in vae.metric_fc21.parameters() if p.requires_grad) + sum(p.numel() for p in vae.metric_fc22.parameters() if p.requires_grad) + sum(p.numel() for p in vae.metric_fc1.parameters() if p.requires_grad) 
        num_cov_param = sum(p.numel() for p in vae.fc22.parameters() if p.requires_grad)
        print(f"Encoder num params: {sum(p.numel() for p in vae.fc21.parameters() if p.requires_grad) + sum(p.numel() for p in vae.conv.parameters() if p.requires_grad)} + log_var: {num_cov_param} + metric: {num_metric_param}")
        print(f"Decoder num params: {sum(p.numel() for p in vae.fc3.parameters() if p.requires_grad) + sum(p.numel() for p in vae.deconv.parameters() if p.requires_grad)}")

        print("Nu params", sum(p.numel() for p in vae.parameters() if p.requires_grad) - num_cov_param - num_metric_param)

    elif train_args.architecture == 'convnet' and args.model == 'vamp':
        print(f"Encoder num params: {sum(p.numel() for p in vae.fc21.parameters() if p.requires_grad) + sum(p.numel() for p in vae.conv.parameters() if p.requires_grad)} + log_var: {sum(p.numel() for p in vae.fc22.parameters() if p.requires_grad)}")
        print(f"Decoder num params: {sum(p.numel() for p in vae.fc3.parameters() if p.requires_grad) + sum(p.numel() for p in vae.deconv.parameters() if p.requires_grad)}")

        print("Nu params", sum(p.numel() for p in vae.parameters() if p.requires_grad) - sum(p.numel() for p in vae.fc21.parameters() if p.requires_grad) - sum(p.numel() for p in vae.means.parameters() if p.requires_grad) )

    elif train_args.architecture == 'mlp' and (not args.model=='vamp' and not args.model=='ae'):
        print("Nu params", sum(p.numel() for p in vae.parameters() if p.requires_grad) - sum(p.numel() for p in vae.fc21.parameters() if p.requires_grad))

    elif train_args.architecture == 'mlp' and args.model=='vamp':
        print("Nu params", sum(p.numel() for p in vae.parameters() if p.requires_grad) - sum(p.numel() for p in vae.fc21.parameters() if p.requires_grad) - sum(p.numel() for p in vae.means.parameters() if p.requires_grad))

    else:
        print("Nu params", sum(p.numel() for p in vae.parameters() if p.requires_grad))

    optimizer = optim.Adam(vae.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=patience, verbose=True)


    best_loss = 1e10
    torch.manual_seed(8)
    torch.cuda.manual_seed(8)
    e = 0
    for epoch in range(1, n_epochs+1):
        print(f"Epoch {epoch}")

        if args.model == 'vae' or args.model=='ae' or args.model =='rhvae' or args.model == 'geoae':

            vae, train_loss, train_loss_rec, train_loss_kld = train_vae(epoch, train_args, vae, train_loader, optimizer)
            val_loss, val_loss_rec, val_loss_kld = eval_vae(epoch, train_args, vae, val_loader)
        
        scheduler.step(val_loss)
        if val_loss < best_loss:
            e = 0
            best_model_dict = {
                'state_dict': deepcopy(vae.state_dict()),
                'args': train_args
            }
            best_loss = val_loss

        if epoch % 1== 0:
            print('----------------------------------------------------------------------------------------------------------------')
            print(f'Epoch {epoch}: Train loss: {np.round(train_loss, 10)}\t Rec Loss: {np.round(train_loss_rec, 10)}\t KLD Loss: {np.round(train_loss_kld, 10)}')
            print(f'Epoch {epoch}: Eval loss: {np.round(val_loss, 10)}\t Rec Loss: {np.round(val_loss_rec, 10)}\t KLD Loss: {np.round(val_loss_kld, 10)}')
            print('----------------------------------------------------------------------------------------------------------------')

    torch.save(best_model_dict, os.path.join(path_to_save, 'best_model.pt'))
    print('<<<<<<<<<<<<<<<<<<<<< Saved best model >>>>>>>>>>>>>>>>>>>>>>>>')


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    
    
    parser.add_argument("path_to_train", type=str,
        help='Path to the training data .npz files')
    parser.add_argument("--model", type=str, choices=['ae', 'vae', 'vamp', 'rhvae'],
        help='Model to train', default='vae')
   
    args = parser.parse_args()

    main(args)
