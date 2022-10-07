import torch
from sklearn import mixture
from imageio import imwrite
import numpy as np
from models.vae import VAE
from utils import Digits
import os
from scipy.io import loadmat
from config import *
from sampling import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(args):

    

    checkpoint = torch.load(args.model_path)

    print(checkpoint['args'])
    if 'rhvae' in args.model_path:
        from models.vae import RHVAE
        best_model = RHVAE(checkpoint['args'])

    else:
        best_model = VAE(checkpoint['args'])

    best_model.to(device)
    best_model.load_state_dict(checkpoint['state_dict'])
    print(best_model)

    print("Nu params", sum(p.numel() for p in best_model.parameters() if p.requires_grad) - sum(p.numel() for p in best_model.fc21.parameters() if p.requires_grad) )

    path_to_train = args.data_path

    dataset_name = args.data_path.split('/')[-2]


    if dataset_name == 'mnist':
        eps_lf = 0.01
        lbd = 0.01

        mnist_digits = np.load(path_to_train)

        train_data = torch.tensor(mnist_digits['x_train'][:-10000]).type(torch.float).permute(0, 3, 1, 2) / 255.
        train_targets = torch.tensor(mnist_digits['y_train'][:-10000])
        

    elif dataset_name == 'cifar':

        cifar_digits = np.load(path_to_train)

        train_data = torch.tensor(cifar_digits['x_train'][:-10000]).type(torch.float).permute(0, 3, 1, 2) / 255.
        train_targets = torch.tensor(cifar_digits['y_train'][:-10000])

        eps_lf = 0.01
        lbd = 0.1


    elif dataset_name == 'celeba':

        train_data = torch.load(os.path.join(path_to_train, 'train_data.pt'))
        train_targets = torch.ones(len(train_data))

        eps_lf = 0.01
        lbd = 1

    elif dataset_name == 'oasis':

        oasis_data = np.load(path_to_train)

        train_targets = torch.tensor(oasis_data['y_train'][:])
        train_data = torch.tensor(oasis_data['x_train']).type(torch.float).permute(0, 3, 1, 2) / 255.

        eps_lf = 0.01
        lbd = 1

    elif dataset_name == 'svhn':
        eps_lf = 0.01
        lbd= 0.01

        svnh_digits = loadmat(path_to_train)['X']
        svnh_targets = loadmat(path_to_train)['y']

        svnh_digits = np.transpose(svnh_digits, (3, 0, 1, 2))

        train_data = torch.tensor(svnh_digits[:-10000]).type(torch.float).permute(0, 3, 1, 2) / 255.
        train_targets = torch.tensor(svnh_targets[:-10000])

    train = Digits(train_data, train_targets)
    train_loader = torch.utils.data.DataLoader(
    dataset=train, batch_size=100, shuffle=False
    )

    if args.generation == 'hmc':

        path_to_save = f"generated_data/vae/{dataset_name}/manifold_sampling/"
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
            print(f"Created folder {path_to_save}. Data will be saved here")

        mu = []
        log_var = []

        with torch.no_grad():
            for _ , (data, _) in enumerate(train_loader):

                mu_data, log_var_data = best_model.encode(data.to(device))

                mu.append(mu_data)
                log_var.append(log_var_data)

        mu = torch.cat(mu)
        log_var = torch.cat(log_var)

        if dataset_name == 'cifar' or dataset_name=='mnist' or dataset_name == 'svhn':
            print('Running Kmedoids')
            from sklearn_extra.cluster import KMedoids
            kmedoids = KMedoids(n_clusters=100).fit(mu.detach().cpu())
            medoids = torch.tensor(kmedoids.cluster_centers_).to(device)
            centroids_idx = kmedoids.medoid_indices_ #

        elif dataset_name == 'oasis':
            centroids_idx = torch.arange(0, 50)
            medoids = mu[centroids_idx]

        else:
            centroids_idx = torch.arange(0, 100).to(device)
            medoids = mu[centroids_idx]

        print("Finding temperature")

        T = 0
        T_is = []
        for i in range(len(medoids)-1):
            mask = torch.tensor([k for k in range(len(medoids)) if k != i])
            dist = torch.norm(medoids[i].unsqueeze(0) - medoids[mask], dim=-1)
            T_i =torch.min(dist, dim=0)[0]
            T_is.append(T_i.item())

        T = np.max(T_is)
        print('Best temperature found: ', T)

        print('Building metric')
        best_model = build_metrics(best_model, mu, log_var, centroids_idx, T=T, lbd=lbd)

        if args.n_samples % args.batch_size > 0:
            print('Cropping batch for now....')

        print('Launching generation HMC')
        for j in range(0, int(args.n_samples / args.batch_size)):
            z, p = hmc_sampling(best_model, mu, n_samples=args.batch_size, eps_lf=eps_lf, mcmc_steps_nbr=100)
            recon_x = best_model.decode(z)
            for i in range(args.batch_size):
                img = (255. * torch.movedim(recon_x[i], 0, 2).cpu().detach().numpy())

                if img.shape[-1]==1:
                    img = np.repeat(img, repeats=3, axis=-1)
                img = img.astype('uint8')
                imwrite(os.path.join(path_to_save, '%08d.png' % int(args.batch_size*j + i)), img)


    elif args.generation == 'gmm' or args.generation == 'GMM' :
        print('Launching generation GMM')

        mu = []

        with torch.no_grad():
            for _ , (data, _) in enumerate(train_loader):

                mu_data, _ = best_model.encode(data.to(device))

                mu.append(mu_data)

        mu = torch.cat(mu)
        print(mu.shape)

        gmm = mixture.GaussianMixture(n_components=args.n_components, covariance_type='full', max_iter=2000,
                                                verbose=2, tol=1e-3)
        gmm.fit(mu.cpu().detach())

        for j in range(0, int(args.n_samples / args.batch_size)):

            idx = np.array(range(args.batch_size))
            np.random.shuffle(idx)

            z = torch.tensor(gmm.sample(args.batch_size)[0][idx, :]).to(device).type(torch.float)

            recon_x = best_model.decode(z)
            for i in range(args.batch_size):
                img = (255. * torch.movedim(recon_x[i], 0, 2).cpu().detach().numpy())

                if img.shape[-1]==1:
                    img = np.repeat(img, repeats=3, axis=-1)
                img = img.astype('uint8')

                if best_model.model_name == 'AE':

                    path_to_save = f"generated_data/ae/{dataset_name}/gmm/"
                    if not os.path.exists(path_to_save):
                        os.makedirs(path_to_save)
                        print(f"Created folder {path_to_save}. Data will be saved here")
                    imwrite(os.path.join(path_to_save, '%08d.png' % int(args.batch_size*j + i)), img)

                elif best_model.model_name == 'RHVAE':
                    path_to_save = f"generated_data/rhvae/{dataset_name}/gmm/"
                    if not os.path.exists(path_to_save):
                        os.makedirs(path_to_save)
                        print(f"Created folder {path_to_save}. Data will be saved here")
                    imwrite(os.path.join(path_to_save, '%08d.png' % int(args.batch_size*j + i)), img)
                else:
                    path_to_save = f"generated_data/vae/{dataset_name}/gmm/"
                    if not os.path.exists(path_to_save):
                        os.makedirs(path_to_save)
                        print(f"Created folder {path_to_save}. Data will be saved here")
                    imwrite(os.path.join(path_to_save, '%08d.png' % int(args.batch_size*j + i)), img)
            

    else:
        print('Launching generation Gaussian')
        for j in range(0, int(args.n_samples / args.batch_size)):
            
            z = torch.randn(args.batch_size, best_model.latent_dim).to(device)
            recon_x = best_model.decode(z)

            for i in range(args.batch_size):
                
                img = (255. * torch.movedim(recon_x[i], 0, 2).cpu().detach().numpy())

                if img.shape[-1]==1:
                    img = np.repeat(img, repeats=3, axis=-1)
                img = img.astype('uint8')

                if best_model.model_name == 'AE':
                    path_to_save = f"generated_data/ae/{dataset_name}/gaussian_prior/"
                    if not os.path.exists(path_to_save):
                        os.makedirs(path_to_save)
                        print(f"Created folder {path_to_save}. Data will be saved here")
                    imwrite(os.path.join(path_to_save, '%08d.png' % int(args.batch_size*j + i)), img)

                elif best_model.model_name == 'RHVAE':
                    path_to_save = f"generated_data/rhvae/{dataset_name}/gaussian_prior/"
                    if not os.path.exists(path_to_save):
                        os.makedirs(path_to_save)
                        print(f"Created folder {path_to_save}. Data will be saved here")
                    imwrite(os.path.join(path_to_save, '%08d.png' % int(args.batch_size*j + i)), img)
                else:
                    path_to_save = f"generated_data/vae/{dataset_name}/gaussian_prior/"
                    if not os.path.exists(path_to_save):
                        os.makedirs(path_to_save)
                        print(f"Created folder {path_to_save}. Data will be saved here")
                    imwrite(os.path.join(path_to_save, '%08d.png' % int(args.batch_size*j + i)), img)

if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    
    
    parser.add_argument("--model_path", type=str,
        help='Path to the model')
    parser.add_argument("--data_path", type=str,
        help='Path to the training data .npz files')
    parser.add_argument("--generation", type=str,
        help='Generation type', default='hmc')
    parser.add_argument("--n_samples", type=int,
        help='Number of samples', default=10000)
    parser.add_argument("--batch_size", type=int,
        help='Batch size', default=500)
    parser.add_argument("--n_components", type=int,
        help='Number of comp for gmm', default=10)
   
   
    args = parser.parse_args()


    np.random.seed(8)
    torch.manual_seed(8)
    torch.cuda.manual_seed(8)

    main(args)
