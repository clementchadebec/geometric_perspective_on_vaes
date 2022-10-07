import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

from .base import BaseVAE


class VAE(BaseVAE, nn.Module):
    def __init__(self, args):

        BaseVAE.__init__(self, args)
        nn.Module.__init__(self)

        self.model_name = args.model_name
        self.dataset = args.dataset
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = args.n_channels
        self.beta = args.beta 

        if args.architecture == 'convnet':

            if args.dataset == 'mnist' or args.dataset == 'cifar': # SAME as From Var. to Deter. AE (check num of params)
                self.conv = torch.nn.Sequential(
                        nn.Conv2d(self.n_channels, 128, 4, 2, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(128, 256, 4, 2, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256, 512, 4, 2, padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512, 1024, 4, 2, padding=1),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    )

                #self.fc1 = nn.Linear(1024*2*2, 16)
                self.fc21 = nn.Linear(1024*2*2, args.latent_dim)
                if not args.model_name == 'AE':
                    self.fc22 = nn.Linear(1024*2*2, args.latent_dim)
                #self.fc22 = nn.Linear(400, args.latent_dim)

                self.fc3 = nn.Linear(args.latent_dim, 1024*8*8)
                #self.fc4 = nn.Linear(16, 1024*8*8)
                self.deconv = nn.Sequential(
                        nn.ConvTranspose2d(1024, 512, 4, 2, padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.ConvTranspose2d(512, 256, 4, 2, padding=1, output_padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.ConvTranspose2d(256, self.n_channels, 4, 1, padding=2),
                        #nn.BatchNorm2d(self.n_channels),
                        #nn.ReLU(),
                        #nn.ConvTranspose2d(self.n_channels, self.n_channels, 3, 1, padding=1),
                        #nn.BatchNorm2d(self.n_channels),
                        nn.Sigmoid()
                )

            elif args.dataset == 'celeba':
                self.conv = torch.nn.Sequential(
                        nn.Conv2d(self.n_channels, 128, 5, 2, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(128, 256, 5, 2, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256, 512, 5, 2, padding=2),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512, 1024, 5, 2, padding=2),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    )

                #self.fc1 = nn.Linear(1024*2*2, 128)
                self.fc21 = nn.Linear(1024*4*4, args.latent_dim)
                if not args.model_name == 'AE':
                    self.fc22 = nn.Linear(1024*4*4, args.latent_dim)
                #self.fc22 = nn.Linear(400, args.latent_dim)

                self.fc3 = nn.Linear(args.latent_dim, 1024*8*8)
                #self.fc4 = nn.Linear(128, 1024*8*8)
                self.deconv = nn.Sequential(
                              nn.ConvTranspose2d(1024, 512, 5, 2, padding=2),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.ConvTranspose2d(512, 256, 5, 2, padding=1, output_padding=0),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                              nn.ConvTranspose2d(256, 128, 5, 2, padding=2, output_padding=1),
                              nn.BatchNorm2d(128),
                              nn.ReLU(),
                              nn.ConvTranspose2d(128, 3, 5, 1, padding=1),
                              #nn.BatchNorm2d(self.n_channels),
                              #nn.ReLU(),
                              #nn.ConvTranspose2d(self.n_channels, self.n_channels, 3, 1, padding=1),
                              #nn.BatchNorm2d(self.n_channels),
                              nn.Sigmoid())


            elif args.dataset == 'oasis':

                self.conv = torch.nn.Sequential(
                    nn.Conv2d(self.n_channels, 64, 5, 2, padding=1),
                    #nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 5, 2, padding=1),
                    #nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, 5, 2, padding=1),
                    #nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.Conv2d(256, 512, 5, 2, padding=(1, 2)),
                    #nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.Conv2d(512, 1024, 5, 2, padding=0),
                    #nn.BatchNorm2d(1024),
                    nn.ReLU(),
                    #nn.Conv2d(1024, 20, 5, 2, padding=1),
                    #nn.BatchNorm2d(1024),
                    #nn.ReLU(),
                )
                self.fc21 = nn.Linear(1024*4*4, args.latent_dim)
                if not args.model_name == 'AE':
                    self.fc22 = nn.Linear(1024*4*4, args.latent_dim)

                self.fc3 = nn.Linear(args.latent_dim, 1024*8*8)
                self.deconv = nn.Sequential(
                              nn.ConvTranspose2d(1024, 512, 5, (3, 2), padding=(1, 0), output_padding=(0, 0)),
                              #nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.ConvTranspose2d(512, 256, 5, 2, padding=(1, 0), output_padding=(0, 0)),
                              #nn.BatchNorm2d(256),
                              nn.ReLU(),
                              nn.ConvTranspose2d(256, 128, 5, 2, padding=0, output_padding=(0, 0)),
                              #nn.BatchNorm2d(128),
                              nn.ReLU(),
                              nn.ConvTranspose2d(128, 64, 5, 2, padding=0, output_padding=(1, 1)),
                              #nn.BatchNorm2d(64),
                              nn.ReLU(),
                              nn.ConvTranspose2d(64, self.n_channels, 5, 1, padding=1),
                              #nn.BatchNorm2d(self.n_channels),
                              nn.Sigmoid())

        else:
            # encoder network
            self.fc1 = nn.Linear(args.input_dim*args.n_channels, 1000)
            self.fc11 = nn.Linear(1000, 500)
            #self.fc111 = nn.Linear(500, 500)
            #self.fc1111 = nn.Linear(500, 500)
            self.fc21 = nn.Linear(500, args.latent_dim)
            self.fc22 = nn.Linear(500, args.latent_dim)

            # decoder network
            self.fc3 = nn.Linear(args.latent_dim, 500)
            self.fc33 = nn.Linear(500, 1000)
            #self.fc333 = nn.Linear(500, 500)
            #self.fc3333 = nn.Linear(500, 500)
            self.fc4 = nn.Linear(1000, args.input_dim*args.n_channels)

        if args.architecture == 'convnet':
            self._encoder = self._encode_convnet
            self._decoder = self._decode_convnet

        else:
            self._encoder = self._encode_mlp
            self._decoder = self._decode_mlp



        # define a N(0, I) distribution
        self.normal = torch.distributions.MultivariateNormal(
            loc=torch.zeros(args.latent_dim).to(self.device),
            covariance_matrix=torch.eye(args.latent_dim).to(self.device),
        )

    def forward(self, x):
        """
        The VAE model
        """
        mu, log_var = self.encode(x)

        if self.model_name == 'AE':
            std = 0
            eps = 0
            z = mu

        else:
            std = torch.exp(0.5 * log_var)
            z, eps = self._sample_gauss(mu, std)

        recon_x = self.decode(z)

        return recon_x, z, eps, mu, log_var

    def loss_function(self, recon_x, x, mu, log_var, z):
        BCE =  F.mse_loss(
            recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='none'
        ).sum()
        if self.model_name == 'VAE':
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        elif self.model_name =='AE':
            KLD = torch.zeros(1).to(self.device)

        elif self.model_name =='GeoAE':
            KLD = torch.linalg.norm(log_var.exp() - torch.ones_like(log_var), dim=1).sum()

        return BCE + self.beta * KLD, BCE, KLD

    def encode(self, x):
        print(x.max())
        return self._encoder(x)

    def decode(self, z):
        x_prob = self._decoder(z)
        return x_prob

    def sample_img(
        self,
        z=None,
        n_samples=1
    ):
        """
        Generate an image
        """
        if z is None:
            z = self.normal.sample(sample_shape=(n_samples,)).to(self.device)

        else:
            n_samples = z.shape[0]

        recon_x = self.decode(z)

        return recon_x

    def encode(self, x):
        return self._encoder(x)

    def decode(self, z):
        x_prob = self._decoder(z)
        return x_prob

    def _encode_mlp(self, x):
        h1 = F.relu(self.fc1(x.reshape(-1, self.input_dim*self.n_channels)))
        h1 = F.relu(self.fc11(h1))
        #h1 = F.relu(self.fc111(h1))
        #h1 = F.relu(self.fc1111(h1))
        if self.model_name == 'AE':
            return self.fc21(h1), 0
        else:
            return self.fc21(h1), self.fc22(h1)

    def _decode_mlp(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = F.relu(self.fc33(h3))
        #h3 = F.relu(self.fc333(h3))
        #h3 = F.relu(self.fc3333(h3))
        if self.dataset == "oasis":
            return torch.sigmoid(self.fc4(h3)).reshape(z.shape[0], self.n_channels, 208,176)
        else:
            return torch.sigmoid(self.fc4(h3)).reshape(z.shape[0], self.n_channels, int(self.input_dim**0.5), int(self.input_dim**0.5))

    def _encode_convnet(self, x):
        h1 = self.conv(x).reshape(x.shape[0], -1)
        #h1 = self.fc1(h1.reshape(h1.shape[0], -1))#F.relu(self.fc1(h1.reshape(h1.shape[0], -1)))

        if self.model_name == 'AE':
            return self.fc21(h1), 0
        if self.dataset == 'oasis':
            return self.fc21(h1), torch.tanh(self.fc22(h1))
        else:
            return self.fc21(h1), self.fc22(h1)

    def _decode_convnet(self, z):
        h3 = self.fc3(z)
        #h3 = F.relu(self.fc4(h3))
        #h3 = F.relu(self.fc4(h3))

        h3 = self.deconv(h3.reshape(-1, 1024, 8, 8))

        return h3

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def _tempering(self, k, K):
        """Perform tempering step"""

        beta_k = (
            (1 - 1 / self.beta_zero_sqrt) * (k / K) ** 2
        ) + 1 / self.beta_zero_sqrt

        return 1 / beta_k

    ########## Estimate densities ##########

    def log_p_x_given_z(self, recon_x, x, reduction="none"):
        """
        Estimate the decoder's log-density modelled as follows:
            p(x|z)     = \prod_i Bernouilli(x_i|pi_{theta}(z_i))
            p(x = s|z) = \prod_i (pi(z_i))^x_i * (1 - pi(z_i)^(1 - x_i))"""
        return -F.binary_cross_entropy(
            recon_x.reshape(x.shape[0], -1), x.view(x.shape[0], -1), reduction=reduction
        ).sum(dim=1)

    def log_z(self, z):
        """
        Return Normal density function as prior on z
        """
        return self.normal.log_prob(z)

    def log_p_z_given_x(self, z, recon_x, x, sample_size=10):
        """
        Estimate log(p(z|x)) using Bayes rule and Importance Sampling for log(p(x))
        """
        logpx = self.log_p_x(x, sample_size)
        lopgxz = self.log_p_x_given_z(recon_x, x)
        logpz = self.log_z(z)
        return lopgxz + logpz - logpx

    def log_p_xz(self, recon_x, x, z):
        """
        Estimate log(p(x, z)) using Bayes rule
        """
        logpxz = self.log_p_x_given_z(recon_x, x)
        logpz = self.log_z(z)
        return logpxz + logpz

    ########## Kullback-Leiber divergences estimates ##########

    def kl_prior(self, mu, log_var):
        """KL[q(z|y) || p(z)] : exact formula"""
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    def kl_cond(self, recon_x, x, z, mu, log_var, sample_size=10):
        """
        KL[p(z|x) || q(z|x)]

        Note:
        -----
        p(z|x) is approximated using IS on log(p(x))
        """
        logpzx = self.log_p_z_given_x(z, recon_x, x, sample_size=sample_size)
        logqzx = torch.distributions.MultivariateNormal(
            loc=mu, covariance_matrix=torch.diag_embed(torch.exp(log_var))
        ).log_prob(z)

        return (logqzx - logpzx).sum()

class HVAE(VAE):
    def __init__(self, args):
        """
        Inputs:
        -------

        n_lf (int): Number of leapfrog steps to perform
        eps_lf (float): Leapfrog step size
        beta_zero (float): Initial tempering
        tempering (str): Tempering type (free, fixed)
        model_type (str): Model type for VAR (mlp, convnet)
        latent_dim (int): Latentn dimension
        """
        VAE.__init__(self, args)

        self.vae_forward = super().forward
        self.n_lf = args.n_lf

        self.eps_lf = nn.Parameter(torch.Tensor([args.eps_lf]), requires_grad=False)

        assert 0 < args.beta_zero <= 1, "Tempering factor should belong to [0, 1]"

        self.beta_zero_sqrt = nn.Parameter(
            torch.Tensor([args.beta_zero]), requires_grad=False
        )

    def forward(self, x):
        """
        The HVAE model
        """

        recon_x, z0, eps0, mu, log_var = self.vae_forward(x)
        gamma = torch.randn_like(z0, device=self.device)
        rho = gamma / self.beta_zero_sqrt
        z = z0
        beta_sqrt_old = self.beta_zero_sqrt

        recon_x = self.decode(z)

        for k in range(self.n_lf):

            # perform leapfrog steps

            # computes potential energy
            U = -self.log_p_xz(recon_x, x, z).sum()

            # Compute its gradient
            g = grad(U, z, create_graph=True)[0]

            # 1st leapfrog step
            rho_ = rho - (self.eps_lf / 2) * g

            # 2nd leapfrog step
            z = z + self.eps_lf * rho_

            recon_x = self.decode(z)

            U = -self.log_p_xz(recon_x, x, z).sum()
            g = grad(U, z, create_graph=True)[0]

            # 3rd leapfrog step
            rho__ = rho_ - (self.eps_lf / 2) * g

            # tempering steps
            beta_sqrt = self._tempering(k + 1, self.n_lf)
            rho = (beta_sqrt_old / beta_sqrt) * rho__
            beta_sqrt_old = beta_sqrt

        return recon_x, z, z0, rho, eps0, gamma, mu, log_var

    def loss_function(self, recon_x, x, z0, zK, rhoK, eps0, gamma, mu, log_var):

        logpxz = self.log_p_xz(recon_x.reshape(x.shape[0], -1), x, zK)  # log p(x, z_K)
        logrhoK = self.normal.log_prob(rhoK)  # log p(\rho_K)
        logp = logpxz + logrhoK

        logq = self.normal.log_prob(eps0) - 0.5 * log_var.sum(dim=1)  # q(z_0|x)

        return -(logp - logq).sum()


    def hamiltonian(self, recon_x, x, z, rho, G_inv=None, G_log_det=None):
        """
        Computes the Hamiltonian function.
        used for HVAE and RHVAE
        """
        if self.model_name == "HVAE":
            return -self.log_p_xz(recon_x.reshape(x.shape[0], -1), x, z).sum()

        # norm = (torch.solve(rho[:, :, None], G).solution[:, :, 0] * rho).sum()
        norm = (
            torch.transpose(rho.unsqueeze(-1), 1, 2) @ G_inv @ rho.unsqueeze(-1)
        ).sum()

        return -self.log_p_xz(recon_x.reshape(x.shape[0], -1), x, z).sum() + 0.5 * norm + 0.5 * G_log_det.sum()

    def _tempering(self, k, K):
        """Perform tempering step"""

        beta_k = (
            (1 - 1 / self.beta_zero_sqrt) * (k / K) ** 2
        ) + 1 / self.beta_zero_sqrt

        return 1 / beta_k


class RHVAE(HVAE):
    def __init__(self, args):

        HVAE.__init__(self, args)
        # defines the Neural net to compute the metric

        # first layer
        self.metric_fc1 = nn.Linear(self.input_dim*self.n_channels, args.metric_fc)

        # diagonal
        self.metric_fc21 = nn.Linear(args.metric_fc, self.latent_dim)
        # remaining coefficients
        k = int(self.latent_dim * (self.latent_dim - 1) / 2)
        self.metric_fc22 = nn.Linear(args.metric_fc, k)

        self.T = nn.Parameter(torch.Tensor([args.temperature]), requires_grad=False)
        self.lbd = nn.Parameter(
            torch.Tensor([args.regularization]), requires_grad=False
        )

        # this is used to store the matrices and centroids throughout trainning for
        # further use in metric update (L is the cholesky decomposition of M)
        self.M = []
        self.centroids = []

        # define a starting metric (gamma_i = 0 & L = I_d)
        def G(z):
            return (
                torch.eye(self.latent_dim, device=self.device).unsqueeze(0)
                * torch.exp(-torch.norm(z.unsqueeze(1), dim=-1) ** 2)
                .unsqueeze(-1)
                .unsqueeze(-1)
            ).sum(dim=1) + self.lbd * torch.eye(self.latent_dim).to(self.device)

        self.G = G

    def metric_forward(self, x):
        """
        This function returns the outputs of the metric neural network

        Outputs:
        --------

        L (Tensor): The L matrix as used in the metric definition
        M (Tensor): L L^T
        """

        h1 = torch.relu(self.metric_fc1(x.reshape(-1, self.input_dim*self.n_channels)))
        h21, h22 = self.metric_fc21(h1), self.metric_fc22(h1)

        L = torch.zeros((x.shape[0], self.latent_dim, self.latent_dim)).to(self.device)
        indices = torch.tril_indices(
            row=self.latent_dim, col=self.latent_dim, offset=-1
        )

        # get non-diagonal coefficients
        L[:, indices[0], indices[1]] = h22

        # add diagonal coefficients
        L = L + torch.diag_embed(h21.exp())

        return L, L @ torch.transpose(L, 1, 2)

    def update_metric(self):
        """
        As soon as the model has seen all the data points (i.e. at the end of 1 loop)
        we update the final metric function using \mu(x_i) as centroids
        """
        # convert to 1 big tensor
        self.M_tens = torch.cat(self.M)
        self.centroids_tens = torch.cat(self.centroids)

        # define new metric
        def G(z):
            return torch.inverse(
                (
                    self.M_tens.unsqueeze(0)
                    * torch.exp(
                        -torch.norm(
                            self.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1
                        )
                        ** 2
                        / (self.T ** 2)
                    )
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                ).sum(dim=1)
                + self.lbd * torch.eye(self.latent_dim).to(self.device)
            )

        def G_inv(z):
            return (
                self.M_tens.unsqueeze(0)
                * torch.exp(
                    -torch.norm(
                        self.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1
                    )
                    ** 2
                    / (self.T ** 2)
                )
                .unsqueeze(-1)
                .unsqueeze(-1)
            ).sum(dim=1) + self.lbd * torch.eye(self.latent_dim).to(self.device)

        self.G = G
        self.G_inv = G_inv
        self.M = []
        self.centroids = []

    def forward(self, x):
        """
        The RHVAE model
        """

        recon_x, z0, eps0, mu, log_var = self.vae_forward(x)

        z = z0

        if self.training:

            # update the metric using batch data points
            L, M = self.metric_forward(x)

            # store LL^T and mu(x_i) to update final metric
            self.M.append(M.clone().detach())
            self.centroids.append(mu.clone().detach())

            G_inv = (
                M.unsqueeze(0)
                * torch.exp(
                    -torch.norm(mu.unsqueeze(0) - z.unsqueeze(1), dim=-1) ** 2
                    / (self.T ** 2)
                )
                .unsqueeze(-1)
                .unsqueeze(-1)
            ).sum(dim=1) + self.lbd * torch.eye(self.latent_dim).to(self.device)

        else:
            G = self.G(z)
            G_inv = self.G_inv(z)
            L = torch.cholesky(G)

        G_log_det = -torch.logdet(G_inv)

        gamma = torch.randn_like(z0, device=self.device)
        rho = gamma / self.beta_zero_sqrt
        beta_sqrt_old = self.beta_zero_sqrt

        # sample \rho from N(0, G)
        rho = (L @ rho.unsqueeze(-1)).squeeze(-1)

        recon_x = self.decode(z)

        for k in range(self.n_lf):

            # perform leapfrog steps

            # step 1
            rho_ = self.leap_step_1(recon_x, x, z, rho, G_inv, G_log_det)

            # step 2
            z = self.leap_step_2(recon_x, x, z, rho_, G_inv, G_log_det)

            recon_x = self.decode(z)

            if self.training:
                G_inv = (
                    M.unsqueeze(0)
                    * torch.exp(
                        -torch.norm(mu.unsqueeze(0) - z.unsqueeze(1), dim=-1) ** 2
                        / (self.T ** 2)
                    )
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                ).sum(dim=1) + self.lbd * torch.eye(self.latent_dim).to(self.device)

            else:
                # compute metric value on new z using final metric
                G = self.G(z)
                G_inv = self.G_inv(z)

            G_log_det = -torch.logdet(G_inv)

            # step 3
            rho__ = self.leap_step_3(recon_x, x, z, rho_, G_inv, G_log_det)

            # tempering
            beta_sqrt = self._tempering(k + 1, self.n_lf)
            rho = (beta_sqrt_old / beta_sqrt) * rho__
            beta_sqrt_old = beta_sqrt

        return recon_x, z, z0, rho, eps0, gamma, mu, log_var, G_inv, G_log_det

    def leap_step_1(self, recon_x, x, z, rho, G_inv, G_log_det, steps=3):
        """
        Resolves first equation of generalized leapfrog integrator
        using fixed point iterations
        """

        def f_(rho_):
            H = self.hamiltonian(recon_x, x, z, rho_, G_inv, G_log_det)
            gz = grad(H, z, retain_graph=True)[0]
            return rho - 0.5 * self.eps_lf * gz

        rho_ = rho.clone()
        for _ in range(steps):
            rho_ = f_(rho_)
        return rho_

    def leap_step_2(self, recon_x, x, z, rho, G_inv, G_log_det, steps=3):
        """
        Resolves second equation of generalized leapfrog integrator
        using fixed point iterations
        """
        H0 = self.hamiltonian(recon_x, x, z, rho, G_inv, G_log_det)
        grho_0 = grad(H0, rho)[0]

        def f_(z_):
            H = self.hamiltonian(recon_x, x, z_, rho, G_inv, G_log_det)
            grho = grad(H, rho, retain_graph=True)[0]
            return z + 0.5 * self.eps_lf * (grho_0 + grho)

        z_ = z.clone()
        for _ in range(steps):
            z_ = f_(z_)
        return z_

    def leap_step_3(self, recon_x, x, z, rho, G_inv, G_log_det, steps=3):
        """
        Resolves third equation of generalized leapfrog integrator
        using fixed point iterations
        """
        H = self.hamiltonian(recon_x, x, z, rho, G_inv, G_log_det)
        gz = grad(H, z, create_graph=True)[0]
        return rho - 0.5 * self.eps_lf * gz

    def loss_function(
        self, recon_x, x, z0, zK, rhoK, eps0, gamma, mu, log_var, G_inv, G_log_det
    ):

        logpxz = self.log_p_xz(recon_x.reshape(x.shape[0], -1), x, zK)  # log p(x, z_K)
        logrhoK = (
            -0.5
            * (torch.transpose(rhoK.unsqueeze(-1), 1, 2) @ G_inv @ rhoK.unsqueeze(-1))
            .squeeze()
            .squeeze()
            - 0.5 * G_log_det
        ) - torch.log(
            torch.tensor([2 * np.pi]).to(self.device)
        ) * self.latent_dim / 2  # log p(\rho_K)

        logp = logpxz + logrhoK

        logq = self.normal.log_prob(eps0) - 0.5 * log_var.sum(dim=1)  # log(q(z_0|x))

        return -(logp - logq).sum()
