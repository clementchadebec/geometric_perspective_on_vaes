import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def build_metrics(model, mu, log_var, idx=None, T=0.3, lbd=0.0001):

    if idx is not None:
        mu = mu[idx]
        log_var = log_var[idx]

    with torch.no_grad():
        model.M_i = torch.diag_embed((-log_var).exp()).detach()
        model.M_i_flat = (-log_var).exp().detach()
        model.M_i_inverse_flat = (log_var).exp().detach()
        model.centroids = mu.detach()
        model.T = T
        model.lbd = lbd


        def G_sampl(z):
            omega = (
                -(
                    torch.transpose(
                                (model.centroids.unsqueeze(0) - z.unsqueeze(1)).unsqueeze(-1), 2, 3) @ torch.diag_embed(model.M_i_flat).unsqueeze(0) @ (model.centroids.unsqueeze(0) -                                      z.unsqueeze(1)).unsqueeze(-1)
                            ) / model.T**2
                ).exp()

            return (torch.diag_embed(model.M_i_flat).unsqueeze(0) * omega
            ).sum(dim=1) + model.lbd * torch.eye(model.latent_dim).to(device)

        model.G_sampl = G_sampl
        
    return model


def d_log_sqrt_det_G(z, model):
    with torch.no_grad():
        omega = (
                -(
                    torch.transpose(
                                (model.centroids.unsqueeze(0) - z.unsqueeze(1)).unsqueeze(-1), 2, 3) @ model.M_i.unsqueeze(0) @ (model.centroids.unsqueeze(0) - z.unsqueeze(1)).unsqueeze(-1)
                            ) / model.T**2
                ).exp()
        d_omega_dz = ((-2 * model.M_i_flat * (z.unsqueeze(1) - model.centroids.unsqueeze(0)) / (model.T ** 2)).unsqueeze(-2) * omega).squeeze(-2)
        num = (d_omega_dz.unsqueeze(-2) * (model.M_i_flat.unsqueeze(0).unsqueeze(-1))).sum(1)
        denom = (model.M_i_flat.unsqueeze(0) * omega.squeeze(-1) + model.lbd).sum(1)

    return torch.transpose(num / denom.unsqueeze(-1), 1, 2).sum(-1)

def log_pi(model, z):
    return 0.5 * (torch.clamp(model.G_sampl(z).det(), 0, 1e10)).log()


def hmc_sampling(model, mu, n_samples=1, mcmc_steps_nbr=1000, n_lf=10, eps_lf=0.01):

    acc_nbr = torch.zeros(n_samples, 1).to(device)
    path = torch.zeros(n_samples, mcmc_steps_nbr, model.latent_dim)
    with torch.no_grad():

        idx = torch.randint(0, len(mu), (n_samples,))
        z0 = mu[idx]
        z = z0
        for i in range(mcmc_steps_nbr):
            #print(i)
            gamma = 0.5*torch.randn_like(z, device=device)
            rho = gamma# / self.beta_zero_sqrt

            H0 = -log_pi(model, z) + 0.5 * torch.norm(rho, dim=1) ** 2
            #print(H0)
            # print(model.G_inv(z).det())
            for k in range(n_lf):

                g = -d_log_sqrt_det_G(z, model).reshape(
                    n_samples, model.latent_dim
                )
                # step 1
                rho_ = rho - (eps_lf / 2) * g

                # step 2
                z = z + eps_lf * rho_
                g = -d_log_sqrt_det_G(z, model).reshape(
                    n_samples, model.latent_dim
                )

                # step 3
                rho__ = rho_ - (eps_lf / 2) * g

                # tempering
                beta_sqrt = 1

                rho =  rho__
                #beta_sqrt_old = beta_sqrt

            H = -log_pi(model, z) + 0.5 * torch.norm(rho, dim=1) ** 2
            alpha = torch.exp(-H) / (torch.exp(-H0))

            acc = torch.rand(n_samples).to(device)
            moves = (acc < alpha).type(torch.int).reshape(n_samples, 1)

            acc_nbr += moves

            z = z * moves + (1 - moves) * z0
            path[:, i] = z
            z0 = z

        return z.detach(), path.detach().cpu()