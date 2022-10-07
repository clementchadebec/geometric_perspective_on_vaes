import torch
from torch.utils import data


class Digits(data.Dataset):
    def __init__(self, digits, labels, mask=None, binarize=False):

        self.labels = labels

        if binarize:
            self.data = (torch.rand_like(digits) < digits).type(torch.float)

        else:
            self.data = digits.type(torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        X = self.data[index]
        y = self.labels[index]


        return X, y

def create_metric(model, device='cpu'):
    """
    Metric creation for RHVAE model
    """
    def G(z):
        return torch.inverse(
            (
                model.M_tens.unsqueeze(0)
                * torch.exp(
                    -torch.norm(
                        model.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1
                    )
                    ** 2
                    / (model.T ** 2)
                )
                .unsqueeze(-1)
                .unsqueeze(-1)
            ).sum(dim=1)
            + model.lbd * torch.eye(model.latent_dim).to(device)
        )

    return G

def create_metric_inv(model, device='cpu'):
    """
    Metric creation for RHVAE model
    """
    def G_inv(z):
        return (
            model.M_tens.unsqueeze(0)
            * torch.exp(
                -torch.norm(model.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1)
                ** 2
                / (model.T ** 2)
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        ).sum(dim=1) + model.lbd * torch.eye(model.latent_dim).to(device)

    return G_inv

def create_dH_dz(model):
    """
    Computation of derivative of Hamiltonian for RHVAE model
    """
    def dH_dz(z, q):

        a = (
        torch.transpose(q.unsqueeze(-1).unsqueeze(1), 2, 3)
        @ model.M_tens.unsqueeze(0)
        @ q.unsqueeze(-1).unsqueeze(1)
        )

        b = centroids_tens.unsqueeze(0) - z.unsqueeze(1)

        return (
            -1
            / (model.T ** 2)
            * b.unsqueeze(-1)
            @ a
            * (
                torch.exp(
                    -torch.norm(
                        model.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1
                    )
                    ** 2
                    / (model.T ** 2)
                )
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        ).sum(dim=1)
    return dH_dz

