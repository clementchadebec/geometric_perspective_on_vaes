import torch
from tqdm import tqdm

def train_vae(epoch, args, model, train_loader, optimizer):
    train_loss = 0
    train_loss_rec = 0
    train_loss_kld = 0

    # Set model on training mode
    model.train()

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        if args.dynamic_binarization:
            x = torch.bernoulli(data)

        else:
            x = data

        # reset gradients
        optimizer.zero_grad()

        if args.model_name == "RHVAE":
            # forward pass
            (
                recon_batch,
                z,
                z0,
                rho,
                eps0,
                gamma,
                mu,
                log_var,
                G_inv,
                G_log_det,
            ) = model(data)
            # loss computation
            loss = model.loss_function(
                recon_batch,
                data,
                z0,
                z,
                rho,
                eps0,
                gamma,
                mu,
                log_var,
                G_inv,
                G_log_det,
            )

            loss_rec = torch.zeros(1)
            loss_kld = torch.zeros(1)

        else:

            # forward pass
            recon_batch, z, _, mu, log_var = model(data)
            # loss computation
            loss, loss_rec, loss_kld = model.loss_function(recon_batch, data, mu, log_var, z)

        # backward pass
        loss.backward()
        # optimization
        optimizer.step()

        train_loss += loss.item() / len(train_loader.dataset)
        train_loss_rec += loss_rec.item() / len(train_loader.dataset)
        train_loss_kld += loss_kld.item() / len(train_loader.dataset)

    if args.model_name == "RHVAE":
        model.update_metric()
    # calculate final loss

    return model, train_loss, train_loss_rec, train_loss_kld
