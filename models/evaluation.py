import torch


def eval_vae(epoch, args, model, val_loader):
    val_loss = 0
    val_loss_rec = 0
    val_loss_kld = 0

    # Set model on eval mode
    model.eval()

    for batch_idx, (data, target) in enumerate(val_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        if args.dynamic_binarization:
            x = torch.bernoulli(data)

        else:
            x = data

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
            with torch.no_grad():

                # forward pass
                recon_batch, z, _, mu, log_var = model(data)
                # loss computation
                loss, loss_rec, loss_kld = model.loss_function(recon_batch, data, mu, log_var, z)



        val_loss += loss.item() / len(val_loader.dataset)
        val_loss_rec += loss_rec.item() / len(val_loader.dataset)
        val_loss_kld += loss_kld.item() / len(val_loader.dataset)

    # calculate final loss

    return val_loss, val_loss_rec, val_loss_kld
