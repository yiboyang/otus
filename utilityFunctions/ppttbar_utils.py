import torch
from torch import optim
import torch.nn as nn
from func_utils import sliced_wd, latent_loss, data_loss, anchor_loss

from configs import device, EVAL_NUM_SLICES, TRAIN_NUM_SLICES
from ppttbar_constraints import threshold_check

# Use the following cost fucntion for the (S)WD: c(u, v) = |u-v|^2
P = 2

##########! Check that the above are necessary imports

def train_epoch(model, optimizer, x_loader, z_loader, beta, lamb, tau=0, rho=0, nu_e=0, nu_d=0, z_loss_fun=None, x_constraint_loss_fun=None, device=device):
    """
    Train model for one epoch
    :param model:
    :param optimizer:
    :param x_loader:
    :param z_loader:
    :param beta: coefficient in front of data loss, E[c(x, x reconstructed)], where c is typically the 2-norm.
    :param lamb: coefficient in front of latent loss, SWD between p(z) and Q(z):=\int_x p(x) Q(z|x)
    :param tau: coefficient in front of "alt_x_loss", which is the SWD between p(x) and p_G(x):=\int_z p(z) p_G(x|z);
    this loss is not part of the WAE formulation and is discouraged.
    :param rho: coefficient in front of decoder constraint loss (now abandoned)
    :param nu_e: coefficient in front of encoder "anchor loss"
    :param nu_d: coefficient in front of decoder "anchor loss"
    :param num_slices: number of rand projections for computing SWD
    :param device:
    :return:
    """

    assert rho == 0, 'no longer using decoder soft constraint loss, as all constraints are now hardcoded in model'
    model.train()
    if z_loss_fun is None:
        z_loss_fun = lambda z, z_tilde: sliced_wd(z, z_tilde, TRAIN_NUM_SLICES, P)
    for batch_idx, (x, z) in enumerate(zip(x_loader, z_loader)):
        x, z = x.to(device), z.to(device)
        optimizer.zero_grad()
        z_tilde = model.encode(x)
        z_loss = z_loss_fun(z, z_tilde) if lamb > 0 else 0

        x_loss = 0
        passing_rate = -1
        if beta > 0 or tau > 0 or rho > 0 or nu_d > 0:
            model_x = model.decode(z)
            marg_good_mask = threshold_check(model_x)  # masks on marginal samples from p_D(x)
            passing_rate = marg_good_mask.type(torch.FloatTensor).mean()  # this is the normalizing constant of \bar{p}_D(x)
            #if batch_idx % 1000 == 0:
            #print('passing rate', passing_rate)
            
            if beta > 0:
                x_tilde = model.decode(z_tilde)  # full "autoencoding chian":  x -> z_tilde -> x_tilde
                good_mask = threshold_check(x_tilde)
                x_loss = data_loss(x[good_mask], x_tilde[good_mask], P) / passing_rate

        encoder_anchor_loss = anchor_loss(z_tilde, x) if nu_e > 0 else 0
        decoder_anchor_loss = 0
        if tau > 0 or rho > 0 or nu_d > 0:
            alt_x_loss = z_loss_fun(x[marg_good_mask], model_x[marg_good_mask]) if tau > 0 else 0  # SWD(p(x), p_D(x))
            #x_constraint_loss = x_constraint_loss_fun(model_x) if rho > 0 else 0  # x_constraint_loss directly takes neural net output
            x_constraint_loss = 0
            decoder_anchor_loss = anchor_loss(z[marg_good_mask], model_x[marg_good_mask]) / passing_rate if nu_d > 0 else 0
        else:
            alt_x_loss = 0
            x_constraint_loss = 0

        loss = beta * x_loss + lamb * z_loss + tau * alt_x_loss + rho * x_constraint_loss + nu_e * encoder_anchor_loss + nu_d * decoder_anchor_loss
        loss.backward()
        optimizer.step()
    losses = dict(passing_rate=passing_rate, loss=loss, x_loss=x_loss, z_loss=z_loss, alt_x_loss=alt_x_loss, x_constraint_loss=x_constraint_loss,
                  encoder_anchor_loss=encoder_anchor_loss, decoder_anchor_loss=decoder_anchor_loss)
    return losses


def eval_epoch(model, x_loader, z_loader, z_loss_fun=None, device=device):
    """
    Evaluate model on one epoch's worth of data.
    :param model:
    :param x_loader:
    :param z_loader:
    :param num_slices:
    :param device:
    :return:
    """
    if z_loss_fun is None:
        z_loss_fun = lambda z, z_tilde: sliced_wd(z, z_tilde, EVAL_NUM_SLICES, P)
    model.eval()
    num_samples = 0
    total_losses = {key: 0. for key in ['passing_rate', 'x_loss', 'z_loss', 'alt_x_loss', 'encoder_anchor_loss', 'decoder_anchor_loss']}
    with torch.no_grad():
        for batch_idx, (x, z) in enumerate(zip(x_loader, z_loader)):
            x, z = x.to(device), z.to(device)
            
            model_x = model.decode(z)
            marg_good_mask = threshold_check(model_x)
            passing_rate = marg_good_mask.type(torch.FloatTensor).mean()  # this is the normalizing constant of \bar{p}_D(x)
            z_tilde = model.encode(x)
            x_tilde = model.decode(z_tilde)
            good_mask = threshold_check(x_tilde)

            losses = {key: 0. for key in
                      ['passing_rate', 'x_loss', 'z_loss', 'alt_x_loss', 'encoder_anchor_loss', 'decoder_anchor_loss']}
            losses['passing_rate'] = passing_rate
            z_loss = z_loss_fun(z, z_tilde)
            losses['z_loss'] = z_loss
            losses['x_loss'] = data_loss(x[good_mask], x_tilde[good_mask], P) / passing_rate
            # losses['alt_x_loss'] = sliced_wd(x, model.decode(z), num_slices, P)
            losses['alt_x_loss'] = z_loss_fun(x[marg_good_mask], model_x[marg_good_mask])
            losses['encoder_anchor_loss'] = anchor_loss(z_tilde, x)
            losses['decoder_anchor_loss'] = anchor_loss(z[marg_good_mask], model_x[marg_good_mask]) / passing_rate
            for key in losses:
                total_losses[key] += losses[key] * len(x)
            num_samples += len(x)
    avg_losses = total_losses.copy()
    for key in total_losses:
        avg_losses[key] = float(total_losses[key] / num_samples)
    avg_losses['loss'] = avg_losses['z_loss'] + avg_losses['alt_x_loss']  # overall loss for model selection
    return avg_losses

def train_and_val(model, train_loaders, eval_loaders, config, optimizer=None,
        z_loss_fun=None, x_constraint_loss_fun=None, verbose=False, prev_hist=None,
                   log_freq=10, lr_decay=False, device=device):
    """
    :param model:
    :param train_loaders:
    :param eval_loaders:
    :param config:
    :param optimizer:
    :param verbose:
    :param prev_hist:
    :param log_freq:
    :param device:
    :return:
    """

    if z_loss_fun is None:
        num_slices = TRAIN_NUM_SLICES
        z_loss_fun = lambda z, z_hat: sliced_wd(z, z_hat, num_slices, P)
    print(config)
    if 'beta' not in config:
        config['beta'] = 1.
        print('Defaulting beta (data loss coefficient) to 1.')
    if 'nu_e' not in config:
        config['nu_e'] = config['nu']
    if 'nu_d' not in config:
        config['nu_d'] = config['nu']
    model.to(device)
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    if lr_decay:
        from torch.optim.lr_scheduler import LambdaLR
        # base_lr = config['lr']
        halflife = 10  # half about every 10 epochs
        lr_lambda = lambda epoch: 1 / (1 + 1 / halflife * epoch)  # this computes the factor to multiply the base_lr by
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=lr_decay_patience, verbose=True)
    epochs = config['epochs']
    if prev_hist:  # if resuming training
        history = prev_hist
    else:
        history = {}
    for key in ('epoch', 'passing_rate', 'train_loss', 'train_x_loss', 'train_z_loss', 'train_x_constraint_loss',
        'train_alt_x_loss', 'eval_loss', 'eval_x_loss', 'eval_z_loss',
        'eval_alt_x_loss'):
        if key not in history:
            history[key] = []

    for i in range(epochs):
        train_losses = train_epoch(model, optimizer, *train_loaders, beta=config['beta'], lamb=config['lamb'],
                                   tau=config['tau'], rho=config['rho'], nu_e=config['nu_e'], nu_d=config['nu_d'],
                                   z_loss_fun=z_loss_fun, x_constraint_loss_fun=x_constraint_loss_fun)
        # eval_losses = eval_epoch(model, *eval_loaders, num_slices=EVAL_NUM_SLICES)
        if lr_decay:
            # scheduler.step(eval_losses['loss']) # ReduceLROnPlateau
            scheduler.step()  # no need for eval_loss with LambdaLR

        if verbose and (i % log_freq == 0 or i == epochs - 1):
            print('epoch:\t%d' % i)
            tmpd = train_losses
            #print('train -- prate:%.3g, loss:%.6g, x_loss:%.6g, z_loss:%.6g, alt_x_loss:%.6g, x_constraint_loss:%.6g, anchor_loss:%g' \
            #      % (tmpd['loss'], tmpd['x_loss'], tmpd['z_loss'], tmpd['alt_x_loss'], tmpd['x_constraint_loss'],
            #tmpd['encoder_anchor_loss'] + tmpd['decoder_anchor_loss']))
            print('train -- loss:%.6g, prate:%.3g, x_loss:%.6g, z_loss:%.6g, anchor_loss:%g' \
                  % (tmpd['loss'], tmpd['passing_rate'], tmpd['x_loss'], tmpd['z_loss'], tmpd['encoder_anchor_loss'] + tmpd['decoder_anchor_loss']))
            if history['epoch'] == []:
                history['epoch'].append(i)
            else:
                history['epoch'].append(history['epoch'][-1] + log_freq)
            history['passing_rate'].append(tmpd['passing_rate'])
            history['train_loss'].append(tmpd['loss'])
            history['train_x_loss'].append(tmpd['x_loss'])
            history['train_z_loss'].append(tmpd['z_loss'])
            history['train_alt_x_loss'].append(tmpd['alt_x_loss'])
            history['train_x_constraint_loss'].append(tmpd['x_constraint_loss'])

            eval_losses = eval_epoch(model, *eval_loaders, z_loss_fun=z_loss_fun)
            history['eval_loss'].append(eval_losses['loss'])
            history['eval_x_loss'].append(eval_losses['x_loss'])
            history['eval_z_loss'].append(eval_losses['z_loss'])
            history['eval_alt_x_loss'].append(eval_losses['alt_x_loss'])

            tmpd = eval_losses
            print('eval -- loss:%.6g, prate:%.3g, x_loss:%.6g, z_loss:%.6g, alt_x_loss:%.6g, anchor_loss:%g' \
                  % (tmpd['loss'], tmpd['passing_rate'], tmpd['x_loss'], tmpd['z_loss'], tmpd['alt_x_loss'],
                     tmpd['encoder_anchor_loss'] + tmpd['decoder_anchor_loss']))

    return eval_losses, history
