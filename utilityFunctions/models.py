import numpy as np

import torch
import torch.nn as nn


def get_mlp(layer_dims, activation=nn.ReLU, last_layer_activation=False):
    """
    Simple helper to create a fully connected network, given a list of
    number of hidden units in each layer.
    """
    layers = []
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        layers.append(activation())
    if not last_layer_activation:  # if not using activation in the last (output) layer
        layers = layers[:-1]

    return nn.Sequential(*layers)


from configs import float_type

if float_type == 'float32':
    float_type = torch.float32
else:
    float_type = torch.float

torch_softplus = torch.nn.Softplus()


class CondNoiseMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_dims, stoch,
                 activation=nn.ReLU, noise_activation=nn.ReLU, input_stats=None,
                 inv_masses=None, output_stats=None, output_raw=False,
                 sigma_fun='exp'):
        """
        Generic model of an implicit and sample-able conditional distribution, p(output | input), parameterized by MLP.
        In the deterministic case (stoch=False), output = nn(input).
        In the stochastic case, we first sample a conditional noise from p(\epsilon|input) = N(mu(input), sigma2(input))
        where mu() and sigma() are neural nets, then we deterministically transform [\epsilon, input] into output with
        another nn. We set \epsilon to have the same dimension as the output, as in a conditional normalizing flow.
        As neural nets tend to be easier to optimize with SGD when the input and output are standardized, this class
        provides the option to either train on standardized data (with input_stats=None, output_raw=False), or raw data
        (with input_stats, output_stats specified as np/torch arrays, and output_raw=True). When training on raw data,
        the input is standardized by input_stats before feeding to the NN, and the model output is the NN output
        un-standardized by output_stats.
        :param input_dim:
        :param output_dim:
        :param hidden_layer_dims:
        :param stoch:
        :param activation: type of actiavtion function in the NNs
        :param noise_activation: overriding option for the actiavtion function in the conditional noise NN
        :param input_stats: if specified, we assume the input is 'raw' (not standardized), and input_stats=[mean, std]
        of the raw input in the form of a 2 x D torch tensor, with inputs_stats[0]=mean, input_stats[1]=std; raw input
        will then be standardized by input_stats before being fed to the NN
        :param inv_masses: a vector of invariant masses of particles. If provided, then output_stats must also be
        provided, and we assume the (raw) model output consists of the 4-momentum (px,py,pz,E) of K particles concatenated
        together, and inv_masses is a vector of length K, such that for each particle, the invariant mass relation
        E^2 = px^2 + py^2 + pz^2 + inv_mass^2 holds.
        :param output_stats: [mean, std] of the raw output space in the form of a 2 x D torch tensor.
        :param output_raw: boolean; if True, the model output will be "raw"; otherwise, the NN output will be directly
        taken to be the model output.
        :param sigma_fun: activation ('squash') function that converts the unrestricted NN output to positively valued
        sigma; default is 'exp' (so the NN parameterizes log sigma), can also use 'softplus' for more numerical stability
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim  # dimensionality of the output space
        self.hidden_layer_dims = hidden_layer_dims
        if sigma_fun == 'exp':
            sigma_fun = torch.exp
        elif sigma_fun == 'softplus':
            sigma_fun = torch_softplus
        else:
            raise NotImplementedError
        self.sigma_fun = sigma_fun

        if isinstance(input_stats, np.ndarray):
            input_stats = torch.Tensor(input_stats)
        self.input_stats = input_stats
        # self.inv_masses = torch.Tensor(inv_masses)
        self.inv_masses = inv_masses
        if inv_masses is not None:
            assert output_stats is not None, "Must provide output stats in order to enforce invariant mass relation."
            num_particles = len(inv_masses)
            assert 4 * num_particles == self.output_dim, "Output space dimensionality != 4 times the number of particles"
            self.num_particles = num_particles
            output_dim -= num_particles  # these dimensions correspond to energy and will be manually computed
        if isinstance(output_stats, np.ndarray):
            output_stats = torch.Tensor(output_stats)
        self.output_stats = output_stats
        self.output_raw = output_raw

        self.stoch = stoch
        if not stoch:
            self.output_nn = get_mlp([input_dim, *hidden_layer_dims, output_dim], activation=activation,
                                     last_layer_activation=False)
        else:
            noise_dim = output_dim
            self.noise_dim = noise_dim
            self.output_nn = get_mlp([input_dim + noise_dim, *hidden_layer_dims, output_dim], activation=activation,
                                     last_layer_activation=False)
            # currently using the reverse architecture of the output_nn;
            # 2 * output_dim for mu and logsigma
            self.cond_noise_nn = get_mlp([input_dim,
                                          *list(reversed(hidden_layer_dims)), 2 * output_dim],
                                         activation=noise_activation, last_layer_activation=False)
            # self.cond_noise_nn = get_mlp([input_dim, *hidden_layer_dims, 2 * output_dim], activation=activation, last_layer_activation=False)

            # custom (hack) initialization to make sure logsigma won't be too
            # small at the beginning of training to cause NaN issues
            for layer in self.cond_noise_nn.modules():
                if isinstance(layer, nn.Linear):
                    final_bias = layer.bias
            final_bias.data.fill_(1.0)

    def forward(self, input):
        """
        Given input batch vector, produce samples from the conditional distribution p(output|input).
        :param input:
        :return:
        """
        if self.input_stats is not None:
            input_mean, input_std = self.input_stats.to(input.device)
            input = (input - input_mean) / input_std
        if self.stoch:
            noise_dim = self.noise_dim
            mean, logsigma = self.cond_noise_nn(input).split(noise_dim, dim=-1)
            # sigma = torch.exp(logsigma)
            # sigma = torch_softplus(logsigma)
            sigma = self.sigma_fun(logsigma)
            input_shape = input.shape
            eps = torch.randn([*input_shape[:-1], noise_dim], dtype=input.dtype, device=input.device)
            eps = mean + eps * sigma
            input = torch.cat((input, eps), dim=-1)
        X = self.output_nn(input)
        if self.inv_masses is not None:
            num_particles = self.num_particles
            batch = len(X)
            output = torch.zeros([batch, self.output_dim], dtype=X.dtype, device=X.device)
            output_mean, output_std = self.output_stats.to(X.device)
            idx_noE = np.hstack([np.arange(4 * i, 4 * i + 3) for i in range(num_particles)])  # NN output doesn't have E
            X = (X * output_std[idx_noE]) + output_mean[idx_noE]  # unstandardize
            for i in range(num_particles):
                # for the ith particle
                p = X[:, 3 * i:3 * (i + 1)]  # 3-momentum, (px, py, pz)
                m = self.inv_masses[i]
                E = ((p ** 2).sum(axis=-1) + m ** 2) ** 0.5
                output[:, 4 * i:4 * i + 3] = p
                output[:, 4 * i + 3] = E

            if self.output_raw:  # output in "raw" (data observation) space
                pass
            else:
                output = (output - output_mean) / output_std
        else:
            output = X
            if self.output_raw:
                output_mean, output_std = self.output_stats.to(X.device)
                output = (output * output_std) + output_mean

        return output


class CondNoiseAutoencoder(nn.Module):
    """
    Consists of a probabilistic encoder, q(z|x), and decoder p_G(x|z). In case they're deterministic (dirac delta
    functions), they're implemented by neural nets z -> x and x -> z. In the stochastic case, they're defined implicitly
    as (z, eps) -> x and (x, eps) -> z.
    """

    def __init__(self, x_dim, z_dim, hidden_layer_dims, stoch_enc=False, stoch_dec=False, activation=nn.ReLU,
                 raw_io=False, x_inv_masses=None, x_stats=None,
                 z_inv_masses=None, z_stats=None, **kwargs):
        """

        :param x_dim:
        :param z_dim:
        :param hidden_layer_dims:
        :param stoch_enc:
        :param stoch_dec:
        :param activation:
        :param raw_io: whether or not the model inputs/outputs raw observations; if False, the model assumes standardized
        input and output.
        :param x_inv_masses:
        :param x_stats:
        :param z_inv_masses:
        :param z_stats:
        """
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.stoch_enc = stoch_enc
        self.stoch_dec = stoch_dec
        if raw_io:
            assert x_stats is not None and z_stats is not None

        self.encoder = CondNoiseMLP(x_dim, z_dim, hidden_layer_dims, stoch_enc, activation,
                                    input_stats=x_stats if raw_io else None,
                                    inv_masses=z_inv_masses, output_stats=z_stats,
                                    output_raw=raw_io, **kwargs)
        self.decoder = CondNoiseMLP(z_dim, x_dim, list(reversed(hidden_layer_dims)), stoch_dec, activation,
                                    input_stats=z_stats if raw_io else None,
                                    inv_masses=x_inv_masses, output_stats=x_stats,
                                    output_raw=raw_io, **kwargs)

    def encode(self, x):
        """
        Draw a sample from p_G(z|x) (which might be a dirac delta fn in case of det encoder)
        :param x:
        :return:
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Draw a sample from q(x|z) (which might be a dirac delta fn in case of det deccoder)
        :param z:
        :return:
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Given data batch, return latent code and reconstruction.
        :param x:
        :return:
        """
        z_tilde = self.encode(x)
        x_tilde = self.decode(z_tilde)
        return z_tilde, x_tilde


class BasicAutoencoder(nn.Module):
    """
    Basic (encoder, decoder) implemented by MLPs.
    Consists of a probabilistic encoder, q(z|x), and decoder p_G(x|z). In case they're deterministic (dirac delta
    functions), they're implemented by neural nets z -> x and x -> z. In the stochastic case, they're defined implicitly
    as (z, eps) -> x and (x, eps) -> z.
    """

    def __init__(self, x_dim, z_dim, hidden_layer_dims, stoch_enc=False, stoch_dec=False, activation=nn.ReLU):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.stoch_enc = stoch_enc
        self.stoch_dec = stoch_dec
        layer_dims = [x_dim, *hidden_layer_dims, z_dim]
        if stoch_enc:
            layer_dims[0] *= 2  # for additional noise input
        self.encoder = get_mlp(layer_dims, activation=activation, last_layer_activation=False)
        layer_dims = [z_dim, *hidden_layer_dims, x_dim]
        if stoch_dec:
            layer_dims[0] *= 2  # for additional noise input
        self.decoder = get_mlp(layer_dims, activation=activation, last_layer_activation=False)

    def sample_nosie(self, shape, dtype=float_type):
        eps = torch.randn(shape, dtype=dtype)
        return eps

    def encode(self, x):
        """
        Draw a sample from p_G(z|x) (which might be a dirac delta fn in case of det encoder)
        :param x:
        :return:
        """
        if self.stoch_enc:
            eps = self.sample_nosie(x.shape)
            eps = eps.to(dtype=x.dtype, device=x.device)
            x = torch.cat((x, eps), dim=1)
        return self.encoder(x)

    def decode(self, z):
        """
        Draw a sample from q(x|z) (which might be a dirac delta fn in case of det deccoder)
        :param z:
        :return:
        """
        if self.stoch_dec:
            eps = self.sample_nosie(z.shape)
            eps = eps.to(dtype=z.dtype, device=z.device)
            z = torch.cat((z, eps), dim=1)
        return self.decoder(z)

    def forward(self, x):
        """
        Given data batch, return latent code and reconstruction.
        :param x:
        :return:
        """
        z_tilde = self.encode(x)
        x_tilde = self.decode(z_tilde)
        return z_tilde, x_tilde


class ResBlock(torch.nn.Module):
    def __init__(self, input_dim, activation=nn.ReLU, depth=1):
        """
        A residual block that implements f(x) = MLP(x) + x, where the MLP consists of a series of (BN, activation, linear)
        operations. The output has the same dimensionality as input.
        Architecture (e) 'full pre-activation' of https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
        """
        super().__init__()
        layers = []
        for _ in range(depth):
            layers += [
                nn.BatchNorm1d(num_features=input_dim),
                activation(),
                nn.Linear(input_dim, input_dim)  # linear layer preserves dimensionality for residual to work
            ]
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x) + x


class StochasticResNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_dims, stoch,
                 activation=nn.ReLU, input_stats=None,
                 inv_masses=None, output_stats=None, output_raw=False,
                 io_residual=False, res_mlp_depth=1):
        """
        Generic model of an implicit and sample-able conditional distribution, p(output | input), implemented by a ResNet.
        In the deterministic case (stoch=False), output = nn(input).
        In the stochastic case, we sample \epsilon from standard Normal, concatenate it with input, and pass
        [\epsilon, input] through a ResNet to get output vector.
        We set \epsilon to have the same dimension as the output, as in a conditional normalizing flow.
        As neural nets tend to be easier to optimize with SGD when the input and output are standardized, this class
        provides the option to either train on standardized data (with input_stats=None, output_raw=False), or raw data
        (with input_stats, output_stats specified as np/torch arrays, and output_raw=True). When training on raw data,
        the input is standardized by input_stats before feeding to the NN, and the model output is the NN output
        un-standardized by output_stats.
        :param input_dim:
        :param output_dim:
        :param hidden_layer_dims: list of dimensions of the ResBlocks; whenever two adjacent numbers are equal, a ResBlock
        with that number of dimensions is created (since a ResBlock must preserve input output dimensionality); otherwise,
        if m!=n, a simple linear layer R^m -> R^n is created to operate between the two dimensions.
        :param stoch:
        :param activation: type of actiavtion function in the NNs
        :param noise_activation: overriding option for the actiavtion function in the conditional noise NN
        :param input_stats: if specified, we assume the input is 'raw' (not standardized), and input_stats=[mean, std]
        of the raw input in the form of a 2 x D torch tensor, with inputs_stats[0]=mean, input_stats[1]=std; raw input
        will then be standardized by input_stats before being fed to the NN
        :param inv_masses: a vector of invariant masses of particles. If provided, then output_stats must also be
        provided, and we assume the (raw) model output consists of the 4-momentum (px,py,pz,E) of K particles concatenated
        together, and inv_masses is a vector of length K, such that for each particle, the invariant mass relation
        E^2 = px^2 + py^2 + pz^2 + inv_mass^2 holds.
        :param output_stats: [mean, std] of the raw output space in the form of a 2 x D torch tensor.
        :param output_raw: boolean; if True, the model output will be "raw"; otherwise, the NN output will be directly
        taken to be the model output.
        :param io_residual: if True, will add a residual connection from input directly to the output.
        If inv-mass constraint is enforced to output so that the NN output is only 18 dimensional, the corresponding 18
        elements of the input (instead of the entire input vector) will be added to the NN output.
        :param res_mlp_depth: depth of MLP within each ResBlock.

        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim  # dimensionality of the output space
        self.hidden_layer_dims = hidden_layer_dims
        self.io_residual = io_residual

        if isinstance(input_stats, np.ndarray):
            input_stats = torch.Tensor(input_stats)
        self.input_stats = input_stats
        # self.inv_masses = torch.Tensor(inv_masses)
        self.inv_masses = inv_masses
        if inv_masses is not None:
            assert output_stats is not None, "Must provide output stats in order to enforce invariant mass relation."
            num_particles = len(inv_masses)
            assert 4 * num_particles == self.output_dim, "Output space dimensionality != 4 times the number of particles"
            self.num_particles = num_particles
            output_dim -= num_particles  # these dimensions correspond to energy and will be manually computed
        if isinstance(output_stats, np.ndarray):
            output_stats = torch.Tensor(output_stats)
        self.output_stats = output_stats
        self.output_raw = output_raw

        self.stoch = stoch

        if stoch:
            self.noise_dim = output_dim
            input_dim += self.noise_dim
        layer_dims = [input_dim, *hidden_layer_dims, output_dim]
        nn_layers = []
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
            if in_dim == out_dim:  # use residual block
                layer = ResBlock(in_dim, activation, depth=res_mlp_depth)
            else:  # just apply a simple linear transform to change dimensionality
                layer = nn.Linear(in_dim, out_dim)
            nn_layers.append(layer)
        self.nn = nn.Sequential(*nn_layers)

    def forward(self, input):
        """
        Given input batch vector, produce samples from the conditional distribution p(output|input).
        :param input:
        :return:
        """
        if self.input_stats is not None:
            raw_input = input.clone()
            input_mean, input_std = self.input_stats.to(input.device)
            input = (input - input_mean) / input_std
        if self.stoch:
            noise_dim = self.noise_dim
            input_shape = input.shape
            eps = torch.randn([*input_shape[:-1], noise_dim], dtype=input.dtype, device=input.device)
            input = torch.cat((input, eps), dim=-1)
        X = self.nn(input)
        if self.inv_masses is not None:
            num_particles = self.num_particles
            batch = len(X)
            output = torch.zeros([batch, self.output_dim], dtype=X.dtype, device=X.device)
            output_mean, output_std = self.output_stats.to(X.device)
            idx_noE = np.hstack([np.arange(4 * i, 4 * i + 3) for i in range(num_particles)])  # NN output doesn't have E
            X = (X * output_std[idx_noE]) + output_mean[idx_noE]  # unstandardize
            if self.io_residual:  # residual connection from the corresponding elements of the raw input
                X = X + raw_input[:, idx_noE]  # X is [batch_size, 18], raw_input is [batch_size, 24]
            for i in range(num_particles):
                # for the ith particle
                p = X[:, 3 * i:3 * (i + 1)]  # 3-momentum, (px, py, pz)
                m = self.inv_masses[i]
                E = ((p ** 2).sum(axis=-1) + m ** 2) ** 0.5
                output[:, 4 * i:4 * i + 3] = p
                output[:, 4 * i + 3] = E

            if self.output_raw:  # output in "raw" (data observation) space
                pass
            else:
                output = (output - output_mean) / output_std
        else:
            output = X
            if self.output_raw:
                output_mean, output_std = self.output_stats.to(X.device)
                output = (output * output_std) + output_mean
                if self.io_residual:  # residual connection from the corresponding elements of the input
                    output = output + raw_input

        return output


class MaskedStochasticResNet(StochasticResNet):
    """
    A wrapper around StochasticResNet that allows optionally applying a mask/filter to the output.
    """

    def __init__(self, mask_fun, *args, **kwargs):
        """

        :param mask_fun: given N x D matrix of "raw" data, returns N dimensional vector indicating whether each row is
        "valid". This was used for filtering out ppttbar data that don't pass the detector threshold.
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.mask_fun = mask_fun

    def forward(self, input, mask_output=False, return_mask=True):
        """
        :param input:
        :param mask_output: whether to apply mask to output
        :param return_mask: whether to also return the boolean that indicats which input passed the threshold
        :return:
        """
        output = super().forward(input)
        if (not mask_output) and (not return_mask):
            return output

        if self.output_raw:  # output is in raw space already
            raw_output = output
        else:
            output_mean, output_std = self.output_stats.to(input.device)
            raw_output = (output * output_std) + output_mean

        mask = self.mask_fun(raw_output)

        if mask_output:  # apply mask
            output = output[mask]
        if return_mask:
            return output, mask
        else:
            return output


class Autoencoder(nn.Module):
    """
    A generic autoencoder module that wraps around an (encoder, decoder) pair, with encoder/decoder implemented by
        a user defined ConditionalModel class. Subsumes the legacy CondNoiseAutoencoder (specialized to CondNoiseMLP).
    """

    def __init__(self, x_dim, z_dim, ConditionalModel, encoder_hidden_layer_dims, stoch_enc=True, stoch_dec=True,
                 activation=nn.ReLU,
                 raw_io=False, x_inv_masses=None, x_stats=None,
                 z_inv_masses=None, z_stats=None, **kwargs):
        """
        :param x_dim:
        :param z_dim:
        :param ConditionalModel: a class that implements the (stochastic) encoder and decoder model.
        :param encoder_hidden_layer_dims: list of hidden layer dims for the encoder; the reverse architecture is used
        for decoder.
        :param stoch_enc:
        :param stoch_dec:
        :param activation:
        :param raw_io: whether or not the model inputs/outputs raw observations; if False, the model assumes standardized
        input and output.
        :param x_inv_masses:
        :param x_stats:
        :param z_inv_masses:
        :param z_stats:
        """
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.stoch_enc = stoch_enc
        self.stoch_dec = stoch_dec
        assert ConditionalModel in (StochasticResNet, CondNoiseMLP)
        if raw_io:
            assert x_stats is not None and z_stats is not None

        self.encoder = ConditionalModel(x_dim, z_dim, encoder_hidden_layer_dims, stoch_enc, activation,
                                        input_stats=x_stats if raw_io else None,
                                        inv_masses=z_inv_masses, output_stats=z_stats,
                                        output_raw=raw_io, **kwargs)
        self.decoder = ConditionalModel(z_dim, x_dim, list(reversed(encoder_hidden_layer_dims)), stoch_dec, activation,
                                        input_stats=z_stats if raw_io else None,
                                        inv_masses=x_inv_masses, output_stats=x_stats,
                                        output_raw=raw_io, **kwargs)

    def encode(self, x, *args, **kwargs):
        """
        Draw a sample from p_G(z|x) (which might be a dirac delta fn in case of det encoder)
        :param x:
        :return:
        """
        return self.encoder(x, *args, **kwargs)

    def decode(self, z, *args, **kwargs):
        """
        Draw a sample from q(x|z) (which might be a dirac delta fn in case of det deccoder)
        :param z:
        :return:
        """
        return self.decoder(z, *args, **kwargs)

    def forward(self, x):
        """
        Given data batch, return latent code and reconstruction.
        :param x:
        :return:
        """
        z_tilde = self.encode(x)
        x_tilde = self.decode(z_tilde)
        return z_tilde, x_tilde

