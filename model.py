import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pdb

VISUALIZE_DASHBOARD = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# {'hidden': 100, 'dense': 100, 'conv1': 2, 'conv2': 3, 'conv3': 4}
USE_MOLECULE_VAE_PARAMS = True
LATENT_REP_SIZE = 10
HIDDEN_MOL_VAE = 100

class Decoder(nn.Module):
    def __init__(self, input_size=200, hidden_n=200, output_feature_size=12, max_seq_length=15):
        super(Decoder, self).__init__()
        self.max_seq_length = max_seq_length
        if USE_MOLECULE_VAE_PARAMS:
            hidden_n = HIDDEN_MOL_VAE
            input_size = HIDDEN_MOL_VAE
        self.hidden_n = hidden_n
        self.output_feature_size = output_feature_size
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.fc_input = nn.Linear(input_size, hidden_n)
        if USE_MOLECULE_VAE_PARAMS:
            self.batch_norm = nn.BatchNorm1d(LATENT_REP_SIZE)
            self.fc_input = nn.Linear(LATENT_REP_SIZE, hidden_n)
        # we specify each layer manually, so that we can do teacher forcing on the last layer.
        # we also use no drop-out in this version.
        self.gru_1 = nn.GRU(input_size=input_size, hidden_size=hidden_n, batch_first=True).to(device)
        self.gru_2 = nn.GRU(input_size=input_size, hidden_size=hidden_n, batch_first=True).to(device)
        self.gru_3 = nn.GRU(input_size=input_size, hidden_size=hidden_n, batch_first=True).to(device)
        self.fc_out = nn.Linear(hidden_n, output_feature_size)

    def forward(self, encoded, hidden_1, hidden_2, hidden_3, beta=0.3, target_seq=None):
        _batch_size = encoded.size()[0]

        embedded = F.relu(self.fc_input(self.batch_norm(encoded))) \
            .view(_batch_size, 1, -1) \
            .repeat(1, self.max_seq_length, 1).to(device)
        # batch_size, seq_length, hidden_size; batch_size, hidden_size
        # pdb.set_trace()
        out_1, hidden_1 = self.gru_1(embedded, hidden_1)
        out_2, hidden_2 = self.gru_2(out_1, hidden_2)
        # NOTE: need to combine the input from previous layer with the expected output during training.
        if self.training and target_seq:
            out_2 = out_2 * (1 - beta) + target_seq * beta
        out_3, hidden_3 = self.gru_3(out_2, hidden_3)
        out = self.fc_out(out_3.contiguous().view(-1, self.hidden_n)).view(_batch_size, self.max_seq_length,
                                                                           self.output_feature_size)
        return F.relu(torch.sigmoid(out)), hidden_1, hidden_2, hidden_3

    def init_hidden(self, batch_size):
        # NOTE: assume only 1 layer no bi-direction
        h1 = Variable(torch.zeros(1, batch_size, self.hidden_n), requires_grad=False)
        h2 = Variable(torch.zeros(1, batch_size, self.hidden_n), requires_grad=False)
        h3 = Variable(torch.zeros(1, batch_size, self.hidden_n), requires_grad=False)
        return h1, h2, h3


class Encoder(nn.Module):
    def __init__(self, L, k1=2, k2=3, k3=4, hidden_n=200):
        super(Encoder, self).__init__()
        # NOTE: GVAE implementation does not use max-pooling. Original DCNN implementation uses max-k pooling.
        self.conv_1 = nn.Conv1d(in_channels=12, out_channels=12, kernel_size=k1, groups=12)
        self.bn_1 = nn.BatchNorm1d(12)
        self.conv_2 = nn.Conv1d(in_channels=12, out_channels=12, kernel_size=k2, groups=12)
        self.bn_2 = nn.BatchNorm1d(12)
        self.conv_3 = nn.Conv1d(in_channels=12, out_channels=12, kernel_size=k3, groups=12)
        self.bn_3 = nn.BatchNorm1d(12)
        latent_rep_size = hidden_n
        if USE_MOLECULE_VAE_PARAMS:
            latent_rep_size = LATENT_REP_SIZE
            hidden_n = HIDDEN_MOL_VAE
        # todo: harded coded because I can LOL
        self.fc_0 = nn.Linear(12 * 9, hidden_n)
        
        self.fc_mu = nn.Linear(hidden_n, latent_rep_size)
        
        if GrammarVariationalAutoEncoder.VAE_MODE:
            self.fc_var = nn.Linear(hidden_n, latent_rep_size)

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.transpose(1, 2).contiguous()
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))
        x_ = x.view(batch_size, -1)
        h = self.fc_0(x_)
        if GrammarVariationalAutoEncoder.VAE_MODE:
            return self.fc_mu(h), self.fc_var(h)
        else:
            return self.fc_mu(h), None

from visdom_helper.visdom_helper import Dashboard


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.bce_loss.size_average = False
        if VISUALIZE_DASHBOARD:
            self.dashboard = Dashboard('Variational-Autoencoder-experiment')

    # question: how is the loss function using the mu and variance?
    def forward(self, x, recon_x, mu=None, log_var=None):
        """gives the batch normalized Variational Error."""

        batch_size = x.size()[0]
        BCE = self.bce_loss(recon_x, x)
        KLD = 0
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        if GrammarVariationalAutoEncoder.VAE_MODE:
            KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
            KLD = torch.sum(KLD_element).mul_(-0.5)

        return (BCE + KLD) / batch_size


class GrammarVariationalAutoEncoder(nn.Module):
    VAE_MODE = False
    def __init__(self):
        super(GrammarVariationalAutoEncoder, self).__init__()
        self.encoder = Encoder(15).to(device)
        self.decoder = Decoder().to(device)

    def forward(self, x):
        batch_size = x.size()[0]
        mu = None
        log_var = None
        if GrammarVariationalAutoEncoder.VAE_MODE:
            mu, log_var = self.encoder(x)
            z = self.reparameterize(mu, log_var)
        else:
            z, _none = self.encoder(x)
        h1, h2, h3 = self.decoder.init_hidden(batch_size)
        h1 = h1.to(device)
        h2 = h2.to(device)
        h3 = h3.to(device)
        output, h1, h2, h3 = self.decoder(z, h1, h2, h3)
        if GrammarVariationalAutoEncoder.VAE_MODE:
            return output, mu, log_var
        else: 
            return output

    def save_model(self, root_path):
        torch.save(self.state_dict(), root_path+'grammar_ae_model.pt')
        torch.save(self.decoder.state_dict(), root_path+'grammar_decoder.pt')
        torch.save(self.encoder.state_dict(), root_path+'grammar_autoencoder.pt')

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        vector_size = log_var.size()
        eps = Variable(torch.FloatTensor(vector_size).normal_())
        std = log_var.mul(0.5).exp_()
        return eps.mul(std).add_(mu)

    def decode(self, z):
        batch_size = z.size()[0]
        h1, h2, h3 = self.decoder.init_hidden(batch_size)
        h1 = h1.to(device)
        h2 = h2.to(device)
        h3 = h3.to(device)
        output, h1, h2, h3 = self.decoder(z, h1, h2, h3)
        return output
