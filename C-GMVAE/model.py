import torch
import torch.nn as nn 
import torch.nn.functional as F 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(p=args.keep_prob)

        '''Feature encoder'''
        self.fx = nn.Sequential(
            nn.Linear(args.feature_dim, 256),
            nn.ReLU(),
            self.dropout,
            nn.Linear(256, 512),
            nn.ReLU(),
            self.dropout,
            nn.Linear(512, 256),
            nn.ReLU(),
            self.dropout
        )
        self.fx_mu = nn.Linear(256, args.latent_dim)
        self.fx_logvar = nn.Linear(256, args.latent_dim)

        '''Latent encoder'''
        self.label_lookup = nn.Linear(args.label_dim, args.emb_size)
        self.fe = nn.Sequential(
            nn.Linear(args.emb_size, 512),
            nn.ReLU(),
            self.dropout,
            nn.Linear(512, 256),
            nn.ReLU(),
            self.dropout
        )

        self.fe_mu = nn.Linear(256, args.latent_dim)
        self.fe_logvar = nn.Linear(256, args.latent_dim)

        '''Decoder'''
        self.fd = nn.Sequential(
            nn.Linear(args.feature_dim + args.latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, args.emb_size),
            nn.LeakyReLU()
        )
    
    def label_encode(self, x):
        # L, L -> L, emb_size
        h0 = self.dropout(F.relu(self.label_lookup(x)))
        # L, emb_size -> L, 256
        h = self.fe(h0)
        # L, 256 -> L, latent_dim
        mu = self.fe_mu(h)
        # L, 256 -> L, latent_dim
        logvar = self.fe_logvar(h)
        fe_output = {
            'fe_mu': mu,
            'fe_logvar': logvar
        }
        return fe_output
    
    def feat_encode(self, x):
        # B, F --> B, 256
        h = self.fx(x)
        # B, 256 --> B, latent_dim
        mu = self.fx_mu(h)
        # B, 256 --> B, latent_dim
        logvar = self.fx_logvar(h)
        fx_output = {
            'fx_mu': mu,
            'fx_logvar': logvar
        }
        return fx_output
    
    def decode(self, z):
        # B, F + latent_dim --> B, emb_size
        d = self.fd(z)
        d = F.normalize(d, dim=1)
        return d
    
    def label_forward(self, x, feat):
        # x : B, L
        # feat : B, F

        # L
        n_label = x.shape[1]
        # L, L
        all_labels = torch.eye(n_label).to(device)
        # L, latent_dim and L, latent_dim
        fe_output = self.label_encode(all_labels)
        # L, latent_dim
        mu = fe_output['fe_mu']
        # B, L @ L, latent_dim --> B , latent_dim
        z = torch.matmul(x, mu) / x.sum(1, keepdim=True)
        # B, F + latent_dim --> B, emb_size
        label_emb = self.decode(torch.cat((feat, z), 1))
        # B, emb_size
        fe_output['label_emb'] = label_emb 
        return fe_output


    def feat_forward(self, x):
        # B, F
        fx_output = self.feat_encode(x) 
        # B, latent_dim
        mu = fx_output['fx_mu']
        # B, latent_dim
        logvar = fx_output['fx_logvar']

        if not self.training:
            z = mu 
            z2 = mu

        else:
            # B, latent_dim
            z = reparameterize(mu, logvar)
            # B, latent_dim
            z2 = reparameterize(mu, logvar)

        # B, F + latent_dim --> B, emb_size
        feat_emb = self.decode(torch.cat((x, z), 1))
        # B, F + latent_dim --> B, emb_size
        feat_emb2 = self.decode(torch.cat((x, z2), 1))
        fx_output['feat_emb'] = feat_emb
        fx_output['feat_emb2'] = feat_emb2
        return fx_output
    
    def forward(self, label, feature):
        # label: B, L
        # feature: B, F

        fe_output = self.label_forward(label, feature)
        # B, emb_size
        label_emb = fe_output['label_emb']

        fx_output = self.feat_forward(feature)
        # B, emb_size and B, emb_size
        feat_emb, feat_emb2 = fx_output['feat_emb'], fx_output['feat_emb2']
        # self.label_lookup.weight = nn.Linear(args.label_dim, args.emb_size)
        # embs : emb_size, L
        embs = self.label_lookup.weight
        # B, emb_size @ emb_size, L --> B, L
        label_out = torch.matmul(label_emb, embs)
        # B, emb_size @ emb_size, L --> B, L
        feat_out = torch.matmul(feat_emb, embs)
        # B, emb_size @ emb_size, L --> B, L
        feat_out2 = torch.matmul(feat_emb2, embs)

        fe_output.update(fx_output)
        output = fe_output
        output['embs'] = embs # emb_size, L
        output['label_out'] = label_out # B, L
        output['feat_out'] = feat_out # B, L
        output['feat_out2'] = feat_out2 # B, L
        output['feat'] = feature # B, F

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
        

def compute_loss(input_label, output, args=None):
    fe_out, fe_mu, fe_logvar, label_emb = \
        output['label_out'], output['fe_mu'], output['fe_logvar'], output['label_emb']
    
    fx_out, fx_mu, fx_logvar, feat_emb = \
        output['feat_out'], output['fx_mu'], output['fx_logvar'], output['feat_emb']
    
    fx_out2 = output['feat_out2']
    embs = output['embs']
    fx_sample = reparameterize(fx_mu, fx_logvar)
    fx_var = torch.exp(fx_logvar)
    fe_var = torch.exp(fe_logvar)
    
