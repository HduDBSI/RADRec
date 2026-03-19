import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from modules import *
from utils import extract



class Diffusion():
    def __init__(self, args):
        self.timesteps = args.timesteps
        self.beta_start = args.beta_start
        self.beta_end = args.beta_end
        self.args = args

        if self.args.beta_sche == 'linear':
            self.betas = self.linear_beta_schedule(self.timesteps, self.beta_start, self.beta_end)
        elif self.args.beta_sche == 'exp':
            self.betas = self.exp_beta_schedule(self.timesteps)
        elif self.args.beta_sche =='cosine':
            self.betas = self.cosine_beta_schedule(self.timesteps)
        elif self.args.beta_sche =='sqrt':
            self.betas = torch.tensor(self.betas_for_alpha_bar(self.timesteps, lambda t: 1-np.sqrt(t + 0.0001),)).float()

        # define alphas（timesteps, ）
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)


        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def linear_beta_schedule(self, timesteps, beta_start, beta_end):
        beta_start = beta_start
        beta_end = beta_end
        return torch.linspace(beta_start, beta_end, timesteps)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def exp_beta_schedule(self, timesteps, beta_min=0.1, beta_max=10):
        x = torch.linspace(1, 2 * timesteps + 1, timesteps)
        betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
        return betas

    def betas_for_alpha_bar(self, num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)


class RADRec(nn.Module):

    def __init__(self, device, args):
        super(RADRec, self).__init__()
        self.hidden_size = args.hidden_size
        self.diffuser_type = args.diffuser_type
        self.device = device
        self.w = args.w

        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)

        self.seq_model = SASRecModel(self.item_embeddings, args=args)

        self.diff = Diffusion(args)


        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.none_embedding.weight, 0, 1)


        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )

        if self.diffuser_type =='mlp1':
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size*3, self.hidden_size)
        )
        elif self.diffuser_type =='mlp2':
            self.diffuser = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size)
        )
            
    def forward(self, x, s, step):
        #  x:(batch, dim) , h:(batch, dim), t:(batch, )
        t = self.step_mlp(step) # t:(batch, dim)


        if self.diffuser_type == 'mlp1':
            res = self.diffuser(torch.cat((x, s, t), dim=1)) # (batch, 3 x dim) -->diffuser-->(batch, dim)
        elif self.diffuser_type == 'mlp2':
            res = self.diffuser(torch.cat((x, s, t), dim=1))
        return res # (batch, dim)
    
    def forward_uncon(self, x, step):
        h = self.none_embedding(torch.tensor([0], device=self.device)).expand(x.shape[0], -1)

        t = self.step_mlp(step)

        if self.diffuser_type == 'mlp1':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        elif self.diffuser_type == 'mlp2':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
            
        return res
    
    def calculate_x0(self, x0):
        return self.item_embeddings(x0)

    def calculate_s(self, condition_rep, p=0.0, force_empty_mask=None):
        s = condition_rep
        none_embedding = self.none_embedding(torch.tensor([0], device=self.device))
        if p > 0:
            B, D = s.shape
            keep_mask_1d = (torch.rand(B, device=self.device) > p).float()
            keep_mask = keep_mask_1d.view(B, 1).expand(B, D)
            s = s * keep_mask + none_embedding * (1 - keep_mask)
        if force_empty_mask is not None:
            B = s.shape[0]
            force_empty_mask = force_empty_mask.to(self.device).float().view(B, 1)
            s = s * (1 - force_empty_mask) + none_embedding * force_empty_mask

        return s

    def forward_process(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.diff.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.diff.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    @torch.no_grad()
    def sample_from_reverse_process(self, s):
        x = torch.randn_like(s)
        for n in reversed(range(0, self.diff.timesteps)):
            t = torch.full((s.shape[0], ), n, device=self.device, dtype=torch.long)
            # x:(batch, dim), s:(batch, dim), t:(batch, ), n:1~timesteps
            x = self.p_sample_with_guidance(self.forward, self.forward_uncon, x, s, t, n)
        
        return x
    
    @torch.no_grad()
    def p_sample_with_guidance(self, model_forward, model_forward_uncon, x, s, t, t_index):

        # guidance, (batch, dim)
        x_start = (1 + self.w) * model_forward(x, s, t) - self.w * model_forward_uncon(x, t)
        x_t = x

        # model_mean = (batch,1) x (batch, dim) + (batch,1) x (batch, dim) = (batch, dim)
        model_mean = (
            extract(self.diff.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.diff.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.diff.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)

            return model_mean + torch.sqrt(posterior_variance_t) * noise 
class SASRecModel(nn.Module):
    def __init__(self, item_embeddings, args):
        super(SASRecModel, self).__init__()
        self.item_embeddings = item_embeddings
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    def item_embedding(self, sequence):
        return self.item_embeddings(sequence)
    
    # Positional Embedding
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)

        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # model same as SASRec
    def forward(self, input_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda == 0:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)

        sequence_output = item_encoded_layers[-1]
        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
