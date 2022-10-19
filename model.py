'''
Taken from https://github.com/bshall/VectorQuantizedVAE and used on pairs of MNIST images
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, RelaxedOneHotCategorical
import math


class VQEmbeddingEMA(nn.Module):
    def __init__(self, latent_dim, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        embedding = torch.zeros(latent_dim, num_embeddings, embedding_dim)
        embedding.uniform_(-1/num_embeddings, 1/num_embeddings)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(latent_dim, num_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def forward(self, x):
        B, C, H, W = x.size()
        N, M, D = self.embedding.size()
        assert C == N * D

        x = x.view(B, N, D, H, W).permute(1, 0, 3, 4, 2)
        x_flat = x.detach().reshape(N, -1, D)

        distances = torch.baddbmm(torch.sum(self.embedding ** 2, dim=2).unsqueeze(1) +
                                  torch.sum(x_flat ** 2, dim=2, keepdim=True),
                                  x_flat, self.embedding.transpose(1, 2),
                                  alpha=-2.0, beta=1.0)
        # print('x shape',x.shape)
        # print('dist shae',distances.shape)
        indices = torch.argmin(distances, dim=-1)
        encodings = F.one_hot(indices, M).float()
        # print('indices shape',indices.shape)
        # print('ecodings shape',encodings.shape)
        quantized = torch.gather(self.embedding, 1, indices.unsqueeze(-1).expand(-1, -1, D))
        quantized = quantized.view_as(x)
        # print('quantized size',quantized.shape, self.embedding.shape)

        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=1)

            n = torch.sum(self.ema_count, dim=-1, keepdim=True)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.bmm(encodings.transpose(1, 2), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=1)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1))

        return quantized.permute(1, 0, 4, 2, 3).reshape(B, C, H, W), loss, perplexity.sum()

    def getDiscLatent(self, x):
        B, C, H, W = x.size()
        N, M, D = self.embedding.size()
        assert C == N * D

        x = x.view(B, N, D, H, W).permute(1, 0, 3, 4, 2)
        x_flat = x.detach().reshape(N, -1, D)

        distances = torch.baddbmm(torch.sum(self.embedding ** 2, dim=2).unsqueeze(1) +
                                  torch.sum(x_flat ** 2, dim=2, keepdim=True),
                                  x_flat, self.embedding.transpose(1, 2),
                                  alpha=-2.0, beta=1.0)
        indices = torch.argmin(distances, dim=-1)

        return indices.permute(1, 0)


class VQEmbeddingGSSoft(nn.Module):
    def __init__(self, latent_dim, num_embeddings, embedding_dim):
        super(VQEmbeddingGSSoft, self).__init__()

        self.embedding = nn.Parameter(torch.Tensor(latent_dim, num_embeddings, embedding_dim))
        nn.init.uniform_(self.embedding, -1/num_embeddings, 1/num_embeddings)

    def forward(self, x):
        B, C, H, W = x.size()
        N, M, D = self.embedding.size()
        assert C == N * D

        x = x.view(B, N, D, H, W).permute(1, 0, 3, 4, 2)
        x_flat = x.reshape(N, -1, D)

        distances = torch.baddbmm(torch.sum(self.embedding ** 2, dim=2).unsqueeze(1) +
                                  torch.sum(x_flat ** 2, dim=2, keepdim=True),
                                  x_flat, self.embedding.transpose(1, 2),
                                  alpha=-2.0, beta=1.0)
        distances = distances.view(N, B, H, W, M)

        dist = RelaxedOneHotCategorical(0.5, logits=-distances)
        # print('distances shape',distances.shape)
        if self.training:
            samples = dist.rsample().view(N, -1, M)
        else:
            samples = torch.argmax(dist.probs, dim=-1)
            samples = F.one_hot(samples, M).float()
            samples = samples.view(N, -1, M)

        # print('sample size',samples.shape)

        quantized = torch.bmm(samples, self.embedding)
        quantized = quantized.view_as(x)

        KL = dist.probs * (dist.logits + math.log(M))
        KL[(dist.probs == 0).expand_as(KL)] = 0
        KL = KL.sum(dim=(0, 2, 3, 4)).mean()

        avg_probs = torch.mean(samples, dim=1)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1))

        return quantized.permute(1, 0, 4, 2, 3).reshape(B, C, H, W), KL, perplexity.sum()

    def getDiscLatent(self, x):
        B, C, H, W = x.size()
        N, M, D = self.embedding.size()
        assert C == N * D

        x = x.view(B, N, D, H, W).permute(1, 0, 3, 4, 2)
        x_flat = x.detach().reshape(N, -1, D)

        distances = torch.baddbmm(torch.sum(self.embedding ** 2, dim=2).unsqueeze(1) +
                                  torch.sum(x_flat ** 2, dim=2, keepdim=True),
                                  x_flat, self.embedding.transpose(1, 2),
                                  alpha=-2.0, beta=1.0)
        indices = torch.argmin(distances, dim=-1)

        return indices.permute(1, 0)




class Residual(nn.Module):
    def __init__(self, channels):
        super(Residual, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


# class Encoder(nn.Module):
#     def __init__(self, channels, latent_dim, embedding_dim):
#         super(Encoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(2, channels, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(True),
#             nn.Conv2d(channels, channels, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(channels),
#             Residual(channels),
#             Residual(channels),
#             nn.Conv2d(channels, latent_dim * embedding_dim, 1)
#         )
#         self.linear = nn.Linear(latent_dim * embedding_dim * 7 * 7, latent_dim * embedding_dim)  # input
#         self.channels = channels
#         self.latent_dim = latent_dim
#         self.embedding_dim = embedding_dim

#     def forward(self, x):
#         batchSize = x.shape[0]
#         # print('encoder output',self.encoder(x).shape, self.latent_dim, self.embedding_dim, self.channels)
#         y = self.encoder(x)
#         print('encoder shape',y.shape)
#         z = self.linear(torch.flatten(y, start_dim=1))
#         return z.view((batchSize, self.latent_dim * self.embedding_dim, 1, 1))


class Decoder(nn.Module):
    def __init__(self, channels, latent_dim, embedding_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim * embedding_dim, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            Residual(channels),
            Residual(channels),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, 2 * 256, 1)
        )
        self.linear = nn.Linear(latent_dim * embedding_dim, latent_dim * embedding_dim * 7 * 7)  # input

        self.channels = channels
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim

    def forward(self, x):
        batchSize = x.shape[0]
        x = self.linear(x.view((batchSize, -1)))
        x = self.decoder(x.view((batchSize, self.latent_dim * self.embedding_dim, 7, 7)))
        # print('dencoder output',x.shape, self.latent_dim, self.embedding_dim, self.channels)
        B, _, H, W = x.size()
        x = x.view(B, 2, 256, H, W).permute(0, 1, 3, 4, 2)
        dist = Categorical(logits=x)
        return dist


class VQVAE(nn.Module):
    def __init__(self, channels, latent_dim, num_embeddings, embedding_dim):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(channels, latent_dim, embedding_dim)
        self.codebook = VQEmbeddingEMA(latent_dim, num_embeddings, embedding_dim)
        self.decoder = Decoder(channels, latent_dim, embedding_dim)

    def forward(self, x):
        x = self.encoder(x)
        x, loss, perplexity = self.codebook(x)
        # print('sample size',x.shape)
        dist = self.decoder(x)
        return dist, loss, perplexity

    def getDiscreteLatent(self, x):
        enc = self.encoder(x)
        return self.codebook.getDiscLatent(enc)


class GSSOFT(nn.Module):
    def __init__(self, channels, latent_dim, num_embeddings, embedding_dim):
        super(GSSOFT, self).__init__()
        self.encoder = Encoder(channels, latent_dim, embedding_dim)
        self.codebook = VQEmbeddingGSSoft(latent_dim, num_embeddings, embedding_dim)
        self.decoder = Decoder(channels, latent_dim, embedding_dim)

    def forward(self, x):
        x = self.encoder(x)
        x, KL, perplexity = self.codebook(x)
        dist = self.decoder(x)
        return dist, KL, perplexity

    def getDiscreteLatent(self, x):
        enc = self.encoder(x)
        return self.codebook.getDiscLatent(enc)




# This is the same network used in RPM recognition network

class Encoder(nn.Module):
    def __init__(self, channels, latent_dim, embedding_dim):
        super(Encoder, self).__init__()
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(2, 10, kernel_size=5),
        #     F.max_pool2d()
        #     nn.Conv2d(10, 20, kernel_size=5),
        #     nn.Dropout2d()
        # )
        self.linear2 = nn.Linear(50, latent_dim * embedding_dim)  # input
        self.channels = channels
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim

        self.conv1 = nn.Conv2d(2, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(4*4*20, 50)
        self.fc2 = nn.Linear(50, 10)


    def forward(self, x):
        batchSize = x.shape[0]
        a = F.relu(F.max_pool2d(self.conv1(x), 2))
        a = F.relu(F.max_pool2d(self.conv2(a), 2))
        a = a.view(-1, 4*4*20)
        y = F.relu(self.fc1(a))
        z2 = self.linear2(y)
        return z2.view((batchSize, self.latent_dim * self.embedding_dim, 1, 1))

class Net(nn.Module):
    # Convolutional Neural Network shared across independent factors
    def __init__(self, num_digits):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(4*4*20, 50)
        self.fc2 = nn.Linear(50, num_digits)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 4*4*20)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)