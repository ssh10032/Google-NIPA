import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=5.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, positive_dist, negative_dist):
        # Positive sample 쌍에 대해 거리를 줄이고, Negative 쌍에 대해 거리를 늘리는 손실 계산
        positive_loss = torch.mean(torch.pow(positive_dist, 2))
        negative_loss = torch.mean(torch.pow(torch.clamp(self.margin - negative_dist, min=0.0), 2))
        loss = positive_loss + negative_loss
        return loss

class TripletContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(TripletContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negative):
        # Normalize features
        anchor, positive, negative = F.normalize(anchor, dim=1), F.normalize(positive, dim=1), F.normalize(negative,
                                                                                                           dim=1)

        # Calculate similarities
        pos_sim = torch.exp(torch.sum(anchor * positive, dim=1) / self.temperature)  # Anchor-Positive similarity
        neg_sim = torch.exp(torch.sum(anchor * negative, dim=1) / self.temperature)  # Anchor-Negative similarity

        # Loss calculation: maximize anchor-positive similarity, minimize anchor-negative similarity
        loss = -torch.log(pos_sim / (pos_sim + neg_sim)).mean()
        # print(f'loss check: {pos_sim / (pos_sim + neg_sim)}')
        return loss