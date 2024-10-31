import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict

class ContrastiveDataset_bc(Dataset):
    ...
    def __getitem__(self, idx):
        segment, dance_name, timestamp, _ = self.data[idx]
        anchor = torch.tensor(segment, dtype=torch.float32)

        # Check for NaN or Inf values in anchor
        if torch.isnan(anchor).any() or torch.isinf(anchor).any():
            raise ValueError(f"NaN or Inf value detected in anchor segment at index {idx}")

        # Positive pair (same dance, same timestamp)
        pos_indices = self.timestamp_to_indices.get((dance_name, timestamp), [])
        if len(pos_indices) > 1:
            pos_idx = np.random.choice([i for i in pos_indices if i != idx])
        else:
            pos_idx = idx  # fallback to self if no other positive sample available
        pos_segment, _, _, _ = self.data[pos_idx]
        pos = torch.tensor(pos_segment, dtype=torch.float32)

        # Check for NaN or Inf values in positive segment
        if torch.isnan(pos).any() or torch.isinf(pos).any():
            raise ValueError(f"NaN or Inf value detected in positive segment at index {pos_idx}")

        # Negative pair (different dance)
        all_labels = list(set(self.labels))
        neg_dance = np.random.choice([l for l in all_labels if l != dance_name])
        neg_indices = [i for i in range(len(self.data)) if self.labels[i] == neg_dance]
        if neg_indices:
            neg_idx = np.random.choice(neg_indices)
        else:
            neg_idx = idx  # fallback to self if no negative sample available
        neg_segment, _, _, _ = self.data[neg_idx]
        neg = torch.tensor(neg_segment, dtype=torch.float32)

        # Check for NaN or Inf values in negative segment
        if torch.isnan(neg).any() or torch.isinf(neg).any():
            raise ValueError(f"NaN or Inf value detected in negative segment at index {neg_idx}")

        # Label for classification
        label = self.labels[idx]

        return anchor, pos, neg, label
