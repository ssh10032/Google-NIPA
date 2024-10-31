import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict

angle_pairs = [
    ['left_biceps', 'left_forearm'],
    ['right_biceps', 'right_forearm'],
    ['between_shoulders', 'left_body'],
    ['between_shoulders', 'right_body'],
    ['between_shoulders', 'right_neck'],
    ['between_shoulders', 'left_neck'],
    ['between_pelvis', 'left_thigh'],
    ['between_pelvis', 'right_thigh'],
    ['right_thigh', 'right_calf'],
    ['left_thigh', 'left_calf'],
    ['right_body', 'right_thigh'],
    ['left_body', 'left_thigh']
]

body_parts = {
    'left_biceps': [11, 13],
    'left_forearm': [13, 15],
    'right_biceps': [12, 14],
    'right_forearm': [14, 16],
    'between_shoulders': [11, 12],
    'left_body': [11, 23],
    'right_body': [12, 24],
    'between_pelvis': [23, 24],
    'left_thigh': [23, 25],
    'left_calf': [25, 27],
    'right_thigh': [24, 26],
    'right_calf': [26, 28],
    'left_neck': [9, 11],
    'right_neck': [10, 12]
}


class LandmarkDataset(Dataset):
    def __init__(self, segment_length):
        self.root_dir = '/media/baebro/NIPA_data/Train/landmarks'
        self.segment_length = segment_length
        self.data = []
        self.labels = []
        self.timestamp_to_indices = {}

        for dance_name in os.listdir(self.root_dir):
            dance_path = os.path.join(self.root_dir, dance_name)
            if os.path.isdir(dance_path):
                for csv_file in os.listdir(dance_path):
                    if csv_file.endswith(".csv") and not csv_file.endswith("_F.csv"):
                        file_path = os.path.join(dance_path, csv_file)
                        df = pd.read_csv(file_path)
                        df = df.drop(columns=['filename'], errors='ignore')
                        self.process_file(df, dance_name, file_path)

    def calculate_angles(self, df):
        angle_data = []

        for part1, part2 in angle_pairs:
            vec1 = df.iloc[:, body_parts[part1][0] * 3 + 1:body_parts[part1][0] * 3 + 4].values - \
                   df.iloc[:, body_parts[part1][1] * 3 + 1:body_parts[part1][1] * 3 + 4].values
            vec2 = df.iloc[:, body_parts[part2][0] * 3 + 1:body_parts[part2][0] * 3 + 4].values - \
                   df.iloc[:, body_parts[part2][1] * 3 + 1:body_parts[part2][1] * 3 + 4].values
            dot_product = np.einsum('ij,ij->i', vec1, vec2)
            norms = np.linalg.norm(vec1, axis=1) * np.linalg.norm(vec2, axis=1)
            angles = np.arccos(np.clip(dot_product / norms, -1.0, 1.0))
            angle_data.append(angles)

        return np.stack(angle_data, axis=1)

    def process_file(self, df, dance_name, file_path):
        angle_df = self.calculate_angles(df)

        # Segment angle data and store with dance and timestamp
        for i in range(0, len(angle_df) - self.segment_length + 1, self.segment_length):
            segment = angle_df[i:i + self.segment_length]
            timestamp = i
            if len(segment) == self.segment_length:
                self.data.append((segment, dance_name, timestamp, file_path))
                self.labels.append(dance_name)

                # Positive pair dictionary
                if (dance_name, timestamp) not in self.timestamp_to_indices:
                    self.timestamp_to_indices[(dance_name, timestamp)] = []
                self.timestamp_to_indices[(dance_name, timestamp)].append(len(self.data) - 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        segment, dance_name, timestamp, _ = self.data[idx]
        anchor = torch.tensor(segment, dtype=torch.float32)

        # Positive pair (same dance, same timestamp)
        pos_idx = np.random.choice(self.timestamp_to_indices[(dance_name, timestamp)])
        while pos_idx == idx:  # Avoid sampling self
            pos_idx = np.random.choice(self.timestamp_to_indices[(dance_name, timestamp)])
        pos_segment, _, _, _ = self.data[pos_idx]
        pos = torch.tensor(pos_segment, dtype=torch.float32)

        # Negative pair (different dance)
        all_labels = list(set(self.labels))
        neg_dance = np.random.choice([l for l in all_labels if l != dance_name])
        neg_indices = [i for i in range(len(self.data)) if self.labels[i] == neg_dance]
        neg_idx = np.random.choice(neg_indices)
        neg_segment, _, _, _ = self.data[neg_idx]
        neg = torch.tensor(neg_segment, dtype=torch.float32)

        return anchor, pos, neg


class ContrastiveDataset(Dataset):
    def __init__(self, segment_length):
        self.root_dir = '/media/baebro/NIPA_data/Train/landmarks/'
        self.segment_length = segment_length
        self.data = []
        self.labels = []
        self.timestamp_to_indices = {}

        for dance_name in os.listdir(self.root_dir):
            dance_path = os.path.join(self.root_dir, dance_name)
            if os.path.isdir(dance_path):
                for csv_file in os.listdir(dance_path):
                    if csv_file.endswith("_angles.csv"):
                        file_path = os.path.join(dance_path, csv_file)
                        df = pd.read_csv(file_path)
                        df = self.interpolate_missing_values(df)
                        self.process_file(df, dance_name, file_path)

    def interpolate_missing_values(self, df):
        # Check for NaN or Inf values and interpolate
        if df.isnull().values.any() or np.isinf(df.values).any():
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.interpolate(method='linear', limit_direction='both', axis=0)
            # Fill any remaining NaN values (e.g., at the start or end) with 0
            df = df.fillna(0)
        return df

    def process_file(self, df, dance_name, file_path):
        # Segment angle data and store with dance and timestamp
        for i in range(0, len(df) - self.segment_length + 1, self.segment_length):
            segment = df.iloc[i:i + self.segment_length].values
            timestamp = i
            if len(segment) == self.segment_length:
                self.data.append((segment, dance_name, timestamp, file_path))
                self.labels.append(dance_name)

                # Positive pair dictionary
                if (dance_name, timestamp) not in self.timestamp_to_indices:
                    self.timestamp_to_indices[(dance_name, timestamp)] = []
                self.timestamp_to_indices[(dance_name, timestamp)].append(len(self.data) - 1)

    def __len__(self):
        return len(self.data)

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

        return anchor, pos, neg

class ContrastiveDataset_bc(Dataset):
    def __init__(self, segment_length):
        self.root_dir = '/media/baebro/NIPA_data/Train/landmarks/'
        self.segment_length = segment_length
        self.data = []
        self.labels = []
        self.timestamp_to_indices = {}

        for dance_name in os.listdir(self.root_dir):
            dance_path = os.path.join(self.root_dir, dance_name)
            if os.path.isdir(dance_path):
                for csv_file in os.listdir(dance_path):
                    if csv_file.endswith("_angles.csv"):
                        file_path = os.path.join(dance_path, csv_file)
                        df = pd.read_csv(file_path)
                        df = self.interpolate_missing_values(df)
                        self.process_file(df, dance_name, file_path)

    def interpolate_missing_values(self, df):
        # Check for NaN or Inf values and interpolate
        if df.isnull().values.any() or np.isinf(df.values).any():
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.interpolate(method='linear', limit_direction='both', axis=0)
            # Fill any remaining NaN values (e.g., at the start or end) with 0
            df = df.fillna(0)
        return df

    def process_file(self, df, dance_name, file_path):
        # Segment angle data and store with dance and timestamp
        for i in range(0, len(df) - self.segment_length + 1, self.segment_length):
            segment = df.iloc[i:i + self.segment_length].values
            timestamp = i
            if len(segment) == self.segment_length:
                self.data.append((segment, dance_name, timestamp, file_path))
                self.labels.append(dance_name)

                # Positive pair dictionary
                if (dance_name, timestamp) not in self.timestamp_to_indices:
                    self.timestamp_to_indices[(dance_name, timestamp)] = []
                self.timestamp_to_indices[(dance_name, timestamp)].append(len(self.data) - 1)

    def __len__(self):
        return len(self.data)
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

