import os
import csv
import msgspec
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule


def parse_csv_file(file_path):
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            data = {row[0]: row[1:] for row in csv_reader}
        return data
    except FileNotFoundError:
        print(f'File not found: {file_path}')
        return {}
    except Exception as e:
        print(f'Failed to read the file {file_path}: {e}')
        return {}


def load_processed_data(processed_file_path):
    if not os.path.exists(processed_file_path):
        print(f'Processed file not found: {processed_file_path}')
        return None
    with open(processed_file_path, 'rb') as f:
        data = msgspec.load(f)
    return data


class VideoDataset(Dataset):
    def __init__(self, data_dir, csv_data):
        self.data_dir = data_dir
        self.csv_data = csv_data
        self.video_files = [f for f in os.listdir(data_dir) if f.endswith('.msgpack')]
        self.gloss_to_index = {
            gloss: index for index, gloss in enumerate(self.csv_data.keys())
        }

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):
        video_file = self.video_files[index]
        video_id = os.path.splitext(video_file)[0]
        video_path = os.path.join(self.data_dir, video_file)
        video_data = load_processed_data(video_path)

        # Get the gloss label from the CSV data
        gloss = None
        for key, value in self.csv_data.items():
            if video_id in value:
                gloss = key
                break

        # Convert gloss to index
        gloss_index = self.gloss_to_index[gloss]

        # Process the video data
        left_positions = video_data['left_positions']
        right_positions = video_data['right_positions']

        # Ensure padding of empty lists with zero vectors (21 keypoints per frame)
        left_positions = [
            frame if frame else [[0] * 3] * 21 for frame in left_positions
        ]
        right_positions = [
            frame if frame else [[0] * 3] * 21 for frame in right_positions
        ]

        # Convert to tensor
        left_positions = torch.tensor(left_positions, dtype=torch.float32)
        right_positions = torch.tensor(right_positions, dtype=torch.float32)

        return left_positions, right_positions, gloss_index


class SignLanguageDataModule(LightningDataModule):
    def __init__(self, data_dir, csv_file, batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        csv_data = parse_csv_file(self.csv_file)
        full_dataset = VideoDataset(self.data_dir, csv_data)

        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
