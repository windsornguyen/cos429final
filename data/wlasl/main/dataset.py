import torch
from torch.utils.data import Dataset, DataLoader
import json
from typing import List, Tuple, Dict

class SignLanguageDataset(Dataset):
    def __init__(self, data: List[Dict], max_length: int):
        self.data = data
        self.max_length = max_length
        self.word_to_idx = self.build_word_to_idx()

    def __len__(self) -> int:
        return len(self.data)

    def pad_sequence(self, sequence: List[List[float]], num_keypoints: int, num_features: int) -> torch.Tensor:
        padded_sequence = sequence[:self.max_length]
        padding = torch.zeros((self.max_length - len(padded_sequence), num_keypoints, num_features), dtype=torch.float32)
        padded_sequence = torch.tensor(padded_sequence, dtype=torch.float32)
        padded_sequence = torch.cat((padded_sequence, padding), dim=0)
        return padded_sequence

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.data[index]
        left_pos = item["positions"]["leftpositions"]
        right_pos = item["positions"]["rightpositions"]
        left_disp = item["positions"]["leftdisplacements"]
        right_disp = item["positions"]["rightdisplacements"]
        word = item["word"]

        num_keypoints = len(left_pos[0]) if left_pos else 0
        num_features = len(left_pos[0][0]) if left_pos and left_pos[0] else 0

        left_pos = self.pad_sequence(left_pos, num_keypoints, num_features)
        right_pos = self.pad_sequence(right_pos, num_keypoints, num_features)
        left_disp = self.pad_sequence(left_disp, num_keypoints, num_features)
        right_disp = self.pad_sequence(right_disp, num_keypoints, num_features)

        word_idx = torch.tensor(self.word_to_idx[word], dtype=torch.long)

        return left_pos, right_pos, left_disp, right_disp, word_idx

    def build_word_to_idx(self) -> Dict[str, int]:
        words = sorted(set(item["word"] for item in self.data))
        return {word: idx for idx, word in enumerate(words)}

    @property
    def num_classes(self) -> int:
        return len(self.word_to_idx)

# Load the JSON datasets using json
def load_json_file(filename: str) -> List[Dict]:
    with open(filename, "r") as f:
        return json.load(f)

train_data = load_json_file("/scratch/gpfs/mn4560/final/data/wlasl/main/wlasl_landmark_data_disp_train.json")
val_data = load_json_file("/scratch/gpfs/mn4560/final/data/wlasl/main/wlasl_landmark_data_disp_val.json")
test_data = load_json_file("/scratch/gpfs/mn4560/final/data/wlasl/main/wlasl_landmark_data_disp_test.json")

# Initialize datasets with the decoded data
max_length = 84
train_dataset = SignLanguageDataset(train_data, max_length)
val_dataset = SignLanguageDataset(val_data, max_length)
test_dataset = SignLanguageDataset(test_data, max_length)

# Optimized DataLoaders with parallel processing
batch_size = 48
num_workers = 4
prefetch_factor = 2

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    prefetch_factor=prefetch_factor,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    prefetch_factor=prefetch_factor,
    persistent_workers=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    prefetch_factor=prefetch_factor,
    persistent_workers=True,
)