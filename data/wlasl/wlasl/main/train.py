# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
# from pytorch_lightning import LightningModule, Trainer
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning.loggers import TensorBoardLogger
# from model import ASLInterpreter
# from dataset import SignLanguageDataModule
# from keypoints import process_videos_in_directory


# class SignLanguageModel(LightningModule):
#     def __init__(
#         self, num_classes: int, learning_rate: float = 1e-3, max_length: int = 500
#     ):
#         super().__init__()
#         self.max_length = max_length
#         self.model = ASLInterpreter(
#             embed_dim=512,
#             hidden_dim=1024,
#             num_heads=8,
#             num_layers=6,
#             num_classes=num_classes,
#             dropout=0.1,
#             max_length=max_length,
#         )
#         self.learning_rate = learning_rate
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx: int):
#         left_positions, right_positions, gloss_indices = batch
#         outputs = self(left_positions, right_positions)
#         loss = self.criterion(outputs, gloss_indices)
#         self.log('train_loss', loss)
#         return loss

#     def prepare_input(self, positions):
#         left_positions = positions['leftpositions']
#         right_positions = positions['rightpositions']

#         # Concatenate left and right positions
#         concatenated_positions = left_positions + right_positions

#         # Pad or truncate the positions to match the maximum sequence length
#         padded_positions = torch.zeros(self.max_length, 4)
#         sequence_length = min(len(concatenated_positions), self.max_length)
#         padded_positions[:sequence_length] = torch.tensor(concatenated_positions[:sequence_length])

#         # Flatten the positions
#         flattened_positions = padded_positions.view(-1)

#         return flattened_positions

#     def training_step(self, batch, batch_idx: int):
#         words, positions = batch
#         x = torch.stack([self.prepare_input(pos) for pos in positions])
#         labels = torch.tensor([self.trainer.datamodule.label_encoder.transform([word])[0] for word in words])

#         outputs = self(x)
#         loss = self.criterion(outputs, labels)
#         self.log('train_loss', loss)
#         return loss

#     def validation_step(self, batch: int):
#         words, positions = batch
#         x = torch.stack([self.prepare_input(pos) for pos in positions])
#         labels = torch.tensor([self.trainer.datamodule.label_encoder.transform([word])[0] for word in words])

#         outputs = self(x)
#         loss = self.criterion(outputs, labels)
#         _, predicted = torch.max(outputs, dim=1)
#         accuracy = (predicted == labels).float().mean().item()
#         return {
#             'val_loss': loss,
#             'val_accuracy': accuracy,
#             'labels': labels.cpu().detach(),
#             'predictions': predicted.cpu().detach(),
#         }


#     def validation_epoch_end(self, outputs):
#         avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
#         avg_accuracy = torch.stack([x['val_accuracy'] for x in outputs]).mean()
#         labels = torch.cat([x['labels'] for x in outputs])
#         predictions = torch.cat([x['predictions'] for x in outputs])
#         precision = precision_score(
#             labels.numpy(), predictions.numpy(), average='macro', zero_division=0
#         )
#         recall = recall_score(
#             labels.numpy(), predictions.numpy(), average='macro', zero_division=0
#         )
#         f1 = f1_score(
#             labels.numpy(), predictions.numpy(), average='macro', zero_division=0
#         )
#         conf_matrix = confusion_matrix(labels.numpy(), predictions.numpy())
#         self.log_dict(
#             {
#                 'val_loss': avg_loss,
#                 'val_accuracy': avg_accuracy,
#                 'val_precision': precision,
#                 'val_recall': recall,
#                 'val_f1': f1,
#             }
#         )
#         self.logger.experiment.add_figure(
#             'confusion_matrix', plot_confusion_matrix(conf_matrix), self.current_epoch
#         )

#     def configure_optimizers(self):
#         return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


# def plot_confusion_matrix(conf_matrix):
#     fig = plt.figure(figsize=(10, 10))
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title('Confusion Matrix')
#     return fig


# def main():
#     csv_file = './start_kit/gloss_video_ids.csv'
#     video_dir = './data'
#     output_dir = './processed_videos'

#     # Process videos and save the extracted data
#     num_videos_to_process = 200
#     process_videos_in_directory(video_dir, output_dir, num_videos=num_videos_to_process)

#     # Set the device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Create the DataModule
#     data_module = SignLanguageDataModule(output_dir, csv_file)
#     data_module.setup()

#     num_glosses = len(data_module.gloss_to_index)
#     expected_loss = torch.log(torch.tensor(num_glosses)).item()

#     # Print information in a formatted box
#     print(f"""
#     +-----------------------------------+
#     | Device: {str(device):<25} |
#     | Number of glosses: {num_glosses:<14} |
#     | Expected loss: {expected_loss:.4f}             |
#     +-----------------------------------+
#     """)

#     model = SignLanguageModel(num_glosses).to(device)

#     # Define callbacks
#     checkpoint_callback = ModelCheckpoint(
#         dirpath='checkpoints', save_top_k=2, monitor='val_loss'
#     )
#     early_stop_callback = EarlyStopping(monitor='val_loss', patience=5)

#     # Define logger
#     logger = TensorBoardLogger('logs', name='sign_language_model')

#     # Create the Trainer
#     trainer = Trainer(
#         max_epochs=100,
#         accelerator='auto',
#         devices='auto',
#         callbacks=[checkpoint_callback, early_stop_callback],
#         logger=logger,
#     )

#     # Fine-tune the model
#     trainer.fit(model, datamodule=data_module)

#     # Save the fine-tuned model
#     trainer.save_checkpoint('fine_tuned_model.ckpt')


# if __name__ == '__main__':
#     main()

#######################


import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.preprocessing import LabelEncoder
import pytorch_lightning as pl
import json
from torch.utils.data import DataLoader, TensorDataset


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class KeypointTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_encoder_layers,
        num_classes,
        dropout=0.5,
        max_length=500,
    ):
        super(KeypointTransformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim * max_length, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_length)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_model * 2, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_encoder_layers
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=0)  # Apply mean reduction to match target shape
        output = self.classifier(output)
        return output


def prepare_data(data_list, max_length):
    X, y = [], []
    label_encoder = LabelEncoder()
    all_words = [data['word'] for data in data_list]
    labels = label_encoder.fit_transform(all_words)
    label_dict = dict(zip(all_words, labels, strict=True))

    for data in data_list:
        word = data['word']
        positions = data['positions']
        lefts = [point for sublist in positions['leftpositions'] for point in sublist]
        rights = [point for sublist in positions['rightpositions'] for point in sublist]
        features = lefts + rights
        if len(features) < max_length:
            features += [(0, 0)] * (max_length - len(features))
        else:
            features = features[:max_length]  # Truncate if longer than max_length
        features_flat = [item for sublist in features for item in sublist]
        X.append(features_flat)
        y.append(label_dict[word])

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
        label_encoder,
    )


class SignLanguageDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, test_data, batch_size, max_length):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.max_length = max_length

    def setup(self, stage=None):
        self.train_X, self.train_y, _ = prepare_data(self.train_data, self.max_length)
        self.val_X, self.val_y, _ = prepare_data(self.val_data, self.max_length)
        self.test_X, self.test_y, _ = prepare_data(self.test_data, self.max_length)

    def train_dataloader(self):
        train_dataset = torch.utils.data.TensorDataset(self.train_X, self.train_y)
        return torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        val_dataset = torch.utils.data.TensorDataset(self.val_X, self.val_y)
        return torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        test_dataset = torch.utils.data.TensorDataset(self.test_X, self.test_y)
        return torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size)


class SignLanguageClassifier(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_encoder_layers,
        num_classes,
        dropout,
        max_length,
    ):
        super().__init__()
        self.model = KeypointTransformer(
            input_dim,
            d_model,
            nhead,
            num_encoder_layers,
            num_classes,
            dropout,
            max_length,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        self.log(
            'train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=5e-4)
        return optimizer

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Log gradient norm
        grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm**0.5
        self.log('grad_norm', grad_norm, on_step=True, prog_bar=True, logger=True)


# Load data
train_file = '/Users/nguyen/Desktop/princeton/cos/cos429/final/data/wlasl/wlasl_train_centered.json'
val_file = '/Users/nguyen/Desktop/princeton/cos/cos429/final/data/wlasl/wlasl_val_centered.json'
test_file = '/Users/nguyen/Desktop/princeton/cos/cos429/final/data/wlasl/wlasl_test_centered.json'

train_data = []
val_data = []
test_data = []

with open(train_file, 'r') as f:
    train_data = json.load(f)
with open(val_file, 'r') as f:
    val_data = json.load(f)
with open(test_file, 'r') as f:
    test_data = json.load(f)

# Set hyperparameters
input_dim = 2
max_length = 84
d_model = 128
nhead = 4
num_encoder_layers = 4
dropout = 0.50
num_classes = 2000
batch_size = 48

# Initialize DataModule and model
data_module = SignLanguageDataModule(
    train_data, val_data, test_data, batch_size, max_length
)
model = SignLanguageClassifier(
    input_dim, d_model, nhead, num_encoder_layers, num_classes, dropout, max_length
)

# Set the device
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')
model.to(device)

# Initialize Trainer
trainer = pl.Trainer(
    max_epochs=25, accelerator='auto', devices=1, val_check_interval=100
)

# Train the model
trainer.fit(model, data_module)

# Test the model
trainer.test(model, data_module)

# Load the best model weights
best_model_path = trainer.checkpoint_callback.best_model_path
model.load_state_dict(torch.load(best_model_path)['state_dict'])

# Prepare test data for inference
test_X, test_y, label_encoder = prepare_data(test_data, max_length)
test_dataset = TensorDataset(test_X, test_y)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Run inference on test data
model.eval()
num_examples = 10

with torch.no_grad():
    for batch in test_dataloader:
        input_batch, label_batch = batch
        input_batch = input_batch.to(device)
        label_batch = label_batch.to(device)

        output = model(input_batch)
        probabilities = torch.softmax(output, dim=1)
        predicted_labels = torch.argmax(probabilities, dim=1)

        for i in range(min(num_examples, input_batch.size(0))):
            actual_word = label_encoder.inverse_transform([label_batch[i].item()])[0]
            predicted_word = label_encoder.inverse_transform(
                [predicted_labels[i].item()]
            )[0]

            print(f'Example {i+1}:')
            print(f'Actual Word: {actual_word}')
            print(f'Predicted Word: {predicted_word}')
            print('Probabilities:')
            for label, prob in zip(
                label_encoder.classes_, probabilities[i], strict=True
            ):
                print(f'{label}: {prob.item():.4f}')
            print()

        num_examples -= input_batch.size(0)
        if num_examples <= 0:
            break
