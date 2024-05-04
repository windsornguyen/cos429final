import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import pytorch_lightning as pl
import json
from torch.utils.data import DataLoader, TensorDataset
from model import ASLInterpreter


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
        self.model = ASLInterpreter(
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


class SignLanguageClassifier(pl.LightningModule):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_heads,
        num_layers,
        num_classes,
        dropout,
        max_length,
    ):
        super().__init__()
        self.model = ASLInterpreter(
            embed_dim,
            hidden_dim,
            num_heads,
            num_layers,
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
train_file = '../wlasl/wlasl_train.json'
val_file = '../wlasl/wlasl_val.json'
test_file = '../wlasl/wlasl_test.json'

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
embed_dim = 128
hidden_dim = 256
num_heads = 4
num_layers = 4
dropout = 0.50
num_classes = 2000
max_length = 84
batch_size = 48

# Initialize DataModule and model
data_module = SignLanguageDataModule(
    train_data, val_data, test_data, batch_size, max_length
)
model = SignLanguageClassifier(
    embed_dim,
    hidden_dim,
    num_heads,
    num_layers,
    num_classes,
    dropout,
    max_length,
)

# Set the device
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

model.to(device)

# Initialize Trainer
trainer = pl.Trainer(
    max_epochs=32, accelerator='auto', devices=1, val_check_interval=100
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
