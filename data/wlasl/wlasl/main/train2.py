import torch
import pytorch_lightning as pl
import json
from torch.utils.data import DataLoader, TensorDataset
from data.wlasl.main.model_backup import SignLanguageClassifier, SignLanguageDataModule, prepare_data

# Load data
train_file = '../wlasl_train.json'
val_file = '../wlasl_val.json'
test_file = '../wlasl_test.json'

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

# Set the device
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

data_module = SignLanguageDataModule(
    train_data, val_data, test_data, batch_size, max_length
)

model = SignLanguageClassifier(
    input_dim, d_model, nhead, num_encoder_layers, num_classes, dropout, max_length
)
model.to(device)

# Initialize DataModule and Trainer
data_module = SignLanguageDataModule(
    train_data, val_data, test_data, batch_size, max_length
)
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
