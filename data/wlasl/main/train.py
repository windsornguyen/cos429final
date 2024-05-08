import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from dataset import SignLanguageDataset, train_loader, val_loader, test_loader
from model import Transformer, MoeArgs, RotatingBufferCache
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

torch.set_float32_matmul_precision("high")

# Set hyperparameters
max_length = 84
d_model = 16
nhead = 2
num_encoder_layers = 2
dropout = 0.1
batch_size = 48
sliding_window = 84
num_experts = 1
num_experts_per_tok = 1
num_epochs = 1
num_keypoints=21

cache = RotatingBufferCache(
    n_layers=num_encoder_layers,
    max_batch_size=batch_size,
    sliding_window=sliding_window,
    n_kv_heads=nhead // 2,
    head_dim=d_model // nhead,
    num_keypoints=num_keypoints,
)

# Initialize model arguments
model_args = {
    'dim': d_model,
    'num_glosses': train_loader.dataset.num_classes,
    'n_layers': num_encoder_layers,
    'head_dim': d_model // nhead,
    'hidden_dim': d_model * 1,
    'n_heads': nhead,
    'n_kv_heads': nhead // 2,
    'norm_eps': 1e-5,
    'max_batch_size': batch_size,
    'sliding_window': sliding_window,
    'moe': {
        'num_experts': num_experts,
        'num_experts_per_tok': num_experts_per_tok
    },
    'rope_theta': 8_192.0,
    'pos_emb_dim': 2,
    'disp_emb_dim': 2,
}

# Set the device
device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

# Initialize model
model = Transformer(model_args)
model.to(device)

# Initialize optimizer and schedulers
optimizer = AdamW(model.parameters(), lr=5e-4)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints',
    filename='best_model',
    monitor='val_loss',
    mode='min',
    save_top_k=1,
)
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=15,
    mode='min'
)
lr_monitor = LearningRateMonitor(logging_interval='step')

class SignLanguageModel(pl.LightningModule):
    def __init__(self, model, lr_scheduler):
        super().__init__()
        self.model = model
        self.lr_scheduler = lr_scheduler
        # self.cache = cache

    def training_step(self, batch, batch_idx):
        left_pos, right_pos, left_disp, right_disp, glosses = batch
        seqlens = [len(lp) for lp in left_pos]
        outputs = self.model(left_pos, right_pos, left_disp, right_disp, seqlens)

        # Reshape outputs and glosses to have compatible shapes
        batch_size = outputs.size(0)
        num_glosses = outputs.size(-1)
        outputs = outputs.view(batch_size, -1, num_glosses)
        glosses = glosses.view(batch_size, -1)

        loss = torch.nn.functional.cross_entropy(outputs.transpose(1, 2), glosses)
        _, predicted_glosses = torch.max(outputs, dim=-1)
            
        accuracy = accuracy_score(glosses.cpu(), predicted_glosses.view(-1).cpu())
        precision = precision_score(glosses.cpu(), predicted_glosses.view(-1).cpu(), average='weighted')
        f1 = f1_score(glosses.cpu(), predicted_glosses.view(-1).cpu(), average='weighted')
        
        self.log('train_loss', loss)
        self.log('train_accuracy', accuracy)
        self.log('train_precision', precision)
        self.log('train_f1', f1)
        
        return loss

    def validation_step(self, batch, batch_idx):
        left_pos, right_pos, left_disp, right_disp, glosses = batch
        seqlens = [len(lp) for lp in left_pos]
        outputs = self.model(left_pos, right_pos, left_disp, right_disp, seqlens)

        # Reshape outputs and glosses to have compatible shapes
        batch_size = outputs.size(0)
        num_glosses = outputs.size(-1)
        outputs = outputs.view(batch_size, -1, num_glosses)
        glosses = glosses.view(batch_size, -1)

        loss = torch.nn.functional.cross_entropy(outputs.transpose(1, 2), glosses)
        _, predicted_glosses = torch.max(outputs, dim=-1)
        
        accuracy = accuracy_score(glosses.cpu().view(-1), predicted_glosses.cpu().view(-1))
        precision = precision_score(glosses.cpu().view(-1), predicted_glosses.cpu().view(-1), average='weighted')
        f1 = f1_score(glosses.cpu().view(-1), predicted_glosses.cpu().view(-1), average='weighted')
        
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy)
        self.log('val_precision', precision)
        self.log('val_f1', f1)
        
        return {'loss': loss, 'accuracy': accuracy, 'precision': precision, 'f1': f1}

    def test_step(self, batch, batch_idx):
        left_pos, right_pos, left_disp, right_disp, glosses = batch
        seqlens = [len(lp) for lp in left_pos]  
        outputs = self.model(left_pos, right_pos, left_disp, right_disp, seqlens)

        # Reshape outputs and glosses to have compatible shapes
        batch_size = outputs.size(0)
        num_glosses = outputs.size(-1)
        outputs = outputs.view(batch_size, -1, num_glosses)
        glosses = glosses.view(batch_size, -1)

        loss = torch.nn.functional.cross_entropy(outputs.transpose(1, 2), glosses)
        _, predicted_glosses = torch.max(outputs, dim=-1)
        
        accuracy = accuracy_score(glosses.cpu(), predicted_glosses.cpu())
        precision = precision_score(glosses.cpu(), predicted_glosses.cpu(), average='weighted')
        f1 = f1_score(glosses.cpu(), predicted_glosses.cpu(), average='weighted')
        
        self.log('test_loss', loss)
        self.log('test_accuracy', accuracy)
        self.log('test_precision', precision)
        self.log('test_f1', f1)
        
        return {'loss': loss, 'accuracy': accuracy, 'precision': precision, 'f1': f1, 'true_labels': glosses.cpu(), 'predicted_labels': predicted_glosses.cpu()}

    def test_epoch_end(self, outputs):
        true_labels = torch.cat([x['true_labels'] for x in outputs]).view(-1)
        predicted_labels = torch.cat([x['predicted_labels'] for x in outputs]).view(-1)
        cm = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig('confusion_matrix.png')

    def train_dataloader(self):
        return train_loader

    def val_dataloader(self):
        return val_loader

    def test_dataloader(self):
        return test_loader

    def configure_optimizers(self):
        return [optimizer], [self.lr_scheduler]

# model = SignLanguageModel(model, lr_scheduler, cache)
model = SignLanguageModel(model, lr_scheduler)

# Initialize Trainer with distributed GPU training
trainer = pl.Trainer(
    max_epochs=num_epochs,
    accelerator='gpu',
    devices=torch.cuda.device_count(),
    strategy='ddp',
    default_root_dir='checkpoints',
    callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
    log_every_n_steps=10,
    gradient_clip_val=1.0,
)

# Run training using PyTorch Lightning's Trainer
trainer.fit(model)

# Load the best checkpoint
best_model_path = checkpoint_callback.best_model_path
model = Transformer.load_from_checkpoint(best_model_path)

# Evaluate the model on the test set
trainer.test(model)
