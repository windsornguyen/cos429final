# TODO: THIS FILE IS NOT USED

import os
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from model import SignLanguageTransformer
from generator import parse_csv_file, VideoDataset, collate_fn
import msgspec.msgpack


class CustomTrainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        num_epochs,
        train_dataloader,
        val_dataloader,
        logging_dir,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.logging_dir = logging_dir
        self.tensorboard_writer = SummaryWriter(log_dir=logging_dir)

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            for batch in self.train_dataloader:
                left_positions = batch['left_positions'].to(self.device)
                right_positions = batch['right_positions'].to(self.device)
                labels = batch['glosses'].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(
                    left_positions=left_positions, right_positions=right_positions
                )
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                self.optimizer.step()

            self.evaluate(epoch)
            self.save_checkpoint(epoch)

    def evaluate(self, epoch):
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        total_precision = 0
        total_recall = 0
        total_f1 = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                left_positions = batch['left_positions'].to(self.device)
                right_positions = batch['right_positions'].to(self.device)
                labels = batch['glosses'].to(self.device)

                outputs = self.model(
                    left_positions=left_positions, right_positions=right_positions
                )
                loss = nn.CrossEntropyLoss()(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, dim=1)
                accuracy = (predicted == labels).float().mean().item()
                total_accuracy += accuracy

                labels_np = labels.cpu().detach().numpy()
                predicted_np = predicted.cpu().detach().numpy()
                precision = precision_score(
                    labels_np, predicted_np, average='macro', zero_division=0
                )
                recall = recall_score(
                    labels_np, predicted_np, average='macro', zero_division=0
                )
                f1 = f1_score(labels_np, predicted_np, average='macro', zero_division=0)
                total_precision += precision
                total_recall += recall
                total_f1 += f1

        avg_loss = total_loss / len(self.val_dataloader)
        avg_accuracy = total_accuracy / len(self.val_dataloader)
        avg_precision = total_precision / len(self.val_dataloader)
        avg_recall = total_recall / len(self.val_dataloader)
        avg_f1 = total_f1 / len(self.val_dataloader)

        print(
            f'Epoch {epoch+1}/{self.num_epochs} - Validation Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}'
        )

        self.tensorboard_writer.add_scalar('val_loss', avg_loss, epoch)
        self.tensorboard_writer.add_scalar('val_accuracy', avg_accuracy, epoch)
        self.tensorboard_writer.add_scalar('val_precision', avg_precision, epoch)
        self.tensorboard_writer.add_scalar('val_recall', avg_recall, epoch)
        self.tensorboard_writer.add_scalar('val_f1', avg_f1, epoch)

    def save_checkpoint(self, epoch):
        checkpoint_dir = os.path.join(self.logging_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

        msgspec_path = os.path.join(
            checkpoint_dir, f'checkpoint_epoch_{epoch+1}.msgpack'
        )
        with open(msgspec_path, 'wb') as f:
            msgspec.msgpack.dump(checkpoint, f)


def main():
    # Set the device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Parse the CSV file
    csv_file = './start_kit/gloss_video_ids.csv'
    parsed_data = parse_csv_file(csv_file)

    # Create the datasets
    video_dir = './processed_videos'
    train_dataset = VideoDataset(video_dir, parsed_data)
    val_dataset = VideoDataset(video_dir, parsed_data)

    # Create data loaders
    batch_size = 4
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Initialize the model
    num_glosses = len(parsed_data)
    model = SignLanguageTransformer(num_glosses).to(device)

    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Define training parameters
    num_epochs = 1000
    logging_dir = './logs'

    # Create the custom trainer
    trainer = CustomTrainer(
        model,
        optimizer,
        device,
        num_epochs,
        train_dataloader,
        val_dataloader,
        logging_dir,
    )

    # Train the model
    trainer.train()


if __name__ == '__main__':
    main()
