# Copyright (c) Roberto Del Prete. All rights reserved.

"""PyTorch Lightning Training Script

This script demonstrates a complete training pipeline using PyTorch Lightning.
It includes data loading, model definition, training, validation, and testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


# ============================================================================
# 1. DATASET DEFINITION
# ============================================================================

class CustomDataset(Dataset):
    """Custom dataset example.
    
    Args:
        data_path: Path to data files
        transform: Optional transform to apply to samples
    """
    
    def __init__(self, data_path, transform=None):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        # Load your data here
        self.data = self._load_data()
    
    def _load_data(self):
        """Load and preprocess data."""
        # Implement data loading logic
        pass
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            tuple: (input_data, target)
        """
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


# ============================================================================
# 2. LIGHTNINGMODULE DEFINITION
# ============================================================================

class MyModel(pl.LightningModule):
    """PyTorch Lightning Module for training.
    
    This class encapsulates the model architecture, training logic,
    validation logic, and optimizer configuration.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (number of classes)
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization parameter
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        super().__init__()
        
        # Save hyperparameters to self.hparams
        # This allows them to be logged and restored from checkpoints
        self.save_hyperparameters()
        
        # Define model architecture
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Define loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        """Forward pass - defines model inference behavior.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output predictions
        """
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Defines a single training step.
        
        This method is called for each batch during training.
        Lightning handles the backward pass and optimizer step automatically.
        
        Args:
            batch: Tuple of (inputs, targets) from DataLoader
            batch_idx: Index of current batch
            
        Returns:
            Loss value (Lightning automatically calls .backward() on this)
        """
        # Unpack batch
        x, y = batch
        
        # Forward pass
        logits = self(x)
        
        # Compute loss
        loss = self.criterion(logits, y)
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics - these are automatically averaged over the epoch
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Defines a single validation step.
        
        Called for each batch during validation.
        No gradients are computed (Lightning handles this).
        
        Args:
            batch: Tuple of (inputs, targets) from DataLoader
            batch_idx: Index of current batch
        """
        x, y = batch
        
        # Forward pass
        logits = self(x)
        
        # Compute loss
        loss = self.criterion(logits, y)
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log validation metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Defines a single test step.
        
        Called for each batch during testing.
        Similar to validation_step but for final evaluation.
        
        Args:
            batch: Tuple of (inputs, targets) from DataLoader
            batch_idx: Index of current batch
        """
        x, y = batch
        
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log test metrics
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Defines prediction behavior.
        
        Used for inference on new data.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            dataloader_idx: DataLoader index (if using multiple)
            
        Returns:
            Model predictions
        """
        x, _ = batch
        return self(x)
    
    def configure_optimizers(self):
        """Configure optimizer(s) and learning rate scheduler(s).
        
        Returns:
            Dictionary containing optimizer and scheduler configuration
        """
        # Define optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Define learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Return optimizer and scheduler configuration
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  # Metric to monitor
                'interval': 'epoch',     # When to update lr
                'frequency': 1           # How often to check
            }
        }


# ============================================================================
# 3. DATA MODULE (Optional but Recommended)
# ============================================================================

class MyDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for organizing data loading.
    
    DataModules encapsulate all data loading logic in one place,
    making it reusable and easier to maintain.
    
    Args:
        data_dir: Directory containing data
        batch_size: Batch size for DataLoaders
        num_workers: Number of workers for DataLoaders
    """
    
    def __init__(
        self,
        data_dir: str = './',
        batch_size: int = 32,
        num_workers: int = 4
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def prepare_data(self):
        """Download and prepare data.
        
        Called only once and on a single GPU.
        Use this for downloading datasets, tokenizing, etc.
        Do not set state here (e.g., self.x = y).
        """
        # Download data if needed
        # This is called only once
        pass
    
    def setup(self, stage=None):
        """Set up datasets for each stage.
        
        Called on every GPU in distributed training.
        Set state here (e.g., self.train_dataset = ...).
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict')
        """
        if stage == 'fit' or stage is None:
            # Load full dataset
            full_dataset = CustomDataset(self.data_dir)
            
            # Split into train and validation
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [train_size, val_size]
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = CustomDataset(self.data_dir)
    
    def train_dataloader(self):
        """Return training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """Return validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        """Return test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


# ============================================================================
# 4. CALLBACKS SETUP
# ============================================================================

def setup_callbacks(checkpoint_dir='./checkpoints'):
    """Set up training callbacks.
    
    Callbacks are hooks that can be used to add custom behavior
    during training (e.g., checkpointing, early stopping).
    
    Args:
        checkpoint_dir: Directory to save model checkpoints
        
    Returns:
        List of callback objects
    """
    callbacks = [
        # Model checkpointing - saves best model based on validation loss
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='model-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,              # Save top 3 models
            monitor='val_loss',        # Metric to monitor
            mode='min',                # 'min' for loss, 'max' for accuracy
            save_last=True,            # Also save last checkpoint
            verbose=True
        ),
        
        # Early stopping - stops training when metric stops improving
        EarlyStopping(
            monitor='val_loss',
            patience=10,               # Number of epochs to wait
            mode='min',
            verbose=True
        ),
        
        # Learning rate monitor - logs learning rate
        LearningRateMonitor(
            logging_interval='epoch'
        )
    ]
    
    return callbacks


# ============================================================================
# 5. MAIN TRAINING FUNCTION
# ============================================================================

def train_model(
    model=None,
    data_module=None,
    max_epochs=100,
    accelerator='gpu',
    devices=1,
    log_dir='./logs',
    checkpoint_dir='./checkpoints'
):
    """Main training function.
    
    This function sets up the trainer and runs training.
    
    Args:
        model: LightningModule instance (created if None)
        data_module: DataModule instance (created if None)
        max_epochs: Maximum number of training epochs
        accelerator: Hardware accelerator ('cpu', 'gpu', 'tpu')
        devices: Number of devices to use
        log_dir: Directory for logging
        checkpoint_dir: Directory for checkpoints
        
    Returns:
        Trained model
    """
    # Create model if not provided
    if model is None:
        model = MyModel(
            input_dim=784,
            hidden_dim=256,
            output_dim=10,
            learning_rate=1e-3,
            weight_decay=1e-5
        )
    
    # Create data module if not provided
    if data_module is None:
        data_module = MyDataModule(
            data_dir='./data',
            batch_size=32,
            num_workers=4
        )
    
    # Set up callbacks
    callbacks = setup_callbacks(checkpoint_dir)
    
    # Set up logger (TensorBoard example)
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name='my_model'
    )
    
    # Alternative: WandB logger
    # logger = WandbLogger(project='my_project', name='my_run')
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=logger,
        
        # Additional useful arguments:
        log_every_n_steps=10,           # Log every N training steps
        val_check_interval=1.0,         # Check validation every epoch
        gradient_clip_val=1.0,          # Clip gradients
        accumulate_grad_batches=1,      # Gradient accumulation
        precision='16-mixed',           # Mixed precision training (faster)
        deterministic=False,            # Set True for reproducibility (slower)
        
        # For debugging:
        # fast_dev_run=True,            # Run 1 batch of train/val/test
        # overfit_batches=10,           # Overfit on N batches for debugging
        # limit_train_batches=100,      # Limit training batches per epoch
        # limit_val_batches=50,         # Limit validation batches
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    # Test the model on test set
    trainer.test(model, data_module)
    
    return model


# ============================================================================
# 6. INFERENCE FUNCTION
# ============================================================================

def run_inference(model_checkpoint_path, data_module):
    """Run inference on new data.
    
    Args:
        model_checkpoint_path: Path to trained model checkpoint
        data_module: DataModule with data to predict on
        
    Returns:
        Predictions
    """
    # Load model from checkpoint
    model = MyModel.load_from_checkpoint(model_checkpoint_path)
    
    # Create trainer for inference
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1
    )
    
    # Run predictions
    predictions = trainer.predict(model, data_module)
    
    return predictions


# ============================================================================
# 7. MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # Set seed for reproducibility
    pl.seed_everything(42)
    
    # Train model
    trained_model = train_model(
        max_epochs=50,
        accelerator='gpu',
        devices=1
    )
    
    # Optional: Run inference
    # predictions = run_inference(
    #     model_checkpoint_path='./checkpoints/best_model.ckpt',
    #     data_module=MyDataModule()
    # )