from pathlib import Path
from typing import Optional, Tuple
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim

from utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)
logger.info(f"PyTorch version: {torch.__version__}")

# Directory configuration
base_dir: Path = Path(__file__).resolve().parent.parent
base_data_dir: Path = base_dir / "Data" / "p2-heart-disease"
models_dir: Path = base_dir / "Models" / "P2"
images_dir: Path = base_dir / "Output" / "P2"

class CustomManualFNN(nn.Module):
    """
    Custom Feed-Forward Neural Network for Heart Disease Classification.
    
    This class implements a simple feed-forward neural network with:
    - First activation: ReLU (after first linear layer)
    - Second activation: Sigmoid (after second linear layer, for binary classification)
    
    Architecture: Input(13) -> Linear(16) -> ReLU -> Linear(1) -> Sigmoid
    
    Attributes:
        fc1 (nn.Linear): First fully connected layer (13 -> 16 units).
        fc2 (nn.Linear): Second fully connected layer (16 -> 1 unit).
        activation1 (nn.Module): ReLU activation for first hidden layer.
        activation2 (nn.Module): Sigmoid activation for output layer.
    """
    
    def __init__(self) -> None:
        """
        Initialize the CustomManualFNN model.
        
        Sets up:
        - First layer: 13 input features -> 16 hidden units
        - Second layer: 16 hidden units -> 1 output (binary classification)
        - Activation 1: ReLU (for hidden layer)
        - Activation 2: Sigmoid (for output layer, produces probabilities 0-1)
        """
        super(CustomManualFNN, self).__init__()
        try:
            # Define layers with clear comments on dimensions
            self.fc1: nn.Linear = nn.Linear(13, 16)      # Input layer: 13 -> 16 units
            self.fc2: nn.Linear = nn.Linear(16, 1)       # Output layer: 16 -> 1 unit
            
            # Define activation functions
            self.activation1: nn.Module = nn.ReLU()      # ReLU for hidden layer
            self.activation2: nn.Module = nn.Sigmoid()   # Sigmoid for output (binary classification)
            
            logger.info("CustomManualFNN created with architecture: (13-16-1) with ReLU->Sigmoid activations")
        except Exception as e:
            logger.error("Failed to initialize CustomManualFNN: %s", e)
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Flow:
        1. Linear transformation: 13 -> 16
        2. ReLU activation
        3. Linear transformation: 16 -> 1
        4. Sigmoid activation (for binary classification output)

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 13).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1) with values in [0, 1].

        Raises:
            RuntimeError: If forward pass fails due to shape mismatch or computation error.
        """
        try:
            # First layer with ReLU
            x = self.fc1(x)              # (batch_size, 13) -> (batch_size, 16)
            x = self.activation1(x)      # ReLU activation
            
            # Second layer with Sigmoid (for binary classification)
            x = self.fc2(x)              # (batch_size, 16) -> (batch_size, 1)
            x = self.activation2(x)      # Sigmoid activation (output: 0-1 probabilities)
            
            return x
        except RuntimeError as e:
            logger.error("Forward pass failed: %s", e)
            raise

    def get_architecture(self) -> str:
        """
        Generate a string representation of the model architecture.
        
        Returns:
            str: Architecture string showing layer dimensions and activation functions.
                Format: (input_dim-hidden_dim-output_dim,activation1-activation2)
        """
        return f"(13-16-1,ReLU-Sigmoid)"

class HeartDiseaseAnalysis:
    def __init__(self):
        """
            Initialize the HeartDiseaseAnalysis instance.

            Sets up device configuration, initializes data containers, and configures
            the loss function and model placeholders.
        """
        try:
            self.data: Optional[pd.DataFrame] = None
            self.inputs: Optional[np.ndarray] = None
            self.targets: Optional[np.ndarray] = None
            self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model: Optional[nn.Module] = None
            self.fn_loss: nn.Module = nn.BCELoss()  # Binary Cross Entropy Loss
            self.train_loss: list[float] = []
            self.train_acc: list[float] = []
            self.test_loss: float = np.inf
            self.test_acc: float = 0.0
            logger.info("HeartDiseaseAnalysis initialized. Device: %s", self.device)
        except Exception as e:
            logger.error("Failed to initialize HeartDiseaseAnalysis: %s", e)
            raise

    def process(self) -> Tuple[list[float], list[float], float, float]:
        """
        Execute the complete analysis pipeline.

        Orchestrates data loading, exploration, preprocessing, model training, and evaluation.

        Returns:
            Tuple[list[float], list[float], float, float]: Training losses, training accuracies, 
                                                           test loss, test accuracy.

        Raises:
            RuntimeError: If any step in the pipeline fails.
        """
        try:
            logger.info("Starting heart disease classification pipeline...")

            # Load and prepare data
            self.data = self.read_data()
            self.inputs = self.data.iloc[:, :-1].values
            self.targets = (self.data.iloc[:, -1] > 0).astype(int).values

            # Explore data
            logger.info("Exploring data with feature analysis...")
            self.explore_data(self.inputs, self.targets)

            # Preprocess
            self.inputs = self.scale_and_transform(self.inputs)
            self.targets = torch.tensor(self.targets, dtype=torch.float32)

            # Split data
            X_train, y_train, X_test, y_test = self.split_data(self.inputs, self.targets, 0.2)

            # Train model
            train_losses, train_accs = self.train_model(X_train, y_train)
            
            # Evaluate model
            test_loss = self.evaluate_model(X_test, y_test)
            
            logger.info("Pipeline completed successfully.")
            return train_losses, train_accs, test_loss, self.test_acc
            
        except Exception as e:
            logger.exception("Pipeline failed: %s", e)
            raise

    @staticmethod
    def split_data(inputs: torch.Tensor, targets: torch.Tensor, 
                   test_ratio: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split data into training and testing sets with reproducibility.

        Args:
            inputs (torch.Tensor): Input features tensor.
            targets (torch.Tensor): Target variable tensor.
            test_ratio (float): Fraction of data to use for testing (0.0 to 1.0).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
                Training inputs, training targets, test inputs, test targets.

        Raises:
            ValueError: If test_ratio is invalid or data is empty.
            Exception: If split operation fails.
        """
        try:
            if not 0.0 < test_ratio < 1.0:
                raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")
            
            torch.manual_seed(42)
            n_samples = inputs.shape[0]
            
            if n_samples == 0:
                raise ValueError('No samples available to split')
            
            indices = torch.randperm(n_samples)
            split_idx = int(n_samples * (1 - test_ratio))
            
            logger.debug('train_test_split n_samples=%d, split_idx=%d', n_samples, split_idx)
            
            train_indices = indices[:split_idx]
            test_indices = indices[split_idx:]
            
            X_train = inputs[train_indices]
            y_train = targets[train_indices]
            X_test = inputs[test_indices]
            y_test = targets[test_indices]
            
            logger.info('Data split: train=%d samples, test=%d samples', X_train.shape[0], X_test.shape[0])
            return X_train, y_train, X_test, y_test
            
        except ValueError as e:
            logger.error("Validation error in split_data: %s", e)
            raise
        except Exception as e:
            logger.exception('Failed to split data: %s', e)
            raise

    @staticmethod
    def read_data(filename: str = 'heart.csv', drop_missing: bool = True) -> pd.DataFrame:
        """
        Read dataset CSV into a pandas DataFrame.

        Args:
            filename (str): CSV filename relative to the data directory.
            drop_missing (bool): Whether to drop rows with missing values.

        Returns:
            pd.DataFrame: Loaded dataset.

        Raises:
            FileNotFoundError: If the CSV file cannot be found.
            pd.errors.ParserError: If the CSV cannot be parsed.
        """
        filepath = base_data_dir / filename
        try:
            data = pd.read_csv(filepath)
            logger.info("Data loaded successfully. Shape: %s", data.shape)
            logger.info("Column names: %s", data.columns.tolist())

            missing = data.isnull().sum()
            if missing.any():
                logger.warning('Missing data found in columns:\n%s', missing[missing > 0])
                if drop_missing:
                    data = data.dropna()
            else:
                logger.debug('No missing data detected.')
            logger.info('Finalised Input Data. Shape: %s', data.shape)
            return data
        except FileNotFoundError as e:
            logger.error('Data file not found at: %s', filepath)
            raise
        except pd.errors.ParserError as e:
            logger.error('Failed to parse CSV at %s: %s', filepath, e)
            raise
        except Exception as e:
            logger.exception('Unexpected error while reading data from %s: %s', filepath, e)
            raise

    @staticmethod
    def explore_data(inputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Explore and visualize input features showing class distribution.

        Creates scatter plots for all 2D feature combinations:
        - Blue points: No disease (target=0)
        - Red points: Disease (target=1)

        Args:
            inputs (np.ndarray): Input feature matrix of shape (n_samples, n_features).
            targets (np.ndarray): Target variable array (0=No Disease, 1=Disease).

        Raises:
            ValueError: If inputs and targets have incompatible shapes.
            Exception: If visualization creation fails.
        """
        try:
            if inputs.shape[0] != targets.shape[0]:
                raise ValueError(f"Shape mismatch: inputs {inputs.shape[0]} vs targets {targets.shape[0]}")

            logger.info("Exploring input features...")
            num_features = inputs.shape[1]
            num_combinations = (num_features * (num_features - 1)) // 2
            
            if num_combinations == 0:
                logger.warning("Only 1 feature, cannot create 2D scatter plots")
                return
            
            logger.debug('Creating %d feature combination plots', num_combinations)
            
            # Create grid layout
            cols = 3
            rows = (num_combinations + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
            fig.suptitle('Heart Disease Classification - Feature Analysis', 
                        fontsize=14, fontweight='bold')
            
            # Flatten axes
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            
            # Create scatter plots for all feature combinations
            plot_idx = 0
            for i in range(num_features):
                for j in range(i + 1, num_features):
                    ax = axes[plot_idx]
                    
                    # Separate by disease status
                    no_disease = inputs[targets == 0]
                    disease = inputs[targets == 1]
                    
                    # Plot both classes
                    ax.scatter(no_disease[:, i], no_disease[:, j], 
                              label='No Disease', alpha=0.7, s=50, color='blue', edgecolors='navy')
                    ax.scatter(disease[:, i], disease[:, j], 
                              label='Disease', alpha=0.7, s=50, color='red', edgecolors='darkred')
                    
                    ax.set_xlabel(f'Feature {i}', fontweight='bold')
                    ax.set_ylabel(f'Feature {j}', fontweight='bold')
                    ax.legend(loc='best', fontsize=8)
                    ax.grid(True, alpha=0.3)
                    
                    plot_idx += 1
            
            # Hide unused subplots
            for idx in range(plot_idx, len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            
            # Save plot
            images_dir.mkdir(parents=True, exist_ok=True)
            out_path = images_dir / "Heart_Disease_Feature_Analysis.png"
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            logger.info('Saved feature analysis to %s', out_path)
            plt.close(fig)
        
        except ValueError as e:
            logger.error("Validation error: %s", e)
            raise
        except Exception as e:
            logger.exception('Failed to create analysis plots: %s', e)
            raise
        
    @staticmethod
    def scale_and_transform(data: np.ndarray) -> torch.Tensor:
        """
        Standardize data using StandardScaler and convert to PyTorch tensor.

        Args:
            data (np.ndarray): Input data array.

        Returns:
            torch.Tensor: Scaled and transformed tensor of dtype float32.

        Raises:
            ValueError: If data is empty or invalid.
            Exception: If scaling or transformation fails.
        """
        try:
            if data.size == 0:
                raise ValueError("Cannot scale empty array")

            scaler = StandardScaler()
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            scaled_data = scaler.fit_transform(data)
            scaled_tensor = torch.tensor(scaled_data, dtype=torch.float32)

            logger.debug('Data scaled and converted to tensor. Shape: %s', scaled_tensor.shape)
            return scaled_tensor
        except ValueError as e:
            logger.error("Validation error in scale_and_transform: %s", e)
            raise
        except Exception as e:
            logger.exception('Failed to transform data: %s', e)
            raise

    def train_model(self, inputs: torch.Tensor, targets: torch.Tensor,
                    num_epochs: int = 500, learning_rate: float = 0.001) -> Tuple[list[float], list[float]]:
        """
        Train the neural network model.

        Args:
            inputs (torch.Tensor): Training input features.
            targets (torch.Tensor): Training target values.
            num_epochs (int): Number of training epochs (default: 500).
            learning_rate (float): Learning rate for optimizer (default: 0.001).

        Returns:
            Tuple[list[float], list[float]]: Training loss history and accuracy history.

        Raises:
            RuntimeError: If model is not initialized or training fails.
            Exception: If gradient computation or optimization fails.
        """
        if self.model is None:
            raise RuntimeError("Model must be initialized before training")

        try:
            logger.info('Starting model training: %s', self.model.__class__.__name__)

            # Get model parameters for optimization
            if isinstance(self.model, nn.Module):
                param_to_optimize = self.model.parameters()
            else:
                param_to_optimize = self.model.layers.parameters()

            optimizer = optim.Adam(param_to_optimize, lr=learning_rate)

            for epoch in range(num_epochs):
                try:
                    # Forward pass
                    if isinstance(self.model, nn.Module):
                        self.model.train()
                        outputs = self.model(inputs)
                    else:
                        self.model.layers.train()
                        outputs = self.model.forward(inputs)

                    # Apply sigmoid to get probabilities (0-1 range)
                    outputs_probs = torch.sigmoid(outputs.squeeze())
                    
                    # Calculate loss using probabilities
                    # BCELoss expects tensor inputs with values between 0 and 1
                    loss = self.fn_loss(outputs_probs, targets.squeeze())

                    # Convert probabilities to binary predictions (0 or 1) for accuracy
                    outputs_binary = (outputs_probs >= 0.5).float()

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Calculate accuracy using binary predictions
                    self.train_loss.append(loss.item())
                    self.train_acc.append(accuracy_score(targets.squeeze().detach().cpu().numpy(), 
                                                        outputs_binary.detach().cpu().numpy()))

                    if (epoch + 1) % (num_epochs // 10) == 0:
                        logger.info('Epoch [%d/%d], Loss: %.4f, Accuracy: %.2f%%', 
                                   epoch + 1, num_epochs, loss.item(), self.train_acc[-1] * 100)

                except RuntimeError as e:
                    logger.error("Error in epoch %d: %s", epoch + 1, e)
                    raise

            logger.info("Training completed successfully.")
            return self.train_loss, self.train_acc

        except Exception as e:
            logger.exception('Failed to train model: %s', e)
            raise

    def evaluate_model(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Evaluate the model on test data.

        Args:
            inputs (torch.Tensor): Test input features.
            targets (torch.Tensor): Test target values.

        Returns:
            float: Test loss value.

        Raises:
            RuntimeError: If model is not initialized or evaluation fails.
            Exception: If forward pass or loss computation fails.
        """
        if self.model is None:
            raise RuntimeError("Model must be initialized before evaluation")

        try:
            with torch.no_grad():
                if isinstance(self.model, nn.Module):
                    self.model.eval()
                    predictions = self.model(inputs)
                else:
                    self.model.layers.eval()
                    predictions = self.model.forward(inputs)

                # Apply sigmoid to get probabilities (0-1 range)
                predictions_probs = torch.sigmoid(predictions.squeeze())
                
                # Calculate loss using probabilities
                loss = self.fn_loss(predictions_probs, targets.squeeze())
                
                # Convert probabilities to binary predictions (0 or 1) for accuracy
                predictions_binary = (predictions_probs >= 0.5).float()
                
                # Convert to numpy for metrics calculation
                targets_np = targets.squeeze().detach().cpu().numpy()
                predictions_np = predictions_binary.detach().cpu().numpy()
                
                acc = accuracy_score(targets_np, predictions_np)
                self.test_acc = acc
                logger.info('Test Loss: %.4f | Test Accuracy: %.2f%%', loss.item(), acc * 100)

                # Create and display confusion matrix
                cm = confusion_matrix(targets_np, predictions_np)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
                disp.plot(cmap="Blues")
                plt.title("Confusion Matrix - Test Set")
                plt.grid(False)
                plt.tight_layout()
                
                # Save confusion matrix
                images_dir.mkdir(parents=True, exist_ok=True)
                cm_path = images_dir / "Confusion_Matrix.png"
                plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                logger.info('Saved confusion matrix: %s', cm_path)
                plt.close()

            return loss.item()

        except RuntimeError as e:
            logger.error("Runtime error during model evaluation: %s", e)
            raise
        except Exception as e:
            logger.exception('Failed to evaluate model: %s', e)
            raise

def main() -> None:
    """
    Main function to execute heart disease classification pipeline.

    Creates model, runs training/evaluation, and saves results.

    Raises:
        RuntimeError: If the analysis process fails.
    """
    try:
        logger.info("=" * 70)
        logger.info("HEART DISEASE CLASSIFICATION ANALYSIS")
        logger.info("=" * 70)
        
        # Initialize analysis
        analysis = HeartDiseaseAnalysis()
        
        # Create model
        model = CustomManualFNN()
        analysis.model = model
        
        model_name = model.__class__.__name__ + model.get_architecture()
        logger.info("Training model: %s", model_name)
        
        # Run pipeline
        train_losses, train_accs, test_loss, test_acc = analysis.process()
        
        # Save model
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{model_name}.pth"
        torch.save(model.state_dict(), model_path)
        logger.info("Model saved: %s", model_path)
        
        # Plot training losses
        fig = plt.figure(figsize=(12, 6))
        plt.plot(train_losses, linewidth=2, label='Training Loss', color='blue')
        plt.title(f'Training Progress - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontweight='bold')
        plt.ylabel('Loss (BCE)', fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        images_dir.mkdir(parents=True, exist_ok=True)
        plot_path = images_dir / "Train_Loss_Curve.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info("Training plot saved: %s", plot_path)
        plt.close(fig)
        
        # Summary
        logger.info("=" * 70)
        logger.info("RESULTS SUMMARY")
        logger.info("Model: %s", model_name)
        logger.info("Train Loss: %.4f | Train Accuracy: %.2f%%", train_losses[-1], train_accs[-1] * 100)
        logger.info("Test Loss:  %.4f | Test Accuracy:  %.2f%%", test_loss, test_acc * 100)
        logger.info("=" * 70)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 70)

    except Exception as e:
        logger.error("=" * 70)
        logger.error("PIPELINE FAILED")
        logger.exception("Fatal error: %s", e)
        logger.error("=" * 70)
        raise
    finally:
        plt.close('all')
        logger.debug("Resources cleaned up")



if __name__ == "__main__":
    try:
        main()
        logger.info("Program execution completed successfully.")
    except Exception as e:
        logger.critical("Program terminated due to fatal error: %s", e)
        exit(1)