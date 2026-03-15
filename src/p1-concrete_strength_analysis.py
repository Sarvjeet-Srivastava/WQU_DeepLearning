"""
Concrete Strength Prediction using Deep Learning Models.

This module implements multiple neural network architectures (Simple, Manual, and Deep MLPs)
to predict concrete strength from 8 concrete composition features. It provides data preprocessing,
model training, evaluation, and comparative analysis across different model configurations.

Key Features:
    - Multiple model architectures with configurable activation functions
    - Data exploration and visualization
    - Model training with validation
    - Comparative analysis across models
    - Model persistence and logging

Author: WQU DeepLearning Project
Version: 2.0
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import math
import re
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim

from utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)
logger.info(f"PyTorch version: {torch.__version__}")

# Directory configuration
base_dir: Path = Path(__file__).resolve().parent.parent
base_data_dir: Path = base_dir / "Data" / "p1-concrete-data"
models_dir: Path = base_dir / "Models" / "P1"
images_dir: Path = base_dir / "Images" / "P1"


class CustomManualMLPModel:
    """
    Manual Multi-Layer Perceptron model with ReLU activation.
    
    This class implements a simple feed-forward neural network with configurable
    input, hidden, and output layer sizes. The model uses ReLU activation between
    layers and is suitable for regression tasks.
    
    Attributes:
        input_size (int): Number of input features.
        hidden_size (int): Number of neurons in the hidden layer.
        output_size (int): Number of output neurons (typically 1 for regression).
        layers (nn.Sequential): Sequential container of the model layers.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """
        Initialize the Manual MLP model.
        
        Args:
            input_size (int): Dimension of input features.
            hidden_size (int): Number of hidden units in the middle layer.
            output_size (int): Dimension of output (typically 1 for regression).
            
        Raises:
            ValueError: If any size parameter is non-positive.
        """
        try:
            if input_size <= 0 or hidden_size <= 0 or output_size <= 0:
                raise ValueError(f"All sizes must be positive. Got input_size={input_size}, "
                               f"hidden_size={hidden_size}, output_size={output_size}")
            
            self.input_size: int = input_size
            self.hidden_size: int = hidden_size
            self.output_size: int = output_size
            
            self.layers: nn.Sequential = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.output_size)
            )
            logger.info("CustomManualMLPModel created: %s", self.layers)
        except ValueError as e:
            logger.error("Failed to initialize CustomManualMLPModel: %s", e)
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
            
        Raises:
            RuntimeError: If forward pass fails due to shape mismatch or computation error.
        """
        try:
            return self.layers(x)
        except RuntimeError as e:
            logger.error("Forward pass failed: %s", e)
            raise

    def get_architecture(self) -> str:
        """
        Generate a string representation of the model architecture.
        
        Returns:
            str: Architecture string in format '(input_size,hidden_size,output_size)'.
        """
        return f"({self.input_size},{self.hidden_size},{self.output_size})"


class CustomSimpleMLPModel(nn.Module):
    """
    Simple Multi-Layer Perceptron model with configurable activation function.
    
    This class implements a straightforward neural network with one hidden layer
    and a configurable activation function. It extends PyTorch's nn.Module for
    compatibility with standard PyTorch workflows.
    
    Attributes:
        input_size (int): Number of input features.
        hidden_size (int): Number of neurons in the hidden layer.
        output_size (int): Number of output neurons.
        fn_activation (nn.Module): Activation function to use (default: ReLU).
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 fn_activation: Optional[nn.Module] = None) -> None:
        """
        Initialize the Simple MLP model.
        
        Args:
            input_size (int): Dimension of input features.
            hidden_size (int): Number of hidden units.
            output_size (int): Dimension of output.
            fn_activation (Optional[nn.Module]): Activation function. Defaults to ReLU if None.
            
        Raises:
            ValueError: If any size parameter is non-positive.
            TypeError: If fn_activation is not None and not an nn.Module.
        """
        super(CustomSimpleMLPModel, self).__init__()
        try:
            if input_size <= 0 or hidden_size <= 0 or output_size <= 0:
                raise ValueError(f"All sizes must be positive. Got input_size={input_size}, "
                               f"hidden_size={hidden_size}, output_size={output_size}")
            
            if fn_activation is not None and not isinstance(fn_activation, nn.Module):
                raise TypeError(f"fn_activation must be nn.Module, got {type(fn_activation)}")
            
            self.input_size: int = input_size
            self.hidden_size: int = hidden_size
            self.output_size: int = output_size
            self.fn_activation: nn.Module = fn_activation if fn_activation is not None else nn.ReLU()
            
            self.fc1: nn.Linear = nn.Linear(input_size, hidden_size)
            self.fc2: nn.Linear = nn.Linear(hidden_size, output_size)
            
            logger.info("CustomSimpleMLPModel created with activation: %s", 
                       self.fn_activation.__class__.__name__)
        except (ValueError, TypeError) as e:
            logger.error("Failed to initialize CustomSimpleMLPModel: %s", e)
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
            
        Raises:
            RuntimeError: If forward pass fails due to shape mismatch or computation error.
        """
        try:
            x = self.fc1(x)
            x = self.fn_activation(x)
            x = self.fc2(x)
            return x
        except RuntimeError as e:
            logger.error("Forward pass failed: %s", e)
            raise

    def get_architecture(self) -> str:
        """
        Generate a string representation of the model architecture.
        
        Returns:
            str: Architecture string including activation function.
        """
        activation_name = self.fn_activation.__class__.__name__
        return f"({self.input_size},{self.hidden_size},{self.output_size},{activation_name})"


class CustomDeepMLPModel(nn.Module):
    """
    Deep Multi-Layer Perceptron model with configurable activation function.
    
    This class implements a deeper neural network architecture with multiple hidden
    layers (64 and 32 units respectively) and a configurable activation function.
    It extends PyTorch's nn.Module for standard PyTorch integration.
    
    Architecture: Input(8) -> Hidden1(64) -> Hidden2(32) -> Output(1)
    
    Attributes:
        fn_activation (nn.Module): Activation function to use (default: ReLU).
        fc1 (nn.Linear): Input layer (8 -> 64 units).
        fc2 (nn.Linear): Second hidden layer (64 -> 32 units).
        fc3 (nn.Linear): Output layer (32 -> 1 unit).
    """
    
    def __init__(self, fn_activation: Optional[nn.Module] = None) -> None:
        """
        Initialize the Deep MLP model.
        
        Args:
            fn_activation (Optional[nn.Module]): Activation function. Defaults to ReLU if None.
            
        Raises:
            TypeError: If fn_activation is not None and not an nn.Module.
        """
        super(CustomDeepMLPModel, self).__init__()
        try:
            if fn_activation is not None and not isinstance(fn_activation, nn.Module):
                raise TypeError(f"fn_activation must be nn.Module, got {type(fn_activation)}")
            
            self.fn_activation: nn.Module = fn_activation if fn_activation is not None else nn.ReLU()
            
            # Define layers with clear comments on dimensions
            self.fc1: nn.Linear = nn.Linear(8, 64)   # Input layer: 8 -> 64 units
            self.fc2: nn.Linear = nn.Linear(64, 32)  # Hidden layer: 64 -> 32 units
            self.fc3: nn.Linear = nn.Linear(32, 1)   # Output layer: 32 -> 1 unit
            
            logger.info("CustomDeepMLPModel created with activation: %s", 
                       self.fn_activation.__class__.__name__)
        except TypeError as e:
            logger.error("Failed to initialize CustomDeepMLPModel: %s", e)
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 8).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
            
        Raises:
            RuntimeError: If forward pass fails due to shape mismatch or computation error.
        """
        try:
            x = self.fc1(x)
            x = self.fn_activation(x)
            x = self.fc2(x)
            x = self.fn_activation(x)
            x = self.fc3(x)
            return x
        except RuntimeError as e:
            logger.error("Forward pass failed: %s", e)
            raise

    def get_architecture(self) -> str:
        """
        Generate a string representation of the model architecture.
        
        Returns:
            str: Architecture string including activation function.
        """
        activation_name = self.fn_activation.__class__.__name__
        return f"(8-64-32-1,{activation_name})"


class ConcreteStrengthAnalysis:
    """
    Main analysis class for concrete strength prediction.
    
    This class orchestrates the entire machine learning pipeline including data loading,
    exploration, preprocessing, model training, and evaluation.
    
    Attributes:
        data (Optional[pd.DataFrame]): Loaded dataset.
        inputs (Optional[np.ndarray]): Input features.
        targets (Optional[np.ndarray]): Target variable (concrete strength).
        device (torch.device): Computation device (CUDA or CPU).
        model (Optional[nn.Module]): Neural network model.
        fn_loss (nn.Module): Loss function (MSELoss by default).
        train_loss (List[float]): Training loss history per epoch.
        test_loss (float): Test loss value.
    """
    
    def __init__(self) -> None:
        """
        Initialize the ConcreteStrengthAnalysis instance.
        
        Sets up device configuration, initializes data containers, and configures
        the loss function and model placeholders.
        """
        try:
            self.data: Optional[pd.DataFrame] = None
            self.inputs: Optional[np.ndarray] = None
            self.targets: Optional[np.ndarray] = None
            self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model: Optional[nn.Module] = None
            self.fn_loss: nn.Module = nn.MSELoss()
            self.train_loss: list[float] = []
            self.test_loss: float = 0.0
            
            logger.info("ConcreteStrengthAnalysis initialized. Device: %s", self.device)
        except Exception as e:
            logger.error("Failed to initialize ConcreteStrengthAnalysis: %s", e)
            raise

    def process(self) -> Tuple[list[float], float]:
        """
        Execute the complete analysis pipeline.
        
        Orchestrates data loading, exploration, preprocessing, and model training/evaluation.
        
        Returns:
            Tuple[list[float], float]: Training loss history and test loss value.
            
        Raises:
            RuntimeError: If any step in the pipeline fails.
        """
        try:
            logger.info("Starting concrete strength analysis pipeline...")
            
            self.data = self.read_data()
            self.inputs = self.data.iloc[:, :-1].values
            self.targets = self.data.iloc[:, -1].values
            
            columns = self.data.columns.tolist()
            col_names = [self.parse_column_name(col)['feature_name'] for col in columns]
            units = [self.parse_column_name(col)['unit'] for col in columns]
            
            self.explore_data(self.inputs, self.targets, list(zip(col_names, units)))
            
            self.inputs = self.scale_and_transform(self.inputs)
            self.targets = self.scale_and_transform(self.targets)
            
            X_train, y_train, X_test, y_test = self.split_data(self.inputs, self.targets, 0.2)
            
            self.train_loss = self.train_model(X_train, y_train)
            self.test_loss = self.evaluate_model(X_test, y_test)
            
            logger.info("Pipeline completed successfully.")
            return self.train_loss, self.test_loss
            
        except Exception as e:
            logger.exception("Pipeline failed: %s", e)
            raise

    @staticmethod
    def read_data(filename: str = 'Concrete_Data.csv') -> pd.DataFrame:
        """
        Read dataset CSV into a pandas DataFrame.

        Args:
            filename (str): CSV filename relative to the data directory.

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
            else:
                logger.info('No missing data detected.')
            
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
    def parse_column_name(col_name: str) -> Dict[str, Optional[str]]:
        """
        Parse a column name into feature name and unit using regex.
        
        Expected format: "Feature Name (unit)" or "Feature Name (component) (unit)"
        
        Args:
            col_name (str): Column name to parse.
            
        Returns:
            Dict[str, Optional[str]]: Dictionary with keys 'feature_name' and 'unit'.
        """
        col_name = col_name.strip()
        # Regex pattern: captures feature name, optional component, and unit in parentheses
        pattern = re.compile(r'([^(]+)\s*(?:\((component\s*\d+)\))?\s*\((.*)\)')

        match = pattern.match(col_name)

        if match:
            groups = match.groups()
            feature_name = groups[0].strip()
            unit = groups[2].strip()

            # Special case for target variable where unit contains comma
            if ',' in unit:
                unit = unit.split(',')[0]

            return {'feature_name': feature_name, 'unit': unit}

        # Fallback for names that might not match pattern
        logger.warning("Column name did not match expected pattern: %s", col_name)
        return {'feature_name': col_name, 'unit': None}

    @staticmethod
    def explore_data(inputs: np.ndarray, targets: np.ndarray, 
                     feature_names: list[tuple[str, str]]) -> None:
        """
        Explore and visualize input features and target distribution.
        
        Creates a comprehensive visualization including target histogram and scatter
        plots for each feature against the target variable.
        
        Args:
            inputs (np.ndarray): Input feature matrix of shape (n_samples, n_features).
            targets (np.ndarray): Target variable array of shape (n_samples,).
            feature_names (list[tuple[str, str]]): List of tuples (feature_name, unit).
            
        Raises:
            ValueError: If inputs and targets have incompatible shapes.
            Exception: If visualization creation fails.
        """
        try:
            if inputs.shape[0] != targets.shape[0]:
                raise ValueError(f"Inputs and targets must have same number of samples. "
                               f"Got {inputs.shape[0]} and {targets.shape[0]}")
            
            logger.info("Exploring input features...")
            num_features = inputs.shape[1]
            rows = math.ceil(num_features / 2)
            logger.debug('inputs.shape=%s, targets.shape=%s, num_features=%d, rows=%d',
                        inputs.shape, targets.shape, num_features, rows)

            fig = plt.figure(figsize=(20, 7 + 7 * rows))
            gs = fig.add_gridspec(nrows=rows+1, ncols=2)
            gs.update(hspace=1, wspace=0.3)

            # Configure smaller fonts for clarity
            plt.rc('font', size=8)
            plt.rc('axes', titlesize=9)
            plt.rc('axes', labelsize=8)
            plt.rc('xtick', labelsize=7)
            plt.rc('ytick', labelsize=7)
            plt.rc('legend', fontsize=8)

            # Target histogram on first axis
            ax0 = fig.add_subplot(gs[0, :])
            t_mean = float(targets.mean().item() if hasattr(targets.mean(), 'item') else targets.mean())
            t_std = float(targets.std().item() if hasattr(targets.std(), 'item') else targets.std())
            ax0.hist(targets, bins=100, edgecolor='k')
            ax0.set_title(f'Distribution of Concrete Strength — mean={t_mean:.2f}, std={t_std:.2f}')
            ax0.set_xlabel(f"{feature_names[-1][0]}\n({feature_names[-1][1]})")
            ax0.set_ylabel('Count')

            # Feature scatter plots
            idx = 0
            for r in range(rows):
                for c in range(2):
                    ax = fig.add_subplot(gs[r + 1, c])
                    if idx >= num_features:
                        ax.axis('off')
                        idx += 1
                        continue
                    
                    feat = inputs[:, idx]
                    f_mean = float(feat.mean().item() if hasattr(feat.mean(), 'item') else feat.mean())
                    f_std = float(feat.std().item() if hasattr(feat.std(), 'item') else feat.std())
                    
                    ax.scatter(feat, targets, c='b', alpha=0.5, s=20)
                    ax.set_title(f'{feature_names[idx][0]} — mean={f_mean:.2f}, std={f_std:.2f}')
                    ax.set_xlabel(f'{feature_names[idx][0]} ({feature_names[idx][1]})')
                    ax.set_ylabel(f"{feature_names[-1][0]}\n({feature_names[-1][1]})")
                    idx += 1
            
            out_dir = base_dir / "Images" / "P1"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "Concrete_Strength_Analysis.png"
            fig.savefig(out_path, dpi=300)
            logger.info('Saved analysis figure to %s', out_path)
            plt.close(fig)
            
        except ValueError as e:
            logger.error("Validation error in explore_data: %s", e)
            raise
        except Exception as e:
            logger.exception('Failed to produce analysis plots: %s', e)
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

    def train_model(self, inputs: torch.Tensor, targets: torch.Tensor,
                    num_epochs: int = 500, learning_rate: float = 0.001) -> list[float]:
        """
        Train the neural network model.
        
        Args:
            inputs (torch.Tensor): Training input features.
            targets (torch.Tensor): Training target values.
            num_epochs (int): Number of training epochs (default: 500).
            learning_rate (float): Learning rate for SGD optimizer (default: 0.001).
            
        Returns:
            list[float]: List of loss values per epoch.
            
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

            optimizer = optim.SGD(param_to_optimize, lr=learning_rate)

            for epoch in range(num_epochs):
                try:
                    # Forward pass
                    if isinstance(self.model, nn.Module):
                        self.model.train()
                        outputs = self.model(inputs)
                    else:
                        self.model.layers.train()
                        outputs = self.model.forward(inputs)
                    
                    loss = self.fn_loss(outputs.squeeze(), targets.squeeze())

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    self.train_loss.append(loss.item())
                    
                    if (epoch + 1) % (num_epochs // 10) == 0:
                        logger.info('Epoch [%d/%d], Loss: %.4f', epoch + 1, num_epochs, loss.item())
                
                except RuntimeError as e:
                    logger.error("Error in epoch %d: %s", epoch + 1, e)
                    raise
            
            logger.info("Training completed successfully.")
            return self.train_loss
            
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

                loss = self.fn_loss(predictions.squeeze(), targets.squeeze())
                logger.info('Test Loss: %.4f', loss.item())
                
            return loss.item()
            
        except RuntimeError as e:
            logger.error("Runtime error during model evaluation: %s", e)
            raise
        except Exception as e:
            logger.exception('Failed to evaluate model: %s', e)
            raise


def main() -> None:
    """
    Main execution function for concrete strength prediction.
    
    Orchestrates the training of 9 different neural network model variations:
    
    **CustomSimpleMLPModel variations (3 models):**
        - ReLU activation function
        - Sigmoid activation function
        - Tanh activation function
    
    **CustomManualMLPModel variations (3 models):**
        - Hidden layer size: 32 (ReLU activation)
        - Hidden layer size: 64 (ReLU activation)
        - Hidden layer size: 128 (ReLU activation)
    
    **CustomDeepMLPModel variations (3 models):**
        - ReLU activation function
        - Sigmoid activation function
        - Tanh activation function
    
    For each model:
        1. Trains on the concrete strength dataset
        2. Saves the trained model to Models/P1/
        3. Records training loss history
    
    Finally, creates a comparative visualization of all training loss curves
    and saves it to Images/P1/Train Loss Comparison.png
    
    Raises:
        Exception: If any critical step in the pipeline fails.
    """
    ca = None
    models_dir = None
    images_dir = None
    
    try:
        # Initialize directories
        models_dir = base_dir / "Models" / "P1"
        images_dir = base_dir / "Images" / "P1"
        models_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Directories initialized. Models: %s, Images: %s", models_dir, images_dir)

        # Instantiate analysis orchestrator
        ca = ConcreteStrengthAnalysis()
        logger.info("ConcreteStrengthAnalysis instance created.")

        train_losses: Dict[str, list[float]] = {}
        test_losses: Dict[str, float] = {}
        model_list: list = []

        # ============================================
        # CustomSimpleMLPModel variations (3 models)
        # ============================================
        logger.info("=" * 60)
        logger.info("Creating CustomSimpleMLPModel variations...")
        logger.info("=" * 60)
        
        try:
            model_simple_relu = CustomSimpleMLPModel(8, 32, 1, fn_activation=nn.ReLU())
            model_simple_sigmoid = CustomSimpleMLPModel(8, 32, 1, fn_activation=nn.Sigmoid())
            model_simple_tanh = CustomSimpleMLPModel(8, 32, 1, fn_activation=nn.Tanh())
            model_list.extend([model_simple_relu, model_simple_sigmoid, model_simple_tanh])
            logger.info("Successfully created 3 CustomSimpleMLPModel variations.")
        except Exception as e:
            logger.exception("Failed to create CustomSimpleMLPModel variations: %s", e)
            raise

        # ============================================
        # CustomManualMLPModel variations (3 models)
        # ============================================
        logger.info("=" * 60)
        logger.info("Creating CustomManualMLPModel variations...")
        logger.info("=" * 60)
        
        try:
            model_manual_32 = CustomManualMLPModel(8, 32, 1)
            model_manual_64 = CustomManualMLPModel(8, 64, 1)
            model_manual_128 = CustomManualMLPModel(8, 128, 1)
            model_list.extend([model_manual_32, model_manual_64, model_manual_128])
            logger.info("Successfully created 3 CustomManualMLPModel variations.")
        except Exception as e:
            logger.exception("Failed to create CustomManualMLPModel variations: %s", e)
            raise

        # ============================================
        # CustomDeepMLPModel variations (3 models)
        # ============================================
        logger.info("=" * 60)
        logger.info("Creating CustomDeepMLPModel variations...")
        logger.info("=" * 60)
        
        try:
            model_deep_relu = CustomDeepMLPModel(fn_activation=nn.ReLU())
            model_deep_sigmoid = CustomDeepMLPModel(fn_activation=nn.Sigmoid())
            model_deep_tanh = CustomDeepMLPModel(fn_activation=nn.Tanh())
            model_list.extend([model_deep_relu, model_deep_sigmoid, model_deep_tanh])
            logger.info("Successfully created 3 CustomDeepMLPModel variations.")
        except Exception as e:
            logger.exception("Failed to create CustomDeepMLPModel variations: %s", e)
            raise

        # ============================================
        # Train all models
        # ============================================
        logger.info("=" * 60)
        logger.info("Starting training of %d models...", len(model_list))
        logger.info("=" * 60)
        
        for idx, model in enumerate(model_list, 1):
            model_name = None
            model_path = None
            
            try:
                model_name = model.__class__.__name__ + model.get_architecture()
                logger.info("[%d/%d] Training model: %s", idx, len(model_list), model_name)
                
                ca.model = model
                ca.train_loss = []  # Reset train loss for each model
                train_loss, test_loss = ca.process()

                # Store losses
                train_losses[model_name] = train_loss
                test_losses[model_name] = test_loss
                logger.debug("Stored losses for model: %s", model_name)

                # Save model
                model_path = models_dir / f"{model_name}.pth"
                state_dict = model.state_dict() if isinstance(model, nn.Module) else model.layers.state_dict()
                torch.save(state_dict, model_path)
                logger.info("Model saved successfully to: %s", model_path)
                
            except Exception as e:
                logger.exception("Failed to train or save model %s: %s", model_name, e)
                raise

        logger.info("=" * 60)
        logger.info("All models trained successfully!")
        logger.info("=" * 60)
        logger.info("Test Losses Summary:\n%s", 
                   "\n".join([f"  {name}: {loss:.4f}" for name, loss in test_losses.items()]))

        # ============================================
        # Plot training losses comparison
        # ============================================
        logger.info("=" * 60)
        logger.info("Creating training loss comparison plot...")
        logger.info("=" * 60)
        
        try:
            fig = plt.figure(figsize=(16, 9))
            
            for model_name, losses in train_losses.items():
                plt.plot(losses, label=model_name, linewidth=2.5, alpha=0.8)

            plt.title('Training Loss Comparison - All Models', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Epochs', fontsize=13, fontweight='bold')
            plt.ylabel('Loss (MSE)', fontsize=13, fontweight='bold')
            plt.legend(fontsize=11, loc='best', framealpha=0.95)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            
            # Save plot
            plot_path = images_dir / "Train Loss Comparison.png"
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info("Training loss comparison plot saved to: %s", plot_path)
            plt.close(fig)
            
        except Exception as e:
            logger.exception("Failed to create or save comparison plot: %s", e)
            raise

        logger.info("=" * 60)
        logger.info("Pipeline execution completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error("PIPELINE EXECUTION FAILED")
        logger.exception("Fatal error in main function: %s", e)
        logger.error("=" * 60)
        raise
    finally:
        # Cleanup resources
        try:
            plt.close('all')
            logger.debug("Matplotlib resources cleaned up.")
        except Exception as e:
            logger.warning("Error during cleanup: %s", e)


if __name__ == "__main__":
    try:
        main()
        logger.info("Program execution completed successfully.")
    except Exception as e:
        logger.critical("Program terminated due to fatal error: %s", e)
        exit(1)
