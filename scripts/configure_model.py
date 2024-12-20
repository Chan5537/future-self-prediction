# scripts/configure_model.py

import os
import torch
from transformers import T5ForConditionalGeneration, T5Config
import logging

def setup_logging():
    """
    Configures the logging settings.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def initialize_model(model_name='t5-base'):
    """
    Initializes the T5 model with the specified variant.

    Parameters:
    - model_name (str): Name of the T5 model variant.

    Returns:
    - model (T5ForConditionalGeneration): The initialized T5 model.
    """
    logger.info(f"Initializing the {model_name} model...")
    try:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        logger.info(f"{model_name} model initialized successfully.")
        return model
    except Exception as e:
        logger.error(f"Error initializing the model: {e}")
        raise

def configure_model(model, learning_rate=3e-4, weight_decay=0.01):
    """
    Configures model parameters if needed.

    Parameters:
    - model (T5ForConditionalGeneration): The T5 model to configure.
    - learning_rate (float): Learning rate for the optimizer.
    - weight_decay (float): Weight decay for the optimizer.

    Returns:
    - config (dict): Configuration dictionary.
    """
    logger.info("Configuring model parameters...")
    config = {
        'learning_rate': learning_rate,
        'weight_decay': weight_decay
    }
    logger.info(f"Model configured with learning_rate={learning_rate} and weight_decay={weight_decay}.")
    return config

def move_model_to_gpu(model):
    """
    Moves the model to GPU if available.

    Parameters:
    - model (T5ForConditionalGeneration): The T5 model to move.

    Returns:
    - device (torch.device): The device the model is moved to.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if device.type == 'cuda':
        logger.info(f"Model moved to GPU: {torch.cuda.get_device_name(0)}.")
    else:
        logger.warning("GPU not available. Model is on CPU, which will be slower.")
    return model, device

def save_model_configuration(config, save_path="../models/config"):
    """
    Saves the model configuration to a file.

    Parameters:
    - config (dict): Configuration dictionary.
    - save_path (str): Path to save the configuration.
    """
    os.makedirs(save_path, exist_ok=True)
    config_file = os.path.join(save_path, 'model_config.txt')
    with open(config_file, 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    logger.info(f"Model configuration saved to {config_file}.")

def main():
    """
    Main function to initialize, configure, and move the T5 model to GPU.
    """
    # Define the model variant
    model_name = 't5-base'  # Change to 't5-small' or 't5-large' as needed

    # Initialize the model
    model = initialize_model(model_name)

    # Configure model parameters
    config = configure_model(model, learning_rate=3e-4, weight_decay=0.01)

    # Move the model to GPU
    model, device = move_model_to_gpu(model)

    # Save the model configuration
    save_model_configuration(config)

    # Optionally, save the initialized model for later use
    # Uncomment the following lines if you wish to save the model
    model_save_path = "../models/t5_model"
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    logger.info(f"Model saved to {model_save_path}.")

    # Verify model device
    current_device = next(model.parameters()).device
    logger.info(f"Model is on device: {current_device}")

if __name__ == "__main__":
    main()
