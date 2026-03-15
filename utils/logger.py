import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a custom logger with a specific format including
    timestamp, logger name (script), function name, level, and message.
    
    Args:
        name (str): The name of the logger, typically __name__ of the calling script.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Check if handlers are already added to avoid duplicate logs
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        
        # Define format: Time - Script Name - Function Name - Level - Message
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger
