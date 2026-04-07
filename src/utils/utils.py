import logging

def setup_logger(name: str, log_file: str, level=logging.INFO):
    """
    Creates and configures a logger.

    Args:
        name (str): Logger name.
        log_file (str): File to log to.
        level: Logging level.
    """
    handler = logging.FileHandler(log_file)        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger