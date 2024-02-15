import logging

def setup_logger():
    logger = logging.getLogger(__name__)

    if not logger.handlers:  # Check if the logger already has handlers
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "[%(asctime)s] [%(filename)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)
        logger.propagate = False

    return logger 