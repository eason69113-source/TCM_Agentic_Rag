from concurrent_log_handler import ConcurrentRotatingFileHandler
import logging

def Logger():

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    logger.handlers = []
    handler = ConcurrentRotatingFileHandler(
        filename="output/app.log",
        maxBytes=1024 * 1024 * 5,
        backupCount=3,
        encoding="utf-8"
    )

    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(handler)

    return logger