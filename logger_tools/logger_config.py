import logging


def get_logger(name):
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        log_format = "[%(asctime)s] [%(filename)s > %(funcName)s():%(lineno)d] [%(levelname)s] - %(message)s"
        formatter = logging.Formatter(log_format)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
