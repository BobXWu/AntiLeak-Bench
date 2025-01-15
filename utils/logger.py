import logging


def get_logger(name='AntiLeak-Bench', level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def add_handlers(logger, file_path=None):
    LOG_FORMAT = "%(asctime)s - %(filename)s[ln:%(lineno)d] - %(levelname)s: %(message)s"
    DATE_FORMAT = "%Y/%m/%d %H:%M:%S"

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    if file_path:
        fh = logging.FileHandler(file_path)
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    return logger
