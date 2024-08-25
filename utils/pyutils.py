import logging


def print_progress(a, b):
    progress = a / b
    bar_length = 20
    filled_length = int(progress * bar_length)
    bar = '=' * filled_length + '>' + '-' * (bar_length - filled_length - 1)
    print(f'\r|{bar}| {progress * 100:.1f}% | {a} / {b}  ', end='')


def setup_logger(filename='test.log'):
    ## setup logger
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)
