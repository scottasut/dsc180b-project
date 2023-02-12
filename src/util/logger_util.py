import logging

def configure_logger(to):
    """Helper to configure logger settings

    Args:
        to (str): path of file to log to.
    """
    logging.basicConfig(filename=to, 
        filemode='a', 
        level=logging.INFO,
        datefmt='%H:%M:%S',
        format='%(asctime)s %(levelname)s %(name)s: %(message)s')