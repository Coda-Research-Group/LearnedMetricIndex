import logging


def get_logger_config() -> str:
    return "[%(asctime)s][%(levelname)-5.5s][%(name)-.20s] %(message)s"


def remove_logger_handlers():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


class Logger:
    @property
    def logger(self, level=logging.INFO):
        component = "{}.{}".format(type(self).__module__, type(self).__name__)
        logging.basicConfig(level=level, format=get_logger_config())
        return logging.getLogger(component)
