from logging import getLogger, handlers, Formatter, INFO

def set_logger():
    root_logger = getLogger()
    root_logger.setLevel(INFO)
    rotating_handler = handlers.RotatingFileHandler(
        r'./logs/app.log',
        mode="a",
        maxBytes=100 * 1024,
        backupCount=3,
        encoding="utf-8"
    )
    format = Formatter("="*60+"\n%(asctime)s: [%(levelname)s]: %(message)s")
    rotating_handler.setFormatter(format)
    root_logger.addHandler(rotating_handler)
