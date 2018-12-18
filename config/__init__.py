import os


try:
    from config.configs import Config
except ImportError:
    from config.default import Config

config = Config()