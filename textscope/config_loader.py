import yaml
import os

class Config:
    def __init__(
        self
    ):
        self.profiles = {}
        self.subthemes = {}

    def load_from_file(
        self,
        filepath:str
    )-> None:
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)
            self.profiles = data.get('PROFILES', {})
            self.subthemes = data.get('SUBTHEMES', {})

config = Config()

def load_config(
        filepath:str=None
)-> None:
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), 'data', 'config.yaml')
    config.load_from_file(filepath)