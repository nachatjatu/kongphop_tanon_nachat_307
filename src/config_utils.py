# config_utils.py
import json
from dataclasses import dataclass, asdict

@dataclass
class ScenarioConfig:
    num_depots: int
    day: str       
    time: str      

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

def load_config(path):
    with open(path) as f:
        return json.load(f)
