# Standard library imports
from abc import ABC, abstractmethod
from datetime import datetime

# Third party imports

# Local imports

class SimulationObject(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def update_state(self, epoch: datetime):
        pass
