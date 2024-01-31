from abc import ABC, abstractmethod
from datetime import datetime

class SimulationObject(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def update_state(self, epoch: datetime):
        pass
