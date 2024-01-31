from abc import ABC, abstractmethod

class SpacecraftModule(ABC):
    def __init__(self, name):
        self.name = name
        self.__power_state = "OFF"

    @property
    def power_state(self) -> str:
        return self.__power_state

    @abstractmethod
    def turn_on(self):
        self.__power_state = "ON"

    @abstractmethod
    def turn_off(self):
        self.__power_state = "OFF"

    @abstractmethod
    def set_idle(self):
        self.__power_state = "IDLE"

