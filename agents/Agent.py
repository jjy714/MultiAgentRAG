from abc import ABC, abstractmethod


### Abstract base class that all agent implementations must inherit from
class Agent(ABC):
    """
    args   : {}
    return : {
        "Agent": "abstract base instance requiring 'communicate' to be implemented"
    }
    """

    ## Initialize the agent (no-op in base class)
    def __init__():
        """
        args   : {}
        return : {
            "None": "no-op initialization"
        }
        """
        pass

    ## Abstract method that all agent subclasses must implement to handle communication
    @abstractmethod
    def communicate(self):
        """
        args   : {}
        return : {
            "Any": "implementation-defined response from the agent"
        }
        """
        pass