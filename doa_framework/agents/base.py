"""
Base agent interface for the DOA Framework.
"""

from abc import ABC, abstractmethod
from ..structs import SystemState, AgentOutput


class AgentInterface(ABC):
    """Abstract base class for all agents in the DOA framework."""
    
    def __init__(self, name: str):
        self._name = name
    
    @property
    def name(self) -> str:
        """Get the agent's name."""
        return self._name
    
    @abstractmethod
    def execute(self, state: SystemState) -> AgentOutput:
        """
        Execute the agent's logic given the current system state.
        
        Args:
            state: Current system state containing task specification and history
            
        Returns:
            AgentOutput containing the agent's response, cost, and metadata
        """
        pass
