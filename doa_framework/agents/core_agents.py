"""
Core agent implementations for the DOA Framework.
"""

from .base import AgentInterface
from ..structs import SystemState, AgentOutput


class TerminatorAgent(AgentInterface):
    """Agent that signals episode termination."""
    
    def __init__(self, name: str = "TerminatorAgent"):
        super().__init__(name)
    
    def execute(self, state: SystemState) -> AgentOutput:
        """
        Signal episode termination.
        
        Args:
            state: Current system state
            
        Returns:
            AgentOutput indicating termination with zero cost
        """
        return AgentOutput(
            content="TERMINATE",
            cost=0.0,
            metadata={"termination_reason": "agent_selected"}
        )


class EchoAgent(AgentInterface):
    """Simple test agent that echoes the task specification."""
    
    def __init__(self, name: str = "EchoAgent"):
        super().__init__(name)
    
    def execute(self, state: SystemState) -> AgentOutput:
        """
        Echo the task specification back.
        
        Args:
            state: Current system state
            
        Returns:
            AgentOutput containing the task specification with fixed cost
        """
        return AgentOutput(
            content=f"Echo: {state.task_specification}",
            cost=1.0,
            metadata={
                "agent_type": "echo",
                "step": state.current_step,
                "history_length": len(state.history)
            }
        )
