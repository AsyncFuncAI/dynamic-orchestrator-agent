"""
Core data structures for the DOA Framework.
"""

from dataclasses import dataclass
from typing import Any, List, Tuple, Optional, Dict
import torch


@dataclass
class AgentOutput:
    """Standardized output from agent execution."""
    content: Any
    cost: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass 
class SystemState:
    """Global system state representing current task context."""
    task_specification: str
    history: List[Tuple[str, AgentOutput]]  # (agent_name, agent_output)
    current_step: int
    max_steps: int
    custom_data: Optional[Dict[str, Any]] = None


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    lambda_cost_penalty: float = 0.1
    gamma_discount_factor: float = 0.99
    task_success_bonus: float = 1.0
    task_failure_penalty: float = -1.0
    step_cost_scale_factor: float = 1.0


@dataclass
class TrajectoryStep:
    """Single step in an episode trajectory for RL training."""
    state_embedding: torch.Tensor
    agent_index: int
    log_prob: torch.Tensor
    reward: float
    next_state_embedding: Optional[torch.Tensor]
    is_terminal_step: bool


@dataclass
class EpisodeTrajectory:
    """Complete trajectory of an episode."""
    steps: List[TrajectoryStep]
    total_undiscounted_reward: float
    task_successful: bool
