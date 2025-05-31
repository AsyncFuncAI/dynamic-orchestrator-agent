"""
Dynamic Orchestrator Agent (DOA) Framework

A framework for adaptive multi-agent LLM collaboration with reinforcement learning-based orchestration.
"""

__version__ = "0.1.0"

from .structs import AgentOutput, SystemState, RewardConfig, TrajectoryStep, EpisodeTrajectory
from .agents.base import AgentInterface
from .agents.core_agents import TerminatorAgent, EchoAgent
from .orchestrator import Orchestrator
from .policy import PolicyNetwork
from .trainer import REINFORCETrainer
from .rewards import calculate_reward

__all__ = [
    "AgentOutput",
    "SystemState", 
    "RewardConfig",
    "TrajectoryStep",
    "EpisodeTrajectory",
    "AgentInterface",
    "TerminatorAgent",
    "EchoAgent", 
    "Orchestrator",
    "PolicyNetwork",
    "REINFORCETrainer",
    "calculate_reward",
]
