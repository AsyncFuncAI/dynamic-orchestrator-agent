"""
REINFORCE trainer for policy optimization in the DOA Framework.
"""

from typing import List
import torch
import torch.optim as optim

from .structs import EpisodeTrajectory, RewardConfig
from .policy import PolicyNetwork


class REINFORCETrainer:
    """REINFORCE-based trainer for optimizing the orchestrator policy."""
    
    def __init__(
        self, 
        policy_network: PolicyNetwork, 
        optimizer: optim.Optimizer, 
        reward_config: RewardConfig
    ):
        """
        Initialize the REINFORCE trainer.
        
        Args:
            policy_network: Policy network to optimize
            optimizer: PyTorch optimizer
            reward_config: Reward configuration for discount factor
        """
        self.policy_network = policy_network
        self.optimizer = optimizer
        self.reward_config = reward_config
    
    def train_batch(self, trajectories: List[EpisodeTrajectory]) -> float:
        """
        Train the policy on a batch of trajectories using REINFORCE.
        
        Args:
            trajectories: List of episode trajectories
            
        Returns:
            Total policy loss
        """
        if not trajectories:
            return 0.0
        
        total_loss = 0.0
        
        for trajectory in trajectories:
            # Calculate discounted returns for each step
            returns = self._calculate_discounted_returns(trajectory)
            
            # Calculate policy loss for this trajectory
            trajectory_loss = 0.0
            
            for step, G_t in zip(trajectory.steps, returns):
                # Policy gradient: -log_prob * return
                loss_term = -step.log_prob * G_t
                trajectory_loss += loss_term
            
            total_loss += trajectory_loss
        
        # Average loss over batch
        avg_loss = total_loss / len(trajectories)
        
        # Perform optimization step
        self.optimizer.zero_grad()
        avg_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return avg_loss.item()
    
    def _calculate_discounted_returns(self, trajectory: EpisodeTrajectory) -> List[float]:
        """
        Calculate discounted returns G_t = Σ_{k=t}^{T} γ^{k-t} * r_k for each step.
        
        Args:
            trajectory: Episode trajectory
            
        Returns:
            List of discounted returns for each step
        """
        returns = []
        G = 0.0
        
        # Calculate returns backwards from the end
        for step in reversed(trajectory.steps):
            G = step.reward + self.reward_config.gamma_discount_factor * G
            returns.insert(0, G)
        
        return returns
