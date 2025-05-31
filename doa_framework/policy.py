"""
Policy network for agent selection in the DOA Framework.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib

from .structs import SystemState


class PolicyNetwork(nn.Module):
    """Neural network policy for selecting agents based on system state."""
    
    def __init__(self, state_embedding_dim: int, num_agents: int, hidden_dim: int = 128):
        """
        Initialize the policy network.
        
        Args:
            state_embedding_dim: Dimension of state embeddings
            num_agents: Number of available agents
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.state_embedding_dim = state_embedding_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        
        # Simple MLP for agent selection
        self.network = nn.Sequential(
            nn.Linear(state_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents)
        )
    
    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get agent selection logits.
        
        Args:
            state_embedding: Embedded representation of system state
            
        Returns:
            Logits for agent selection
        """
        return self.network(state_embedding)
    
    def select_action(self, system_state: SystemState) -> Tuple[int, torch.Tensor]:
        """
        Select an agent based on the current system state.
        
        Args:
            system_state: Current system state
            
        Returns:
            Tuple of (selected_agent_index, log_probability)
        """
        # Get state embedding
        state_embedding = self._embed_state(system_state)
        
        # Forward pass to get logits
        logits = self.forward(state_embedding)
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample action
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob
    
    def _embed_state(self, system_state: SystemState) -> torch.Tensor:
        """
        Embed system state into a fixed-size tensor.
        
        This is a basic implementation for MVP. In production, this would use
        more sophisticated encoding of task specification and history.
        
        Args:
            system_state: System state to embed
            
        Returns:
            State embedding tensor
        """
        # Create a basic embedding based on state features
        embedding = torch.zeros(self.state_embedding_dim)
        
        # Encode task specification using hash
        task_hash = hashlib.md5(system_state.task_specification.encode()).hexdigest()
        task_features = [int(task_hash[i:i+2], 16) / 255.0 for i in range(0, min(32, len(task_hash)), 2)]
        
        # Fill first part with task features
        task_dim = min(len(task_features), self.state_embedding_dim // 2)
        embedding[:task_dim] = torch.tensor(task_features[:task_dim])
        
        # Encode history information
        if len(system_state.history) > 0:
            # Use simple features: history length, average cost, step count
            history_features = [
                len(system_state.history) / 10.0,  # Normalized history length
                sum(output.cost for _, output in system_state.history) / len(system_state.history) / 10.0,  # Avg cost
                system_state.current_step / system_state.max_steps,  # Progress
            ]
            
            # Fill remaining dimensions with history features
            hist_start = self.state_embedding_dim // 2
            hist_dim = min(len(history_features), self.state_embedding_dim - hist_start)
            embedding[hist_start:hist_start + hist_dim] = torch.tensor(history_features[:hist_dim])
        
        return embedding
