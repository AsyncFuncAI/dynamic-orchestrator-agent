"""
Orchestrator for dynamic agent coordination in the DOA Framework.
"""

from typing import List
import torch

from .structs import SystemState, AgentOutput, EpisodeTrajectory, TrajectoryStep, RewardConfig
from .agents.base import AgentInterface
from .policy import PolicyNetwork
from .rewards import calculate_reward


class Orchestrator:
    """Central orchestrator that coordinates agent execution using a learnable policy."""
    
    def __init__(
        self, 
        agents: List[AgentInterface], 
        policy: PolicyNetwork, 
        reward_config: RewardConfig
    ):
        """
        Initialize the orchestrator.
        
        Args:
            agents: List of available agents
            policy: Policy network for agent selection
            reward_config: Configuration for reward calculation
        """
        self.agents = agents
        self.policy = policy
        self.reward_config = reward_config
        
        # Create agent name to index mapping
        self.agent_name_to_idx = {agent.name: idx for idx, agent in enumerate(agents)}
    
    def run_episode(self, initial_state: SystemState) -> EpisodeTrajectory:
        """
        Run a complete episode using the policy to select agents.
        
        Args:
            initial_state: Starting system state
            
        Returns:
            Complete episode trajectory for training
        """
        current_state = initial_state
        trajectory_steps = []
        total_reward = 0.0
        
        while not self._is_terminated(current_state, None):
            # Get current state embedding
            state_embedding = self.policy._embed_state(current_state)
            
            # Select agent using policy
            agent_idx, log_prob = self.policy.select_action(current_state)
            selected_agent = self.agents[agent_idx]
            
            # Execute selected agent
            agent_output = selected_agent.execute(current_state)
            
            # Update state
            next_state = self._update_state(current_state, selected_agent.name, agent_output)
            
            # Check if this is terminal step
            is_terminal = self._is_terminated(next_state, selected_agent.name)
            
            # Calculate task success for terminal step
            task_successful = False
            if is_terminal:
                task_successful = (
                    selected_agent.name == "TerminatorAgent" and 
                    current_state.current_step < current_state.max_steps
                )
            
            # Calculate reward
            reward = calculate_reward(
                agent_output, 
                is_terminal, 
                task_successful, 
                self.reward_config
            )
            total_reward += reward
            
            # Get next state embedding (None if terminal)
            next_state_embedding = None if is_terminal else self.policy._embed_state(next_state)
            
            # Create trajectory step
            step = TrajectoryStep(
                state_embedding=state_embedding,
                agent_index=agent_idx,
                log_prob=log_prob,
                reward=reward,
                next_state_embedding=next_state_embedding,
                is_terminal_step=is_terminal
            )
            trajectory_steps.append(step)
            
            # Move to next state
            current_state = next_state
        
        # Determine overall task success
        final_task_successful = (
            len(trajectory_steps) > 0 and 
            trajectory_steps[-1].is_terminal_step and
            self.agents[trajectory_steps[-1].agent_index].name == "TerminatorAgent" and
            current_state.current_step <= current_state.max_steps
        )
        
        return EpisodeTrajectory(
            steps=trajectory_steps,
            total_undiscounted_reward=total_reward,
            task_successful=final_task_successful
        )
    
    def _update_state(
        self, 
        current_state: SystemState, 
        agent_name: str, 
        agent_output: AgentOutput
    ) -> SystemState:
        """
        Update system state after agent execution.
        
        Args:
            current_state: Current system state
            agent_name: Name of the executed agent
            agent_output: Output from the agent
            
        Returns:
            Updated system state
        """
        new_history = current_state.history + [(agent_name, agent_output)]
        
        return SystemState(
            task_specification=current_state.task_specification,
            history=new_history,
            current_step=current_state.current_step + 1,
            max_steps=current_state.max_steps,
            custom_data=current_state.custom_data
        )
    
    def _is_terminated(self, current_state: SystemState, last_agent_name: str) -> bool:
        """
        Check if episode should terminate.
        
        Args:
            current_state: Current system state
            last_agent_name: Name of the last executed agent
            
        Returns:
            True if episode should terminate
        """
        # Terminate if max steps reached
        if current_state.current_step >= current_state.max_steps:
            return True
        
        # Terminate if TerminatorAgent was selected
        if last_agent_name == "TerminatorAgent":
            return True
        
        return False
