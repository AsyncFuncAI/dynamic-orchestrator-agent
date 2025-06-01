"""
Reward calculation for the DOA Framework.
"""

from .structs import AgentOutput, RewardConfig


def calculate_reward(
    agent_output: AgentOutput, 
    is_terminal: bool, 
    task_was_successful: bool, 
    reward_config: RewardConfig
) -> float:
    """
    Calculate reward for an agent's action.
    
    Implements the reward function from the paper:
    - Terminal step: (success_bonus or failure_penalty) - λ * cost
    - Intermediate step: -λ * cost
    
    Args:
        agent_output: Output from the executed agent
        is_terminal: Whether this is the terminal step
        task_was_successful: Whether the task was completed successfully (only relevant for terminal)
        reward_config: Reward configuration parameters
        
    Returns:
        Calculated reward value
    """
    # Base cost penalty
    cost_penalty = (
        reward_config.lambda_cost_penalty * 
        agent_output.cost * 
        reward_config.step_cost_scale_factor
    )
    
    if is_terminal:
        # Terminal reward: success/failure bonus/penalty minus cost
        if task_was_successful:
            base_reward = reward_config.task_success_bonus
        else:
            base_reward = reward_config.task_failure_penalty
        
        return base_reward - cost_penalty
    else:
        # Intermediate step: only cost penalty
        return -cost_penalty
