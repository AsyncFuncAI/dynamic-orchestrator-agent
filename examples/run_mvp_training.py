#!/usr/bin/env python3
"""
MVP training script for the Dynamic Orchestrator Agent Framework.

This script demonstrates the basic training loop for the DOA framework,
showing how the orchestrator learns to select agents more effectively over time.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
from typing import List
import numpy as np

from doa_framework import (
    AgentInterface, TerminatorAgent, EchoAgent, Orchestrator, 
    PolicyNetwork, REINFORCETrainer, SystemState, RewardConfig
)


def create_test_task(task_id: int) -> str:
    """Create a simple test task specification."""
    tasks = [
        "Solve the math problem: 2 + 2 = ?",
        "Explain what is machine learning",
        "Write a haiku about programming",
        "Calculate the factorial of 5",
        "Describe the water cycle"
    ]
    return tasks[task_id % len(tasks)]


def main():
    """Main training loop for the MVP."""
    
    # Configuration
    N_EPOCHS = 50
    K_EPISODES_PER_EPOCH = 10
    STATE_EMBEDDING_DIM = 64
    HIDDEN_DIM = 128
    LEARNING_RATE = 0.001
    MAX_STEPS = 4
    
    print("🚀 Starting DOA Framework MVP Training")
    print(f"Epochs: {N_EPOCHS}, Episodes per epoch: {K_EPISODES_PER_EPOCH}")
    print(f"State embedding dim: {STATE_EMBEDDING_DIM}, Hidden dim: {HIDDEN_DIM}")
    print(f"Learning rate: {LEARNING_RATE}, Max steps: {MAX_STEPS}")
    print("-" * 60)
    
    # Initialize agents
    agents: List[AgentInterface] = [
        EchoAgent(name="EchoAgent"),
        TerminatorAgent(name="TerminatorAgent")
    ]
    print(f"Initialized {len(agents)} agents: {[agent.name for agent in agents]}")
    
    # Initialize reward configuration
    reward_config = RewardConfig(
        lambda_cost_penalty=0.1,
        gamma_discount_factor=0.99,
        task_success_bonus=1.0,
        task_failure_penalty=-1.0,
        step_cost_scale_factor=1.0
    )
    print(f"Reward config: λ={reward_config.lambda_cost_penalty}, γ={reward_config.gamma_discount_factor}")
    
    # Initialize policy network
    policy_network = PolicyNetwork(
        state_embedding_dim=STATE_EMBEDDING_DIM,
        num_agents=len(agents),
        hidden_dim=HIDDEN_DIM
    )
    print(f"Policy network: {sum(p.numel() for p in policy_network.parameters())} parameters")
    
    # Initialize optimizer
    optimizer = optim.Adam(policy_network.parameters(), lr=LEARNING_RATE)
    
    # Initialize orchestrator and trainer
    orchestrator = Orchestrator(agents, policy_network, reward_config)
    trainer = REINFORCETrainer(policy_network, optimizer, reward_config)
    
    print("All components initialized successfully!")
    print("=" * 60)
    
    # Training loop
    best_avg_reward = float('-inf')
    
    for epoch in range(N_EPOCHS):
        batch_trajectories = []
        epoch_rewards = []
        epoch_success_rate = 0.0
        
        # Collect episodes for this epoch
        for episode in range(K_EPISODES_PER_EPOCH):
            # Create initial system state
            task_spec = create_test_task(epoch * K_EPISODES_PER_EPOCH + episode)
            initial_state = SystemState(
                task_specification=task_spec,
                history=[],
                current_step=0,
                max_steps=MAX_STEPS
            )
            
            # Run episode
            trajectory = orchestrator.run_episode(initial_state)
            batch_trajectories.append(trajectory)
            
            # Track metrics
            epoch_rewards.append(trajectory.total_undiscounted_reward)
            if trajectory.task_successful:
                epoch_success_rate += 1.0
        
        # Calculate epoch metrics
        avg_reward = np.mean(epoch_rewards)
        success_rate = epoch_success_rate / K_EPISODES_PER_EPOCH
        
        # Train on batch
        loss = trainer.train_batch(batch_trajectories)
        
        # Update best reward
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_marker = " 🌟"
        else:
            best_marker = ""
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{N_EPOCHS} | "
              f"Avg Reward: {avg_reward:6.3f}{best_marker} | "
              f"Success Rate: {success_rate:5.1%} | "
              f"Loss: {loss:8.5f}")
        
        # Detailed logging every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"  📊 Detailed stats:")
            print(f"     Reward range: [{min(epoch_rewards):.3f}, {max(epoch_rewards):.3f}]")
            print(f"     Reward std: {np.std(epoch_rewards):.3f}")
            
            # Analyze agent selection patterns
            agent_selections = {}
            for traj in batch_trajectories:
                for step in traj.steps:
                    agent_name = agents[step.agent_index].name
                    agent_selections[agent_name] = agent_selections.get(agent_name, 0) + 1
            
            total_selections = sum(agent_selections.values())
            print(f"     Agent usage: ", end="")
            for agent_name, count in agent_selections.items():
                percentage = count / total_selections * 100
                print(f"{agent_name}: {percentage:.1f}% ", end="")
            print()
            print()
    
    print("=" * 60)
    print("🎉 Training completed!")
    print(f"Best average reward achieved: {best_avg_reward:.3f}")
    
    # Final evaluation
    print("\n🔍 Final Policy Evaluation:")
    test_trajectories = []
    for i in range(5):
        task_spec = create_test_task(i)
        initial_state = SystemState(
            task_specification=task_spec,
            history=[],
            current_step=0,
            max_steps=MAX_STEPS
        )
        trajectory = orchestrator.run_episode(initial_state)
        test_trajectories.append(trajectory)
        
        print(f"  Test {i+1}: Task='{task_spec[:30]}...' | "
              f"Reward={trajectory.total_undiscounted_reward:.3f} | "
              f"Success={trajectory.task_successful} | "
              f"Steps={len(trajectory.steps)}")
    
    final_avg_reward = np.mean([t.total_undiscounted_reward for t in test_trajectories])
    final_success_rate = np.mean([t.task_successful for t in test_trajectories])
    
    print(f"\n📈 Final Performance:")
    print(f"   Average Reward: {final_avg_reward:.3f}")
    print(f"   Success Rate: {final_success_rate:.1%}")
    print(f"   Policy learned to balance exploration and exploitation!")


if __name__ == "__main__":
    main()
