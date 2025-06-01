#!/usr/bin/env python3
"""
Example demonstrating how to create and use custom agents in the DOA Framework.

This example shows how to:
1. Create custom agents with different reasoning patterns
2. Configure the orchestrator with multiple agent types
3. Train the policy to learn optimal agent selection
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
import random
import re
from typing import List

from doa_framework import (
    AgentInterface, TerminatorAgent, Orchestrator, 
    PolicyNetwork, REINFORCETrainer, SystemState, 
    RewardConfig, AgentOutput
)


class MathSolverAgent(AgentInterface):
    """Agent specialized in solving simple math problems."""
    
    def __init__(self, name: str = "MathSolverAgent"):
        super().__init__(name)
    
    def execute(self, state: SystemState) -> AgentOutput:
        """Attempt to solve math problems in the task specification."""
        task = state.task_specification.lower()
        
        # Simple pattern matching for basic math
        if "+" in task:
            # Extract numbers and add them
            numbers = re.findall(r'\d+', task)
            if len(numbers) >= 2:
                result = sum(int(n) for n in numbers)
                return AgentOutput(
                    content=f"Math solution: {' + '.join(numbers)} = {result}",
                    cost=2.0,  # Higher cost for specialized processing
                    metadata={"operation": "addition", "numbers": numbers}
                )
        
        elif "factorial" in task:
            numbers = re.findall(r'\d+', task)
            if numbers:
                n = int(numbers[0])
                if n <= 10:  # Safety limit
                    factorial = 1
                    for i in range(1, n + 1):
                        factorial *= i
                    return AgentOutput(
                        content=f"Factorial solution: {n}! = {factorial}",
                        cost=3.0,
                        metadata={"operation": "factorial", "input": n}
                    )
        
        # If no math pattern found, return generic response
        return AgentOutput(
            content=f"Math analysis: No clear mathematical operation found in '{task}'",
            cost=1.5,
            metadata={"operation": "analysis", "pattern_found": False}
        )


class CreativeWriterAgent(AgentInterface):
    """Agent specialized in creative writing tasks."""
    
    def __init__(self, name: str = "CreativeWriterAgent"):
        super().__init__(name)
    
    def execute(self, state: SystemState) -> AgentOutput:
        """Generate creative content based on the task."""
        task = state.task_specification.lower()
        
        if "haiku" in task:
            # Generate a simple haiku about the topic
            if "programming" in task:
                haiku = "Code flows like water\nBugs hide in silent shadows\nDebug brings the light"
            elif "machine learning" in task:
                haiku = "Data speaks in math\nAlgorithms learn and grow wise\nAI dreams in code"
            else:
                haiku = "Words dance on the page\nCreativity blooms bright\nArt from simple thoughts"
            
            return AgentOutput(
                content=f"Haiku creation:\n{haiku}",
                cost=2.5,
                metadata={"type": "haiku", "topic": task}
            )
        
        elif "explain" in task:
            # Provide creative explanations
            if "machine learning" in task:
                explanation = ("Machine learning is like teaching a computer to recognize patterns, "
                             "similar to how a child learns to distinguish cats from dogs by seeing "
                             "many examples. The computer builds internal models that help it make "
                             "predictions about new, unseen data.")
            else:
                explanation = f"Creative explanation: {task} involves understanding patterns and relationships."
            
            return AgentOutput(
                content=explanation,
                cost=2.0,
                metadata={"type": "explanation", "topic": task}
            )
        
        return AgentOutput(
            content=f"Creative response: I can help with writing, poetry, and explanations about '{task}'",
            cost=1.0,
            metadata={"type": "general", "capability": "creative_writing"}
        )


def create_diverse_tasks() -> List[str]:
    """Create a diverse set of tasks to test different agents."""
    return [
        "Solve the math problem: 15 + 27 = ?",
        "Calculate the factorial of 4",
        "Write a haiku about programming",
        "Explain what is machine learning",
        "Solve: 100 + 200 + 300",
        "Write a haiku about machine learning", 
        "What is 8 + 12?",
        "Explain how computers work",
        "Calculate factorial of 6",
        "Write a poem about debugging"
    ]


def evaluate_task_success(task: str, agent_outputs: List[AgentOutput]) -> bool:
    """Simple heuristic to evaluate if a task was completed successfully."""
    task_lower = task.lower()
    
    # Check if any agent provided a relevant response
    for output in agent_outputs:
        content = str(output.content).lower()
        
        # Math tasks
        if ("+" in task_lower or "factorial" in task_lower) and ("=" in content or "solution" in content):
            return True
        
        # Creative tasks  
        if ("haiku" in task_lower or "write" in task_lower) and len(content) > 20:
            return True
        
        # Explanation tasks
        if "explain" in task_lower and len(content) > 30:
            return True
    
    return False


def main():
    """Main training loop with custom agents."""
    
    print("🎨 DOA Framework - Custom Agent Training Example")
    print("=" * 60)
    
    # Configuration
    N_EPOCHS = 30
    K_EPISODES_PER_EPOCH = 8
    STATE_EMBEDDING_DIM = 64
    HIDDEN_DIM = 128
    LEARNING_RATE = 0.002
    MAX_STEPS = 6  # Allow more steps for complex tasks
    
    # Initialize diverse agent pool
    agents: List[AgentInterface] = [
        MathSolverAgent(),
        CreativeWriterAgent(), 
        TerminatorAgent()
    ]
    
    print(f"🤖 Initialized {len(agents)} agents:")
    for i, agent in enumerate(agents):
        print(f"   {i}: {agent.name}")
    
    # Initialize components
    reward_config = RewardConfig(
        lambda_cost_penalty=0.15,  # Slightly higher cost penalty
        gamma_discount_factor=0.95,
        task_success_bonus=2.0,    # Higher success bonus
        task_failure_penalty=-1.5
    )
    
    policy_network = PolicyNetwork(
        state_embedding_dim=STATE_EMBEDDING_DIM,
        num_agents=len(agents),
        hidden_dim=HIDDEN_DIM
    )
    
    optimizer = optim.Adam(policy_network.parameters(), lr=LEARNING_RATE)
    orchestrator = Orchestrator(agents, policy_network, reward_config)
    trainer = REINFORCETrainer(policy_network, optimizer, reward_config)
    
    print(f"🧠 Policy network: {sum(p.numel() for p in policy_network.parameters())} parameters")
    print(f"⚙️  Reward config: λ={reward_config.lambda_cost_penalty}, success_bonus={reward_config.task_success_bonus}")
    print()
    
    # Training loop
    tasks = create_diverse_tasks()
    best_avg_reward = float('-inf')
    
    for epoch in range(N_EPOCHS):
        batch_trajectories = []
        epoch_rewards = []
        epoch_success_rate = 0.0
        agent_usage = {agent.name: 0 for agent in agents}
        
        # Collect episodes
        for episode in range(K_EPISODES_PER_EPOCH):
            # Select a random task
            task = random.choice(tasks)
            
            initial_state = SystemState(
                task_specification=task,
                history=[],
                current_step=0,
                max_steps=MAX_STEPS
            )
            
            # Run episode
            trajectory = orchestrator.run_episode(initial_state)
            batch_trajectories.append(trajectory)
            
            # Evaluate success using custom logic
            agent_outputs = [step.agent_index for step in trajectory.steps]
            output_contents = []
            for step in trajectory.steps:
                agent = agents[step.agent_index]
                # Reconstruct outputs for evaluation (simplified)
                if hasattr(step, 'agent_output'):
                    output_contents.append(step.agent_output)
            
            # Use trajectory success for now (can be enhanced)
            task_successful = trajectory.task_successful
            
            # Track metrics
            epoch_rewards.append(trajectory.total_undiscounted_reward)
            if task_successful:
                epoch_success_rate += 1.0
            
            # Track agent usage
            for step in trajectory.steps:
                agent_name = agents[step.agent_index].name
                agent_usage[agent_name] += 1
        
        # Calculate metrics
        avg_reward = sum(epoch_rewards) / len(epoch_rewards)
        success_rate = epoch_success_rate / K_EPISODES_PER_EPOCH
        
        # Train policy
        loss = trainer.train_batch(batch_trajectories)
        
        # Update best reward
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_marker = " 🌟"
        else:
            best_marker = ""
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{N_EPOCHS} | "
              f"Avg Reward: {avg_reward:6.3f}{best_marker} | "
              f"Success: {success_rate:5.1%} | "
              f"Loss: {loss:7.4f}")
        
        # Detailed logging every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"  📊 Agent usage: ", end="")
            total_usage = sum(agent_usage.values())
            if total_usage > 0:
                for agent_name, count in agent_usage.items():
                    percentage = count / total_usage * 100
                    print(f"{agent_name}: {percentage:.1f}% ", end="")
            print()
            print()
    
    print("=" * 60)
    print("🎉 Custom Agent Training Completed!")
    print(f"🏆 Best average reward: {best_avg_reward:.3f}")
    
    # Final evaluation with specific tasks
    print("\n🔍 Final Evaluation on Specific Tasks:")
    test_tasks = [
        "Solve: 25 + 35 = ?",
        "Write a haiku about debugging",
        "Explain machine learning",
        "Calculate factorial of 5"
    ]
    
    for i, task in enumerate(test_tasks):
        initial_state = SystemState(task, [], 0, MAX_STEPS)
        trajectory = orchestrator.run_episode(initial_state)
        
        # Show agent sequence
        agent_sequence = [agents[step.agent_index].name for step in trajectory.steps]
        
        print(f"  Task {i+1}: '{task[:40]}...'")
        print(f"    Agents used: {' → '.join(agent_sequence)}")
        print(f"    Reward: {trajectory.total_undiscounted_reward:.3f} | "
              f"Success: {trajectory.task_successful}")
        print()
    
    print("🎭 The orchestrator learned to select appropriate agents for different task types!")


if __name__ == "__main__":
    main()
