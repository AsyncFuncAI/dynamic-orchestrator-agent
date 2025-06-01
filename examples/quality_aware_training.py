#!/usr/bin/env python3
"""
Quality-aware training example for the DOA Framework.

This example demonstrates how to train the orchestrator to actually solve tasks
by evaluating the quality of agent outputs, not just termination behavior.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
import random
import re
from typing import List, Tuple

from doa_framework import (
    AgentInterface, TerminatorAgent, Orchestrator, 
    PolicyNetwork, REINFORCETrainer, SystemState, 
    RewardConfig, AgentOutput, EpisodeTrajectory
)


class QualityAwareOrchestrator(Orchestrator):
    """Extended orchestrator that evaluates task completion quality."""
    
    def run_episode(self, initial_state: SystemState) -> EpisodeTrajectory:
        """Run episode with quality-aware reward calculation."""
        # Run the standard episode
        trajectory = super().run_episode(initial_state)
        
        # Evaluate actual task completion quality
        task_quality_score = self._evaluate_task_quality(
            initial_state.task_specification, 
            trajectory
        )
        
        # Adjust the final step reward based on quality
        if trajectory.steps:
            final_step = trajectory.steps[-1]
            
            # Replace the terminal reward with quality-based reward
            quality_bonus = task_quality_score * 5.0  # Scale quality to reward
            cost_penalty = self.reward_config.lambda_cost_penalty * sum(
                self._get_step_cost(step, trajectory) for step in trajectory.steps
            )
            
            # Update final step reward
            final_step.reward = quality_bonus - cost_penalty
            
            # Recalculate total reward
            trajectory.total_undiscounted_reward = sum(step.reward for step in trajectory.steps)
            
            # Update success based on quality
            trajectory.task_successful = task_quality_score > 0.7
        
        return trajectory
    
    def _get_step_cost(self, step, trajectory) -> float:
        """Get the cost for a step by re-executing the agent (simplified)."""
        # In a real implementation, you'd store this in the trajectory
        # For now, use default costs based on agent type
        agent = self.agents[step.agent_index]
        if agent.name == "MathSolverAgent":
            return 2.0
        elif agent.name == "CreativeWriterAgent":
            return 2.5
        else:  # TerminatorAgent
            return 0.0
    
    def _evaluate_task_quality(self, task: str, trajectory: EpisodeTrajectory) -> float:
        """Evaluate how well the task was completed (0.0 to 1.0)."""
        task_lower = task.lower()
        
        # Collect all agent outputs from the trajectory
        agent_outputs = []
        for step in trajectory.steps:
            agent = self.agents[step.agent_index]
            if agent.name != "TerminatorAgent":
                # Simulate agent execution to get output content
                # In practice, you'd store this in the trajectory
                state = SystemState(task, [], 0, 4)  # Simplified state
                output = agent.execute(state)
                agent_outputs.append(output)
        
        if not agent_outputs:
            return 0.0  # No actual work done
        
        # Evaluate based on task type
        best_score = 0.0
        
        for output in agent_outputs:
            content = str(output.content).lower()
            
            # Math problem evaluation
            if "+" in task_lower and "=" in content:
                # Extract expected and actual results
                task_numbers = [int(x) for x in re.findall(r'\d+', task)]
                if len(task_numbers) >= 2:
                    expected = sum(task_numbers)
                    if str(expected) in content:
                        best_score = max(best_score, 1.0)  # Perfect math
                    elif "solution" in content:
                        best_score = max(best_score, 0.6)  # Attempted solution
            
            elif "factorial" in task_lower and ("factorial" in content or "!" in content):
                # Check factorial calculation
                task_numbers = [int(x) for x in re.findall(r'\d+', task)]
                if task_numbers:
                    n = task_numbers[0]
                    expected_factorial = 1
                    for i in range(1, n + 1):
                        expected_factorial *= i
                    if str(expected_factorial) in content:
                        best_score = max(best_score, 1.0)  # Perfect factorial
                    elif "factorial" in content:
                        best_score = max(best_score, 0.7)  # Attempted factorial
            
            # Creative writing evaluation
            elif "haiku" in task_lower:
                lines = content.split('\n')
                # Simple haiku structure check (3 lines)
                if len([line for line in lines if line.strip()]) >= 3:
                    best_score = max(best_score, 0.9)  # Good haiku structure
                elif len(content) > 30:
                    best_score = max(best_score, 0.6)  # Some creative content
            
            elif "explain" in task_lower:
                # Explanation quality based on length and keywords
                if len(content) > 50 and any(word in content for word in ["learn", "pattern", "data", "algorithm"]):
                    best_score = max(best_score, 0.9)  # Good explanation
                elif len(content) > 30:
                    best_score = max(best_score, 0.6)  # Basic explanation
        
        return best_score


# Import the custom agents from the previous example
class MathSolverAgent(AgentInterface):
    """Agent specialized in solving simple math problems."""
    
    def __init__(self, name: str = "MathSolverAgent"):
        super().__init__(name)
    
    def execute(self, state: SystemState) -> AgentOutput:
        task = state.task_specification.lower()
        
        if "+" in task:
            numbers = re.findall(r'\d+', task)
            if len(numbers) >= 2:
                result = sum(int(n) for n in numbers)
                return AgentOutput(
                    content=f"Math solution: {' + '.join(numbers)} = {result}",
                    cost=2.0,
                    metadata={"operation": "addition", "numbers": numbers}
                )
        
        elif "factorial" in task:
            numbers = re.findall(r'\d+', task)
            if numbers:
                n = int(numbers[0])
                if n <= 10:
                    factorial = 1
                    for i in range(1, n + 1):
                        factorial *= i
                    return AgentOutput(
                        content=f"Factorial solution: {n}! = {factorial}",
                        cost=3.0,
                        metadata={"operation": "factorial", "input": n}
                    )
        
        return AgentOutput(
            content=f"Math analysis: No clear mathematical operation found",
            cost=1.5,
            metadata={"operation": "analysis", "pattern_found": False}
        )


class CreativeWriterAgent(AgentInterface):
    """Agent specialized in creative writing tasks."""
    
    def __init__(self, name: str = "CreativeWriterAgent"):
        super().__init__(name)
    
    def execute(self, state: SystemState) -> AgentOutput:
        task = state.task_specification.lower()
        
        if "haiku" in task:
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
            if "machine learning" in task:
                explanation = ("Machine learning teaches computers to recognize patterns, "
                             "like how children learn to distinguish cats from dogs through examples. "
                             "Algorithms build models that help make predictions about new data.")
            else:
                explanation = f"Creative explanation: Understanding patterns and relationships in {task}"
            
            return AgentOutput(
                content=explanation,
                cost=2.0,
                metadata={"type": "explanation", "topic": task}
            )
        
        return AgentOutput(
            content=f"Creative response: I can help with writing and explanations",
            cost=1.0,
            metadata={"type": "general"}
        )


def main():
    """Quality-aware training demonstration."""
    
    print("🎯 DOA Framework - Quality-Aware Training")
    print("=" * 60)
    
    # Configuration
    N_EPOCHS = 25
    K_EPISODES_PER_EPOCH = 6
    STATE_EMBEDDING_DIM = 64
    HIDDEN_DIM = 128
    LEARNING_RATE = 0.003
    MAX_STEPS = 4
    
    # Initialize agents
    agents = [
        MathSolverAgent(),
        CreativeWriterAgent(),
        TerminatorAgent()
    ]
    
    print(f"🤖 Agents: {[agent.name for agent in agents]}")
    
    # Initialize components with quality-aware orchestrator
    reward_config = RewardConfig(
        lambda_cost_penalty=0.1,
        gamma_discount_factor=0.95,
        task_success_bonus=0.0,  # Will be replaced by quality score
        task_failure_penalty=0.0
    )
    
    policy_network = PolicyNetwork(STATE_EMBEDDING_DIM, len(agents), HIDDEN_DIM)
    optimizer = optim.Adam(policy_network.parameters(), lr=LEARNING_RATE)
    
    # Use quality-aware orchestrator
    orchestrator = QualityAwareOrchestrator(agents, policy_network, reward_config)
    trainer = REINFORCETrainer(policy_network, optimizer, reward_config)
    
    print(f"🧠 Policy network: {sum(p.numel() for p in policy_network.parameters())} parameters")
    print("⚙️  Using quality-aware reward evaluation")
    print()
    
    # Diverse tasks that require different agents
    tasks = [
        "Solve: 12 + 8 = ?",
        "Calculate factorial of 4",
        "Write a haiku about programming",
        "Explain machine learning",
        "Solve: 25 + 15 = ?",
        "Write a haiku about debugging"
    ]
    
    # Training loop
    for epoch in range(N_EPOCHS):
        batch_trajectories = []
        epoch_rewards = []
        agent_usage = {agent.name: 0 for agent in agents}
        
        for episode in range(K_EPISODES_PER_EPOCH):
            task = random.choice(tasks)
            initial_state = SystemState(task, [], 0, MAX_STEPS)
            
            trajectory = orchestrator.run_episode(initial_state)
            batch_trajectories.append(trajectory)
            epoch_rewards.append(trajectory.total_undiscounted_reward)
            
            # Track agent usage
            for step in trajectory.steps:
                agent_usage[agents[step.agent_index].name] += 1
        
        # Train and report
        avg_reward = sum(epoch_rewards) / len(epoch_rewards)
        success_rate = sum(1 for t in batch_trajectories if t.task_successful) / len(batch_trajectories)
        loss = trainer.train_batch(batch_trajectories)
        
        print(f"Epoch {epoch+1:2d}/{N_EPOCHS} | "
              f"Avg Reward: {avg_reward:6.3f} | "
              f"Success: {success_rate:5.1%} | "
              f"Loss: {loss:7.4f}")
        
        # Show agent usage every 5 epochs
        if (epoch + 1) % 5 == 0:
            total_usage = sum(agent_usage.values())
            if total_usage > 0:
                print(f"  📊 Agent usage: ", end="")
                for name, count in agent_usage.items():
                    pct = count / total_usage * 100
                    print(f"{name}: {pct:.1f}% ", end="")
                print()
    
    print("\n" + "=" * 60)
    print("🎉 Quality-Aware Training Completed!")
    
    # Final evaluation
    print("\n🔍 Final Evaluation:")
    test_tasks = [
        "Solve: 30 + 45 = ?",
        "Write a haiku about machine learning", 
        "Calculate factorial of 5",
        "Explain how neural networks work"
    ]
    
    for task in test_tasks:
        initial_state = SystemState(task, [], 0, MAX_STEPS)
        trajectory = orchestrator.run_episode(initial_state)
        
        agent_sequence = [agents[step.agent_index].name for step in trajectory.steps]
        quality_score = orchestrator._evaluate_task_quality(task, trajectory)
        
        print(f"  Task: '{task}'")
        print(f"    Agents: {' → '.join(agent_sequence)}")
        print(f"    Quality: {quality_score:.2f} | Reward: {trajectory.total_undiscounted_reward:.2f}")
        print()
    
    print("🎭 The orchestrator learned to select agents based on task quality!")


if __name__ == "__main__":
    main()
