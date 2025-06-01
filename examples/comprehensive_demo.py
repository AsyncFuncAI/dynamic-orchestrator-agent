#!/usr/bin/env python3
"""
Comprehensive demonstration of the DOA Framework capabilities.

This example showcases:
1. Multiple specialized agents with different capabilities
2. Quality-aware reward evaluation
3. Complex task types requiring agent coordination
4. Visualization of learning progress and agent selection patterns
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
import random
import re
import json
from typing import List, Dict, Any
from collections import defaultdict

from doa_framework import (
    AgentInterface, TerminatorAgent, Orchestrator, 
    PolicyNetwork, REINFORCETrainer, SystemState, 
    RewardConfig, AgentOutput, EpisodeTrajectory
)


class AdvancedMathAgent(AgentInterface):
    """Advanced mathematical problem solver."""
    
    def execute(self, state: SystemState) -> AgentOutput:
        task = state.task_specification.lower()
        
        # Addition
        if "+" in task:
            numbers = [int(x) for x in re.findall(r'\d+', task)]
            if len(numbers) >= 2:
                result = sum(numbers)
                return AgentOutput(
                    content=f"Addition: {' + '.join(map(str, numbers))} = {result}",
                    cost=1.5,
                    metadata={"operation": "addition", "result": result}
                )
        
        # Multiplication
        elif "*" in task or "×" in task:
            numbers = [int(x) for x in re.findall(r'\d+', task)]
            if len(numbers) >= 2:
                result = 1
                for n in numbers:
                    result *= n
                return AgentOutput(
                    content=f"Multiplication: {' × '.join(map(str, numbers))} = {result}",
                    cost=2.0,
                    metadata={"operation": "multiplication", "result": result}
                )
        
        # Factorial
        elif "factorial" in task:
            numbers = [int(x) for x in re.findall(r'\d+', task)]
            if numbers and numbers[0] <= 12:
                n = numbers[0]
                result = 1
                for i in range(1, n + 1):
                    result *= i
                return AgentOutput(
                    content=f"Factorial: {n}! = {result}",
                    cost=2.5,
                    metadata={"operation": "factorial", "input": n, "result": result}
                )
        
        return AgentOutput(
            content="Math analysis: No supported operation found",
            cost=1.0,
            metadata={"operation": "analysis"}
        )


class CreativeWritingAgent(AgentInterface):
    """Advanced creative writing and explanation agent."""
    
    def execute(self, state: SystemState) -> AgentOutput:
        task = state.task_specification.lower()
        
        # Haiku generation
        if "haiku" in task:
            topic_haikus = {
                "programming": "Code flows like water\nBugs hide in silent shadows\nDebug brings the light",
                "machine learning": "Data speaks in math\nAlgorithms learn and grow wise\nAI dreams in code",
                "debugging": "Errors lurk in code\nPatience guides the searching mind\nSolution emerges",
                "ai": "Silicon minds think\nPatterns emerge from chaos\nWisdom from data"
            }
            
            # Find matching topic
            haiku = "Words dance on the page\nCreativity blooms bright\nArt from simple thoughts"
            for topic, topic_haiku in topic_haikus.items():
                if topic in task:
                    haiku = topic_haiku
                    break
            
            return AgentOutput(
                content=f"Haiku:\n{haiku}",
                cost=2.0,
                metadata={"type": "haiku", "topic": task}
            )
        
        # Explanations
        elif "explain" in task:
            explanations = {
                "machine learning": "Machine learning is like teaching a computer to recognize patterns. Just as a child learns to identify animals by seeing many examples, ML algorithms learn from data to make predictions about new, unseen information.",
                "neural networks": "Neural networks are inspired by how brain neurons connect. They consist of layers of interconnected nodes that process information, learning to recognize complex patterns through training on examples.",
                "programming": "Programming is the art of giving precise instructions to computers. It involves breaking down complex problems into smaller, manageable steps that a computer can execute.",
                "ai": "Artificial Intelligence aims to create systems that can perform tasks typically requiring human intelligence, such as learning, reasoning, and problem-solving."
            }
            
            explanation = "This is a complex topic that involves understanding patterns and relationships."
            for topic, topic_explanation in explanations.items():
                if topic in task:
                    explanation = topic_explanation
                    break
            
            return AgentOutput(
                content=f"Explanation: {explanation}",
                cost=2.5,
                metadata={"type": "explanation", "topic": task}
            )
        
        return AgentOutput(
            content="I can help with creative writing, poetry, and explanations.",
            cost=1.0,
            metadata={"type": "general"}
        )


class QualityEvaluatorAgent(AgentInterface):
    """Agent that evaluates and improves upon previous outputs."""
    
    def execute(self, state: SystemState) -> AgentOutput:
        if not state.history:
            return AgentOutput(
                content="Quality evaluation: No previous outputs to evaluate.",
                cost=0.5,
                metadata={"evaluation": "no_input"}
            )
        
        # Analyze previous outputs
        evaluations = []
        for agent_name, output in state.history:
            if agent_name != "QualityEvaluatorAgent":
                content = str(output.content).lower()
                
                # Evaluate quality
                quality_score = 0.0
                feedback = []
                
                if "=" in content and any(char.isdigit() for char in content):
                    quality_score += 0.4
                    feedback.append("Contains mathematical result")
                
                if len(content) > 30:
                    quality_score += 0.3
                    feedback.append("Substantial content")
                
                if any(word in content for word in ["solution", "explanation", "haiku"]):
                    quality_score += 0.3
                    feedback.append("Structured response")
                
                evaluations.append({
                    "agent": agent_name,
                    "quality": quality_score,
                    "feedback": feedback
                })
        
        best_eval = max(evaluations, key=lambda x: x["quality"]) if evaluations else None
        
        if best_eval and best_eval["quality"] > 0.7:
            evaluation_text = f"Quality assessment: Excellent work by {best_eval['agent']} (score: {best_eval['quality']:.1f})"
        else:
            evaluation_text = "Quality assessment: Outputs could be improved with more detail or accuracy"
        
        return AgentOutput(
            content=evaluation_text,
            cost=1.5,
            metadata={"evaluations": evaluations, "best_quality": best_eval["quality"] if best_eval else 0.0}
        )


class SmartOrchestrator(Orchestrator):
    """Enhanced orchestrator with sophisticated quality evaluation."""
    
    def run_episode(self, initial_state: SystemState) -> EpisodeTrajectory:
        trajectory = super().run_episode(initial_state)
        
        # Enhanced quality evaluation
        quality_score = self._evaluate_comprehensive_quality(
            initial_state.task_specification, 
            trajectory
        )
        
        # Adjust rewards based on quality and agent coordination
        if trajectory.steps:
            # Bonus for using appropriate agents
            agent_appropriateness_bonus = self._calculate_agent_appropriateness(
                initial_state.task_specification, trajectory
            )
            
            # Update final reward
            final_step = trajectory.steps[-1]
            base_quality_reward = quality_score * 6.0
            appropriateness_bonus = agent_appropriateness_bonus * 2.0
            cost_penalty = self.reward_config.lambda_cost_penalty * sum(
                self._estimate_step_cost(step) for step in trajectory.steps
            )
            
            final_step.reward = base_quality_reward + appropriateness_bonus - cost_penalty
            trajectory.total_undiscounted_reward = sum(step.reward for step in trajectory.steps)
            trajectory.task_successful = quality_score > 0.6
        
        return trajectory
    
    def _estimate_step_cost(self, step) -> float:
        """Estimate cost based on agent type."""
        agent = self.agents[step.agent_index]
        cost_map = {
            "AdvancedMathAgent": 2.0,
            "CreativeWritingAgent": 2.5,
            "QualityEvaluatorAgent": 1.5,
            "TerminatorAgent": 0.0
        }
        return cost_map.get(agent.name, 1.0)
    
    def _calculate_agent_appropriateness(self, task: str, trajectory: EpisodeTrajectory) -> float:
        """Calculate bonus for using appropriate agents for the task."""
        task_lower = task.lower()
        agent_names = [self.agents[step.agent_index].name for step in trajectory.steps]
        
        appropriateness = 0.0
        
        # Math tasks should use math agents
        if any(op in task_lower for op in ["+", "*", "factorial"]):
            if "AdvancedMathAgent" in agent_names:
                appropriateness += 0.5
        
        # Creative tasks should use creative agents
        if any(word in task_lower for word in ["haiku", "write", "explain"]):
            if "CreativeWritingAgent" in agent_names:
                appropriateness += 0.5
        
        # Quality evaluation adds value
        if "QualityEvaluatorAgent" in agent_names and len(agent_names) > 2:
            appropriateness += 0.3
        
        return min(appropriateness, 1.0)
    
    def _evaluate_comprehensive_quality(self, task: str, trajectory: EpisodeTrajectory) -> float:
        """Comprehensive quality evaluation."""
        if not trajectory.steps:
            return 0.0
        
        task_lower = task.lower()
        quality_scores = []
        
        # Simulate agent outputs for evaluation
        for step in trajectory.steps:
            agent = self.agents[step.agent_index]
            if agent.name != "TerminatorAgent":
                # Simplified state for evaluation
                eval_state = SystemState(task, [], 0, 4)
                output = agent.execute(eval_state)
                content = str(output.content).lower()
                
                score = 0.0
                
                # Task-specific evaluation
                if "+" in task_lower:
                    numbers = [int(x) for x in re.findall(r'\d+', task)]
                    if len(numbers) >= 2:
                        expected = sum(numbers)
                        if str(expected) in content:
                            score = 1.0
                        elif "=" in content:
                            score = 0.6
                
                elif "factorial" in task_lower:
                    if "factorial" in content and "=" in content:
                        score = 0.9
                
                elif "haiku" in task_lower:
                    lines = content.split('\n')
                    if len([l for l in lines if l.strip()]) >= 3:
                        score = 0.9
                
                elif "explain" in task_lower:
                    if len(content) > 50:
                        score = 0.8
                
                quality_scores.append(score)
        
        return max(quality_scores) if quality_scores else 0.0


def main():
    """Comprehensive demonstration."""
    
    print("🎭 DOA Framework - Comprehensive Demonstration")
    print("=" * 70)
    
    # Configuration
    N_EPOCHS = 20
    K_EPISODES_PER_EPOCH = 8
    
    # Initialize sophisticated agent pool
    agents = [
        AdvancedMathAgent("AdvancedMathAgent"),
        CreativeWritingAgent("CreativeWritingAgent"),
        QualityEvaluatorAgent("QualityEvaluatorAgent"),
        TerminatorAgent("TerminatorAgent")
    ]
    
    print(f"🤖 Agent Pool ({len(agents)} agents):")
    for i, agent in enumerate(agents):
        print(f"   {i+1}. {agent.name}")
    
    # Initialize components
    reward_config = RewardConfig(lambda_cost_penalty=0.08, gamma_discount_factor=0.95)
    policy_network = PolicyNetwork(64, len(agents), 128)
    optimizer = optim.Adam(policy_network.parameters(), lr=0.002)
    
    orchestrator = SmartOrchestrator(agents, policy_network, reward_config)
    trainer = REINFORCETrainer(policy_network, optimizer, reward_config)
    
    # Diverse task set
    tasks = [
        "Solve: 15 + 25 + 10 = ?",
        "Calculate 6 * 7 * 2",
        "Calculate factorial of 6",
        "Write a haiku about AI",
        "Explain machine learning",
        "Write a haiku about debugging",
        "Explain neural networks",
        "Solve: 100 + 200 = ?"
    ]
    
    print(f"\n📋 Task Set ({len(tasks)} tasks):")
    for i, task in enumerate(tasks[:4]):  # Show first 4
        print(f"   • {task}")
    print(f"   ... and {len(tasks)-4} more")
    
    print(f"\n🧠 Training Configuration:")
    print(f"   • Epochs: {N_EPOCHS}, Episodes per epoch: {K_EPISODES_PER_EPOCH}")
    print(f"   • Policy network: {sum(p.numel() for p in policy_network.parameters())} parameters")
    print(f"   • Cost penalty λ: {reward_config.lambda_cost_penalty}")
    print("\n" + "=" * 70)
    
    # Training with detailed tracking
    agent_usage_history = []
    reward_history = []
    
    for epoch in range(N_EPOCHS):
        batch_trajectories = []
        epoch_rewards = []
        agent_usage = defaultdict(int)
        
        for episode in range(K_EPISODES_PER_EPOCH):
            task = random.choice(tasks)
            initial_state = SystemState(task, [], 0, 5)  # Allow more steps
            
            trajectory = orchestrator.run_episode(initial_state)
            batch_trajectories.append(trajectory)
            epoch_rewards.append(trajectory.total_undiscounted_reward)
            
            # Track agent usage
            for step in trajectory.steps:
                agent_usage[agents[step.agent_index].name] += 1
        
        # Train and track metrics
        avg_reward = sum(epoch_rewards) / len(epoch_rewards)
        success_rate = sum(1 for t in batch_trajectories if t.task_successful) / len(batch_trajectories)
        loss = trainer.train_batch(batch_trajectories)
        
        reward_history.append(avg_reward)
        agent_usage_history.append(dict(agent_usage))
        
        # Progress display
        print(f"Epoch {epoch+1:2d}/{N_EPOCHS} | "
              f"Reward: {avg_reward:6.3f} | "
              f"Success: {success_rate:5.1%} | "
              f"Loss: {loss:7.4f}")
        
        # Detailed analysis every 5 epochs
        if (epoch + 1) % 5 == 0:
            total_usage = sum(agent_usage.values())
            print(f"  📊 Agent distribution: ", end="")
            for name, count in agent_usage.items():
                pct = count / total_usage * 100 if total_usage > 0 else 0
                print(f"{name.replace('Agent', '')}: {pct:.0f}% ", end="")
            print()
    
    print("\n" + "=" * 70)
    print("🎉 Training Complete!")
    
    # Final comprehensive evaluation
    print("\n🔍 Final Evaluation on Diverse Tasks:")
    
    final_test_tasks = [
        "Solve: 45 + 55 = ?",
        "Calculate 8 * 9",
        "Write a haiku about programming",
        "Explain how AI works",
        "Calculate factorial of 7"
    ]
    
    for i, task in enumerate(final_test_tasks):
        initial_state = SystemState(task, [], 0, 5)
        trajectory = orchestrator.run_episode(initial_state)
        
        agent_sequence = [agents[step.agent_index].name.replace("Agent", "") for step in trajectory.steps]
        quality = orchestrator._evaluate_comprehensive_quality(task, trajectory)
        appropriateness = orchestrator._calculate_agent_appropriateness(task, trajectory)
        
        print(f"\n  Test {i+1}: '{task}'")
        print(f"    Agent sequence: {' → '.join(agent_sequence)}")
        print(f"    Quality score: {quality:.2f}")
        print(f"    Appropriateness: {appropriateness:.2f}")
        print(f"    Final reward: {trajectory.total_undiscounted_reward:.2f}")
        print(f"    Success: {'✅' if trajectory.task_successful else '❌'}")
    
    # Summary statistics
    final_avg_reward = sum(reward_history[-5:]) / 5  # Last 5 epochs
    print(f"\n📈 Training Summary:")
    print(f"   • Final average reward: {final_avg_reward:.3f}")
    print(f"   • Reward improvement: {final_avg_reward - reward_history[0]:.3f}")
    print(f"   • Peak reward: {max(reward_history):.3f}")
    
    print(f"\n🎭 The orchestrator learned sophisticated agent coordination!")
    print(f"   • Task-appropriate agent selection")
    print(f"   • Quality-aware decision making") 
    print(f"   • Multi-agent collaboration patterns")


if __name__ == "__main__":
    main()
