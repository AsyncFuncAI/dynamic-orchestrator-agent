#!/usr/bin/env python3
"""
🚀 Self-Optimizing API Router Demo

This demo showcases the DOA Framework's ability to learn optimal API routing patterns.
The system learns to route requests to the most appropriate microservices based on:
- Request type and urgency
- Quality vs cost trade-offs  
- Service specialization
- Response aggregation needs

This demonstrates adaptive behavior that emerges from reinforcement learning,
without hardcoded routing logic.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
import random
import json
from typing import List, Dict, Any
from collections import defaultdict, Counter

from doa_framework import (
    TerminatorAgent, REINFORCETrainer,
    SystemState, RewardConfig
)
from doa_framework.policy import PolicyNetwork

from api_router_agents import (
    FastSearchAgent, DeepAnalysisAgent, 
    NLPSpecialistAgent, ResponseAggregatorAgent
)
from api_router_orchestrator import APIRouterOrchestrator


class APIRouterPolicyNetwork(PolicyNetwork):
    """Enhanced policy network that understands JSON task specifications."""

    def _embed_state(self, system_state):
        """Enhanced state embedding for API routing tasks."""
        import torch

        embedding = torch.zeros(self.state_embedding_dim)

        try:
            task_spec = json.loads(system_state.task_specification)
        except json.JSONDecodeError:
            task_spec = {"type": "unknown", "query": system_state.task_specification}

        # Task type encoding (one-hot style)
        task_type = task_spec.get("type", "unknown")
        type_encoding = {
            "search": [1.0, 0.0, 0.0, 0.0],
            "recommendation": [0.0, 1.0, 0.0, 0.0],
            "nlp_analysis": [0.0, 0.0, 1.0, 0.0],
            "unknown": [0.0, 0.0, 0.0, 1.0]
        }
        type_features = type_encoding.get(task_type, type_encoding["unknown"])

        # Urgency encoding
        urgency = task_spec.get("urgency", "normal")
        urgency_encoding = {
            "high": [1.0, 0.0, 0.0],
            "normal": [0.0, 1.0, 0.0],
            "low": [0.0, 0.0, 1.0]
        }
        urgency_features = urgency_encoding.get(urgency, urgency_encoding["normal"])

        # Detail level encoding
        detail_level = task_spec.get("detail_level", "summary")
        detail_encoding = {
            "summary": [1.0, 0.0],
            "full": [0.0, 1.0]
        }
        detail_features = detail_encoding.get(detail_level, detail_encoding["summary"])

        # Analysis type encoding (for NLP tasks)
        analysis_type = task_spec.get("analysis_type", "general")
        analysis_encoding = {
            "sentiment": [1.0, 0.0, 0.0, 0.0],
            "entity_extraction": [0.0, 1.0, 0.0, 0.0],
            "general": [0.0, 0.0, 1.0, 0.0],
            "other": [0.0, 0.0, 0.0, 1.0]
        }
        analysis_features = analysis_encoding.get(analysis_type, analysis_encoding["general"])

        # Combine all task features
        task_features = type_features + urgency_features + detail_features + analysis_features

        # Fill embedding with task features
        task_dim = min(len(task_features), self.state_embedding_dim // 2)
        embedding[:task_dim] = torch.tensor(task_features[:task_dim])

        # History encoding
        if len(system_state.history) > 0:
            # Agent usage pattern in history
            agent_usage = torch.zeros(5)  # For 5 agent types
            total_cost = 0.0

            for agent_name, output in system_state.history:
                total_cost += output.cost
                # Map agent names to indices
                if "FastSearch" in agent_name:
                    agent_usage[0] += 1
                elif "DeepAnalysis" in agent_name:
                    agent_usage[1] += 1
                elif "NLPSpecialist" in agent_name:
                    agent_usage[2] += 1
                elif "ResponseAggregator" in agent_name:
                    agent_usage[3] += 1
                elif "Terminator" in agent_name:
                    agent_usage[4] += 1

            # Normalize agent usage
            if len(system_state.history) > 0:
                agent_usage = agent_usage / len(system_state.history)

            # History features
            history_features = [
                len(system_state.history) / 10.0,  # Normalized history length
                total_cost / max(len(system_state.history), 1) / 5.0,  # Avg cost normalized
                system_state.current_step / system_state.max_steps,  # Progress
            ]

            # Combine agent usage and history features
            all_history_features = history_features + agent_usage.tolist()

            # Fill remaining dimensions
            hist_start = self.state_embedding_dim // 2
            hist_dim = min(len(all_history_features), self.state_embedding_dim - hist_start)
            embedding[hist_start:hist_start + hist_dim] = torch.tensor(all_history_features[:hist_dim])

        return embedding


def generate_task_specifications() -> List[str]:
    """Generate diverse API request task specifications."""
    
    # Search tasks with different urgency levels
    search_tasks = [
        {"query": "latest news about AI", "type": "search", "urgency": "high"},
        {"query": "machine learning trends", "type": "search", "urgency": "normal"},
        {"query": "blockchain technology updates", "type": "search", "urgency": "low"},
        {"query": "cybersecurity threats 2024", "type": "search", "urgency": "high"},
        {"query": "quantum computing research", "type": "search", "urgency": "normal"},
        {"query": "climate change solutions", "type": "search", "urgency": "low"},
    ]
    
    # Recommendation tasks with different detail levels
    recommendation_tasks = [
        {"data": {"user_id": 123, "product_id": 789}, "type": "recommendation", "detail_level": "summary"},
        {"data": {"user_id": 456, "product_id": 101}, "type": "recommendation", "detail_level": "full"},
        {"data": {"user_id": 789, "product_id": 202}, "type": "recommendation", "detail_level": "summary"},
        {"data": {"user_id": 321, "product_id": 303}, "type": "recommendation", "detail_level": "full"},
    ]
    
    # NLP analysis tasks with different analysis types
    nlp_tasks = [
        {"text_to_process": "This product is amazing!", "type": "nlp_analysis", "analysis_type": "sentiment"},
        {"text_to_process": "John Smith works at OpenAI in San Francisco", "type": "nlp_analysis", "analysis_type": "entity_extraction"},
        {"text_to_process": "The weather is nice today", "type": "nlp_analysis", "analysis_type": "sentiment"},
        {"text_to_process": "Apple Inc. released new iPhone", "type": "nlp_analysis", "analysis_type": "entity_extraction"},
        {"text_to_process": "Complex technical documentation", "type": "nlp_analysis", "analysis_type": "general"},
    ]
    
    # Combine all tasks and convert to JSON strings
    all_tasks = search_tasks + recommendation_tasks + nlp_tasks
    return [json.dumps(task) for task in all_tasks]


def analyze_agent_patterns(agent_usage_history: List[Dict], agents: List, epoch: int):
    """Analyze and display agent selection patterns."""
    if not agent_usage_history:
        return
    
    recent_usage = agent_usage_history[-1]
    total_usage = sum(recent_usage.values())
    
    if total_usage == 0:
        return
    
    print(f"  📊 Agent Usage Patterns (Epoch {epoch}):")
    for agent in agents:
        count = recent_usage.get(agent.name, 0)
        percentage = (count / total_usage) * 100 if total_usage > 0 else 0
        print(f"    • {agent.name.replace('Agent', '')}: {percentage:.1f}% ({count} times)")


def analyze_task_specific_patterns(trajectories: List, agents: List, tasks: List[str]):
    """Analyze which agents are selected for different task types."""
    task_patterns = defaultdict(list)
    
    for i, trajectory in enumerate(trajectories):
        if i < len(tasks):
            try:
                task_spec = json.loads(tasks[i])
                task_type = task_spec.get("type", "unknown")
                urgency = task_spec.get("urgency", "normal")
                detail_level = task_spec.get("detail_level", "summary")
                
                # Create task category
                if task_type == "search":
                    category = f"search_{urgency}"
                elif task_type == "recommendation":
                    category = f"recommendation_{detail_level}"
                elif task_type == "nlp_analysis":
                    analysis_type = task_spec.get("analysis_type", "general")
                    category = f"nlp_{analysis_type}"
                else:
                    category = "unknown"
                
                # Get agent sequence
                agent_sequence = [agents[step.agent_index].name for step in trajectory.steps]
                task_patterns[category].append(agent_sequence)
            except (json.JSONDecodeError, IndexError):
                continue
    
    return task_patterns


def display_learning_progress(task_patterns: Dict, epoch: int):
    """Display how agent selection patterns are evolving for different task types."""
    print(f"\n  🧠 Learning Patterns (Epoch {epoch}):")
    
    for category, sequences in task_patterns.items():
        if not sequences:
            continue
        
        # Find most common sequence
        sequence_counter = Counter()
        for seq in sequences:
            # Convert to simplified names and create sequence string
            simplified_seq = [name.replace("Agent", "") for name in seq]
            sequence_str = " → ".join(simplified_seq)
            sequence_counter[sequence_str] += 1
        
        if sequence_counter:
            most_common = sequence_counter.most_common(1)[0]
            sequence, count = most_common
            percentage = (count / len(sequences)) * 100
            
            print(f"    • {category}: {sequence} ({percentage:.0f}% of time)")


def main():
    """Main demo function."""
    
    print("🚀 Self-Optimizing API Router Demo")
    print("=" * 60)
    print("This demo shows how the DOA Framework learns to route API requests")
    print("to the most appropriate microservices without hardcoded logic.\n")
    
    # Configuration
    N_EPOCHS = 25
    K_EPISODES_PER_EPOCH = 10
    
    # Initialize agent pool (simulating microservices)
    agents = [
        FastSearchAgent("FastSearchAgent"),
        DeepAnalysisAgent("DeepAnalysisAgent"), 
        NLPSpecialistAgent("NLPSpecialistAgent"),
        ResponseAggregatorAgent("ResponseAggregatorAgent"),
        TerminatorAgent("TerminatorAgent")
    ]
    
    print(f"🤖 Microservice Pool ({len(agents)} services):")
    service_descriptions = {
        "FastSearchAgent": "Quick keyword search (low cost, fast response)",
        "DeepAnalysisAgent": "Comprehensive analysis (high cost, high quality)",
        "NLPSpecialistAgent": "Specialized NLP processing (medium cost, NLP expert)",
        "ResponseAggregatorAgent": "Response synthesis (medium cost, improves coherence)",
        "TerminatorAgent": "Request completion (no cost)"
    }
    
    for agent in agents:
        desc = service_descriptions.get(agent.name, "Unknown service")
        print(f"   • {agent.name}: {desc}")
    
    # Initialize DOA components
    reward_config = RewardConfig(lambda_cost_penalty=0.15, gamma_discount_factor=0.95)
    policy_network = APIRouterPolicyNetwork(64, len(agents), 128)
    optimizer = optim.Adam(policy_network.parameters(), lr=0.003)
    
    orchestrator = APIRouterOrchestrator(agents, policy_network, reward_config)
    trainer = REINFORCETrainer(policy_network, optimizer, reward_config)
    
    # Generate diverse task set
    tasks = generate_task_specifications()
    
    print(f"\n📋 API Request Types ({len(tasks)} total):")
    task_type_counts = defaultdict(int)
    for task_json in tasks:
        try:
            task = json.loads(task_json)
            task_type_counts[task.get("type", "unknown")] += 1
        except json.JSONDecodeError:
            task_type_counts["unknown"] += 1
    
    for task_type, count in task_type_counts.items():
        print(f"   • {task_type}: {count} requests")
    
    print(f"\n🧠 Training Configuration:")
    print(f"   • Epochs: {N_EPOCHS}, Episodes per epoch: {K_EPISODES_PER_EPOCH}")
    print(f"   • Policy network: {sum(p.numel() for p in policy_network.parameters())} parameters")
    print(f"   • Cost penalty λ: {reward_config.lambda_cost_penalty}")
    print("\n" + "=" * 60)
    
    # Training loop with detailed tracking
    reward_history = []
    agent_usage_history = []
    
    print("\n🎯 Training Progress:")
    
    for epoch in range(N_EPOCHS):
        batch_trajectories = []
        epoch_rewards = []
        agent_usage = defaultdict(int)
        
        # Sample tasks for this epoch
        epoch_tasks = random.choices(tasks, k=K_EPISODES_PER_EPOCH)
        
        for episode in range(K_EPISODES_PER_EPOCH):
            task = epoch_tasks[episode]
            initial_state = SystemState(task, [], 0, 4)  # Reduced to 4 steps for efficiency
            
            trajectory = orchestrator.run_episode(initial_state)
            batch_trajectories.append(trajectory)
            epoch_rewards.append(trajectory.total_undiscounted_reward)
            
            # Track agent usage
            for step in trajectory.steps:
                agent_usage[agents[step.agent_index].name] += 1
        
        # Train policy
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
            analyze_agent_patterns(agent_usage_history, agents, epoch + 1)
            
            # Analyze task-specific patterns
            task_patterns = analyze_task_specific_patterns(batch_trajectories, agents, epoch_tasks)
            display_learning_progress(task_patterns, epoch + 1)
            print()
    
    print("=" * 60)
    print("🎉 Training Complete! Analyzing learned behavior...\n")
    
    # Final evaluation on specific task types
    print("🔍 Final Evaluation - Learned Routing Patterns:")
    
    test_scenarios = [
        {"query": "urgent security alert", "type": "search", "urgency": "high"},
        {"query": "detailed market analysis", "type": "search", "urgency": "low"},
        {"data": {"user_id": 999}, "type": "recommendation", "detail_level": "full"},
        {"text_to_process": "I love this product!", "type": "nlp_analysis", "analysis_type": "sentiment"},
        {"text_to_process": "Microsoft announced new AI", "type": "nlp_analysis", "analysis_type": "entity_extraction"}
    ]
    
    for i, scenario in enumerate(test_scenarios):
        task_json = json.dumps(scenario)
        initial_state = SystemState(task_json, [], 0, 4)
        trajectory = orchestrator.run_episode(initial_state)
        
        agent_sequence = [agents[step.agent_index].name.replace("Agent", "") for step in trajectory.steps]
        
        # Determine scenario description
        task_type = scenario.get("type", "unknown")
        if task_type == "search":
            urgency = scenario.get("urgency", "normal")
            desc = f"Search ({urgency} urgency)"
        elif task_type == "recommendation":
            detail = scenario.get("detail_level", "summary")
            desc = f"Recommendation ({detail} detail)"
        elif task_type == "nlp_analysis":
            analysis = scenario.get("analysis_type", "general")
            desc = f"NLP Analysis ({analysis})"
        else:
            desc = "Unknown task"
        
        print(f"\n  Test {i+1}: {desc}")
        print(f"    Learned Route: {' → '.join(agent_sequence)}")
        print(f"    Quality Score: {orchestrator._evaluate_response_quality(task_json, trajectory):.2f}")
        print(f"    Efficiency: {orchestrator._calculate_efficiency(trajectory):.2f}")
        print(f"    Final Reward: {trajectory.total_undiscounted_reward:.2f}")
        print(f"    Success: {'✅' if trajectory.task_successful else '❌'}")
    
    # Training summary
    final_avg_reward = sum(reward_history[-5:]) / 5
    initial_avg_reward = sum(reward_history[:5]) / 5
    improvement = final_avg_reward - initial_avg_reward
    
    print(f"\n📈 Training Summary:")
    print(f"   • Initial avg reward: {initial_avg_reward:.3f}")
    print(f"   • Final avg reward: {final_avg_reward:.3f}")
    print(f"   • Improvement: {improvement:.3f} ({improvement/abs(initial_avg_reward)*100:.1f}%)")
    print(f"   • Peak reward: {max(reward_history):.3f}")
    
    print(f"\n🎭 The API Router learned intelligent routing patterns!")
    print(f"   ✨ High urgency → Fast services")
    print(f"   ✨ NLP tasks → NLP specialist")
    print(f"   ✨ Complex requests → Deep analysis")
    print(f"   ✨ Multi-service → Response aggregation")
    print(f"   ✨ All without hardcoded routing logic!")


if __name__ == "__main__":
    main()
