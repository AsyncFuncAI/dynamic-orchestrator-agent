#!/usr/bin/env python3
"""
🤖 DOA Framework - Simple LLM Agents Demo

This shows DOA learning to route between different types of LLM agents.
Works without API keys - uses mock responses to demonstrate the concept.

To use with real LLMs:
1. pip install google-generativeai
2. export GOOGLE_API_KEY='your-key'
3. Set USE_REAL_LLM = True below

Run this to see DOA learning LLM routing patterns!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import torch
import torch.optim as optim
import random
from doa_framework import *

# Set to True if you have Gemini API key
USE_REAL_LLM = False

if USE_REAL_LLM:
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
        print("✅ Using real Gemini LLM")
    except:
        USE_REAL_LLM = False
        print("⚠️ Falling back to mock LLM responses")


class QuickLLMAgent(AgentInterface):
    """Fast LLM for simple questions."""
    
    def __init__(self):
        super().__init__("QuickLLM")
    
    def execute(self, state: SystemState) -> AgentOutput:
        request = json.loads(state.task_specification)
        query = request.get("query", "")
        
        if USE_REAL_LLM:
            prompt = f"Give a brief answer (1-2 sentences): {query}"
            response = model.generate_content(prompt)
            result = response.text
        else:
            # Mock quick response
            result = f"Quick answer: {query[:50]}... [Brief explanation]"
        
        return AgentOutput(
            content=result,
            cost=0.5,  # Cheap and fast
            metadata={"type": "quick"}
        )


class DeepLLMAgent(AgentInterface):
    """Thorough LLM for complex analysis."""
    
    def __init__(self):
        super().__init__("DeepLLM")
    
    def execute(self, state: SystemState) -> AgentOutput:
        request = json.loads(state.task_specification)
        query = request.get("query", "")
        
        if USE_REAL_LLM:
            prompt = f"""Provide detailed analysis of: {query}
            
Include:
1. Key considerations
2. Multiple perspectives  
3. Recommendations"""
            response = model.generate_content(prompt)
            result = response.text
        else:
            # Mock detailed response
            result = f"""Detailed analysis of: {query}

1. Key considerations: [Comprehensive analysis]
2. Multiple perspectives: [Different viewpoints]
3. Recommendations: [Actionable insights]

[Detailed explanation continues...]"""
        
        return AgentOutput(
            content=result,
            cost=2.0,  # Expensive but thorough
            metadata={"type": "deep"}
        )


class CreativeLLMAgent(AgentInterface):
    """Creative LLM for writing and brainstorming."""
    
    def __init__(self):
        super().__init__("CreativeLLM")
    
    def execute(self, state: SystemState) -> AgentOutput:
        request = json.loads(state.task_specification)
        query = request.get("query", "")
        
        if USE_REAL_LLM:
            prompt = f"Be creative and imaginative: {query}"
            response = model.generate_content(prompt)
            result = response.text
        else:
            # Mock creative response
            creative_elements = [
                "Once upon a time...",
                "Imagine a world where...",
                "Picture this scenario...",
                "In a creative twist..."
            ]
            result = f"{random.choice(creative_elements)} {query}\n\n[Creative story/response continues with vivid imagery and imagination...]"
        
        return AgentOutput(
            content=result,
            cost=1.5,  # Medium cost for creativity
            metadata={"type": "creative"}
        )


class LLMOrchestrator(Orchestrator):
    """Learns which LLM agent to use for different query types."""
    
    def run_episode(self, initial_state):
        trajectory = super().run_episode(initial_state)
        
        # Reward based on using the right agent for the right task
        quality_score = self._evaluate_agent_choice(initial_state.task_specification, trajectory)
        
        if trajectory.steps:
            # Simple reward: high bonus for good agent choice
            reward = quality_score * 5.0
            trajectory.steps[-1].reward = reward
            trajectory.total_undiscounted_reward = reward
            trajectory.task_successful = quality_score > 0.8
        
        return trajectory
    
    def _evaluate_agent_choice(self, task_spec, trajectory):
        """Reward using the right agent for the task type."""
        try:
            request = json.loads(task_spec)
            query_type = request.get("type", "unknown")
            agent_names = [self.agents[step.agent_index].name for step in trajectory.steps]
            
            # Perfect matches get high scores
            if query_type == "quick" and "QuickLLM" in agent_names:
                return 1.0
            elif query_type == "analysis" and "DeepLLM" in agent_names:
                return 1.0
            elif query_type == "creative" and "CreativeLLM" in agent_names:
                return 1.0
            else:
                return 0.3  # Wrong agent choice
                
        except:
            return 0.0


def demo():
    """Demo DOA learning LLM routing."""
    
    print("🤖 DOA Framework - LLM Agents Demo")
    print("=" * 40)
    
    # Create LLM agents
    agents = [
        QuickLLMAgent(),
        DeepLLMAgent(), 
        CreativeLLMAgent(),
        TerminatorAgent()
    ]
    
    print("🧠 LLM Agents:")
    print("   • QuickLLM: Fast answers to simple questions")
    print("   • DeepLLM: Thorough analysis for complex topics")
    print("   • CreativeLLM: Creative writing and brainstorming")
    
    # DOA setup
    policy = PolicyNetwork(32, len(agents), 64)
    orchestrator = LLMOrchestrator(agents, policy, RewardConfig())
    trainer = REINFORCETrainer(policy, optim.Adam(policy.parameters()), RewardConfig())
    
    # Test queries
    test_queries = [
        {"type": "quick", "query": "What is Python?"},
        {"type": "analysis", "query": "Analyze the impact of AI on society"},
        {"type": "creative", "query": "Write a story about a time-traveling robot"},
        {"type": "quick", "query": "Define machine learning"},
        {"type": "analysis", "query": "Compare remote work vs office work"},
        {"type": "creative", "query": "Brainstorm ideas for a mobile app"}
    ]
    
    print("\n📋 BEFORE Training (random routing):")
    for query in test_queries[:3]:
        task_spec = json.dumps(query)
        state = SystemState(task_spec, [], 0, 3)
        trajectory = orchestrator.run_episode(state)
        agent_sequence = [agents[step.agent_index].name for step in trajectory.steps]
        print(f"   {query['type']} → {' → '.join(agent_sequence)}")
    
    # Training
    print(f"\n🔄 Training DOA...")
    
    for i in range(30):
        query = random.choice(test_queries)
        task_spec = json.dumps(query)
        state = SystemState(task_spec, [], 0, 3)
        trajectory = orchestrator.run_episode(state)
        
        # Train every few episodes
        if i % 5 == 4:
            trainer.train_batch([trajectory])
            print(f"   Episode {i+1}: Training...")
    
    print("✅ Training complete!")
    
    print("\n📋 AFTER Training (learned routing):")
    for query in test_queries[:3]:
        task_spec = json.dumps(query)
        state = SystemState(task_spec, [], 0, 3)
        trajectory = orchestrator.run_episode(state)
        agent_sequence = [agents[step.agent_index].name for step in trajectory.steps]
        quality = orchestrator._evaluate_agent_choice(task_spec, trajectory)
        print(f"   {query['type']} → {' → '.join(agent_sequence)} (Quality: {quality:.1f})")
    
    print(f"\n🎉 DOA learned to route:")
    print(f"   ✨ Quick questions → QuickLLM")
    print(f"   ✨ Analysis tasks → DeepLLM")
    print(f"   ✨ Creative requests → CreativeLLM")
    
    # Show actual LLM responses
    print(f"\n🤖 Sample LLM Responses:")
    sample_query = {"type": "creative", "query": "Write about a robot learning to paint"}
    task_spec = json.dumps(sample_query)
    state = SystemState(task_spec, [], 0, 3)
    trajectory = orchestrator.run_episode(state)
    
    # Get response from the selected agent
    for step in trajectory.steps:
        agent = agents[step.agent_index]
        if agent.name != "TerminatorAgent":
            output = agent.execute(state)
            print(f"\n{agent.name} responds:")
            print(f"{output.content[:200]}...")
            break
    
    print(f"\n🎯 To use with real LLMs:")
    print(f"1. pip install google-generativeai")
    print(f"2. export GOOGLE_API_KEY='your-key'")
    print(f"3. Set USE_REAL_LLM = True in this file")


if __name__ == "__main__":
    demo()
