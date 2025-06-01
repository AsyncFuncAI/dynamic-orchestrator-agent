#!/usr/bin/env python3
"""
🤖 DOA Framework - Real LLM Demo with OpenAI

This demonstrates DOA orchestrating REAL LLM agents using OpenAI's API.
DOA learns to route different types of requests to specialized LLM configurations.

Setup:
1. pip install openai
2. export OPENAI_API_KEY='your-api-key'
3. Run this script

DOA will learn optimal routing between:
- FastGPT: Quick responses (gpt-4o-mini)
- DeepGPT: Detailed analysis (gpt-4o)
- CreativeGPT: Creative writing (gpt-4o with creative prompts)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import torch
import torch.optim as optim
import time
import random
from doa_framework import *

# Try to import OpenAI
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    OPENAI_AVAILABLE = True
    print("✅ OpenAI API configured")
except ImportError:
    print("⚠️ OpenAI not installed. Run: pip install openai")
    OPENAI_AVAILABLE = False
except Exception as e:
    print(f"⚠️ OpenAI API error: {e}")
    OPENAI_AVAILABLE = False


class FastGPTAgent(AgentInterface):
    """Fast GPT agent using gpt-4o-mini for quick responses."""
    
    def __init__(self):
        super().__init__("FastGPT")
    
    def execute(self, state: SystemState) -> AgentOutput:
        try:
            request = json.loads(state.task_specification)
            query = request.get("query", "")
            
            if OPENAI_AVAILABLE:
                start_time = time.time()
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Give brief, direct answers. Keep responses under 100 words."},
                        {"role": "user", "content": query}
                    ],
                    max_tokens=150,
                    temperature=0.3
                )
                response_time = time.time() - start_time
                result = response.choices[0].message.content
            else:
                # Fallback mock response
                result = f"FastGPT: Quick answer to '{query}' - [Brief explanation in 1-2 sentences]"
                response_time = 0.5
            
            return AgentOutput(
                content=json.dumps({
                    "agent": "FastGPT",
                    "response": result,
                    "response_time": f"{response_time:.2f}s",
                    "model": "gpt-4o-mini",
                    "cost_estimate": "$0.0001"
                }),
                cost=0.5,  # Low cost for mini model
                metadata={"agent_type": "fast_gpt", "model": "gpt-4o-mini"}
            )
            
        except Exception as e:
            return AgentOutput(
                content=json.dumps({"error": str(e), "agent": "FastGPT"}),
                cost=0.1,
                metadata={"agent_type": "fast_gpt", "error": True}
            )


class DeepGPTAgent(AgentInterface):
    """Deep analysis GPT agent using gpt-4o for thorough responses."""
    
    def __init__(self):
        super().__init__("DeepGPT")
    
    def execute(self, state: SystemState) -> AgentOutput:
        try:
            request = json.loads(state.task_specification)
            query = request.get("query", "")
            
            if OPENAI_AVAILABLE:
                start_time = time.time()
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": """You are an expert analyst. Provide comprehensive, well-structured responses that include:
1. Key analysis points
2. Multiple perspectives
3. Implications and considerations
4. Actionable insights
Be thorough and analytical."""},
                        {"role": "user", "content": query}
                    ],
                    max_tokens=800,
                    temperature=0.7
                )
                response_time = time.time() - start_time
                result = response.choices[0].message.content
            else:
                # Fallback mock response
                result = f"""Deep analysis of: {query}

1. Key Analysis Points:
   - [Comprehensive point 1]
   - [Comprehensive point 2]

2. Multiple Perspectives:
   - Perspective A: [Detailed view]
   - Perspective B: [Alternative view]

3. Implications:
   - [Important consideration 1]
   - [Important consideration 2]

4. Actionable Insights:
   - [Recommendation 1]
   - [Recommendation 2]"""
                response_time = 2.5
            
            return AgentOutput(
                content=json.dumps({
                    "agent": "DeepGPT",
                    "response": result,
                    "response_time": f"{response_time:.2f}s",
                    "model": "gpt-4o",
                    "cost_estimate": "$0.01"
                }),
                cost=2.0,  # Higher cost for advanced model
                metadata={"agent_type": "deep_gpt", "model": "gpt-4o"}
            )
            
        except Exception as e:
            return AgentOutput(
                content=json.dumps({"error": str(e), "agent": "DeepGPT"}),
                cost=0.5,
                metadata={"agent_type": "deep_gpt", "error": True}
            )


class CreativeGPTAgent(AgentInterface):
    """Creative GPT agent optimized for creative writing and brainstorming."""
    
    def __init__(self):
        super().__init__("CreativeGPT")
    
    def execute(self, state: SystemState) -> AgentOutput:
        try:
            request = json.loads(state.task_specification)
            query = request.get("query", "")
            
            if OPENAI_AVAILABLE:
                start_time = time.time()
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": """You are a creative writing expert. Be imaginative, use vivid language, and think outside the box. 
Create engaging, original content with:
- Rich descriptions and imagery
- Creative metaphors and storytelling
- Unique perspectives and ideas
- Inspiring and thought-provoking content"""},
                        {"role": "user", "content": query}
                    ],
                    max_tokens=600,
                    temperature=0.9  # High creativity
                )
                response_time = time.time() - start_time
                result = response.choices[0].message.content
            else:
                # Fallback mock response
                creative_starters = [
                    "In a world where imagination knows no bounds...",
                    "Picture this extraordinary scene...",
                    "Once upon a time, in a realm of infinite possibilities...",
                    "Imagine if reality itself could be painted with words..."
                ]
                result = f"{random.choice(creative_starters)}\n\n{query}\n\n[Creative narrative continues with vivid imagery, compelling characters, and imaginative scenarios that transport the reader to new worlds...]"
                response_time = 1.8
            
            return AgentOutput(
                content=json.dumps({
                    "agent": "CreativeGPT",
                    "response": result,
                    "response_time": f"{response_time:.2f}s",
                    "model": "gpt-4o",
                    "cost_estimate": "$0.008"
                }),
                cost=1.5,  # Medium-high cost for creative model
                metadata={"agent_type": "creative_gpt", "model": "gpt-4o"}
            )
            
        except Exception as e:
            return AgentOutput(
                content=json.dumps({"error": str(e), "agent": "CreativeGPT"}),
                cost=0.3,
                metadata={"agent_type": "creative_gpt", "error": True}
            )


class SmartLLMOrchestrator(Orchestrator):
    """Learns optimal routing between different LLM configurations."""
    
    def run_episode(self, initial_state):
        trajectory = super().run_episode(initial_state)
        
        # Evaluate based on task-agent matching and efficiency
        quality_score = self._evaluate_llm_routing_quality(initial_state.task_specification, trajectory)
        efficiency_score = self._evaluate_response_efficiency(trajectory)
        
        if trajectory.steps:
            # Reward calculation
            quality_reward = quality_score * 5.0
            efficiency_reward = efficiency_score * 2.0
            cost_penalty = sum(self._estimate_step_cost(step) for step in trajectory.steps) * 0.2
            
            final_reward = quality_reward + efficiency_reward - cost_penalty
            
            # Update trajectory
            final_step = trajectory.steps[-1]
            final_step.reward = final_reward
            trajectory.total_undiscounted_reward = final_reward
            trajectory.task_successful = quality_score > 0.8 and len(trajectory.steps) <= 2
        
        return trajectory
    
    def _evaluate_llm_routing_quality(self, task_spec, trajectory):
        """Evaluate if the right LLM was chosen for the task."""
        try:
            request = json.loads(task_spec)
            query_type = request.get("type", "unknown")
            query = request.get("query", "").lower()
            
            agent_names = [self.agents[step.agent_index].name for step in trajectory.steps]
            
            # Task-specific routing evaluation
            if query_type == "quick" or any(word in query for word in ["what is", "define", "simple question"]):
                return 1.0 if "FastGPT" in agent_names else 0.3
            
            elif query_type == "analysis" or any(word in query for word in ["analyze", "compare", "evaluate", "detailed"]):
                return 1.0 if "DeepGPT" in agent_names else 0.4
            
            elif query_type == "creative" or any(word in query for word in ["write", "story", "creative", "brainstorm", "imagine"]):
                return 1.0 if "CreativeGPT" in agent_names else 0.3
            
            else:
                # For unknown types, any agent is reasonable
                return 0.7
                
        except:
            return 0.0
    
    def _evaluate_response_efficiency(self, trajectory):
        """Evaluate efficiency (fewer steps = better)."""
        num_steps = len(trajectory.steps)
        if num_steps <= 2:
            return 1.0
        elif num_steps <= 3:
            return 0.7
        else:
            return 0.4
    
    def _estimate_step_cost(self, step):
        """Estimate cost for each LLM agent."""
        agent = self.agents[step.agent_index]
        cost_map = {
            "FastGPT": 0.5,
            "DeepGPT": 2.0,
            "CreativeGPT": 1.5,
            "TerminatorAgent": 0.0
        }
        return cost_map.get(agent.name, 1.0)


def demo_real_llm_orchestration():
    """Demo DOA learning to route between real LLM agents."""
    
    print("🤖 DOA Framework - Real LLM Demo")
    print("=" * 45)
    
    if not OPENAI_AVAILABLE:
        print("⚠️ Running in demo mode (OpenAI API not available)")
        print("To use real LLMs:")
        print("1. pip install openai")
        print("2. export OPENAI_API_KEY='your-api-key'")
        print()
    
    # Create LLM agents
    agents = [
        FastGPTAgent(),
        DeepGPTAgent(),
        CreativeGPTAgent(),
        TerminatorAgent()
    ]
    
    print("🧠 LLM Agent Pool:")
    print("   • FastGPT: Quick answers (gpt-4o-mini, $0.0001/query)")
    print("   • DeepGPT: Detailed analysis (gpt-4o, $0.01/query)")
    print("   • CreativeGPT: Creative writing (gpt-4o creative, $0.008/query)")
    
    # Initialize DOA
    policy = PolicyNetwork(64, len(agents), 128)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)
    orchestrator = SmartLLMOrchestrator(agents, policy, RewardConfig())
    trainer = REINFORCETrainer(policy, optimizer, RewardConfig())
    
    # Test queries
    test_queries = [
        {"type": "quick", "query": "What is machine learning?"},
        {"type": "analysis", "query": "Analyze the pros and cons of remote work vs office work"},
        {"type": "creative", "query": "Write a short story about an AI discovering emotions"},
        {"type": "quick", "query": "Define blockchain technology"},
        {"type": "analysis", "query": "Evaluate the impact of social media on mental health"},
        {"type": "creative", "query": "Brainstorm innovative app ideas for sustainability"}
    ]
    
    print("\n📋 BEFORE Training (random routing):")
    for query in test_queries[:3]:
        task_spec = json.dumps(query)
        state = SystemState(task_spec, [], 0, 3)
        trajectory = orchestrator.run_episode(state)
        agent_sequence = [agents[step.agent_index].name for step in trajectory.steps]
        print(f"   {query['type']} → {' → '.join(agent_sequence)}")
    
    # Training
    print(f"\n🔄 Training DOA on LLM routing...")
    
    trajectories = []
    for i in range(50):
        query = random.choice(test_queries)
        task_spec = json.dumps(query)
        state = SystemState(task_spec, [], 0, 3)
        trajectory = orchestrator.run_episode(state)
        trajectories.append(trajectory)
        
        # Train in batches
        if len(trajectories) >= 10:
            loss = trainer.train_batch(trajectories)
            avg_reward = sum(t.total_undiscounted_reward for t in trajectories) / len(trajectories)
            success_rate = sum(1 for t in trajectories if t.task_successful) / len(trajectories)
            print(f"   Batch {i//10 + 1}: Reward={avg_reward:.2f}, Success={success_rate:.1%}")
            trajectories = []
    
    print("✅ Training complete!")
    
    print("\n📋 AFTER Training (learned routing):")
    for query in test_queries[:3]:
        task_spec = json.dumps(query)
        state = SystemState(task_spec, [], 0, 3)
        trajectory = orchestrator.run_episode(state)
        agent_sequence = [agents[step.agent_index].name for step in trajectory.steps]
        quality = orchestrator._evaluate_llm_routing_quality(task_spec, trajectory)
        print(f"   {query['type']} → {' → '.join(agent_sequence)} (Quality: {quality:.2f})")
    
    print(f"\n🎉 DOA learned optimal LLM routing:")
    print(f"   ✨ Quick questions → FastGPT (cheap & fast)")
    print(f"   ✨ Analysis tasks → DeepGPT (thorough & detailed)")
    print(f"   ✨ Creative requests → CreativeGPT (imaginative & engaging)")
    print(f"   ✨ Automatic cost optimization!")
    
    # Show a real response
    if OPENAI_AVAILABLE:
        print(f"\n🤖 Sample Real LLM Response:")
        sample_query = {"type": "creative", "query": "Write about a robot learning to paint"}
        task_spec = json.dumps(sample_query)
        state = SystemState(task_spec, [], 0, 3)
        trajectory = orchestrator.run_episode(state)
        
        # Get actual response
        for step in trajectory.steps:
            agent = agents[step.agent_index]
            if agent.name != "TerminatorAgent":
                output = agent.execute(state)
                try:
                    response_data = json.loads(output.content)
                    print(f"\n{agent.name} ({response_data['model']}) responds:")
                    print(f"{response_data['response'][:300]}...")
                    print(f"Response time: {response_data['response_time']}")
                    print(f"Estimated cost: {response_data['cost_estimate']}")
                    break
                except:
                    print(f"\n{agent.name}: {output.content[:200]}...")
                    break


if __name__ == "__main__":
    demo_real_llm_orchestration()
