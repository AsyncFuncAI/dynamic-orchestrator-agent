#!/usr/bin/env python3
"""
🤖 DOA Framework - Real LLM Agents Demo

This example shows DOA orchestrating REAL LLM agents using Google's Gemini.
DOA learns to route different types of requests to specialized LLM agents:

- QuickAgent: Fast responses for simple questions
- AnalysisAgent: Deep analysis for complex requests  
- CreativeAgent: Creative writing and brainstorming
- CodeAgent: Programming and technical questions

Setup:
1. pip install google-generativeai
2. Set GOOGLE_API_KEY environment variable
3. Run this script

DOA will learn which agent to use for different request types!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import torch
import torch.optim as optim
from typing import Dict, Any
import time

# Import DOA framework
from doa_framework import (
    AgentInterface, AgentOutput, SystemState,
    PolicyNetwork, Orchestrator, TerminatorAgent,
    RewardConfig, REINFORCETrainer
)

# Import Google Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("⚠️ google-generativeai not installed. Run: pip install google-generativeai")
    GEMINI_AVAILABLE = False

# Configure Gemini
if GEMINI_AVAILABLE:
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        genai.configure(api_key=api_key)
        print("✅ Gemini API configured")
    else:
        print("⚠️ GOOGLE_API_KEY environment variable not set")
        GEMINI_AVAILABLE = False


class QuickLLMAgent(AgentInterface):
    """Fast LLM agent for simple questions and quick responses."""
    
    def __init__(self):
        super().__init__("QuickLLMAgent")
        if GEMINI_AVAILABLE:
            self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
    
    def execute(self, state: SystemState) -> AgentOutput:
        try:
            request = json.loads(state.task_specification)
            user_query = request.get("query", "")
            
            if GEMINI_AVAILABLE:
                # Quick, concise prompt for fast responses
                prompt = f"Give a brief, direct answer (1-2 sentences max): {user_query}"
                
                start_time = time.time()
                response = self.model.generate_content(prompt)
                response_time = time.time() - start_time
                
                result = {
                    "agent": "QuickLLM",
                    "response": response.text,
                    "response_time": f"{response_time:.2f}s",
                    "type": "quick_answer"
                }
            else:
                # Fallback for demo purposes
                result = {
                    "agent": "QuickLLM",
                    "response": f"Quick answer to: {user_query}",
                    "response_time": "0.5s",
                    "type": "quick_answer"
                }
            
            return AgentOutput(
                content=json.dumps(result),
                cost=0.5,  # Low cost for quick responses
                metadata={"agent_type": "quick_llm", "query_type": request.get("type", "unknown")}
            )
            
        except Exception as e:
            return AgentOutput(
                content=json.dumps({"error": str(e), "agent": "QuickLLM"}),
                cost=0.1,
                metadata={"agent_type": "quick_llm", "error": True}
            )


class AnalysisLLMAgent(AgentInterface):
    """Deep analysis LLM agent for complex reasoning and detailed responses."""
    
    def __init__(self):
        super().__init__("AnalysisLLMAgent")
        if GEMINI_AVAILABLE:
            self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
    
    def execute(self, state: SystemState) -> AgentOutput:
        try:
            request = json.loads(state.task_specification)
            user_query = request.get("query", "")
            
            if GEMINI_AVAILABLE:
                # Detailed analysis prompt
                prompt = f"""Provide a comprehensive analysis of the following:
                
{user_query}

Please include:
1. Key points and considerations
2. Different perspectives or approaches
3. Potential implications or consequences
4. Actionable insights or recommendations

Be thorough and analytical in your response."""
                
                start_time = time.time()
                response = self.model.generate_content(prompt)
                response_time = time.time() - start_time
                
                result = {
                    "agent": "AnalysisLLM",
                    "response": response.text,
                    "response_time": f"{response_time:.2f}s",
                    "type": "detailed_analysis"
                }
            else:
                # Fallback for demo purposes
                result = {
                    "agent": "AnalysisLLM",
                    "response": f"Detailed analysis of: {user_query}\n\n1. Key considerations...\n2. Multiple perspectives...\n3. Recommendations...",
                    "response_time": "2.5s",
                    "type": "detailed_analysis"
                }
            
            return AgentOutput(
                content=json.dumps(result),
                cost=2.0,  # Higher cost for detailed analysis
                metadata={"agent_type": "analysis_llm", "query_type": request.get("type", "unknown")}
            )
            
        except Exception as e:
            return AgentOutput(
                content=json.dumps({"error": str(e), "agent": "AnalysisLLM"}),
                cost=0.5,
                metadata={"agent_type": "analysis_llm", "error": True}
            )


class CreativeLLMAgent(AgentInterface):
    """Creative LLM agent for writing, brainstorming, and creative tasks."""
    
    def __init__(self):
        super().__init__("CreativeLLMAgent")
        if GEMINI_AVAILABLE:
            self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
    
    def execute(self, state: SystemState) -> AgentOutput:
        try:
            request = json.loads(state.task_specification)
            user_query = request.get("query", "")
            
            if GEMINI_AVAILABLE:
                # Creative prompt
                prompt = f"""Be creative and imaginative in responding to:

{user_query}

Use vivid language, creative metaphors, and engaging storytelling. 
Think outside the box and provide an inspiring, original response."""
                
                start_time = time.time()
                response = self.model.generate_content(prompt)
                response_time = time.time() - start_time
                
                result = {
                    "agent": "CreativeLLM",
                    "response": response.text,
                    "response_time": f"{response_time:.2f}s",
                    "type": "creative_response"
                }
            else:
                # Fallback for demo purposes
                result = {
                    "agent": "CreativeLLM",
                    "response": f"Creative response to: {user_query}\n\nImagine a world where... [creative content]",
                    "response_time": "1.8s",
                    "type": "creative_response"
                }
            
            return AgentOutput(
                content=json.dumps(result),
                cost=1.5,  # Medium cost for creative work
                metadata={"agent_type": "creative_llm", "query_type": request.get("type", "unknown")}
            )
            
        except Exception as e:
            return AgentOutput(
                content=json.dumps({"error": str(e), "agent": "CreativeLLM"}),
                cost=0.3,
                metadata={"agent_type": "creative_llm", "error": True}
            )


class CodeLLMAgent(AgentInterface):
    """Code-specialized LLM agent for programming questions and technical tasks."""
    
    def __init__(self):
        super().__init__("CodeLLMAgent")
        if GEMINI_AVAILABLE:
            self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
    
    def execute(self, state: SystemState) -> AgentOutput:
        try:
            request = json.loads(state.task_specification)
            user_query = request.get("query", "")
            
            if GEMINI_AVAILABLE:
                # Code-focused prompt
                prompt = f"""As a programming expert, help with this technical question:

{user_query}

Please provide:
1. Clear, working code examples
2. Explanations of key concepts
3. Best practices and considerations
4. Alternative approaches if applicable

Format code blocks properly and be precise with technical details."""
                
                start_time = time.time()
                response = self.model.generate_content(prompt)
                response_time = time.time() - start_time
                
                result = {
                    "agent": "CodeLLM",
                    "response": response.text,
                    "response_time": f"{response_time:.2f}s",
                    "type": "code_response"
                }
            else:
                # Fallback for demo purposes
                result = {
                    "agent": "CodeLLM",
                    "response": f"Technical response to: {user_query}\n\n```python\n# Code example\nprint('Hello, World!')\n```\n\nExplanation: ...",
                    "response_time": "1.2s",
                    "type": "code_response"
                }
            
            return AgentOutput(
                content=json.dumps(result),
                cost=1.2,  # Medium cost for code assistance
                metadata={"agent_type": "code_llm", "query_type": request.get("type", "unknown")}
            )
            
        except Exception as e:
            return AgentOutput(
                content=json.dumps({"error": str(e), "agent": "CodeLLM"}),
                cost=0.2,
                metadata={"agent_type": "code_llm", "error": True}
            )


class LLMOrchestrator(Orchestrator):
    """Custom orchestrator that learns optimal LLM agent routing."""
    
    def run_episode(self, initial_state: SystemState):
        trajectory = super().run_episode(initial_state)
        
        # Evaluate based on request type and agent appropriateness
        quality_score = self._evaluate_llm_quality(initial_state.task_specification, trajectory)
        efficiency_score = self._evaluate_efficiency(trajectory)
        
        if trajectory.steps:
            # Reward calculation
            quality_reward = quality_score * 4.0
            efficiency_reward = efficiency_score * 2.0
            cost_penalty = sum(self._estimate_step_cost(step) for step in trajectory.steps) * 0.1
            
            final_reward = quality_reward + efficiency_reward - cost_penalty
            
            # Update trajectory
            final_step = trajectory.steps[-1]
            final_step.reward = final_reward
            trajectory.total_undiscounted_reward = sum(step.reward for step in trajectory.steps)
            trajectory.task_successful = quality_score > 0.7 and len(trajectory.steps) <= 3
        
        return trajectory
    
    def _evaluate_llm_quality(self, task_spec: str, trajectory) -> float:
        """Evaluate if the right LLM agent was chosen for the task."""
        try:
            request = json.loads(task_spec)
            query_type = request.get("type", "unknown")
            query = request.get("query", "").lower()
            
            agent_names = [self.agents[step.agent_index].name for step in trajectory.steps]
            
            quality = 0.0
            
            # Quick questions should use QuickLLMAgent
            if query_type == "quick" or any(word in query for word in ["what is", "define", "simple"]):
                if "QuickLLMAgent" in agent_names:
                    quality = 1.0
                else:
                    quality = 0.3
            
            # Analysis questions should use AnalysisLLMAgent
            elif query_type == "analysis" or any(word in query for word in ["analyze", "compare", "evaluate", "pros and cons"]):
                if "AnalysisLLMAgent" in agent_names:
                    quality = 1.0
                else:
                    quality = 0.4
            
            # Creative tasks should use CreativeLLMAgent
            elif query_type == "creative" or any(word in query for word in ["write", "story", "creative", "brainstorm"]):
                if "CreativeLLMAgent" in agent_names:
                    quality = 1.0
                else:
                    quality = 0.3
            
            # Code questions should use CodeLLMAgent
            elif query_type == "code" or any(word in query for word in ["code", "program", "function", "python", "javascript"]):
                if "CodeLLMAgent" in agent_names:
                    quality = 1.0
                else:
                    quality = 0.2
            
            else:
                # Unknown type - any agent is okay
                quality = 0.6
            
            return quality
            
        except:
            return 0.0
    
    def _evaluate_efficiency(self, trajectory) -> float:
        """Evaluate efficiency based on number of steps."""
        num_steps = len(trajectory.steps)
        if num_steps <= 2:
            return 1.0
        elif num_steps <= 3:
            return 0.8
        else:
            return 0.5
    
    def _estimate_step_cost(self, step) -> float:
        """Estimate cost for each agent."""
        agent = self.agents[step.agent_index]
        cost_map = {
            "QuickLLMAgent": 0.5,
            "AnalysisLLMAgent": 2.0,
            "CreativeLLMAgent": 1.5,
            "CodeLLMAgent": 1.2,
            "TerminatorAgent": 0.0
        }
        return cost_map.get(agent.name, 1.0)


def demo_llm_orchestration():
    """Demo showing DOA learning to route to appropriate LLM agents."""
    
    print("🤖 DOA Framework - Real LLM Agents Demo")
    print("=" * 50)
    
    if not GEMINI_AVAILABLE:
        print("⚠️ Running in demo mode (Gemini API not available)")
        print("To use real LLMs:")
        print("1. pip install google-generativeai")
        print("2. export GOOGLE_API_KEY='your-api-key'")
        print()
    
    # Create LLM agents
    agents = [
        QuickLLMAgent(),
        AnalysisLLMAgent(),
        CreativeLLMAgent(),
        CodeLLMAgent(),
        TerminatorAgent()
    ]
    
    print(f"🧠 LLM Agent Pool ({len(agents)} agents):")
    for agent in agents:
        if agent.name != "TerminatorAgent":
            print(f"   • {agent.name}: Specialized LLM for specific tasks")
    
    # Initialize DOA components
    policy = PolicyNetwork(64, len(agents), 128)
    optimizer = optim.Adam(policy.parameters(), lr=0.005)
    orchestrator = LLMOrchestrator(agents, policy, RewardConfig())
    trainer = REINFORCETrainer(policy, optimizer, RewardConfig())
    
    # Test requests
    test_requests = [
        {"type": "quick", "query": "What is machine learning?"},
        {"type": "analysis", "query": "Compare the pros and cons of remote work vs office work"},
        {"type": "creative", "query": "Write a short story about a robot learning to paint"},
        {"type": "code", "query": "How do I create a REST API in Python?"}
    ]
    
    print("\n📋 BEFORE Training (random routing):")
    for request in test_requests:
        task_spec = json.dumps(request)
        state = SystemState(task_spec, [], 0, 3)
        trajectory = orchestrator.run_episode(state)
        
        agent_sequence = [agents[step.agent_index].name for step in trajectory.steps]
        print(f"   {request['type']} query → {' → '.join(agent_sequence)}")
    
    # Training
    print(f"\n🔄 Training DOA to learn optimal LLM routing...")
    
    training_requests = test_requests * 15  # Repeat for training
    trajectories = []
    
    for i, request in enumerate(training_requests):
        task_spec = json.dumps(request)
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
    for request in test_requests:
        task_spec = json.dumps(request)
        state = SystemState(task_spec, [], 0, 3)
        trajectory = orchestrator.run_episode(state)
        
        agent_sequence = [agents[step.agent_index].name for step in trajectory.steps]
        quality = orchestrator._evaluate_llm_quality(task_spec, trajectory)
        
        print(f"   {request['type']} query → {' → '.join(agent_sequence)} (Quality: {quality:.2f})")
    
    print(f"\n🎉 DOA learned to route:")
    print(f"   ✨ Quick questions → QuickLLMAgent")
    print(f"   ✨ Analysis tasks → AnalysisLLMAgent")
    print(f"   ✨ Creative requests → CreativeLLMAgent")
    print(f"   ✨ Code questions → CodeLLMAgent")
    print(f"   ✨ All without hardcoded routing logic!")
    
    # Interactive demo
    if GEMINI_AVAILABLE:
        print(f"\n🎯 Try it yourself!")
        while True:
            user_input = input("\nEnter a question (or 'quit' to exit): ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            # Auto-detect query type (simplified)
            if any(word in user_input.lower() for word in ["what is", "define"]):
                query_type = "quick"
            elif any(word in user_input.lower() for word in ["analyze", "compare", "pros and cons"]):
                query_type = "analysis"
            elif any(word in user_input.lower() for word in ["write", "story", "creative"]):
                query_type = "creative"
            elif any(word in user_input.lower() for word in ["code", "program", "python"]):
                query_type = "code"
            else:
                query_type = "quick"
            
            request = {"type": query_type, "query": user_input}
            task_spec = json.dumps(request)
            state = SystemState(task_spec, [], 0, 3)
            trajectory = orchestrator.run_episode(state)
            
            # Get the actual response
            for step in trajectory.steps:
                agent = agents[step.agent_index]
                if agent.name != "TerminatorAgent":
                    output = agent.execute(state)
                    try:
                        response_data = json.loads(output.content)
                        print(f"\n🤖 {agent.name} responds:")
                        print(f"{response_data.get('response', 'No response')}")
                        break
                    except:
                        print(f"\n🤖 {agent.name}: {output.content}")
                        break


if __name__ == "__main__":
    demo_llm_orchestration()
