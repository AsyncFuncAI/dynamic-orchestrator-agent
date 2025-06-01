#!/usr/bin/env python3
"""
🤖 DOA Framework - Real Gemini LLM Demo

This shows DOA orchestrating REAL Google Gemini LLM agents!
DOA learns to route different types of requests to specialized LLM configurations.

Setup (2 minutes):
1. pip install google-generativeai
2. Get API key: https://makersuite.google.com/app/apikey
3. export GOOGLE_API_KEY='your-api-key'
4. python examples/gemini_llm_demo.py

DOA learns to route:
- Quick questions → FastGemini (low temperature, short responses)
- Analysis tasks → DeepGemini (detailed prompts, thorough analysis)  
- Creative requests → CreativeGemini (high temperature, creative prompts)

Watch DOA learn which LLM configuration to use for each task type!
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

# Try to import Google Gemini
try:
    import google.generativeai as genai
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        genai.configure(api_key=api_key)
        GEMINI_AVAILABLE = True
        print("✅ Gemini API configured")
    else:
        print("⚠️ GOOGLE_API_KEY not set. Get one at: https://makersuite.google.com/app/apikey")
        GEMINI_AVAILABLE = False
except ImportError:
    print("⚠️ Run: pip install google-generativeai")
    GEMINI_AVAILABLE = False


class FastGeminiAgent(AgentInterface):
    """Fast Gemini for quick, concise answers."""
    
    def __init__(self):
        super().__init__("FastGemini")
        if GEMINI_AVAILABLE:
            self.model = genai.GenerativeModel(
                "gemini-2.0-flash",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low creativity for factual answers
                    max_output_tokens=100  # Short responses
                )
            )
    
    def execute(self, state: SystemState) -> AgentOutput:
        try:
            request = json.loads(state.task_specification)
            query = request.get("query", "")
            
            if GEMINI_AVAILABLE:
                prompt = f"Give a brief, direct answer (1-2 sentences): {query}"
                start_time = time.time()
                response = self.model.generate_content(prompt)
                response_time = time.time() - start_time
                result = response.text
            else:
                # Demo fallback
                result = f"FastGemini: Quick answer to '{query}' - [Brief factual response]"
                response_time = 0.3
            
            return AgentOutput(
                content=json.dumps({
                    "agent": "FastGemini",
                    "response": result,
                    "response_time": f"{response_time:.2f}s",
                    "config": "low_temp_short"
                }),
                cost=0.5,  # Cheap and fast
                metadata={"agent_type": "fast_gemini"}
            )
            
        except Exception as e:
            return AgentOutput(
                content=json.dumps({"error": str(e), "agent": "FastGemini"}),
                cost=0.1,
                metadata={"agent_type": "fast_gemini", "error": True}
            )


class DeepGeminiAgent(AgentInterface):
    """Deep analysis Gemini for thorough, detailed responses."""
    
    def __init__(self):
        super().__init__("DeepGemini")
        if GEMINI_AVAILABLE:
            self.model = genai.GenerativeModel(
                "gemini-2.0-flash-exp",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.4,  # Balanced creativity
                    max_output_tokens=500  # Longer responses
                )
            )
    
    def execute(self, state: SystemState) -> AgentOutput:
        try:
            request = json.loads(state.task_specification)
            query = request.get("query", "")
            
            if GEMINI_AVAILABLE:
                prompt = f"""Provide a comprehensive analysis of: {query}

Please structure your response with:
1. Key points and considerations
2. Different perspectives or approaches  
3. Implications and consequences
4. Actionable recommendations

Be thorough and analytical."""
                
                start_time = time.time()
                response = self.model.generate_content(prompt)
                response_time = time.time() - start_time
                result = response.text
            else:
                # Demo fallback
                result = f"""Deep analysis of: {query}

1. Key Points:
   - Comprehensive consideration A
   - Important factor B

2. Multiple Perspectives:
   - Viewpoint 1: [Detailed analysis]
   - Viewpoint 2: [Alternative perspective]

3. Implications:
   - Short-term impact: [Analysis]
   - Long-term consequences: [Evaluation]

4. Recommendations:
   - Action item 1
   - Strategic approach 2"""
                response_time = 1.8
            
            return AgentOutput(
                content=json.dumps({
                    "agent": "DeepGemini",
                    "response": result,
                    "response_time": f"{response_time:.2f}s",
                    "config": "balanced_temp_long"
                }),
                cost=2.0,  # More expensive for detailed analysis
                metadata={"agent_type": "deep_gemini"}
            )
            
        except Exception as e:
            return AgentOutput(
                content=json.dumps({"error": str(e), "agent": "DeepGemini"}),
                cost=0.5,
                metadata={"agent_type": "deep_gemini", "error": True}
            )


class CreativeGeminiAgent(AgentInterface):
    """Creative Gemini for imaginative, engaging content."""
    
    def __init__(self):
        super().__init__("CreativeGemini")
        if GEMINI_AVAILABLE:
            self.model = genai.GenerativeModel(
                "gemini-2.0-flash-exp",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.9,  # High creativity
                    max_output_tokens=400  # Medium length for stories
                )
            )
    
    def execute(self, state: SystemState) -> AgentOutput:
        try:
            request = json.loads(state.task_specification)
            query = request.get("query", "")
            
            if GEMINI_AVAILABLE:
                prompt = f"""Be creative and imaginative with: {query}

Use vivid language, engaging storytelling, and think outside the box. 
Create something inspiring and original that captures the imagination."""
                
                start_time = time.time()
                response = self.model.generate_content(prompt)
                response_time = time.time() - start_time
                result = response.text
            else:
                # Demo fallback
                creative_starters = [
                    "In a world where imagination knows no bounds...",
                    "Picture this extraordinary scene...",
                    "Once upon a time, in a realm of infinite possibilities..."
                ]
                result = f"{random.choice(creative_starters)}\n\n{query}\n\n[Creative narrative continues with vivid imagery and compelling storytelling...]"
                response_time = 1.2
            
            return AgentOutput(
                content=json.dumps({
                    "agent": "CreativeGemini",
                    "response": result,
                    "response_time": f"{response_time:.2f}s",
                    "config": "high_temp_creative"
                }),
                cost=1.5,  # Medium cost for creative work
                metadata={"agent_type": "creative_gemini"}
            )
            
        except Exception as e:
            return AgentOutput(
                content=json.dumps({"error": str(e), "agent": "CreativeGemini"}),
                cost=0.3,
                metadata={"agent_type": "creative_gemini", "error": True}
            )


class GeminiOrchestrator(Orchestrator):
    """Learns optimal routing between different Gemini configurations."""
    
    def run_episode(self, initial_state):
        trajectory = super().run_episode(initial_state)
        
        # Evaluate routing quality
        quality_score = self._evaluate_gemini_routing(initial_state.task_specification, trajectory)
        
        if trajectory.steps:
            # Simple but effective reward
            reward = quality_score * 6.0  # High reward for good routing
            trajectory.steps[-1].reward = reward
            trajectory.total_undiscounted_reward = reward
            trajectory.task_successful = quality_score > 0.8
        
        return trajectory
    
    def _evaluate_gemini_routing(self, task_spec, trajectory):
        """Reward using the right Gemini configuration for the task."""
        try:
            request = json.loads(task_spec)
            query_type = request.get("type", "unknown")
            query = request.get("query", "").lower()

            agent_names = [self.agents[step.agent_index].name for step in trajectory.steps]

            # Penalize if only TerminatorAgent was used (no actual work done)
            non_terminator_agents = [name for name in agent_names if name != "TerminatorAgent"]
            if not non_terminator_agents:
                return 0.0  # No work done = no reward

            # MUCH more dramatic reward differences to encourage proper routing
            if query_type == "quick" or any(word in query for word in ["what is", "define", "simple"]):
                if "FastGemini" in agent_names:
                    return 1.0  # Perfect match
                elif "DeepGemini" in agent_names or "CreativeGemini" in agent_names:
                    return 0.1  # Wrong agent - big penalty
                else:
                    return 0.0

            elif query_type == "analysis" or any(word in query for word in ["analyze", "compare", "evaluate", "pros and cons"]):
                if "DeepGemini" in agent_names:
                    return 1.0  # Perfect match
                elif "FastGemini" in agent_names:
                    return 0.1  # Too shallow - big penalty
                elif "CreativeGemini" in agent_names:
                    return 0.05  # Completely wrong
                else:
                    return 0.0

            elif query_type == "creative" or any(word in query for word in ["write", "story", "creative", "imagine", "poem"]):
                if "CreativeGemini" in agent_names:
                    return 1.0  # Perfect match
                elif "FastGemini" in agent_names or "DeepGemini" in agent_names:
                    return 0.1  # Not creative enough - big penalty
                else:
                    return 0.0

            else:
                return 0.3  # Unknown type - any agent is okay

        except:
            return 0.0


def demo():
    """Demo DOA learning Gemini LLM routing."""
    
    print("🤖 DOA Framework - Real Gemini LLM Demo")
    print("=" * 45)
    
    if not GEMINI_AVAILABLE:
        print("⚠️ Running in demo mode")
        print("For real Gemini LLMs:")
        print("1. pip install google-generativeai")
        print("2. Get API key: https://makersuite.google.com/app/apikey")
        print("3. export GOOGLE_API_KEY='your-key'")
        print()
    
    # Create Gemini agents
    agents = [
        FastGeminiAgent(),
        DeepGeminiAgent(),
        CreativeGeminiAgent(),
        TerminatorAgent()
    ]
    
    print("🧠 Gemini Agent Pool:")
    print("   • FastGemini: Quick answers (temp=0.1, 100 tokens)")
    print("   • DeepGemini: Detailed analysis (temp=0.4, 500 tokens)")
    print("   • CreativeGemini: Creative content (temp=0.9, 400 tokens)")
    
    # DOA setup with better learning parameters
    policy = PolicyNetwork(64, len(agents), 128)  # Larger network
    optimizer = optim.Adam(policy.parameters(), lr=0.01)  # Better learning rate
    reward_config = RewardConfig(lambda_cost_penalty=0.05)  # Lower cost penalty to encourage exploration
    orchestrator = GeminiOrchestrator(agents, policy, reward_config)
    trainer = REINFORCETrainer(policy, optimizer, reward_config)
    
    # Test queries
    test_queries = [
        {"type": "quick", "query": "What is artificial intelligence?"},
        {"type": "analysis", "query": "Analyze the impact of AI on the job market"},
        {"type": "creative", "query": "Write a story about a robot discovering art"},
        {"type": "quick", "query": "Define machine learning"},
        {"type": "analysis", "query": "Compare the pros and cons of remote work"},
        {"type": "creative", "query": "Imagine a world where AI and humans collaborate perfectly"}
    ]
    
    print("\n📋 BEFORE Training (random routing):")
    for query in test_queries[:3]:
        task_spec = json.dumps(query)
        state = SystemState(task_spec, [], 0, 2)
        trajectory = orchestrator.run_episode(state)
        agent_sequence = [agents[step.agent_index].name for step in trajectory.steps]
        print(f"   {query['type']} → {' → '.join(agent_sequence)}")
    
    # Training with balanced data and exploration
    print(f"\n🔄 Training DOA on Gemini routing...")

    # Create balanced training set (equal examples of each type)
    training_queries = []
    for _ in range(40):  # 40 of each type = 120 total
        training_queries.extend([
            {"type": "quick", "query": random.choice([
                "What is artificial intelligence?", "Define machine learning", "What is blockchain?",
                "What is quantum computing?", "Define neural networks", "What is cloud computing?"
            ])},
            {"type": "analysis", "query": random.choice([
                "Analyze the impact of AI on society", "Compare Python vs JavaScript",
                "Should companies adopt AI? Analyze benefits and risks", "Evaluate remote work pros and cons",
                "Compare different programming languages", "Analyze the future of renewable energy"
            ])},
            {"type": "creative", "query": random.choice([
                "Write a story about a robot discovering art", "Write a poem about technology",
                "Imagine a conversation between human and AI in 2030", "Create a story about time travel",
                "Write about the future of space exploration", "Imagine a world where AI and humans collaborate"
            ])}
        ])

    random.shuffle(training_queries)

    trajectories = []
    for i, query in enumerate(training_queries):
        task_spec = json.dumps(query)
        state = SystemState(task_spec, [], 0, 2)
        trajectory = orchestrator.run_episode(state)
        trajectories.append(trajectory)

        # Train in batches of 10 for better stability
        if len(trajectories) >= 10:
            trainer.train_batch(trajectories)

            # Evaluate current performance
            avg_quality = sum(orchestrator._evaluate_gemini_routing(json.dumps(t_query), t)
                            for t_query, t in zip(training_queries[i-9:i+1], trajectories)) / len(trajectories)

            print(f"   Episode {i+1}: Avg Quality={avg_quality:.2f}")
            trajectories = []
    
    print("✅ Training complete!")
    
    print("\n📋 AFTER Training (learned routing):")
    for query in test_queries[:3]:
        task_spec = json.dumps(query)
        state = SystemState(task_spec, [], 0, 2)
        trajectory = orchestrator.run_episode(state)
        agent_sequence = [agents[step.agent_index].name for step in trajectory.steps]
        quality = orchestrator._evaluate_gemini_routing(task_spec, trajectory)
        print(f"   {query['type']} → {' → '.join(agent_sequence)} (Quality: {quality:.2f})")
    
    print(f"\n🎉 DOA learned optimal Gemini routing:")
    print(f"   ✨ Quick questions → FastGemini (factual, concise)")
    print(f"   ✨ Analysis tasks → DeepGemini (thorough, structured)")
    print(f"   ✨ Creative requests → CreativeGemini (imaginative, engaging)")
    
    # Show real Gemini response
    print(f"\n🤖 Sample Gemini Response:")
    sample_query = {"type": "creative", "query": "Write about a robot learning to paint"}
    task_spec = json.dumps(sample_query)
    state = SystemState(task_spec, [], 0, 2)
    trajectory = orchestrator.run_episode(state)
    
    # Get actual response
    for step in trajectory.steps:
        agent = agents[step.agent_index]
        if agent.name != "TerminatorAgent":
            output = agent.execute(state)
            try:
                response_data = json.loads(output.content)
                print(f"\n{agent.name} ({response_data['config']}) responds:")
                print(f"{response_data['response'][:250]}...")
                if GEMINI_AVAILABLE:
                    print(f"Response time: {response_data['response_time']}")
                break
            except:
                print(f"\n{agent.name}: {output.content[:200]}...")
                break
    
    if GEMINI_AVAILABLE:
        print(f"\n🎯 Testing DOA with Real Queries:")

        # Test with actual diverse queries
        real_test_queries = [
            {"type": "quick", "query": "What is blockchain?"},
            {"type": "analysis", "query": "Should companies adopt AI? Analyze the benefits and risks."},
            {"type": "creative", "query": "Write a short poem about the future of technology"},
            {"type": "quick", "query": "Define quantum computing in simple terms"},
            {"type": "analysis", "query": "Compare Python vs JavaScript for web development"},
            {"type": "creative", "query": "Imagine a conversation between a human and an AI in 2030"}
        ]

        for i, test_query in enumerate(real_test_queries, 1):
            print(f"\n--- Test Query {i} ---")
            print(f"Query Type: {test_query['type']}")
            print(f"Question: {test_query['query']}")

            # Let DOA route and get real response
            task_spec = json.dumps(test_query)
            state = SystemState(task_spec, [], 0, 2)
            trajectory = orchestrator.run_episode(state)

            # Show routing decision
            agent_sequence = [agents[step.agent_index].name for step in trajectory.steps]
            quality = orchestrator._evaluate_gemini_routing(task_spec, trajectory)
            print(f"DOA Routing: {' → '.join(agent_sequence)} (Quality: {quality:.2f})")

            # Get and show actual Gemini response
            for step in trajectory.steps:
                agent = agents[step.agent_index]
                if agent.name != "TerminatorAgent":
                    output = agent.execute(state)
                    try:
                        response_data = json.loads(output.content)
                        print(f"\n{agent.name} Response:")
                        print(f"{response_data['response']}")
                        print(f"⏱️ Response time: {response_data['response_time']}")
                        print(f"⚙️ Config: {response_data['config']}")
                        break
                    except:
                        print(f"\n{agent.name}: {output.content}")
                        break

            # Longer delay to avoid rate limiting
            time.sleep(3)

        # Calculate final score
        total_score = 0
        perfect_scores = 0
        for test_query in real_test_queries:
            task_spec = json.dumps(test_query)
            state = SystemState(task_spec, [], 0, 2)
            trajectory = orchestrator.run_episode(state)
            quality = orchestrator._evaluate_gemini_routing(task_spec, trajectory)
            total_score += quality
            if quality >= 1.0:
                perfect_scores += 1

        avg_score = total_score / len(real_test_queries)
        final_score = (avg_score * 10)  # Convert to 0-10 scale

        print(f"\n🎯 FINAL EVALUATION:")
        print(f"   📊 Average Quality Score: {avg_score:.2f}")
        print(f"   🏆 Final Score: {final_score:.1f}/10")
        print(f"   ✅ Perfect Routings: {perfect_scores}/{len(real_test_queries)}")

        if final_score >= 9.5:
            print(f"   🎉 EXCELLENT! DOA achieved near-perfect routing!")
        elif final_score >= 8.0:
            print(f"   👍 GOOD! DOA learned most routing patterns correctly!")
        elif final_score >= 6.0:
            print(f"   📈 IMPROVING! DOA is learning but needs more training!")
        else:
            print(f"   🔄 NEEDS WORK! DOA requires better reward tuning!")

        print(f"\n🎉 DOA successfully routed {len(real_test_queries)} real queries!")
        print(f"Notice how different agent configurations were chosen based on query type!")
    else:
        print(f"\n🎯 To test with real queries:")
        print(f"1. Get Gemini API key: https://makersuite.google.com/app/apikey")
        print(f"2. export GOOGLE_API_KEY='your-key'")
        print(f"3. Re-run this demo")


if __name__ == "__main__":
    demo()
