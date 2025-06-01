#!/usr/bin/env python3
"""
🚀 DOA Framework - 5 Minute Quick Start

This is the SIMPLEST possible example to get you started with DOA.
No complex setup, no configuration - just copy, paste, and run!

What this does:
- Shows how to wrap 2 simple functions as DOA agents
- Demonstrates intelligent routing between them
- Proves DOA learns better patterns over time

Run this file and see DOA in action!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import torch
import torch.optim as optim
from doa_framework import (
    AgentInterface, AgentOutput, SystemState, 
    PolicyNetwork, Orchestrator, TerminatorAgent, 
    RewardConfig, REINFORCETrainer
)


# Step 1: Turn your functions into DOA agents (2 minutes)
class FastAgent(AgentInterface):
    """A fast but basic service."""
    
    def __init__(self):
        super().__init__("FastAgent")
    
    def execute(self, state: SystemState) -> AgentOutput:
        # Your existing fast function here
        result = "Quick result"
        
        return AgentOutput(
            content=result,
            cost=0.5,  # Cheap and fast
            metadata={"type": "fast"}
        )


class SlowAgent(AgentInterface):
    """A slow but thorough service."""
    
    def __init__(self):
        super().__init__("SlowAgent")
    
    def execute(self, state: SystemState) -> AgentOutput:
        # Your existing thorough function here
        result = "Detailed analysis with comprehensive results"
        
        return AgentOutput(
            content=result,
            cost=2.0,  # Expensive but thorough
            metadata={"type": "slow"}
        )


# Step 2: Create DOA system (1 minute)
class SimpleDOA:
    """The simplest possible DOA setup."""
    
    def __init__(self):
        # Your agents
        self.agents = [
            FastAgent(),
            SlowAgent(), 
            TerminatorAgent()
        ]
        
        # DOA brain (policy network)
        self.policy = PolicyNetwork(32, len(self.agents), 64)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.01)
        
        # DOA orchestrator
        self.orchestrator = Orchestrator(
            self.agents, 
            self.policy, 
            RewardConfig(lambda_cost_penalty=0.1)
        )
        
        # DOA trainer
        self.trainer = REINFORCETrainer(
            self.policy, 
            self.optimizer, 
            RewardConfig()
        )
    
    def handle(self, request_type):
        """Handle a request using DOA."""
        # Convert request to DOA format
        task = json.dumps({"type": request_type})
        state = SystemState(task, [], 0, 3)
        
        # Let DOA decide which agents to use
        trajectory = self.orchestrator.run_episode(state)
        
        # Extract the result
        results = []
        for step in trajectory.steps:
            agent = self.agents[step.agent_index]
            if agent.name != "TerminatorAgent":
                output = agent.execute(state)
                results.append({
                    "agent": agent.name,
                    "result": output.content,
                    "cost": output.cost
                })
        
        return {
            "results": results,
            "total_cost": sum(r["cost"] for r in results),
            "agent_sequence": [self.agents[s.agent_index].name for s in trajectory.steps]
        }
    
    def train(self, request_types, num_episodes=20):
        """Train DOA to make better decisions."""
        print("🧠 Training DOA...")
        
        trajectories = []
        for i in range(num_episodes):
            # Pick random request type
            import random
            request_type = random.choice(request_types)
            
            # Run episode
            task = json.dumps({"type": request_type})
            state = SystemState(task, [], 0, 3)
            trajectory = self.orchestrator.run_episode(state)
            
            # Simple reward: prefer fast for "urgent", slow for "detailed"
            if request_type == "urgent" and "FastAgent" in [self.agents[s.agent_index].name for s in trajectory.steps]:
                trajectory.steps[-1].reward += 2.0  # Bonus for using fast agent
            elif request_type == "detailed" and "SlowAgent" in [self.agents[s.agent_index].name for s in trajectory.steps]:
                trajectory.steps[-1].reward += 2.0  # Bonus for using slow agent
            
            trajectories.append(trajectory)
            
            # Train every 5 episodes
            if len(trajectories) >= 5:
                loss = self.trainer.train_batch(trajectories)
                print(f"Episode {i+1}: Training loss = {loss:.2f}")
                trajectories = []
        
        print("✅ Training complete!")


# Step 3: Use it! (1 minute)
def demo():
    """Demo showing DOA learning better patterns."""
    
    print("🚀 DOA Framework - Simple Demo")
    print("=" * 40)
    
    # Create DOA system
    doa = SimpleDOA()
    
    # Test BEFORE training
    print("\n📋 BEFORE Training:")
    print("Urgent request:", doa.handle("urgent"))
    print("Detailed request:", doa.handle("detailed"))
    
    # Train DOA
    doa.train(["urgent", "detailed"], num_episodes=20)
    
    # Test AFTER training  
    print("\n📋 AFTER Training:")
    print("Urgent request:", doa.handle("urgent"))
    print("Detailed request:", doa.handle("detailed"))
    
    print("\n🎉 Notice how DOA learned to:")
    print("   • Use FastAgent for urgent requests")
    print("   • Use SlowAgent for detailed requests")
    print("   • All without hardcoded if/else logic!")


# Step 4: Your turn!
def your_integration_template():
    """Template for integrating YOUR functions."""
    
    # Replace these with YOUR actual functions
    def your_fast_function(data):
        return f"Fast processing of {data}"
    
    def your_slow_function(data):
        return f"Thorough analysis of {data}"
    
    # Wrap them as agents
    class YourFastAgent(AgentInterface):
        def __init__(self):
            super().__init__("YourFastAgent")
        
        def execute(self, state: SystemState) -> AgentOutput:
            request = json.loads(state.task_specification)
            result = your_fast_function(request)
            return AgentOutput(content=result, cost=0.5)
    
    class YourSlowAgent(AgentInterface):
        def __init__(self):
            super().__init__("YourSlowAgent")
        
        def execute(self, state: SystemState) -> AgentOutput:
            request = json.loads(state.task_specification)
            result = your_slow_function(request)
            return AgentOutput(content=result, cost=2.0)
    
    # Create DOA with YOUR agents
    class YourDOA:
        def __init__(self):
            self.agents = [YourFastAgent(), YourSlowAgent(), TerminatorAgent()]
            self.policy = PolicyNetwork(32, len(self.agents), 64)
            self.orchestrator = Orchestrator(self.agents, self.policy, RewardConfig())
        
        def handle_your_request(self, data):
            task = json.dumps(data)
            state = SystemState(task, [], 0, 3)
            trajectory = self.orchestrator.run_episode(state)
            # Extract results...
            return "Your results here"
    
    # Use it in your app
    your_doa = YourDOA()
    result = your_doa.handle_your_request({"type": "your_request"})
    print("Your DOA result:", result)


if __name__ == "__main__":
    # Run the demo
    demo()
    
    print("\n" + "=" * 40)
    print("🎯 Next Steps:")
    print("1. Copy this file")
    print("2. Replace FastAgent/SlowAgent with YOUR functions")
    print("3. Run and watch DOA learn YOUR patterns!")
    print("4. Check out examples/integration_example.py for more advanced usage")
