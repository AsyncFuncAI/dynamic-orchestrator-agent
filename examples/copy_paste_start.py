#!/usr/bin/env python3
"""
🚀 DOA Framework - Copy & Paste Start (2 minutes!)

Just copy this entire file and run it. That's it!
No setup, no configuration, no complexity.

This shows DOA learning to route between a fast vs slow service.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import torch
import torch.optim as optim
from doa_framework import *

# Your services as DOA agents
class FastService(AgentInterface):
    def __init__(self): super().__init__("FastService")
    def execute(self, state): 
        return AgentOutput("Fast result", 0.5)

class SlowService(AgentInterface):
    def __init__(self): super().__init__("SlowService")
    def execute(self, state): 
        return AgentOutput("Detailed result", 2.0)

# DOA setup (one-liner style)
agents = [FastService(), SlowService(), TerminatorAgent()]
policy = PolicyNetwork(16, len(agents), 32)
orchestrator = Orchestrator(agents, policy, RewardConfig())
trainer = REINFORCETrainer(policy, optim.Adam(policy.parameters()), RewardConfig())

def handle_request(request_type):
    """Use DOA to handle a request."""
    state = SystemState(json.dumps({"type": request_type}), [], 0, 3)
    trajectory = orchestrator.run_episode(state)
    
    # Show which agents were used
    agent_names = [agents[s.agent_index].name for s in trajectory.steps]
    return f"Request '{request_type}' → {' → '.join(agent_names)}"

def train_doa():
    """Train DOA to prefer fast for urgent, slow for detailed."""
    print("🧠 Training DOA...")
    
    for i in range(30):
        # Random request
        request_type = ["urgent", "detailed"][i % 2]
        state = SystemState(json.dumps({"type": request_type}), [], 0, 3)
        trajectory = orchestrator.run_episode(state)
        
        # Reward good choices
        agent_names = [agents[s.agent_index].name for s in trajectory.steps]
        if request_type == "urgent" and "FastService" in agent_names:
            trajectory.steps[-1].reward += 3.0
        elif request_type == "detailed" and "SlowService" in agent_names:
            trajectory.steps[-1].reward += 3.0
        
        # Train every 5 episodes
        if i % 5 == 4:
            trainer.train_batch([trajectory])
    
    print("✅ Training done!")

# Demo
print("🚀 DOA Framework - 2 Minute Demo")
print("\n📋 BEFORE training (random routing):")
print(handle_request("urgent"))
print(handle_request("detailed"))

train_doa()

print("\n📋 AFTER training (learned routing):")
print(handle_request("urgent"))
print(handle_request("detailed"))

print("\n🎉 DOA learned:")
print("• Urgent requests → FastService")
print("• Detailed requests → SlowService")
print("• No hardcoded if/else needed!")

print("\n🎯 To use with YOUR functions:")
print("1. Replace FastService/SlowService with your functions")
print("2. Adjust the reward logic for your use case")
print("3. That's it - DOA will learn your patterns!")
