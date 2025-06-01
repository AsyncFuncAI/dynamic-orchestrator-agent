#!/usr/bin/env python3
"""
Example: Integrating DOA Framework into an existing application.

This shows how to wrap existing services/functions as DOA agents
and use the framework for intelligent orchestration.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
import json
from typing import Dict, Any, List
from doa_framework import (
    AgentInterface, TerminatorAgent, Orchestrator,
    PolicyNetwork, SystemState, RewardConfig, AgentOutput
)


# Step 1: Wrap your existing services as DOA Agents
class DatabaseServiceAgent(AgentInterface):
    """Wraps your existing database service."""
    
    def __init__(self, db_connection, name: str = "DatabaseService"):
        super().__init__(name)
        self.db = db_connection  # Your existing DB connection
    
    def execute(self, state: SystemState) -> AgentOutput:
        try:
            request = json.loads(state.task_specification)
            query_type = request.get("type", "unknown")
            
            if query_type == "user_lookup":
                # Use your existing database service
                user_id = request.get("user_id")
                result = self.db.get_user(user_id)  # Your existing method
                
                return AgentOutput(
                    content=json.dumps({"user_data": result}),
                    cost=0.5,  # Database query cost
                    metadata={"service": "database", "query_type": "user_lookup"}
                )
            
            elif query_type == "analytics":
                # Your existing analytics query
                result = self.db.run_analytics_query(request.get("query"))
                
                return AgentOutput(
                    content=json.dumps({"analytics": result}),
                    cost=2.0,  # More expensive analytics
                    metadata={"service": "database", "query_type": "analytics"}
                )
            
            else:
                return AgentOutput(
                    content=json.dumps({"error": "Unknown query type"}),
                    cost=0.1,
                    metadata={"service": "database", "query_type": "unknown"}
                )
                
        except Exception as e:
            return AgentOutput(
                content=json.dumps({"error": str(e)}),
                cost=0.1,
                metadata={"service": "database", "error": True}
            )


class CacheServiceAgent(AgentInterface):
    """Wraps your existing cache service."""
    
    def __init__(self, cache_client, name: str = "CacheService"):
        super().__init__(name)
        self.cache = cache_client  # Your existing cache (Redis, etc.)
    
    def execute(self, state: SystemState) -> AgentOutput:
        try:
            request = json.loads(state.task_specification)
            cache_key = request.get("cache_key", "")
            
            # Check if data is in cache
            cached_data = self.cache.get(cache_key)  # Your existing cache method
            
            if cached_data:
                return AgentOutput(
                    content=json.dumps({"cached_data": cached_data, "cache_hit": True}),
                    cost=0.1,  # Very cheap cache hit
                    metadata={"service": "cache", "cache_hit": True}
                )
            else:
                return AgentOutput(
                    content=json.dumps({"cache_hit": False}),
                    cost=0.05,  # Cheap cache miss
                    metadata={"service": "cache", "cache_hit": False}
                )
                
        except Exception as e:
            return AgentOutput(
                content=json.dumps({"error": str(e), "cache_hit": False}),
                cost=0.05,
                metadata={"service": "cache", "error": True}
            )


class ExternalAPIAgent(AgentInterface):
    """Wraps calls to external APIs."""
    
    def __init__(self, api_client, name: str = "ExternalAPI"):
        super().__init__(name)
        self.api = api_client  # Your existing API client
    
    def execute(self, state: SystemState) -> AgentOutput:
        try:
            request = json.loads(state.task_specification)
            api_endpoint = request.get("endpoint", "")
            
            # Call your existing API
            result = self.api.call(api_endpoint, request.get("params", {}))
            
            return AgentOutput(
                content=json.dumps({"api_result": result}),
                cost=1.5,  # External API calls are expensive
                metadata={"service": "external_api", "endpoint": api_endpoint}
            )
            
        except Exception as e:
            return AgentOutput(
                content=json.dumps({"error": str(e)}),
                cost=0.5,  # Still some cost for failed API call
                metadata={"service": "external_api", "error": True}
            )


# Step 2: Create a custom orchestrator for your domain
class ApplicationOrchestrator(Orchestrator):
    """Custom orchestrator with your business logic for rewards."""
    
    def run_episode(self, initial_state: SystemState) -> 'EpisodeTrajectory':
        trajectory = super().run_episode(initial_state)
        
        # Custom reward calculation based on your business metrics
        quality_score = self._evaluate_business_quality(initial_state.task_specification, trajectory)
        latency_score = self._evaluate_latency(trajectory)
        cost_efficiency = self._evaluate_cost_efficiency(trajectory)
        
        if trajectory.steps:
            # Your custom reward formula
            business_reward = quality_score * 3.0  # Quality is important
            latency_reward = latency_score * 2.0   # Speed matters
            efficiency_reward = cost_efficiency * 1.0  # Cost control
            
            final_reward = business_reward + latency_reward + efficiency_reward
            
            # Update final step
            final_step = trajectory.steps[-1]
            final_step.reward = final_reward
            trajectory.total_undiscounted_reward = sum(step.reward for step in trajectory.steps)
            
            # Your success criteria
            trajectory.task_successful = (
                quality_score > 0.7 and 
                latency_score > 0.5 and 
                len(trajectory.steps) <= 4
            )
        
        return trajectory
    
    def _evaluate_business_quality(self, task_spec: str, trajectory) -> float:
        """Evaluate based on your business requirements."""
        try:
            request = json.loads(task_spec)
            request_type = request.get("type", "unknown")

            # Check if appropriate services were used
            agent_names = [self.agents[step.agent_index].name for step in trajectory.steps]

            quality = 0.0

            # Business rule: Always check cache first for user lookups
            if request_type == "user_lookup":
                cache_used = "CacheService" in agent_names
                db_used = "DatabaseService" in agent_names
                external_avoided = "ExternalAPI" not in agent_names

                if cache_used and db_used and external_avoided:
                    quality = 1.0  # Perfect: cache first, then DB, no external API
                elif db_used and external_avoided:
                    quality = 0.8  # Good: got data from DB, avoided external API
                elif db_used:
                    quality = 0.6  # OK: got data but used unnecessary services
                else:
                    quality = 0.2  # Poor: didn't get user data

            # Business rule: Use external API only when necessary
            elif request_type == "external_data":
                cache_used = "CacheService" in agent_names
                external_used = "ExternalAPI" in agent_names

                if cache_used and external_used:
                    quality = 1.0  # Perfect: checked cache, then API
                elif external_used:
                    quality = 0.7  # OK: got data but didn't check cache
                else:
                    quality = 0.3  # Poor: didn't get external data

            # Business rule: Analytics should use database
            elif request_type == "analytics":
                db_used = "DatabaseService" in agent_names
                external_avoided = "ExternalAPI" not in agent_names

                if db_used and external_avoided:
                    quality = 1.0  # Perfect: used DB for analytics
                elif db_used:
                    quality = 0.7  # OK: used DB but also unnecessary services
                else:
                    quality = 0.2  # Poor: didn't use DB for analytics

            return min(quality, 1.0)

        except:
            return 0.0
    
    def _evaluate_latency(self, trajectory) -> float:
        """Evaluate based on expected latency."""
        num_steps = len(trajectory.steps)
        if num_steps <= 2:
            return 1.0
        elif num_steps <= 3:
            return 0.8
        elif num_steps <= 4:
            return 0.6
        else:
            return 0.3
    
    def _evaluate_cost_efficiency(self, trajectory) -> float:
        """Evaluate cost efficiency."""
        total_cost = sum(self._estimate_step_cost(step) for step in trajectory.steps)
        if total_cost <= 1.0:
            return 1.0
        elif total_cost <= 2.0:
            return 0.8
        elif total_cost <= 3.0:
            return 0.6
        else:
            return 0.4
    
    def _estimate_step_cost(self, step) -> float:
        """Estimate cost for each service."""
        agent = self.agents[step.agent_index]
        cost_map = {
            "CacheService": 0.1,
            "DatabaseService": 0.5,
            "ExternalAPI": 1.5,
            "TerminatorAgent": 0.0
        }
        return cost_map.get(agent.name, 1.0)


# Step 3: Integration class for your application
class DOAIntegration:
    """Main integration class for your application."""
    
    def __init__(self, db_connection, cache_client, api_client):
        # Wrap your existing services
        self.agents = [
            CacheServiceAgent(cache_client),
            DatabaseServiceAgent(db_connection),
            ExternalAPIAgent(api_client),
            TerminatorAgent()
        ]
        
        # Initialize DOA components
        self.policy = PolicyNetwork(64, len(self.agents), 128)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.001)
        self.reward_config = RewardConfig(lambda_cost_penalty=0.1)
        
        self.orchestrator = ApplicationOrchestrator(
            self.agents, self.policy, self.reward_config
        )
        
        # Load pre-trained policy if available
        self._load_policy_if_exists()
    
    def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main method to handle requests using DOA orchestration."""
        
        # Convert your request to DOA format
        task_spec = json.dumps(request_data)
        initial_state = SystemState(task_spec, [], 0, 4)
        
        # Run orchestration
        trajectory = self.orchestrator.run_episode(initial_state)
        
        # Extract results from trajectory
        results = []
        for step in trajectory.steps:
            agent = self.agents[step.agent_index]
            if agent.name != "TerminatorAgent":
                # Simulate agent execution to get output
                output = agent.execute(initial_state)
                try:
                    content = json.loads(output.content)
                    results.append({
                        "service": agent.name,
                        "result": content,
                        "cost": output.cost
                    })
                except:
                    results.append({
                        "service": agent.name,
                        "result": output.content,
                        "cost": output.cost
                    })
        
        return {
            "results": results,
            "total_cost": sum(r["cost"] for r in results),
            "success": trajectory.task_successful,
            "agent_sequence": [self.agents[s.agent_index].name for s in trajectory.steps]
        }
    
    def train_online(self, request_data: Dict[str, Any], feedback_score: float):
        """Online training based on user feedback."""
        # This would implement online learning based on real user feedback
        # For now, just a placeholder
        pass
    
    def _load_policy_if_exists(self):
        """Load pre-trained policy weights."""
        try:
            self.policy.load_state_dict(torch.load("doa_policy.pth"))
            print("✅ Loaded pre-trained DOA policy")
        except FileNotFoundError:
            print("ℹ️ No pre-trained policy found, starting fresh")
    
    def save_policy(self):
        """Save trained policy for future use."""
        torch.save(self.policy.state_dict(), "doa_policy.pth")
        print("💾 Saved DOA policy")


# Step 4: Usage example
def example_usage():
    """Example of how to use DOA in your application."""

    # Your existing services (mocked here)
    class MockDB:
        def get_user(self, user_id): return {"id": user_id, "name": "John"}
        def run_analytics_query(self, query): return {"result": "analytics_data"}

    class MockCache:
        def get(self, key): return None  # Cache miss

    class MockAPI:
        def call(self, endpoint, params): return {"data": "external_data"}

    # Initialize DOA integration
    doa = DOAIntegration(MockDB(), MockCache(), MockAPI())

    print("🔄 Training DOA with sample requests...")

    # Training requests
    training_requests = [
        {"type": "user_lookup", "user_id": 123, "cache_key": "user_123"},
        {"type": "external_data", "endpoint": "/api/data", "cache_key": "ext_data"},
        {"type": "analytics", "query": "SELECT COUNT(*) FROM users"},
        {"type": "user_lookup", "user_id": 456, "cache_key": "user_456"},
        {"type": "external_data", "endpoint": "/api/weather", "cache_key": "weather"},
    ] * 10  # Repeat for training

    # Quick training loop
    from doa_framework import REINFORCETrainer
    trainer = REINFORCETrainer(doa.policy, doa.optimizer, doa.reward_config)

    trajectories = []
    for i, request in enumerate(training_requests):
        task_spec = json.dumps(request)
        initial_state = SystemState(task_spec, [], 0, 4)
        trajectory = doa.orchestrator.run_episode(initial_state)
        trajectories.append(trajectory)

        # Train in batches
        if len(trajectories) >= 10:
            loss = trainer.train_batch(trajectories)
            avg_reward = sum(t.total_undiscounted_reward for t in trajectories) / len(trajectories)
            success_rate = sum(1 for t in trajectories if t.task_successful) / len(trajectories)
            print(f"Batch {i//10 + 1}: Avg Reward={avg_reward:.2f}, Success={success_rate:.1%}, Loss={loss:.2f}")
            trajectories = []

    print("\n🎯 Testing trained DOA:")

    # Test different types of requests
    test_requests = [
        {"type": "user_lookup", "user_id": 999, "cache_key": "user_999"},
        {"type": "external_data", "endpoint": "/api/data", "cache_key": "ext_data"},
        {"type": "analytics", "query": "SELECT COUNT(*) FROM orders"}
    ]

    for request in test_requests:
        print(f"\n📋 Request: {request}")
        result = doa.handle_request(request)
        print(f"🎯 Result: Success={result['success']}, Cost={result['total_cost']:.2f}")
        print(f"   Agent Sequence: {' → '.join(result['agent_sequence'])}")

    # Save the learned policy
    doa.save_policy()


if __name__ == "__main__":
    example_usage()
