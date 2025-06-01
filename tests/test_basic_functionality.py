"""
Basic functionality tests for the DOA Framework.
"""

import pytest
import torch
import torch.optim as optim

from doa_framework import (
    AgentOutput, SystemState, RewardConfig, TrajectoryStep, EpisodeTrajectory,
    AgentInterface, TerminatorAgent, EchoAgent, Orchestrator, 
    PolicyNetwork, REINFORCETrainer, calculate_reward
)


def test_agent_output_creation():
    """Test AgentOutput dataclass creation."""
    output = AgentOutput(content="test", cost=1.5, metadata={"key": "value"})
    assert output.content == "test"
    assert output.cost == 1.5
    assert output.metadata["key"] == "value"


def test_system_state_creation():
    """Test SystemState dataclass creation."""
    state = SystemState(
        task_specification="Test task",
        history=[],
        current_step=0,
        max_steps=4
    )
    assert state.task_specification == "Test task"
    assert len(state.history) == 0
    assert state.current_step == 0
    assert state.max_steps == 4


def test_terminator_agent():
    """Test TerminatorAgent functionality."""
    agent = TerminatorAgent()
    state = SystemState("test", [], 0, 4)
    
    output = agent.execute(state)
    assert output.content == "TERMINATE"
    assert output.cost == 0.0
    assert agent.name == "TerminatorAgent"


def test_echo_agent():
    """Test EchoAgent functionality."""
    agent = EchoAgent()
    state = SystemState("Hello World", [], 0, 4)
    
    output = agent.execute(state)
    assert "Hello World" in output.content
    assert output.cost == 1.0
    assert agent.name == "EchoAgent"


def test_policy_network():
    """Test PolicyNetwork basic functionality."""
    policy = PolicyNetwork(state_embedding_dim=32, num_agents=2, hidden_dim=64)
    
    # Test forward pass
    state_embedding = torch.randn(32)
    logits = policy.forward(state_embedding)
    assert logits.shape == (2,)
    
    # Test action selection
    state = SystemState("test", [], 0, 4)
    action_idx, log_prob = policy.select_action(state)
    assert isinstance(action_idx, int)
    assert 0 <= action_idx < 2
    assert isinstance(log_prob, torch.Tensor)


def test_reward_calculation():
    """Test reward calculation function."""
    config = RewardConfig()
    output = AgentOutput("test", cost=2.0)
    
    # Test terminal successful
    reward = calculate_reward(output, is_terminal=True, task_was_successful=True, reward_config=config)
    expected = config.task_success_bonus - config.lambda_cost_penalty * output.cost
    assert abs(reward - expected) < 1e-6
    
    # Test terminal failed
    reward = calculate_reward(output, is_terminal=True, task_was_successful=False, reward_config=config)
    expected = config.task_failure_penalty - config.lambda_cost_penalty * output.cost
    assert abs(reward - expected) < 1e-6
    
    # Test intermediate step
    reward = calculate_reward(output, is_terminal=False, task_was_successful=False, reward_config=config)
    expected = -config.lambda_cost_penalty * output.cost
    assert abs(reward - expected) < 1e-6


def test_orchestrator_basic():
    """Test basic Orchestrator functionality."""
    agents = [EchoAgent(), TerminatorAgent()]
    policy = PolicyNetwork(state_embedding_dim=32, num_agents=2)
    config = RewardConfig()
    
    orchestrator = Orchestrator(agents, policy, config)
    
    initial_state = SystemState("test task", [], 0, 4)
    trajectory = orchestrator.run_episode(initial_state)
    
    assert isinstance(trajectory, EpisodeTrajectory)
    assert len(trajectory.steps) > 0
    assert isinstance(trajectory.total_undiscounted_reward, float)
    assert isinstance(trajectory.task_successful, bool)


def test_reinforce_trainer():
    """Test REINFORCETrainer basic functionality."""
    policy = PolicyNetwork(state_embedding_dim=32, num_agents=2)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    config = RewardConfig()
    
    trainer = REINFORCETrainer(policy, optimizer, config)
    
    # Create dummy trajectory with proper gradients
    steps = [
        TrajectoryStep(
            state_embedding=torch.randn(32),
            agent_index=0,
            log_prob=torch.tensor(-0.5, requires_grad=True),
            reward=1.0,
            next_state_embedding=torch.randn(32),
            is_terminal_step=False
        ),
        TrajectoryStep(
            state_embedding=torch.randn(32),
            agent_index=1,
            log_prob=torch.tensor(-0.3, requires_grad=True),
            reward=0.5,
            next_state_embedding=None,
            is_terminal_step=True
        )
    ]
    
    trajectory = EpisodeTrajectory(steps=steps, total_undiscounted_reward=1.5, task_successful=True)
    
    # Test training
    loss = trainer.train_batch([trajectory])
    assert isinstance(loss, float)


def test_integration():
    """Test full integration of all components."""
    # Setup
    agents = [EchoAgent(), TerminatorAgent()]
    policy = PolicyNetwork(state_embedding_dim=32, num_agents=2)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    config = RewardConfig()
    
    orchestrator = Orchestrator(agents, policy, config)
    trainer = REINFORCETrainer(policy, optimizer, config)
    
    # Run a few episodes and train
    trajectories = []
    for _ in range(3):
        initial_state = SystemState("test task", [], 0, 4)
        trajectory = orchestrator.run_episode(initial_state)
        trajectories.append(trajectory)
    
    # Train on batch
    loss = trainer.train_batch(trajectories)
    assert isinstance(loss, float)
    
    print(f"Integration test passed! Loss: {loss:.4f}")


if __name__ == "__main__":
    pytest.main([__file__])
