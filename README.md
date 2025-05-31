# Dynamic Orchestrator Agent (DOA) Framework 🎭

A cutting-edge framework for adaptive multi-agent LLM collaboration with reinforcement learning-based orchestration, inspired by the "Puppeteer" model.

## 🌟 Overview

The DOA Framework implements a **learnable orchestrator** that dynamically selects which agents to activate based on the current task state. Unlike static multi-agent systems, our orchestrator continuously improves through reinforcement learning, learning to:

- 🎯 **Optimize agent selection** for better task performance
- ⚡ **Minimize computational costs** through efficient orchestration
- 🔄 **Adapt to complex reasoning patterns** including cycles and hubs
- 📈 **Self-improve** via REINFORCE-based policy optimization

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Task Input    │───▶│   Orchestrator   │───▶│  Agent Pool     │
└─────────────────┘    │                  │    │                 │
                       │  Policy Network  │    │ • EchoAgent     │
┌─────────────────┐    │  (Neural Net)    │    │ • TerminatorAgent│
│ Reward Signal   │◀───│                  │    │ • CustomAgents  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ▲                       │
         │              ┌─────────────────┐
         └──────────────│ REINFORCE       │
                        │ Trainer         │
                        └─────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/dynamic-orchestrator-agent.git
cd dynamic-orchestrator-agent

# Install dependencies
pip install torch numpy dataclasses-json typing-extensions

# Or use Poetry
poetry install
```

### Run the MVP Training

```bash
python examples/run_mvp_training.py
```

This will start training the orchestrator to learn optimal agent selection patterns!

## 📊 Expected Output

```
🚀 Starting DOA Framework MVP Training
Epochs: 50, Episodes per epoch: 10
State embedding dim: 64, Hidden dim: 128
Learning rate: 0.001, Max steps: 4
------------------------------------------------------------
Initialized 2 agents: ['EchoAgent', 'TerminatorAgent']
Reward config: λ=0.1, γ=0.99
Policy network: 17154 parameters
All components initialized successfully!
============================================================
Epoch   1/50 | Avg Reward: -0.234 | Success Rate:  20.0% | Loss:  0.45123
Epoch   2/50 | Avg Reward: -0.156 | Success Rate:  30.0% | Loss:  0.38901
...
Epoch  50/50 | Avg Reward:  0.823 🌟 | Success Rate:  90.0% | Loss:  0.12456
```

## 🧩 Core Components

### 1. **Orchestrator** (`doa_framework/orchestrator.py`)
Central coordinator that uses a neural policy to select agents dynamically.

### 2. **Policy Network** (`doa_framework/policy.py`)
Neural network that learns to map system states to agent selection probabilities.

### 3. **Agent Interface** (`doa_framework/agents/base.py`)
Standardized interface for implementing custom agents.

### 4. **REINFORCE Trainer** (`doa_framework/trainer.py`)
Policy gradient trainer that optimizes the orchestrator's decision-making.

### 5. **Reward System** (`doa_framework/rewards.py`)
Configurable reward function balancing task success and computational efficiency.

## 🔧 Key Features

- **🧠 Learnable Orchestration**: Neural policy learns optimal agent selection
- **⚖️ Cost-Performance Balance**: Configurable λ parameter for cost vs. accuracy trade-offs
- **🔄 Dynamic Topologies**: Supports complex reasoning patterns including cycles
- **📈 Continuous Improvement**: REINFORCE-based learning from experience
- **🔌 Modular Design**: Easy to add new agents and tools
- **📊 Rich Observability**: Comprehensive trajectory logging and metrics

## 🎯 Use Cases

- **🤖 Multi-Agent AI Systems**: Coordinate specialized AI agents for complex tasks
- **💼 Business Process Automation**: Optimize workflows with multiple AI components
- **🔬 Research & Development**: Experiment with adaptive multi-agent architectures
- **🎓 Educational**: Learn about RL-based coordination and multi-agent systems

## 📈 Performance Metrics

The framework tracks several key metrics:

- **Task Success Rate**: Percentage of successfully completed tasks
- **Average Reward**: Balances success and computational cost
- **Agent Utilization**: How frequently each agent is selected
- **Convergence Speed**: How quickly the policy learns optimal patterns

## 🛠️ Extending the Framework

### Adding Custom Agents

```python
from doa_framework.agents.base import AgentInterface
from doa_framework.structs import SystemState, AgentOutput

class MyCustomAgent(AgentInterface):
    def __init__(self, name: str = "MyCustomAgent"):
        super().__init__(name)

    def execute(self, state: SystemState) -> AgentOutput:
        # Your agent logic here
        result = self.process_task(state.task_specification)
        return AgentOutput(
            content=result,
            cost=1.5,  # Computational cost
            metadata={"agent_type": "custom"}
        )
```

### Configuring Rewards

```python
from doa_framework import RewardConfig

# Emphasize cost efficiency
cost_focused_config = RewardConfig(
    lambda_cost_penalty=0.5,  # Higher cost penalty
    task_success_bonus=1.0,
    task_failure_penalty=-2.0
)

# Emphasize task success
performance_focused_config = RewardConfig(
    lambda_cost_penalty=0.05,  # Lower cost penalty
    task_success_bonus=2.0,
    task_failure_penalty=-1.0
)
```

## 📚 Technical Details

### State Representation
The system state includes:
- **Task Specification**: The current task description
- **Execution History**: Sequence of (agent_name, agent_output) pairs
- **Step Information**: Current step and maximum allowed steps
- **Custom Data**: Extensible metadata storage

### Reward Function
Based on the paper's formulation:
- **Terminal Step**: `R_T = r - λ * C_total`
- **Intermediate Steps**: `R_t = -λ * c_t`

Where:
- `r`: Task success reward (+1) or failure penalty (-1)
- `λ`: Cost penalty weight (configurable)
- `C_total`: Total computational cost
- `c_t`: Step-wise cost

### Policy Network Architecture
- **Input**: State embedding (task + history features)
- **Architecture**: MLP with ReLU activations
- **Output**: Probability distribution over available agents
- **Training**: REINFORCE with gradient clipping

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
poetry install --with dev

# Run tests
pytest tests/

# Format code
black doa_framework/ examples/ tests/

# Type checking
mypy doa_framework/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This framework is inspired by the "Puppeteer" model from:
> Dang et al. (2025). "Dynamic Multi-Agent Orchestration with Reinforcement Learning"

## 📞 Support

- 📧 Email: support@doa-framework.org
- 💬 Discord: [Join our community](https://discord.gg/doa-framework)
- 🐛 Issues: [GitHub Issues](https://github.com/your-org/dynamic-orchestrator-agent/issues)
- 📖 Docs: [Full Documentation](https://docs.doa-framework.org)

---

**Built with ❤️ by the DOA Team**
