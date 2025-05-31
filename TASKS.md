Okay, this is a great exercise! Here's a granular, step-by-step task plan to build the MVP (P0 features) of the Dynamic Orchestrator Agent (DOA) framework, tailored for a senior software engineer or an engineering LLM.

**Project Setup & Core Data Structures (Phase 0.1)**

1.  **Task 0.1.1: Initialize Project Structure**
    *   Start: Create a new Python project (e.g., using Poetry or pipenv).
    *   End: Basic directory structure (`doa_framework/`, `tests/`, `examples/`, `pyproject.toml` or `requirements.txt`) committed to Git.

2.  **Task 0.1.2: Define `AgentOutput` Dataclass**
    *   Start: Create `doa_framework/structs.py`.
    *   End: Implement `@dataclass AgentOutput(content: Any, cost: float, metadata: Optional[dict] = None)` for standardized agent return.

3.  **Task 0.1.3: Define `SystemState` Dataclass**
    *   Start: In `doa_framework/structs.py`.
    *   End: Implement `@dataclass SystemState(task_specification: str, history: List[Tuple[str, AgentOutput]], current_step: int, max_steps: int, custom_data: Optional[dict] = None)` to represent the global task state. `history` stores (agent_name, agent_output).

4.  **Task 0.1.4: Define `AgentInterface` Abstract Base Class**
    *   Start: Create `doa_framework/agents/base.py`.
    *   End: Implement `class AgentInterface(ABC): @abstractmethod def execute(self, state: SystemState) -> AgentOutput: pass`. Include a `name: str` property.

**Basic Agents (Phase 0.2)**

5.  **Task 0.2.1: Implement `TerminatorAgent`**
    *   Start: Create `doa_framework/agents/core_agents.py`.
    *   End: Implement `class TerminatorAgent(AgentInterface)` whose `execute` method returns an `AgentOutput` indicating termination (e.g., `content="TERMINATE"`, `cost=0.0`). Set `name="TerminatorAgent"`.

6.  **Task 0.2.2: Implement `EchoAgent` (Simple Test Agent)**
    *   Start: In `doa_framework/agents/core_agents.py`.
    *   End: Implement `class EchoAgent(AgentInterface)` whose `execute` method returns an `AgentOutput` with `content` being the `task_specification` from the input `SystemState` and a fixed `cost` (e.g., 1.0). Set `name="EchoAgent"`.

**Orchestrator Core - Serialized Execution (Phase 0.3)**

7.  **Task 0.3.1: Define `Orchestrator` Class Skeleton**
    *   Start: Create `doa_framework/orchestrator.py`.
    *   End: Implement `class Orchestrator:` with an `__init__(self, agents: List[AgentInterface], policy: 'PolicyNetwork', reward_config: 'RewardConfig')` and a `run_episode(self, initial_state: SystemState) -> 'EpisodeTrajectory'` method signature.

8.  **Task 0.3.2: Implement `Orchestrator._update_state` Method**
    *   Start: In `Orchestrator` class.
    *   End: Implement private method `_update_state(self, current_state: SystemState, agent_name: str, agent_output: AgentOutput) -> SystemState` that appends to history and increments `current_step`.

9.  **Task 0.3.3: Implement `Orchestrator._is_terminated` Method**
    *   Start: In `Orchestrator` class.
    *   End: Implement private method `_is_terminated(self, current_state: SystemState, last_agent_name: str) -> bool` checking `current_step >= max_steps` or if `last_agent_name == "TerminatorAgent"`.

10. **Task 0.3.4: Implement Initial `Orchestrator.run_episode` Loop Logic (without RL specifics yet)**
    *   Start: In `Orchestrator.run_episode`.
    *   End: Loop:
        1.  (Placeholder for policy-based agent selection - for now, select agents sequentially or randomly from the `self.agents` list, excluding Terminator until last step for testing).
        2.  Execute selected agent.
        3.  Update state using `_update_state`.
        4.  Check for termination using `_is_terminated`.
        5.  (Placeholder for trajectory data collection).
        Return a dummy/empty trajectory for now.

**Policy Network (Phase 0.4 - PyTorch)**

11. **Task 0.4.1: Define `PolicyNetwork` Class (PyTorch `nn.Module`)**
    *   Start: Create `doa_framework/policy.py`. Import `torch` and `torch.nn`.
    *   End: Implement `class PolicyNetwork(nn.Module):` with `__init__(self, state_embedding_dim: int, num_agents: int, hidden_dim: int)` and a `forward(self, state_embedding: torch.Tensor) -> torch.Tensor` (outputting logits for agent selection). Use a simple MLP (e.g., Linear -> ReLU -> Linear).

12. **Task 0.4.2: Implement Basic `PolicyNetwork._embed_state` Placeholder**
    *   Start: In `PolicyNetwork` class.
    *   End: Implement a private method `_embed_state(self, system_state: SystemState) -> torch.Tensor`. For MVP, this can be a very simple fixed-size random tensor or a tensor of zeros with `state_embedding_dim`. This is a CRITICAL part that will need sophisticated implementation later.

13. **Task 0.4.3: Implement `PolicyNetwork.select_action` Method**
    *   Start: In `PolicyNetwork` class.
    *   End: Implement `select_action(self, system_state: SystemState) -> Tuple[int, torch.Tensor]` that:
        1.  Calls `_embed_state`.
        2.  Passes embedding to `forward` to get logits.
        3.  Applies `F.softmax` to get probabilities.
        4.  Uses `torch.multinomial` to sample an agent index.
        5.  Returns the selected agent index and its log-probability.

**Reward Shaping (Phase 0.5)**

14. **Task 0.5.1: Define `RewardConfig` Dataclass**
    *   Start: In `doa_framework/structs.py`.
    *   End: Implement `@dataclass RewardConfig(lambda_cost_penalty: float = 0.1, gamma_discount_factor: float = 0.99, task_success_bonus: float = 1.0, task_failure_penalty: float = -1.0, step_cost_scale_factor: float = 1.0)`.

15. **Task 0.5.2: Implement `calculate_reward` Function**
    *   Start: Create `doa_framework/rewards.py`.
    *   End: Implement `calculate_reward(agent_output: AgentOutput, is_terminal: bool, task_was_successful: bool, reward_config: RewardConfig) -> float`.
        *   If `is_terminal`: `(task_success_bonus if task_was_successful else task_failure_penalty) - reward_config.lambda_cost_penalty * agent_output.cost * reward_config.step_cost_scale_factor`.
        *   Else (step reward): `-reward_config.lambda_cost_penalty * agent_output.cost * reward_config.step_cost_scale_factor`.

**Trajectory Collection (Phase 0.6)**

16. **Task 0.6.1: Define `TrajectoryStep` Dataclass**
    *   Start: In `doa_framework/structs.py`.
    *   End: Implement `@dataclass TrajectoryStep(state_embedding: torch.Tensor, agent_index: int, log_prob: torch.Tensor, reward: float, next_state_embedding: Optional[torch.Tensor], is_terminal_step: bool)`.

17. **Task 0.6.2: Define `EpisodeTrajectory` Dataclass**
    *   Start: In `doa_framework/structs.py`.
    *   End: Implement `@dataclass EpisodeTrajectory(steps: List[TrajectoryStep], total_undiscounted_reward: float, task_successful: bool)`.

18. **Task 0.6.3: Integrate Trajectory Collection into `Orchestrator.run_episode`**
    *   Start: Modify `Orchestrator.run_episode`.
    *   End:
        1.  Inside the loop, after an agent executes and state updates:
            *   Get `state_embedding` from the policy using `_embed_state(current_state)`.
            *   Call `policy.select_action(current_state)` to get `agent_idx`, `log_prob`.
            *   Execute `self.agents[agent_idx]`.
            *   Calculate `reward` using `calculate_reward`.
            *   Get `next_state_embedding` (or None if terminal).
            *   Store `TrajectoryStep`.
        2.  Return `EpisodeTrajectory` populated with all steps.
        3.  (Note: `task_successful` for `calculate_reward` and `EpisodeTrajectory` needs a simple heuristic for MVP, e.g., if TerminatorAgent was called *not* due to `max_steps`).

**REINFORCE Trainer (Phase 0.7)**

19. **Task 0.7.1: Define `REINFORCETrainer` Class Skeleton**
    *   Start: Create `doa_framework/trainer.py`.
    *   End: Implement `class REINFORCETrainer:` with `__init__(self, policy_network: PolicyNetwork, optimizer: torch.optim.Optimizer, reward_config: RewardConfig)` and a `train_batch(self, trajectories: List[EpisodeTrajectory]) -> float` method (returning total loss).

20. **Task 0.7.2: Implement Discounted Returns Calculation in `REINFORCETrainer`**
    *   Start: In `REINFORCETrainer.train_batch`.
    *   End: For each trajectory, calculate discounted returns `G_t = Σ_{k=t}^{T} γ^{k-t} * r_k` for each step, going backwards from the end of the episode. Store these.

21. **Task 0.7.3: Implement Policy Loss Calculation in `REINFORCETrainer`**
    *   Start: In `REINFORCETrainer.train_batch`.
    *   End: Calculate total policy loss: `L = - Σ_trajectories Σ_steps (log_prob_t * G_t)`. Ensure correct sign for gradient ascent.

22. **Task 0.7.4: Implement Optimizer Step in `REINFORCETrainer`**
    *   Start: In `REINFORCETrainer.train_batch`.
    *   End: Perform `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`.

**Main Training Loop (Phase 0.8 - MVP Script)**

23. **Task 0.8.1: Create `run_mvp_training.py` Script Skeleton**
    *   Start: Create `examples/run_mvp_training.py`.
    *   End: Basic script structure with imports and a `main()` function.

24. **Task 0.8.2: Initialize MVP Components in `run_mvp_training.py`**
    *   Start: In `main()`.
    *   End:
        *   Instantiate `AgentPool`: `agents = [EchoAgent(name="Agent0"), TerminatorAgent(name="Agent1")]`.
        *   Instantiate `RewardConfig`.
        *   Instantiate `PolicyNetwork` (define `state_embedding_dim`, `num_agents=len(agents)`, `hidden_dim`).
        *   Instantiate `torch.optim.Adam` for the `PolicyNetwork`'s parameters.
        *   Instantiate `Orchestrator`.
        *   Instantiate `REINFORCETrainer`.

25. **Task 0.8.3: Implement Outer Training Loop in `run_mvp_training.py`**
    *   Start: In `main()`.
    *   End: Loop for `N_EPOCHS`:
        1.  Initialize `batch_trajectories = []`.
        2.  Loop for `K_EPISODES_PER_EPOCH`:
            *   Create `initial_system_state` (fixed `task_specification`, `max_steps=4`).
            *   `trajectory = orchestrator.run_episode(initial_system_state)`.
            *   Add `trajectory` to `batch_trajectories`.
        3.  `loss = trainer.train_batch(batch_trajectories)`.
        4.  Print epoch number, average total reward for the batch, and loss.

**Final MVP Polish (Phase 0.9)**

26. **Task 0.9.1: Basic Configuration for `max_steps`**
    *   Start: Review `SystemState` and `Orchestrator`.
    *   End: Ensure `max_steps` is passed into `SystemState` and correctly used by `_is_terminated`.

27. **Task 0.9.2: Simple `task_successful` Heuristic for MVP**
    *   Start: Review `Orchestrator.run_episode` and `calculate_reward`.
    *   End: Implement a simple check: `task_successful = True` if the last agent in trajectory was `TerminatorAgent` AND `current_step < max_steps`. Otherwise `False`. This is passed to `calculate_reward` for the terminal step and stored in `EpisodeTrajectory`.

28. **Task 0.9.3: Ensure `PolicyNetwork._embed_state` Handles `SystemState` (Basic)**
    *   Start: Review `PolicyNetwork._embed_state`.
    *   End: Modify the placeholder to at least generate a tensor of consistent shape `state_embedding_dim` based on *some* aspect of `SystemState` (e.g., hash of `task_specification` modulo `embedding_dim` values, or simply number of items in history). The actual content doesn't need to be meaningful for the first runnable MVP, just functional.

29. **Task 0.9.4: Map Agent Index to Agent Name in Orchestrator**
    *   Start: Review `Orchestrator` and `PolicyNetwork.select_action`.
    *   End: Ensure `Orchestrator` uses the `agent_index` from the policy to select the correct agent from its `self.agents` list and uses the agent's `name` when calling `_update_state`.

30. **Task 0.9.5: Write a README.md for MVP**
    *   Start: Create `README.md`.
    *   End: Brief description of the MVP, how to set up, and how to run `examples/run_mvp_training.py`.

This detailed plan should provide enough granularity for systematic implementation of the MVP. Each task is small and focused. Good luck to the engineering LLM!