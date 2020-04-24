"""Agent and DQN training code.
"""
__author__ = 'jstol'

# Standard imports
from collections import deque
from typing import (
    Generator,
    List,
    Optional,
    Tuple,
)
import random

# Third party imports
import torch
import torch.nn.functional as F
from torch import (
    nn,
    optim,
    Tensor,
)
from unityagents import UnityEnvironment


# Global configuration
random.seed(123)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Utility classes
class DQN(nn.Module):
    def __init__(self, state_size: int, num_actions: int, hidden_sizes: Optional[List[int]] = None):
        """Creates a Deep-Q Network to estimate action-values.

        Args:
            state_size: Size of the state space.
            num_actions: Number of actions available to the agent.
            hidden_sizes: List detailing the number/sizes of the hidden layers to use.
        """
        super().__init__()

        self.state_size = state_size
        self.num_actions = num_actions
        self.hidden_sizes = hidden_sizes or [32, 16, 8]

        # Set up DNN with ReLU activations
        # Input
        layers = [
            nn.Linear(state_size, self.hidden_sizes[0]),
            nn.ReLU(),
        ]

        # Hidden layers
        for h1, h2 in [(self.hidden_sizes[i], self.hidden_sizes[i + 1]) for i in range(len(self.hidden_sizes[:-1]))]:
            layers += [
                nn.Linear(h1, h2),
                nn.ReLU(),
            ]

        # Output
        layers += [
            nn.Linear(self.hidden_sizes[-1], self.num_actions),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, input_states: Tensor) -> Tensor:
        return self.layers(input_states)


class Agent:
    def __init__(self, state_size: int, num_actions: int, update_freq: int, buffer_size: Optional[int] = int(2e4),
                 batch_size: Optional[int] = 64, gamma: Optional[float] = 0.99, lr: Optional[float] = 1e-3,
                 tau: Optional[float] = 1e-3):
        """Creates an RL agent that makes use of DQNs to estimate action-values.

        Args:
            state_size: Size of the state space.
            num_actions: Number of (discrete) actions available to the agent.
            update_freq: Frequency to update the DQNs.
            buffer_size: Size of the replay buffer to maintain.
            batch_size: Size of the batch to use for SGD.
            gamma: Reward discount factor.
            lr: Learning rate to use (Adam).
            tau: The mixing factor to use when updating the target DQN. If None, use hard updates.
        """
        # Env variables
        self.state_size = state_size
        self.num_actions = num_actions
        self.update_freq = update_freq

        # Models
        self.dqn = DQN(state_size, num_actions).to(device)
        self.target_dqn = DQN(state_size, num_actions).to(device)

        # Learning vars
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)

        # Episode vars
        self.step_t_mod = 0

    def _act_greedy(self, state: List[float]) -> int:
        """Pick the best possible action based on our estimated action-values (provided by the DQN).

        Returns:
            The best action to take.
        """
        state_tensor = Tensor(state).float().unsqueeze(0).to(device)
        return self.dqn(state_tensor).argmax().item()

    def _act_random(self) -> int:
        """Uniformly pick a random action based on the number of available actions.

        Returns:
            The random action to take.
        """
        return random.randrange(self.num_actions)

    def act(self, state: List[float], eps: Optional[float] = 0.0) -> int:
        """Pick an action based on an epsilon-greedy policy.

        Args:
            state: The state to choose an action for.
            eps: Epsilon value to use when making epsilon-greedy decision (probability of taking a random action).

        Returns:
            The action to take.
        """
        use_greedy = random.random() >= eps
        with torch.no_grad():
            action = self._act_greedy(state) if use_greedy else self._act_random()
        return action

    def _update_target_dqn(self):
        """Update target_dqn model parameters.

        (Function based on Udacity DeepRL DQN homework code).
        θ_target_dqn = [τ * θ_dqn] + [(1 - τ) * θ_target_dqn]
        """
        if self.tau:
            for dqn_param, target_dqn_param in zip(self.dqn.parameters(), self.target_dqn.parameters()):
                target_dqn_param.data.copy_(self.tau * dqn_param.data + (1.0 - self.tau) * target_dqn_param.data)
        else:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

    def step(self, state: List[float], action: int, reward: float, next_state: List[float]):
        """Update the agent's replay buffer and DQNs (if applicable).

        Args:
            state: Initial state.
            action: Action taken in initial state.
            reward: Reward received from taking the action in given state.
            next_state: Next state that was observed.
        """
        # Add experience to the replay buffer
        self.replay_buffer.append((state, action, reward, next_state))
        self.step_t_mod = (self.step_t_mod + 1) % self.update_freq

        # If it's time to learn, run through the learning process
        if len(self.replay_buffer) >= self.batch_size and self.step_t_mod == 0:
            experience_batch = random.sample(self.replay_buffer, k=self.batch_size)

            states, actions, rewards, next_states = zip(*experience_batch)
            states = Tensor(states).float().to(device)
            actions = Tensor(actions).long().to(device).unsqueeze(-1)
            rewards = Tensor(rewards).float().to(device).unsqueeze(-1)
            next_states = Tensor(next_states).float().to(device)

            # Forward pass
            predictions = self.dqn(states).gather(-1, actions)
            target_predictions = self.target_dqn(next_states).detach().max(dim=-1)[0].unsqueeze(-1)
            G = rewards + self.gamma * target_predictions

            # Backprop error and update weights
            self.optimizer.zero_grad()
            loss = F.mse_loss(predictions, G)
            loss.backward()
            self.optimizer.step()

            # Update Target DQN
            self._update_target_dqn()


# Helper functions
def _create_eps_generator(eps_start: float, eps_min: float, eps_decay: float) -> Generator[float, None, None]:
    """Creates a generator that yields a decaying epsilon value.

    Args:
        eps_start: Starting epsilon value.
        eps_min: Minimum allowable epsilon value.
        eps_decay: Epsilon decay factor.

    Returns:
        Generator: Generator that yields sequence of epsilon values.
    """
    eps = eps_start
    while True:
        yield eps
        eps = max(eps_min, eps * eps_decay)


def env_step(env: UnityEnvironment, brain_name: str, action: int) -> Tuple[List[float], float, bool]:
    """Helper function to wrap Unity env.step

    Args:
        env: An instance of the environment.
        brain_name: The name of the Udacity "brain" to use.
        action: The action that has been selected.

    Returns:
        A tuple of the state transitioned to, the reward received, and whether or not the episode has finished.
    """
    env_info = env.step(action)[brain_name]
    reward = env_info.rewards[0]
    state = env_info.vector_observations[0]
    done = env_info.local_done[0]
    return state, reward, done


def create_default_agent(env: UnityEnvironment) -> Agent:
    """Helper function to create a pre-configured Agent based on a given environment

    Args:
        env: Environment to create the agent for.

    Returns:
        The new Agent.
    """
    # Set up hyperparams
    update_freq = 4
    buffer_size = int(2e4)
    batch_size = 64
    gamma = 0.99
    lr = 1e-3

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    num_actions = brain.vector_action_space_size
    state_size = brain.vector_observation_space_size

    # Set up agent
    return Agent(state_size, num_actions, update_freq, buffer_size, batch_size, gamma, lr)


def train(env: UnityEnvironment, eps_start: float = 1.0, eps_min: float = 0.05, eps_decay: float = 0.998,
          max_num_episodes=2000) -> (Generator[float, None, None], Agent):
    """Train a DeepRL agent on the "Banana" task.

    Args:
        env: The Unity environment.
        eps_start: Starting epsilon value.
        eps_min: Minimum allowable epsilon value.
        eps_decay: Epsilon decay factor.
        max_num_episodes: Maximum number of episodes to let the agent run for.

    Returns:
        A tuple containing a generator yielding the total score received for each individual episode and the agent.
    """
    # Fetch the brain name to interact with the environment as we train
    brain_name = env.brain_names[0]

    # Set up agent
    agent = create_default_agent(env)

    # Run each episode and yield the score
    eps_generator = _create_eps_generator(eps_start, eps_min, eps_decay)

    def train_step_generator():
        for _ in range(max_num_episodes):
            # Reset environment
            env_info = env.reset(train_mode=True)[brain_name]
            state = env_info.vector_observations[0]

            # Step through episode
            score = 0.0
            epsilon = next(eps_generator)

            while True:
                # Take an action
                action = agent.act(state, epsilon)

                # Pass to environment
                next_state, reward, done = env_step(env, brain_name, action)

                # Update the agent
                agent.step(state, action, reward, next_state)
                state = next_state

                # Update score
                score += reward

                if done:
                    break

            yield score

    return agent, train_step_generator()
