import torch
import torch.nn as nn
from torch.distributions import Categorical
from collections import namedtuple

from rlcard.utils.utils import remove_illegal

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'legal_actions', 'done'])


class PPOAgent(object):
    def __init__(self,
                 discount_factor=0.99,
                 num_actions=2,
                 state_shape=None,
                 mlp_layers=[256, 256],
                 learning_rate=0.0001,
                 device=None):

        self.use_raw = False
        self.discount_factor = discount_factor
        self.eps_clip = 0.2
        self.k_epochs = 10
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_shape[0], num_actions, mlp_layers[0], mlp_layers[1])
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': learning_rate*0.1},
            {'params': self.policy.critic.parameters(), 'lr': learning_rate}
        ])

        self.policy_old = ActorCritic(state_shape[0], num_actions, mlp_layers[0], mlp_layers[1])
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.mse_loss = nn.MSELoss()
        self.set_device(device)

    def feed(self, ts):
        # store buffer, should be same size in select_action()
        (state, action, reward, next_state, done) = tuple(ts)
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)

    def select_action(self, state, training):
        # self.steps +=1
        # print(state)
        # if self.steps == 3:
        #     exit()
        #TODO: add more feature
        with torch.no_grad():
            obs = torch.FloatTensor(state["obs"]).to(self.device)
            legal_actions = list(state['legal_actions'].keys())
            action, action_logprob = self.policy_old.act(obs, legal_actions)
        if training:
            self.buffer.states.append(obs)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
        return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.discount_factor * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        # if rewards.shape[0] != 1:
        #     rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.k_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.mse_loss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def step(self, state):
        ''' Predict the action for generating training data but
            have the predictions disconnected from the computation graph

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
        '''
        action = self.select_action(state, training=True)
        return action

    def eval_step(self, state):
        ''' Predict the action for evaluation purpose.

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
            info (dict): A dictionary containing information
        '''

        action = self.select_action(state, training=False)
        return action, {}

    def set_device(self, device):
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.device = device
        self.policy.device = device
        self.policy = self.policy.to(self.device)
        self.policy_old.device = device
        self.policy_old = self.policy_old.to(self.device)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, num_actions, actor_dim, critic_dim):
        super(ActorCritic, self).__init__()
        self.device = "cpu"
        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, actor_dim),
            nn.Tanh(),
            nn.Linear(actor_dim, actor_dim),
            nn.Tanh(),
            nn.Linear(actor_dim, num_actions),
            nn.Softmax(dim=-1)
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, critic_dim),
            nn.Tanh(),
            nn.Linear(critic_dim, critic_dim),
            nn.Tanh(),
            nn.Linear(critic_dim, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, obs, legal_actions):
        action_probs = self.actor(obs)
        action_probs = torch.FloatTensor(remove_illegal(action_probs, legal_actions)).to(self.device)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, obs, action):
        action_probs = self.actor(obs)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(obs)
        return action_logprobs, state_values, dist_entropy


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def __str__(self):
        return str({
            "actions": self.actions,
            "states": self.states,
            "logprobs": self.logprobs,
            "rewards": self.rewards,
            "is_terminals": self.is_terminals
        })
