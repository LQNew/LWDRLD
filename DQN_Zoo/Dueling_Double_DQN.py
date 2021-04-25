import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementation of Deep Reinforcement Learning with Double Q-Learning (Double DQN)
# Paper: https://arxiv.org/abs/1509.06461
# Implementation of Dueling Network Architectures for Deep Reinforcement Learning (Dueling DQN)
# Paper: https://arxiv.org/abs/1511.06581

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weight_init(m):
	"""Custom weight init for Conv2D and Linear layers."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		# delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
		assert m.weight.size(2) == m.weight.size(3)
		m.weight.data.fill_(0.0)
		m.bias.data.fill_(0.0)
		mid = m.weight.size(2) // 2
		gain = nn.init.calculate_gain("relu")
		nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


# Convolutional network structure for Atari
class ConvNet(nn.Module):
	def __init__(self, action_dim, in_channels=4):
		super(ConvNet, self).__init__()

		self.feature_dim = 512
		self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)  # [84, 84, 4] --> [20, 20, 32]
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)  # [20, 20, 32] --> [9, 9, 64]
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)  # [9, 9, 64] --> [7, 7, 64]
		self.fc4 = nn.Linear(7*7*64, self.feature_dim)
		# state value
		self.fc_value = nn.Linear(self.feature_dim, 1)
		# advantage value
		self.fc_advantage = nn.Linear(self.feature_dim, action_dim)
		self.apply(weight_init)
		
	def forward(self, state):
		q = F.relu(self.conv1(state / 255.0))
		q = F.relu(self.conv2(q))
		q = F.relu(self.conv3(q))
		q = q.view(q.size(0), -1)
		q = F.relu(self.fc4(q))
		v_value = self.fc_value(q)
		advantange = self.fc_advantage(q)
		q_action = v_value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
		return q_action


class DuelingDoubleDQN(object):
	def __init__(
		self,
		action_dim,
		discount=0.99, 
		learning_rate=1e-4,
		policy_freq=1e4,
		gradient_clip=10.0,
	):
		self.action_dim = action_dim
		self.q_net, self.target_q_net = ConvNet(self.action_dim).to(device), ConvNet(self.action_dim).to(device)
		self.target_q_net.load_state_dict(self.q_net.state_dict())
		self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

		self.discount = discount
		self.policy_freq = policy_freq
		self.gradient_clip = gradient_clip

		self.train_steps = 0
		
	def _range_tensor(self, end):
		return torch.arange(end).long().to(device)

	def select_action(self, states, epsilon):
		states = torch.as_tensor(states, device=device, dtype=torch.float32)
		if np.random.uniform() >= epsilon:
			# greedy action selection
			action_value = self.q_net(states)
			action = torch.argmax(action_value, dim=1).cpu().data.numpy()
		else:
			# random action selection
			action = np.random.randint(0, self.action_dim, (states.size(0)))
		return action

	def train(self, replay_buffer, batch_size=64):
		self.train_steps += 1

		# Sample batches from replay buffer 
		states, actions, next_states, rewards, not_dones = replay_buffer.sample(batch_size)

		batch_indices = self._range_tensor(batch_size)
		# Compute the target Q value
		with torch.no_grad():
			target_Q = self.target_q_net(next_states)
			best_actions = torch.argmax(self.q_net(next_states), dim=-1).long()
			target_Q = target_Q[batch_indices, best_actions]
			target_Q = rewards + not_dones * self.discount * target_Q

		# Get current Q estimates
		current_Q = self.q_net(states)
		current_Q = current_Q[batch_indices, actions.long()]

		# Compute Q loss
		loss = F.mse_loss(current_Q, target_Q)
		
		# Optimize the Q function
		self.optimizer.zero_grad()
		loss.backward()
		nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=self.gradient_clip, norm_type=2.0)  # perform gradient clip
		self.optimizer.step()

		# target parameter update
		if self.train_steps % self.policy_freq == 0:
			self.target_q_net.load_state_dict(self.q_net.state_dict())
	
	# save the model
	def save(self, filename):
		torch.save(self.q_net.state_dict(), filename + "_q_net")

	# load the model
	def load(self, filename):
		self.q_net.load_state_dict(torch.load(filename + "_q_net"))
		self.target_q_net.load_state_dict(self.q_net.state_dict())