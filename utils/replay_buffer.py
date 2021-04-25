import numpy as np
import random
import torch

from utils.segment_tree import SumSegmentTree, MinSegmentTree

class ReplayBuffer(object):
	def __init__(self, max_size):
		self._storage = []
		self._maxsize = max_size
		self.ptr = 0

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def __len__(self):
		return len(self._storage)

	def add(self, state, action, next_state, reward, done):
		data = (state, action, next_state, reward, 1. - done)
		if self.ptr >= len(self._storage):
			self._storage.append(data)
		else:
			self._storage[self.ptr] = data
		self.ptr = (self.ptr + 1) % self._maxsize

	def _encode_sample(self, idxes):
		states, actions, next_states, rewards, not_dones = [], [], [], [], []
		for i in idxes:
			data = self._storage[i]
			state, action, next_state, reward, not_done = data
			states.append(np.array(state, copy=False))
			actions.append(np.array(action, copy=False))
			next_states.append(np.array(next_state, copy=False))
			rewards.append(reward)
			not_dones.append(not_done)
		states, actions, next_states = np.array(states), np.array(actions), np.array(next_states)
		rewards, not_dones = np.array(rewards), np.array(not_dones)
		return (
			torch.as_tensor(states,      device=self.device, dtype=torch.float32),
			torch.as_tensor(actions,     device=self.device, dtype=torch.float32),
			torch.as_tensor(next_states, device=self.device, dtype=torch.float32),
			torch.as_tensor(rewards,     device=self.device, dtype=torch.float32),
			torch.as_tensor(not_dones,   device=self.device, dtype=torch.float32),
		)

	def sample(self, batch_size):
		idxes = [random.randint(0, len(self._storage)-1) for _ in range(batch_size)]
		return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
	def __init__(self, max_size, alpha):
		"""Create Prioritized Replay buffer.

		Parameters
		----------
		max_size: int
			Max number of transitions to store in the buffer. 
			When the buffer overflows the old memories are dropped.
		alpha: float
			how much prioritization is used (0: no prioritization, 1: full prioritization)
		
		See Also
		--------
		ReplayBuffer.__init__
		"""
		super(PrioritizedReplayBuffer, self).__init__(max_size)
		assert alpha >= 0
		self._alpha = alpha

		it_capacity = 1
		while it_capacity < max_size:
			it_capacity *= 2

		self._it_sum = SumSegmentTree(it_capacity)
		self._it_min = MinSegmentTree(it_capacity)
		self._max_priority = 1.0

	def add(self, state, action, next_state, reward, done):
		"""See ReplayBuffer.add"""
		idx = self.ptr
		super().add(state, action, next_state, reward, done)
		self._it_sum[idx] = self._max_priority ** self._alpha
		self._it_min[idx] = self._max_priority ** self._alpha

	def _sample_proportional(self, batch_size):
		res = []
		p_total = self._it_sum.sum(0, len(self._storage)-1)
		every_range_len = p_total / batch_size
		for i in range(batch_size):
			mass = random.random() * every_range_len + i * every_range_len
			idx = self._it_sum.find_prefixsum_idx(mass)
			res.append(idx)
		return res

	def sample(self, batch_size, beta):
		"""Sample a batch of experiences.

		compared to ReplayBuffer.sample
		it also returns importance weights and idxes of sampled experiences.
		
		Parameters
		----------
		batch_size: int
			How many transitions to sample.
		beta: float
			To what degree to use importance weights (0: no corrections, 1: full correction)
		
		Returns
		-------
		obs_batch: np.array
			batch of observations
		act_batch: np.array
			batch of actions executed given obs_batch
		rew_batch: np.array
			rewards received as results of executing act_batch
		next_obs_batch: np.array
			next set of observations seen after executing act_batch
		done_mask: np.array
			done_mask[i] = 1 if executing act_batch[i] resulted in
			the end of an episode and 0 otherwise.
		weights: np.array
			Array of shape (batch_size,) and dtype np.float32
			denoting importance weight of each sampled transition
		idxes: np.array
			Array of shape (batch_size,) and dtype np.int32
			idexes in buffer of sampled experiences
		"""
		assert beta > 0
		idxes = self._sample_proportional(batch_size)

		weights = []
		p_min = self._it_min.min() / self._it_sum.sum()
		max_weight = (p_min * len(self._storage)) ** (-beta)

		for idx in idxes:
			p_sample = self._it_sum[idx] / self._it_sum.sum()
			weight = (p_sample * len(self._storage)) ** (-beta)
			weights.append(weight / max_weight)
		weights = np.array(weights)
		encoded_sample = self._encode_sample(idxes)
		return tuple(list(encoded_sample) + [torch.as_tensor(weights, device=self.device, dtype=torch.float32), idxes])

	def update_priorities(self, idxes, priorities):
		"""Update priorities of sampled transitions.
		Sets priority of transition at index idxes[i] in buffer to priorities[i].
		
		Parameters
		----------
		idxes: [int]
			List of idxes of sampled transitions
		priorities: [float]
			List of updated priorities corresponding to transitions at the sampled idxes denoted 
			by variable `idxes`.
		"""
		assert len(idxes) == len(priorities)
		for idx, priority in zip(idxes, priorities):
			assert priority > 0
			assert 0 <= idx < len(self._storage)
			self._it_sum[idx] = priority ** self._alpha  # update priority
			self._it_min[idx] = priority ** self._alpha

			self._max_priority = max(self._max_priority, priority)