import numpy as np

class Schedule(object):
	def update(self, t):
		raise NotImplementedError


class LinearSchedule(Schedule):
	def __init__(self, val_begin, val_end, nsteps):
		"""
		Args:
			eps_begin: initial exploration
			eps_end: end exploration
			nsteps: number of steps between the two values of eps
		"""
		self.value = val_begin
		self.val_begin = val_begin
		self.val_end = val_end
		self.nsteps = nsteps

	def update(self, t):
		if t <= self.nsteps:
			self.value = self.val_begin + t * (self.val_end - self.val_begin) / self.nsteps
		else:
			self.value = self.val_end


class ConstantSchedule(Schedule):
	def __init__(self, value):
		self.value = value

	def update(self, t):
		return


class PiecewiseSchedule(LinearSchedule):
	def __init__(self, val_begin, val_end, nsteps):
		super(PiecewiseSchedule, self).__init__(val_begin, val_end, nsteps)

	def update(self, t):
		if t <= self.nsteps:
			self.value = self.val_begin + t * (self.val_end - self.val_begin) / self.nsteps
		else:
			self.value = self.val_end
		self.value = int(self.value)


class LinearExploration(LinearSchedule):
	def __init__(self, env, val_begin, val_end, nsteps):
		self.env = env
		super(LinearExploration, self).__init__(val_begin, val_end, nsteps)

	def get_action(self, best_action):
		roll = np.random.rand()
		if roll < self.value:
			return self.env.action_space.sample()
		return best_action
