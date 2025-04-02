from torch.distributions import Categorical
import torch


class BaseGradientEstimator:
	def __init__(self, reward):
		self.reward = reward

	def __call__(self, input_dict, logits):
		"""
			input_dict - whatever is needed to compute the reward, everything should respect logits order
			logits of shape (bsize, N, |Y|)

			Returns surrogate float to minimize and statistics dict, i.e., R_argmax, R_sample and etc, for monitoring reasons
		"""
		raise NotImplementedError


class RaoMarginalized(BaseGradientEstimator):
	def __init__(self, reward, with_argmax_baseline=True):
		super().__init__(reward)
		self.with_argmax_baseline = with_argmax_baseline

	def __call__(self, input_dict, logits):
		bsize, N, K = logits.shape
		dist = Categorical(logits=logits.view(bsize * N, K))  # (bsize * N, |Y|)
		
		# sample
		labels_sample = dist.sample().view(bsize, N) # (bsize, N)
		reward_sample = self.reward(input_dict, labels_sample) # (bsize, N, |Y|)

		log_probs_all = dist.log_prob(dist.enumerate_support()).T.view(bsize, N, K) # (bsize, N, |Y|)
		log_probs_sample = torch.gather(
			log_probs_all,
			2,
			labels_sample.unsqueeze(-1)
		).squeeze(2) # (bsize, N)

		second_term = ( (reward_sample * torch.exp(log_probs_all)).detach() * log_probs_all ).sum((1, 2)).mean()

		# always calculate for monitoring purposes
		labels_argmax = logits.argmax(2).detach()  # (bsize, N)
		reward_argmax = self.reward(input_dict, labels_argmax) # (bsize, N, |Y|)
		if self.with_argmax_baseline:
			baseline = reward_argmax
		else:
			baseline = 0.

		# first term
		marginalized_reward_w_baseline = ( (reward_sample - baseline) * torch.exp(log_probs_all) )[:, 1:, :].sum(2) # (bsize, N - 1)
		first_term = ( marginalized_reward_w_baseline.detach() * log_probs_sample.cumsum(1)[:, :-1] ).sum(1).mean()

		# loss
		surrogate_loss = -1 * (first_term + second_term)

		# to monitor
		reward_sample_to_monitor = torch.gather(
			reward_sample,
			2,
			labels_sample.unsqueeze(-1)
		).squeeze(2).sum(1).mean().item()

		reward_argmax_to_monitor = torch.gather(
			reward_argmax,
			2,
			labels_argmax.unsqueeze(-1)
		).squeeze(2).sum(1).mean().item()
		
		stats = {
			"R_argmax": reward_argmax_to_monitor,
			"R_sample": reward_sample_to_monitor,
			"y_argmax": labels_argmax,
			"y_sample": labels_sample
		}
		return surrogate_loss, stats


class Rao(BaseGradientEstimator):
	def __init__(self, reward, with_argmax_baseline=True):
		super().__init__(reward)
		self.with_argmax_baseline = with_argmax_baseline

	def __call__(self, input_dict, logits):
		bsize, N, K = logits.shape
		dist = Categorical(logits=logits.view(bsize * N, K))  # (bsize * N, |Y|)
		
		# sample
		labels_sample = dist.sample().view(bsize, N) # (bsize, N)
		reward_sample = self.reward(input_dict, labels_sample) # (bsize, N, |Y|)

		reward_sample_sub = torch.gather(
			reward_sample,
			2,
			labels_sample.unsqueeze(-1)
		).squeeze(2) # (bsize, N)
		logprobs_sample = dist.log_prob(labels_sample.view(-1)).view(bsize, N).cumsum(1) # (bsize, N)

		# always calculate for monitoring purposes, however we can put inside if, if we want to monitor only sometimes
		labels_argmax = logits.argmax(2).detach() # (bsize, N)
		reward_argmax = self.reward(input_dict, labels_argmax) # (bsize, N, |Y|)
		if self.with_argmax_baseline:
			baseline = torch.gather(
				reward_argmax,
				2,
				labels_argmax.unsqueeze(-1)
			).squeeze(2) # (bsize, N)
		else:
			baseline = 0.
			
		# loss
		surrogate_loss = -1 * ( (reward_sample_sub - baseline).detach() * logprobs_sample ).sum(1).mean()

		# to monitor
		reward_sample_to_monitor = torch.gather(
			reward_sample,
			2,
			labels_sample.unsqueeze(-1)
		).squeeze(2).sum(1).mean().item()

		reward_argmax_to_monitor = torch.gather(
			reward_argmax,
			2,
			labels_argmax.unsqueeze(-1)
		).squeeze(2).sum(1).mean().item()
		
		stats = {
			"R_argmax": reward_argmax_to_monitor,
			"R_sample": reward_sample_to_monitor,
			"y_argmax": labels_argmax,
			"y_sample": labels_sample
		}
		return surrogate_loss, stats


class Naive(BaseGradientEstimator):
	def __init__(self, reward, with_argmax_baseline=True):
		super().__init__(reward)
		self.with_argmax_baseline = with_argmax_baseline

	def __call__(self, input_dict, logits):
		bsize, N, K = logits.shape
		dist = Categorical(logits=logits.view(bsize * N, K))  # (bsize * N, |Y|)
		
		# sample
		labels_sample = dist.sample().view(bsize, N) # (bsize, N)
		reward_sample = self.reward(input_dict, labels_sample) # (bsize, N, |Y|)

		reward_sample_sub = torch.gather(
			reward_sample,
			2,
			labels_sample.unsqueeze(-1)
		).squeeze(2) # (bsize, N)
		logprobs_sample = dist.log_prob(labels_sample.view(-1)).view(bsize, N) # (bsize, N)

		# always calculate for monitoring purposes, however we can put inside if, if we want to monitor only sometimes
		labels_argmax = logits.argmax(2).detach() # (bsize, N)
		reward_argmax = self.reward(input_dict, labels_argmax) # (bsize, N, |Y|)
		if self.with_argmax_baseline:
			reward_argmax_sub = torch.gather(
				reward_argmax,
				2,
				labels_argmax.unsqueeze(-1)
			).squeeze(2) # (bsize, N)
			baseline = reward_argmax_sub.sum(1) # (bsize, )
		else:
			baseline = 0.
			
		surrogate_loss = -1 * ( (reward_sample_sub.sum(1) - baseline).detach() * logprobs_sample.sum(1) ).mean()

		# to monitor
		reward_sample_to_monitor = torch.gather(
			reward_sample,
			2,
			labels_sample.unsqueeze(-1)
		).squeeze(2).sum(1).mean().item()

		reward_argmax_to_monitor = torch.gather(
			reward_argmax,
			2,
			labels_argmax.unsqueeze(-1)
		).squeeze(2).sum(1).mean().item()
		
		stats = {
			"R_argmax": reward_argmax_to_monitor,
			"R_sample": reward_sample_to_monitor,
			"y_argmax": labels_argmax,
			"y_sample": labels_sample
		}
		return surrogate_loss, stats

