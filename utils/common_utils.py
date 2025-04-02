import math

import numpy as np
import torch
import torch.nn.functional as F

from misc.constants import LOC_FINDER_TOKEN


def get_minimal_to_distinguish(A):
    """
        A: list of lists of integers
    """
    K = len(A)
    res = [1] * K
    for i in range(K):
        for j in range(K):
            if i != j:
                common_prefix_len = find_len_of_common_prefix(A[i], A[j])
                res[i] = max(res[i], common_prefix_len + 1)

    return [A[i][:res[i]] for i in range(K)]


def find_len_of_common_prefix(list1, list2):
    arr1 = np.array(list1)
    arr2 = np.array(list2)
    min_len = min(len(arr1), len(arr2))
    mismatch_idx = np.where(arr1[:min_len] != arr2[:min_len])[0]
    if len(mismatch_idx) > 0:
        return mismatch_idx[0]

    return min_len


def get_location(template, tokenizer, input_dict, y, return_right_location=False):
    """
        template
        tokenizer
        input_dict: dict of str
        y: str
        return_right_location: bool

        This function returns the location of the token preceding y in the tokenized sequence, i.e.,
        we need this locations to compute p(y_{n} | ctx), thus this location corresponds to the 
        last token of ctx. It also returns the tokenized sequence itself.
        
        return_right_location: Useful for multi-token labels y=t_1, ..., t_m. Whether to return left locations only,
                                    i.e., location preceding t_1 or also return right locations, i.e., location preceding t_m.
                                    then basically logits[left:right + 1] will give you logits to predict t_1, ..., t_m
    """

    assert LOC_FINDER_TOKEN in tokenizer.special_tokens_map["additional_special_tokens"]

    tokenized_full = tokenizer(template(input_dict, y))["input_ids"]
    prefix_len = find_len_of_common_prefix(
        tokenizer(template(input_dict, LOC_FINDER_TOKEN))["input_ids"],
        tokenized_full
    )
    if return_right_location:
        prefix_len_right = find_len_of_common_prefix(
            tokenizer(template(input_dict, y + LOC_FINDER_TOKEN))["input_ids"],
            tokenized_full
        )
        return tokenized_full, prefix_len - 1, prefix_len_right - 2
    else:
        return tokenized_full, prefix_len - 1

def get_location_prefix(template, tokenizer, input_dict, y, prefix=""):
	"""
		template
		tokenizer
		input_dict: dict of str
		y: str

		This function returns the location of the token preceding y in the tokenized sequence, i.e.,
		we need this locations to compute p(y_{n} | ctx), thus this location corresponds to the 
		last token of ctx. It also returns the tokenized sequence itself.
	"""

	assert LOC_FINDER_TOKEN in tokenizer.special_tokens_map["additional_special_tokens"]

	def find_len_of_common_prefix(list1, list2):
		arr1 = np.array(list1)
		arr2 = np.array(list2)
		min_len = min(len(arr1), len(arr2))
		mismatch_idx = np.where(arr1[:min_len] != arr2[:min_len])[0]
		if len(mismatch_idx) > 0:
			return mismatch_idx[0]

		return min_len

	tokenized_full = tokenizer(prefix + template(input_dict, y))["input_ids"]
	prefix_len = find_len_of_common_prefix(
		tokenizer(prefix + template(input_dict, LOC_FINDER_TOKEN))["input_ids"],
		tokenized_full
	)
	return tokenized_full, prefix_len - 1

def get_BoT(template, tokenizer, input_dict, y):
    """
        Obtains bag of tokens for the classname
    """
    assert LOC_FINDER_TOKEN in tokenizer.special_tokens_map["additional_special_tokens"]

    tokenized_full, left, right = get_location(template, tokenizer, input_dict, y, return_right_location=True)
    return tokenized_full[left + 1: right + 2]

def find_repeated_indices(a):
    index_map = {}
    for i, elem in enumerate(a):
        if elem in index_map:
            index_map[elem].append(i)
        else:
            index_map[elem] = [i]
    
    result = [tuple(indices) for indices in index_map.values() if len(indices) > 1]
    return result


def entropy_regularizer(logits):
    """
        Numerically stable implementation

        logits: Tensor of shape (*, K)

        The function computes the average probs over the trailing dimensions, and then
        computes entropy of the obtained categorical distribution over K classes
    """
    all_dims_except_last = tuple(range(logits.dim() - 1))
    log_nelems = torch.tensor(logits.shape[:-1]).log().sum()

    log_probs = F.log_softmax(logits, dim=-1)
    avg_log_probs = torch.logsumexp(log_probs, dim=all_dims_except_last) - log_nelems
    H = - (torch.exp(avg_log_probs) * avg_log_probs).sum()
    return H



def cosine_scheduler(lr, final_lr, iters, warmup_iters=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, lr, warmup_iters)

    iters = int(iters)
    warmup_iters = int(warmup_iters)
    iters = np.arange(iters - warmup_iters + 1)
    schedule = np.array(
        [final_lr + 0.5 * (lr - final_lr) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters]
    )

    schedule = np.concatenate((warmup_schedule, schedule))

    return schedule



def reshape_list(lst, new_shape):
    assert len(new_shape) == 2
    # Check that the total number of elements matches
    if len(lst) != new_shape[0] * new_shape[1]:
        raise ValueError(f"Cannot reshape list of size {len(lst)} into shape {new_shape}")

    # Reshape the list
    reshaped = [lst[i * new_shape[1]:(i + 1) * new_shape[1]] for i in range(new_shape[0])]
    return reshaped


def corrupt_tensor(tensor, p, K):
    # Ensure the percentage p is between 0 and 1
    assert 0 <= p <= 1, "Percentage p must be between 0 and 1."
    
    # Get the shape of the input tensor
    shape = tensor.shape
    
    # Create a mask for corruption
    num_elements = tensor.numel()
    num_corrupt = int(p * num_elements)
    
    # Flatten the tensor to simplify indexing
    flat_tensor = tensor.clone().view(-1)
    
    # Randomly choose indices to corrupt
    indices = torch.randperm(num_elements)[:num_corrupt]
    
    # Generate random values for these indices
    random_values = torch.randint(0, K, (num_corrupt,), dtype=flat_tensor.dtype)
    
    # Replace the selected elements with random values
    flat_tensor[indices] = random_values
    
    # Reshape the tensor back to its original shape
    return flat_tensor.view(shape)
