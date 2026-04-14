import math
from collections import deque
from functools import reduce
from math import gcd

import torch
from torch.utils.data import Dataset


class AlternatingDistributedBatchSampler:
    def __init__(
        self,
        lengths,
        batch_sizes,
        world_size=1,
        rank=0,
        shuffle=True,
        drop_last=True,
        seed=42,
    ):
        self.lengths = [int(x) for x in lengths]
        self.batch_sizes = [int(x) for x in batch_sizes]
        if len(self.lengths) != len(self.batch_sizes):
            raise ValueError("lengths and batch_sizes must have same length")
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.offsets = []
        c = 0
        for n in self.lengths:
            self.offsets.append(c)
            c += n
        self._build_epoch_state()

    def _balanced_indices(self, n, g):
        gen = torch.Generator()
        gen.manual_seed(int(g))
        indices = (
            torch.randperm(n, generator=gen).tolist()
            if self.shuffle
            else list(range(n))
        )
        num_samples = (
            math.floor(n / self.world_size)
            if self.drop_last
            else math.ceil(n / self.world_size)
        )
        total_size = num_samples * self.world_size
        if len(indices) < total_size:
            m = (total_size + len(indices) - 1) // len(indices)
            indices = (indices * m)[:total_size]
        return indices, num_samples

    def _chunk(self, xs, bs):
        if self.drop_last:
            n_full = len(xs) // bs
            return [xs[i * bs : (i + 1) * bs] for i in range(n_full)]
        else:
            return [xs[i * bs : (i + 1) * bs] for i in range((len(xs) + bs - 1) // bs)]

    def _build_epoch_state(self, epoch=0):
        g = self.seed + int(epoch)
        self.batches = []
        for i, (n, bs) in enumerate(zip(self.lengths, self.batch_sizes)):
            idx_all, num_per_rank = self._balanced_indices(n, g * 2 + i)
            idx_rank = idx_all[
                self.rank : self.rank + num_per_rank * self.world_size : self.world_size
            ]
            self.batches.append(self._chunk(idx_rank, bs))
        self.length = sum(len(b) for b in self.batches)

    def set_epoch(self, epoch):
        self._build_epoch_state(epoch)

    def __iter__(self):
        heads = [0] * len(self.batches)
        k = 0
        K = len(self.batches)
        while True:
            advanced = False
            for _ in range(K):
                i = k % K
                if heads[i] < len(self.batches[i]):
                    batch = self.batches[i][heads[i]]
                    yield [self.offsets[i] + j for j in batch]
                    heads[i] += 1
                    k = i + 1
                    advanced = True
                    break
                k = i + 1
            if not advanced:
                break

    def __len__(self):
        return self.length


class BlockInterleaveDataset(Dataset):
    def __init__(
        self,
        datasets,
        block_size,
        seed=42,
        shuffle_within=True,
        scheme="round_robin",
        weights=None,
    ):
        assert block_size > 0
        self.datasets = datasets
        self.block_size = int(block_size)
        self.seed = int(seed)
        self.shuffle_within = bool(shuffle_within)
        self.scheme = scheme
        self.weights = weights
        self._rng = torch.Generator()
        self._build_order(self.seed)

    def _shuffled_indices(self, n, g):
        idxs = torch.arange(n)
        if self.shuffle_within:
            idxs = idxs[torch.randperm(n, generator=g)]
        return idxs.tolist()

    def _build_order(self, seed):
        g = torch.Generator()
        g.manual_seed(seed)

        pools = []
        for ds in self.datasets:
            idxs = self._shuffled_indices(len(ds), g)
            pools.append(deque(idxs))

        assert len(self.weights) == len(
            self.datasets
        ), f"weights len ({len(self.weights)}) != num datasets ({len(self.datasets)})"
        assert all(
            isinstance(w, int) and w > 0 for w in self.weights
        ), "weights must be positive ints"
        pattern = []
        for i, w in enumerate(self.weights):
            pattern.extend([i] * w)
        # keep only datasets that still have items
        active = [i for i, q in enumerate(pools) if len(q) > 0]
        pattern = [i for i in pattern if i in active]
        order, k = [], 0
        while active:
            i = pattern[k % len(pattern)]
            if len(pools[i]) == 0:
                # dataset i exhausted -> drop it from active & shrink pattern
                active.remove(i)
                if not active:
                    break
                pattern = [j for j in pattern if j in active]
                k = k % len(pattern)  # re-align index
                continue
            take = min(self.block_size, len(pools[i]))
            block = [pools[i].popleft() for _ in range(take)]
            order.extend([(i, j) for j in block])
            k += 1

        self._order = order
        return

    def reshuffle(self, epoch):
        self._build_order(self.seed + int(epoch) + 1)

    def __len__(self):
        return len(self._order)

    def __getitem__(self, idx):
        ds_id, local_idx = self._order[idx]
        return self.datasets[ds_id][local_idx]
