import random

from torch.utils.data import IterableDataset


class InterleavedIterableDataset(IterableDataset):
    """
    Groups multiple iterable datasets together into a single iterable dataset.
    The single datasets can be iterated through in different ways.
    """
    def __init__(self, datasets, mode = "RoundRobin"):
        """
        :param datasets: a list of iterable datasets to merge
        :param mode: how the datasets should be interleaved, options: 'Random', 'RoundRobin'
        """
        if not mode in ["RoundRobin", "Random"]: raise ValueError(f"Detected unknown mode for dataset: {mode}. Supported modes are 'RoundRobin' and 'Random'")
        self.datasets = datasets
        self.mode = mode

    def __iter__(self):
        # initiate passed datasets
        iterators = [iter(dataset) for dataset in self.datasets]

        if self.mode == "RoundRobin":
            # initiate the list that determines the order in which the individual datasets are sampled
            loop_list = list(range(len(self.datasets)))

            # the working copy of the loop list is used to handle one iteration of a round
            working_list = loop_list.copy()
            # continue sampling until all iterators/datasets are exhausted
            while len(loop_list) > 0:
                # start new round-robin loop, if current one is empty
                if len(working_list) == 0: working_list = loop_list.copy()

                # take first element from current working list
                current_index = working_list.pop(0)
                # try to get a sample, if iterator returns no sample (ie the default value 'None'),
                # then it is exhausted and is removed from the round-robin loop
                sample = next(iterators[current_index], None)
                if sample is None:
                    loop_list.remove(current_index)
                    continue

                yield sample
        elif self.mode == "Random":
            # indexes of iterators that can be used for samples
            valid_indexes = list(range(len(self.datasets)))

            # continue sampling until all iterators/datasets are exhausted
            while len(valid_indexes) > 0:
                # get random sample from random iterator
                # if iterator returns the default value, then remove it from future sampling
                current_index = random.sample(valid_indexes, 1)[0]
                sample = next(iterators[current_index], None)
                if sample is None:
                    valid_indexes.remove(current_index)
                    continue

                yield sample
        else: raise ValueError(f"Detected unknown mode for dataset: {self.mode}. Supported modes are 'RoundRobin' and 'Random'")