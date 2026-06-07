class LabelRemapper:
    """
    Handles the label remapping problem Max described:
    The PNN classifier only accepts labels 0, 1, 2, ..., (n-1).
    But each dataset has its own native label (e.g. Cresci17 returns values like 1, 2, 3...).

    This class keeps a mapping from native dataset labels → sequential classifier indices,
    and grows that mapping as new datasets/classes arrive.

    Example:
        remapper = LabelRemapper()
        remapper.register(native_label=1)   # → maps to index 0
        remapper.register(native_label=3)   # → maps to index 1
        remapper.convert([1, 1, 3])         # → [0, 0, 1]
    """

    def __init__(self):
        self._native_to_index = {}   # e.g. {1: 0, 3: 1, 5: 2}
        self._index_to_native = {}   # reverse map, useful for debugging
        self._next_index = 0

    def register(self, native_label):
        """
        Register a new native label and assign it the next sequential index.
        Safe to call multiple times with the same label — it won't re-register.

        Returns the index assigned to this label.
        """
        if native_label not in self._native_to_index:
            self._native_to_index[native_label] = self._next_index
            self._index_to_native[self._next_index] = native_label
            self._next_index += 1
        return self._native_to_index[native_label]

    def convert(self, native_labels):
        """
        Convert a list of native labels to their sequential classifier indices.
        All labels must have been registered first.

        Args:
            native_labels: a list or iterable of native label values

        Returns:
            list of remapped integer indices
        """
        result = []
        for label in native_labels:
            if label not in self._native_to_index:
                raise ValueError(
                    f"Label '{label}' has not been registered. "
                    f"Call register() before convert(). "
                    f"Known labels: {list(self._native_to_index.keys())}"
                )
            result.append(self._native_to_index[label])
        return result

    def num_registered(self):
        """How many unique labels have been registered so far (= current number of classifier outputs needed)."""
        return self._next_index

    def __repr__(self):
        return f"LabelRemapper({self._native_to_index})"
