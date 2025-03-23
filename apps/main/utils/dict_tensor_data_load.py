import torch


class DictTensorBatchIterator:
    def __init__(self, data_dict, batch_size):
        """
        Initialize the iterator for batching a dictionary containing tensors and strings.

        Args:
            data_dict (dict): Dictionary where keys map to strings or tensors.
            batch_size (int): Desired batch size for the tensors.
        """
        self.data_dict = data_dict
        self.batch_size = batch_size

        # Validate and prepare tensors
        self.tensor_keys = [
            key for key, value in data_dict.items() if isinstance(value, torch.Tensor)
        ]

        self.total_batches = None
        self.current_batch = 0

        # Remove singleton dimensions (first dimension = 1)
        for key in self.tensor_keys:
            tensor = data_dict[key]
            if tensor.shape[0] == 1 and len(tensor.shape) > 4:
                self.data_dict[key] = tensor.squeeze(0)  # Remove singleton dimension

        # Calculate the total number of batches (using the first tensor's shape)
        if self.tensor_keys:
            self.total_batches = (
                self.data_dict[self.tensor_keys[0]].shape[0] // batch_size
            )

    def __iter__(self):
        return self

    def __next__(self):
        """
        Return the next batch of the dictionary.

        Returns:
            dict: A dictionary with batched tensors and unchanged strings.

        Raises:
            StopIteration: If all batches are processed.
        """
        if self.tensor_keys and self.current_batch >= self.total_batches:
            raise StopIteration

        batch = {}
        for key, value in self.data_dict.items():
            start_idx = self.current_batch * self.batch_size
            end_idx = start_idx + self.batch_size
            batch[key] = value[start_idx:end_idx]

        self.current_batch += 1
        return batch
