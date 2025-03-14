from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        """
        Args:
            data (Tensor): The time series data (shape: [num_samples, channels])
            seq_len (int): Length of input sequence
            pred_len (int): Length of prediction sequence
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_samples = len(data) - seq_len - pred_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            sequence (Tensor): Input sequence of shape (seq_len, channels)
            label (Tensor): Target sequence of shape (pred_len, channels)
        """
        seq = self.data[idx : idx + self.seq_len]  # Get the input sequence
        label = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]  # Get target sequence
        
        return seq, label