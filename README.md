## How to Run the Project
To run the project and train the model, follow the steps below:

### Fetch the Data
Run the following command to fetch the necessary data:
`python data/fetch_data.py`

### Process the Data
After fetching the data, run the following script to process, merge, and concatenate the datasets:
`python data/process_data.py`

### Train the Model

## Training Parameters
- **`epochs`** (`int`): Number of training iterations over the dataset. More epochs improve learning but may lead to overfitting.
- **`batch_size`** (`int`): Number of samples processed before the model updates weights. Larger batches provide more stability but require more memory.
- **`seq_len`** (`int`): Length of the input sequence used for training. Determines how much historical data is considered in each training step.
- **`pred_len`** (`int`): Number of future time steps the model predicts.
- **`stride`** (`int`): Step size for sliding the input window across time series data. Affects the number of training samples.
- **`patch_size`** (`int`): Size of each patch extracted from the time series data. Controls how data is divided for processing in the PatchTST model.
- **`d_model`** (`int`): Dimensionality of the model's hidden representations. Higher values increase expressiveness but require more computation.
- **`kernel_size`** (`int`): Size of the convolutional kernel applied in feature extraction layers.
- **`tickers`** (`List[str]`): Size of the convolutional kernel applied in feature extraction layers.
?
## Model Parameters
- **`model_name`** (`str`): Specifies the model type to use. Options include:
- `"xcpatchtst"` (Cross-Channel PatchTST)
- `"patchtst"` (Standard PatchTST)
- `"dlinear"` (Decomposition Linear Model)

## Dataset Parameters
- **`tickers`** (`list[str]`): List of stock ticker symbols to fetch and process data for.
- **`ticker_threshold`** (`int`): Minimum number of data points required for a ticker to be included in training.

## Data Directories
- **`raw_dataset_dir`** (`str`): Directory path where raw (unprocessed) dataset files are stored.
- **`processed_dataset_dir`** (`str`): Directory path for storing preprocessed datasets.
- **`dataset_dir`** (`str`): Comma-separated list of stock tickers (symbols)

Once the data is processed, use the script below to train the model with custom arguments (20 epochs, batch size of 16, and the DLinear model):
`python src/xcpatchtst.py --epochs 20 --batch_size 16 --model_name dlinear`

### Monitor the Training with TensorBoard
To visualize the training process, launch TensorBoard with the following command:
`tensorboard --logdir=runs`
Open a web browser and navigate to http://localhost:6006/ to view the TensorBoard dashboard.