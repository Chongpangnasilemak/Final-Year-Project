## How to Run the Project
To run the project and train the model, follow the steps below:

### Fetch the Data
Run the following command to fetch the necessary data:
`python data/fetch_data.py`

### Process the Data
After fetching the data, run the following script to process, merge, and concatenate the datasets:
`python data/process_data.py`

### Train the Model
Once the data is processed, use the script below to train the model with custom arguments (20 epochs, batch size of 16, and the DLinear model):
`python src/xcpatchtst.py --epochs 20 --batch_size 16 --model_name dlinear`

### Monitor the Training with TensorBoard
To visualize the training process, launch TensorBoard with the following command:
`tensorboard --logdir=runs`
Open a web browser and navigate to http://localhost:6006/ to view the TensorBoard dashboard.