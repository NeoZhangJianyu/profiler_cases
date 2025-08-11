
import logging
import os
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.profiler import profile, ProfilerActivity, record_function


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    device = "cuda"
    activities += [ProfilerActivity.CUDA]
elif torch.xpu.is_available():
    device = "xpu"
    activities += [ProfilerActivity.XPU]
else:
    print(
        "Neither CUDA nor XPU devices are available to demonstrate profiling on acceleration devices"
    )
    import sys

    sys.exit(0)

sort_by_keyword = device + "_time_total"

print(f"device {device}")


# Define the neural network class
class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()
        # Define the layers of the network
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.relu = nn.ReLU()                         # ReLU activation function
        self.fc2 = nn.Linear(hidden_size, output_size) # Second fully connected layer

    def forward(self, x):
        # Define the forward pass (how data flows through the network)
        x = self.fc1(x)
        x = self.relu(x)
        x = x.to('cpu')
        x = x.to(device)
        x = self.fc2(x)
        return x

def main():
    input_dim = 10
    hidden_dim = 20
    output_dim = 1

    # Create an instance of the model
    model = SimpleNeuralNetwork(input_dim, hidden_dim, output_dim)
    model = model.to(device)

    # Create a dummy input tensor
    dummy_input = torch.randn(1, input_dim) # Batch size of 1, with input_dim features
    dummy_input = dummy_input.to(device)
    # Pass the input through the model

    start_time= time.time()

    # Generate logits and outputs
    with torch.no_grad(), profile(activities=activities, record_shapes=True) as prof, \
        record_function("model_inference"):  # Do not compute grad. descent/no training involved
        output = model(dummy_input)

    print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=100))
    prof.export_chrome_trace(f"trace_{device}.json")

    # Calculate the time elapsed for thinking
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    time_str = f"{elapsed_time}"


    # Log the thinking time and final response
    logger.info(f"Thinking time: {time_str}")



if __name__ == "__main__":
    main()
