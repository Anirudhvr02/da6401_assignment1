# FeedForward Neural Network with Backpropagation from Scratch
This repository contains all the files required for Assignment 1 of the DA6401 - Fundamentals of Deep Learning course at IIT Madras. The objective is to build a FeedForward Neural Network with backpropagation entirely from scratch.

## Task
Implement a FeedForward Neural Network with backpropagation from scratch without relying on high-level deep learning frameworks. The implementation must support various optimizers and loss functions and be highly configurable.

## Submission
- **WandB Project:** [DA6401_ASSIGNMENT_1](https://wandb.ai/anirudhvr02-indian-institute-of-technology-madras/DA6401_Assignment1/reports/DA6401-Assignment-1--VmlldzoxMTgzNzEwMQ)
- **WandB Report:** [Report Link](https://wandb.ai/anirudhvr02-indian-institute-of-technology-madras/DA6401_Assignment1/reports/DA6401-Assignment-1--VmlldzoxMTgzNzEwMQ)

## Dataset
The project utilizes the Fashion MNIST and MNIST datasets, which can be imported directly:
```python
from keras.datasets import fashion_mnist
from keras.datasets import mnist
```

## Implementation Details
### Optimizers
The implementation offers multiple optimization strategies:
- **SGD (Stochastic Gradient Descent)**
- **Momentum-based SGD**
- **NAG (Nesterov Accelerated Gradient)**
- **RMSProp**
- **Adam**
- **Nadam**

### Loss Functions
Two primary loss functions are available:
- **Cross Entropy**
- **Mean Squared Error**

### Backpropagation
Backpropagation is executed iteratively. For each layer, errors and deltas are computed and used to calculate gradients by multiplying with the inputs.

### Flexibility
The design allows extensive customization:
- Choice of activation functions per layer.
- Adjustable number of neurons and hidden layers.
- Configurable input batch size and output activation.
- Easy experimentation with different network configurations and optimizers.

## Tools and Libraries

### Packages
- **Python 3.10.1** – Main programming language.
- **WandB** – For experiment tracking and hyperparameter tuning.
- **NumPy** – For numerical operations.
- **Matplotlib** – For data visualization.
- **Keras** – For dataset loading.
- **Scikit-learn** – For data splitting and generating confusion matrices.

### Parameters used
```json
{
    "method": "random",
    "name": "Q4_SWEEP",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "num_layers": {"values": [1, 2, 3]},
        "hidden_size": {"values": [32, 64, 128]},
        "input_size": {"value": 784},
        "output_size": {"value": 10},
        "hidden_layers": {"values": [3, 4, 5]},
        "neurons": {"values": [32, 64, 128]},
        "activation": {"values": ["sigmoid", "tanh", "relu"]},
        "output_activation": {"value": "softmax"},
        "learning_rate": {"values": [1e-3, 1e-4]},
        "weight_decay": {"values": [0, 0.0005, 0.000005]},
        "epochs": {"values": [5, 10]},  
        "optimizer": {"values": ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]},
        "batch_size": {"values": [16, 32, 64]},
        "weight_initialization": {"values": ["xavier", "random"]},
        "beta": {"values": [0.7, 0.8, 0.9]},
        "beta1": {"value": 0.9},
        "beta2": {"value": 0.9999},
        "epsilon": {"value": 1e-8},
        "criterion": {"value": "cross_entropy"}
    }
}
```

## Usage

### Running Manually
To train the model, run:
```bash
Assignment_1.ipynb
```


### Running a Sweep using WandB
To conduct a hyperparameter sweep, adjust parameters as needed and execute:
```bash
DA6401_Assignment1_Sweep.ipynb
```

- **Task:** Build a feedforward neural network with backpropagation.
- **Tracking:** Experiment results are logged on [WandB Project](https://wandb.ai/anirudhvr02-indian-institute-of-technology-madras/DA6401_Assignment1/reports/DA6401-Assignment-1--VmlldzoxMTgzNzEwMQ) and detailed in the [WandB Report](https://wandb.ai/anirudhvr02-indian-institute-of-technology-madras/DA6401_Assignment1/reports/DA6401-Assignment-1--VmlldzoxMTgzNzEwMQ)
