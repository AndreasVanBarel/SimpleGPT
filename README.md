# SimpleGPT

A simple GPT to train on and generate text similar to some provided data.txt. The structure was inspired by [this video](https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy) by Andrej Karpathy.

### Usage

Have a data.txt file in the same directory as simplegpt.py. Running `simplegpt.py` trains a transformer on that data and prints some generated text similar to the provided training data. The parameters of the model can be changed at the top of the `simplegpt.py` file. Currently, they are set to:

```python
# Parameters
batch_size = B = 64 # Batch size - number of examples to process in parallel
block_size = T = 128 # Context length for predictions
n_eds = C = 256 # Number of embedding dimensions (C for channels)
n_heads = 8 # Number of heads. Each head has headsize C//n_heads = 256/8 = 32
n_layers = 6 # Number of transformer blocks
dropout = 0.2 # To prevent overfitting
opt_steps = 5000 # Number of training steps
learning_rate = 4e-4 # Learning rate for the optimizer, AdamW in this case.
estimation_evals = 250 # Number of samples for estimation of loss
eval_interval = 250 # Every eval_interval training steps, the training and validation loss are estimated
n_gen = 2500 # Number of tokens to generate and print to illustrate the model at the end
```
