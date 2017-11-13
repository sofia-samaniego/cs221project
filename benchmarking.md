## Benchmarking
Running the async_dql code on various nodes with the following settings:

GAME = 'Breakout-v0'
BUFFER_SIZE = 4
INPUT_SHAPE = (BUFFER_SIZE, 84,84)
NUM_FRAMES = 100000
DISCOUNT = 0.99
EPSILON_START = 1.0
EPSILON_FRAME = 2000
EPSILON_TRAINING_PERIOD = 100
PRED_UPDATE_RATE = 32
TARGET_UPDATE_RATE = 1000
ACTION_REPEAT = 4
NUM_THREADS = 16
LR = 0.00025 # learning rate
MOMENTUM = 0.95


The projected time/cost is if we ran for 50,000,000 frames.

| Computer       | Time | Cost          | Projected Time | Projected Cost |
|----------------|------|---------------|----------------|----------------|
| Sofia's Laptop |      | $0            |                | $0             |
| Alex's Laptop  |      | $0            |                | $0             |
| Simon's Laptop |      | $0            |                | $0             |
| c4.4xlarge     |      | $0.796 / hour |                |                |
| p2.xlarge      |      | $0.9 / hour   |                |                |
| g3.4xlarge     |      | $1.14 / Hour  |                |                |
