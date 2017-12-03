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

| Computer       | Time (s) | Cost          | Projected Time (h) | Projected Cost |
|----------------|------|---------------|----------------|----------------|
| Sofia's Laptop |      | $0            |                | $0             |
| Alex's Laptop  |2226  | $0            |309.2           | $0             |
| Simon's Laptop |      | $0            |                | $0             |
| c4.4xlarge     |839.7 | $0.796 / hour | 116.63         | $92.4               |
| p2.xlarge      |      | $0.9 / hour   |                |                |
| g3.4xlarge     |      | $1.14 / Hour  |                |                |

Then to do some benchmarking by number of threads. This is done with the following settings:

GAME = 'Breakout-v0'
BUFFER_SIZE = 4
INPUT_SHAPE = (BUFFER_SIZE, 84,84)
NUM_FRAMES = 10000
DISCOUNT = 0.99
EPSILON_START = 1.0
EPSILON_FRAME = 200
EPSILON_TRAINING_PERIOD = 10
PRED_UPDATE_RATE = 32
TARGET_UPDATE_RATE = 100
ACTION_REPEAT = 4
NUM_THREADS = 16
LR = 0.00025 # learning rate
MOMENTUM = 0.95

| Number of threads | Time (s)      |
|-------------------|---------------|
| 1                 |357.38236618   | 
| 2                 |308.572651863  | 
| 4                 |251.068399906  | 
| 6                 |267.027966022  | 
| 8                 |290.010194063  | 
| 12                |291.42575717   | 
| 16                |288.063944101  | 
| 24                |298.016618013  | 
| 32                |350.107196093  | 


