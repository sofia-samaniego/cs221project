# CS221 Project
Code for our CS221 project on improving the performance of RL algorithms in sparse-reward environments through the use of intrinsic motivation.

The code base contains Python scripts for the A-DQL and AIM-DQL models described in the writeup.

## A-DQL
The `async_dql.py` script will run an asynchronous version of deep Q-learning using Python 2.7. It's usage is as so:
```
python async_dql.py
```
There are a large number of hyper-parameters that can be configured and sit at the top of the script file in capitals. The script will save files `worker_[thread_id].csv` for each thread in a folder `data/` (needs to exist) tracking the learning progress of the workers. It will also save Tensorflow sessions containing the graph metadata and weights for matrices in a `trained/` folder. Once the model is finished training to the defined number of frames then the agent will be evaluated over a certain number of episode with the score of each episode saved in a `trained_eval.csv` file in the `data/` directory.

## AIM-DQL
The `gpu_im_async_dql.py` script will run the AIM-DQL model using Python 3. The change in Python was required due to the GPU node used being pre-configured for Tensorflow via Python 3 only. It's usage is as follows:
```
python3 gpu_im_async_dql.py [GAME] [BETA]
```
where GAME is the Atari arcade game you wish to train on and BETA is the tradeoff coefficient between intrinsic and extrinsic rewards. There are a large number of hyper-parameters that can be altered at the top of the file in capitals. Similarly to the A-DQL script, worker files are saved in the `data/` directory for both the initial epsilon-greedy stage and then the intrinsic motivation stage. Tensorflow checkpoint files are saved in the `trained/` directory, and the model is evaluated once training is complete with results saved in a csv file in the `data/` directory. 

To run the AIM-DQL method on a GPU we utilized the SLURM task manager via the `cs221.sbatch` file. GPU node configurations can be changed inside this file and a job is submitted to the GPU via
```
sbatch cs221.sbatch
```

## Plotting Scripts
Various plotting scripts are contained in the `plotting_scripts/` directory and are all written in R. They are all relatively self-explanatory and take the data from the `data/` directory to produce visually appealing graphs via ggplot in R.

## Other
* `requirements.txt` lists the required packages to run the code.

* `benchmark_threads.py` was a short Python script used for the progress report to benchmark the computation times for varying number of threads.

