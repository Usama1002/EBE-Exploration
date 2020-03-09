To run experiments for egreedy exploration, please run the following command with a GPU:


## For environment 'defend the line':

```
	python3 doom.py --train --env defend_the_line --epochs 1000 --save_dir ./save_dir/  --learning_steps_per_epoch 5000 --max_eps_upto 0.0 --decay_eps_upto 1.0
```

## For environment 'defend the center':

```
	python3 doom_hash.py --train --env defend_the_center --epochs 1000 --save_dir ./save_dir/ --learning_steps_per_epoch 5000 --max_eps_upto 0.0 --decay_eps_upto 1.0
```

## For environment 'Seek and Destroy':

```
	python3 doom_hash.py --train --epochs 10  --save_dir ./save_dir/  --learning_steps_per_epoch 2000 --replay_memory_size 10000  --max_eps_upto 0.0 --decay_eps_upto 1.0
```

Please update the flags 'max_eps_upto' and 'decay_eps_upto' to suit the required e-greedy version for the experiments.

To run the experiments for EBE or Boltzmann exploration, please add --entropy or --BZ to the above commands. Note that with any of these flags, 'max_eps_upto' and 'decay_eps_upto' will lose their effect.
