To run experiments for #Exploration, please run the following command with a GPU:


For environment 'defend the line':
''''''''

	python3 doom_hash.py --train --env defend_the_line --epochs 1000 --save_dir ./save_dir/  --learning_steps_per_epoch 5000 --max_eps_upto 0.0 --decay_eps_upto 1.0

''''''''

For environment 'defend the center':
''''''''

	python3 doom_hash.py --train --env defend_the_center --epochs 1000 --save_dir ./save_dir/ --learning_steps_per_epoch 5000 --max_eps_upto 0.0 --decay_eps_upto 1.0

''''''''

For environment 'Seek and Destroy':
''''''''

	python3 doom_hash.py --train --epochs 10  --save_dir ./save_dir/  --learning_steps_per_epoch 2000 --replay_memory_size 10000  --max_eps_upto 0.0 --decay_eps_upto 1.0

''''''''
