For pseudo-count based exploration experiments, please follow the follwoing instructions:

For environment 'defend the line', replace lines 98-106 in file doom_pseudo.py with the following:

	args.train = True
	args.env = 'defend_the_line'
	args.epochs = 10000
	args.learning_steps_per_epoch = 5000
	args.replay_memory_size = 50000
	args.save_dir = './save_dir/'
	args.max_eps_upto = 0.0
	args.decay_eps_upto = 1.0
	args.bonus = True

Then run the following command:

	python3 doom.pseudo.py




For environment 'defend the center', replace lines 98-106 in file doom_pseudo.py with the following:

	args.train = True
	args.env = 'defend_the_center'
	args.epochs = 10000
	args.learning_steps_per_epoch = 5000
	args.replay_memory_size = 50000
	args.save_dir = './save_dir/'
	args.max_eps_upto = 0.0
	args.decay_eps_upto = 1.0
	args.bonus = True

Then run the following command:

	python3 doom.pseudo.py




For environment 'Seek and Destroy', replace lines 98-106 in file doom_pseudo.py with the following:

	args.train = True
	args.env = 'simpler_basic'
	args.epochs = 10
	args.learning_steps_per_epoch = 2000
	args.replay_memory_size = 10000
	args.save_dir = './save_dir/'
	args.max_eps_upto = 0.0
	args.decay_eps_upto = 1.0
	args.bonus = True

Then run the following command:

	python3 doom.pseudo.py