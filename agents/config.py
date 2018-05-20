env = {
    "action_shape": (12,),
    "state_shape": (224, 320, 3),
}

deep_q = {
    'gamma': 0.99,
    'reward_offset': -0.1,
    'noise': {
        'epsilon': {
            'start': 0.4,
            'end': 0.1,
            'test': 0.1,
        },
        'until': 2e4,
    },
    'simple': {
        'learning_rate': 1e-5,
        'num_hidden': 1024,
    },
    'conv': {
        'learning_rate': 1e-4,
        'layers': {
            'hidden': [128, 64],
        },
        'dropout': 0.5,
    },
    'conv_recurrent': {
        'learning_rate': 2.5e-4,
        'embedding': 128,
        'rnn_layers': [64, 32, 16],
        'dropout': 0.5,
        'num_frames': 4,
    },
    'conv_recurrent_rl2': {
        'learning_rate': 2.5e-4,
        'embedding': 128,
        'rnn_layers': [64, 32, 16],
        'rl2_layers': [16, 16, 16],
        'dropout': 0.5,
        'num_frames': 4,
        'conv_layers': [
            {
                'type': 'conv2d',
                'filters': 16,
                'kernel_size': 5,
                'strides': 2,
            },
            {
                'type': 'conv2d',
                'filters': 32,
                'kernel_size': 5,
                'strides': 2,
            },
            {
                'type': 'max_pool2d',
                'pool_size': 2,
                'strides': 2,
                'dropout': True,
            },
            {
                'type': 'conv2d',
                'filters': 64,
                'kernel_size': 5,
                'strides': 2,
            },
            {
                'type': 'conv2d',
                'filters': 128,
                'kernel_size': 5,
                'strides': 2,
            },
            {
                'type': 'max_pool2d',
                'pool_size': 2,
                'strides': 2,
                'dropout': True,
            },
        ],
    },
    'rl2': {
        'learning_rate': 2.5e-4,
        'embedding': 128,
        'rnn_layers': [64, 32, 16],
        'dropout': 0.5,
    },
}
