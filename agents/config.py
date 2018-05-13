env = {
    "action_shape": (12,),
    "state_shape": (224, 320, 3),
}

deep_q = {
    'gamma': 0.99,
    'noise': {
        'epsilon': {
            'start': 1.0,
            'end': 0.1,
        },
        'until': 1e5,
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
        'rnn_layers': [256, 64],
        # 'rnn_depth': 3,
        'dropout': 0.5,
        'num_frames': 4,
    },
}
