env = {
    "action_shape": (12,),
    "state_shape": (224, 320, 3),
}

deep_q = {
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
        'learning_rate': 1e-4,
        'lstm_size': 8,
        'rnn_depth': 3,
        'dropout': 0.5,
        'num_frames': 4,
    },
}
