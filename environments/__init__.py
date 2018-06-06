from .maker import Maker

# Dict[str, Maker]
all_environments = dict()


# OpenAI Gym
try:
    import gym

    for env_spec in gym.envs.registry.all():
        all_environments['gym.' + env_spec.id] = Maker(
            make=lambda state, *args, **kwargs: gym.make(env_spec.id, *args, **kwargs),
            states=[None],
        )
except ImportError:
    pass

# OpenAI Retro
try:
    import retro

    for game in retro.list_games():
        is_installed = True
        try:
            retro.make(game).close()
        except FileNotFoundError:
            is_installed = False
        if is_installed:
            all_environments['retro.' + game] = Maker(
                make=lambda state, *args, **kwargs: retro.make(game, *args, state=state, **kwargs),
                states=retro.list_states(game),
            )
except ImportError:
    pass

# OpenAI Retro Contest
try:
    from retro_contest.local import make as retro_contest_make
    from .retro_contest import get_levels_by_game

    levels_by_game = get_levels_by_game()
    all_levels = [game + '/' + level for game, levels in levels_by_game.items() for level in levels]
    all_environments['retro_contest'] = Maker(
        make=lambda state, *args, **kwargs: retro_contest_make(state.split('/')[0], *args, state=state.split('/')[1], **kwargs),
        states=all_levels,
    )
except ImportError:
    pass