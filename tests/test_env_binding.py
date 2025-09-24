from pufferlib.ocean.breakout import breakout

kwargs = dict(
    frameskip=1,
    width=576,
    height=330,
    paddle_width=62,
    paddle_height=8,
    ball_width=32,
    ball_height=32,
    brick_width=32,
    brick_height=12,
    brick_rows=6,
    brick_cols=18,
    continuous=False,
)


def test_env_binding():
    reference = breakout.Breakout()

    # Correct usage
    c_env = breakout.binding.env_init(
        reference.observations,
        reference.actions,
        reference.rewards,
        reference.terminals,
        reference.truncations,
        0,
        **kwargs,
    )
    c_envs = breakout.binding.vectorize(c_env)
    breakout.binding.vec_reset(c_envs, 0)
    breakout.binding.vec_step(c_envs)
    breakout.binding.vec_close(c_envs)

    # Correct vec usage
    c_envs = breakout.binding.vec_init(
        reference.observations,
        reference.actions,
        reference.rewards,
        reference.terminals,
        reference.truncations,
        reference.num_agents,
        0,
        **kwargs,
    )

    # Correct vec usage
    c_envs = breakout.binding.vec_init(
        reference.observations,
        reference.actions,
        reference.rewards,
        reference.terminals,
        reference.truncations,
        reference.num_agents,
        0,
        **kwargs,
    )
    breakout.binding.vec_reset(c_envs, 0)
    breakout.binding.vec_step(c_envs)
    breakout.binding.vec_close(c_envs)

    try:
        c_env = breakout.binding.env_init()
        raise Exception("init missing args. Should have thrown TypeError")
    except TypeError:
        pass

    try:
        c_env = breakout.binding.env_init(
            reference.observations,
            reference.actions,
            reference.rewards,
            reference.terminals,
            reference.truncations,
            reference.num_agents,
            0,
        )
        raise Exception("init missing kwarg. Should have thrown TypeError")
    except TypeError:
        pass

    try:
        c_envs = breakout.binding.vec_init()
        raise Exception("vec_init missing args. Should have thrown TypeError")
    except TypeError:
        pass

    try:
        c_envs = breakout.binding.vec_init(
            reference.observations,
            reference.actions,
            reference.rewards,
            reference.terminals,
            reference.truncations,
            reference.num_agents,
            0,
        )
        raise Exception("vec_init missing kwarg. Should have thrown TypeError")
    except TypeError:
        pass

    try:
        breakout.binding.vec_reset()
        raise Exception("vec_reset missing arg. Should have thrown TypeError")
    except TypeError:
        pass

    try:
        breakout.binding.vec_step()
        raise Exception("vec_step missing arg. Should have thrown TypeError")
    except TypeError:
        pass


if __name__ == "__main__":
    test_env_binding()
