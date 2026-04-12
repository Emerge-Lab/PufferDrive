
from pufferlib.ocean.breakout import breakout
env = breakout.Breakout()
env.reset()
while True:
    env.step(env.action_space.sample())
    frame = env.render()

