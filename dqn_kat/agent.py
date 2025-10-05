import flappy_bird_gymnasium
import gymnasium
env = gymnasium.make("CartPole-v1", render_mode="human")

obs, _ = env.reset() # Initialized the environment
while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample() # Samples a random action from the Action space (For flappy bird, 0 - Do nothing, 1 - Flap)

    # Processing:
    obs, reward, terminated, _, info = env.step(action) # env.step executes the action, obs is the next state, reward is reward from recent action, terminated = true if game lost, info for debugging
    
    # Checking if the player is still alive
    if terminated: # Exits the while loop, and closes the environment
        break

env.close()