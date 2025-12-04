import matplotlib.pyplot as plt

episodes = []
returns = []
max_episodes = 50000;

with open("50000_ep_output.txt", "r", encoding="utf-16") as f: 
    for line in f:
        parts = line.strip().split()

        if len(parts) > 0 and parts[0] == "ep":
            episode = int(parts[1])
            reward = float(parts[3])

            episodes.append(episode)
            returns.append(reward)
            if episode >= max_episodes:
                break

# Plot
plt.plot(episodes, returns, marker='o', color='#4a96a2')
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title(f"Episode vs Return ({max_episodes})")
plt.grid(True)
plt.show()
