import ast
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def calculate_percentage_in_cluster(cluster_folder):
    person_counts = {}
    total_faces = 0

    for person_folder in os.listdir(cluster_folder):
        if os.path.isdir(os.path.join(cluster_folder, person_folder)):
            person_faces = len(os.listdir(os.path.join(cluster_folder, person_folder)))
            person_counts[person_folder] = person_faces
            total_faces += person_faces

    percentages = {}

    for person, count in person_counts.items():
        percentage = (count / total_faces) * 100
        percentages[person] = percentage

    return percentages


def calculate_percentage_in_series(serie_folder):
    episodes = []

    for episode_folder in os.listdir(serie_folder):
        if os.path.isdir(os.path.join(serie_folder, episode_folder)):
            episode_path = os.path.join(serie_folder, episode_folder)
            episode_data = calculate_percentage_in_cluster(episode_path)
            episode_data["Episode"] = episode_folder
            episodes.append(episode_data)

    return episodes


# Example usage
serie_folder_path = "avrupa_yakasi_folder"
episode_data = calculate_percentage_in_series(serie_folder_path)

# Create a DataFrame from the episode data
df = pd.DataFrame(episode_data)

# Extract episode numbers from the Episode column
df["Episode Number"] = df["Episode"].str.extract(r"e(\d+)").astype(int)
df = df.sort_values(by="Episode Number")
# print(df)

# Get a list of unique persons
persons = list(
    set(person for person in df.columns if person not in ["Episode", "Episode Number"])
)


def plot_percentage_for_names_all_episodes(episode_data, names):
    # Create a DataFrame from the episode data
    df = pd.DataFrame(episode_data)

    # Extract episode numbers from the Episode column
    df["Episode Number"] = df["Episode"].str.extract(r"e(\d+)").astype(int)
    df = df.sort_values(by="Episode Number")

    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)

    for name in names:
        data = df[["Episode", name, "Episode Number"]].sort_values(by="Episode Number")
        ax.bar(
            df["Episode Number"] + 0.2 * names.index(name),
            data[name],
            width=0.2,
            label=name,
        )

    plt.title("Percentage of Appearance for Selected Names in Each Episode")
    plt.xlabel("Episode Number")
    plt.ylabel("Percentage of Appearance")
    plt.legend()
    plt.xticks(df["Episode Number"], rotation=45)
    plt.show()


# plot_percentage_for_names_all_episodes(df, ["cem", "sehsuvar", "asli"])


def get_data_for_episode(episode_data, episode_number):
    # Create a DataFrame from the episode data
    df = pd.DataFrame(episode_data)

    # Extract episode numbers from the Episode column
    df["Episode Number"] = df["Episode"].str.extract(r"e(\d+)").astype(int)

    # Filter the DataFrame for the specified episode number
    episode_df = df[df["Episode Number"] == episode_number]

    return episode_df


# print(df[df["Episode Number"] == 2])

# plot_percentage_for_names_all_episodes(
#     get_data_for_episode(df, 1),
#     [
#         "fatos",
#         "sehsuvar",
#         "iffet",
#         "yaprak",
#         "volkan",
#         "selin",
#         "tahsin",
#         "cem",
#         "asli",
#         "tacettin",
#         "sertac",
#     ],
# )


# Step 2: Prepare Data for Visualization
# This step depends on how you want to structure your data. You might want to create lists or dictionaries.


data = pd.read_csv("episode_percentages.csv")
data["Episode Number"] = data["Episode"].str.extract(r"e(\d+)").astype(int)
data = data.sort_values(by="Episode Number")
# Convert the 'Percentages' column to dictionaries with float values
data["Percentages"] = data["Percentages"].apply(lambda x: ast.literal_eval(x))

# Ensure all values in the 'Percentages' column are dictionaries with float values
data["Percentages"] = data["Percentages"].apply(
    lambda x: {k: float(v) for k, v in x.items()}
)

# Sort the 'Percentages' column based on player's screen time
data["Percentages"] = data["Percentages"].apply(
    lambda x: dict(sorted(x.items(), key=lambda item: item[1], reverse=True))
)
print(data)


# Step 3: Create the Animated Plot
# You'll need to create a function that updates the plot for each episode.
# Use FuncAnimation to iterate over episodes and update the plot.
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))

how_many_people_to_show = 8


# Function to update the animated plot
def animate(i):
    episode_data = data.iloc[i]

    # Get the top 5 visible players for the left graph
    top_players_left = dict(
        sorted(
            episode_data["Percentages"].items(), key=lambda item: item[1], reverse=True
        )[:how_many_people_to_show]
    )

    # Get the cumulative top 5 players for the right graph
    top_players_right = {}
    for j in range(i + 1):
        for player, percentage in data.iloc[j]["Percentages"].items():
            if player in top_players_right:
                top_players_right[player] += percentage
            else:
                top_players_right[player] = percentage
    top_players_right = dict(
        sorted(top_players_right.items(), key=lambda item: item[1], reverse=True)[
            :how_many_people_to_show
        ]
    )

    # Sort the players by most visited for both left and right graphs
    top_players_left = dict(sorted(top_players_left.items(), key=lambda item: item[1]))
    top_players_right = dict(
        sorted(top_players_right.items(), key=lambda item: item[1])
    )

    # Clear the previous plots
    ax_left.clear()
    ax_right.clear()

    # Plot the left graph
    ax_left.barh(list(top_players_left.keys()), top_players_left.values(), color="blue")
    ax_left.set_title(f'Top 5 Visible Players in {episode_data["Episode"]}')

    # Plot the right graph
    ax_right.barh(
        list(top_players_right.keys()), top_players_right.values(), color="red"
    )
    ax_right.set_title(f"Top 5 Players in the Season So Far")


# Create the animation
ani = FuncAnimation(
    fig, animate, frames=len(data), repeat=False, interval=500
)  # Set interval to control animation speed

# Show the animated plot
plt.show()
