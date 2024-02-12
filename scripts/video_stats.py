import ast
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sjvisualizer


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


serie_folder_path = "/Users/mustafagumustas/screen-time/data"
episode_data = calculate_percentage_in_series(serie_folder_path)
# Create a DataFrame from the episode data
df = pd.DataFrame([episode_data]).T
df = df.fillna(0)

df = df.rename(columns={0: "Percentages"})


df["Episodes"] = df["Percentages"].apply(
    lambda x: " ".join(
        [f"{k} {v.split('_')[-1]}" for k, v in x.items() if k == "Episode"]
    )
)
# df = df.apply(
#     lambda row: {key: value for key, value in row.items() if key != "Episode"}, axis=1
# )
df["Percentages"] = df["Percentages"].apply(
    lambda x: {k: v for k, v in x.items() if k != "Episode"}
)

# Get a list of unique persons
persons = list(
    set(person for person in df.columns if person not in ["Episode", "Episodes"])
)


# Step 2: Prepare Data for Visualization
# This step depends on how you want to structure your data. You might want to create lists or dictionaries.
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ast
from scipy.interpolate import interp1d

data = df

# Ensure all values in the 'Percentages' column are dictionaries with float values
data["Percentages"] = data["Percentages"].apply(
    lambda x: {k: float(v) for k, v in x.items()}
)

# Sort the 'Percentages' column based on player's screen time
data["Percentages"] = data["Percentages"].apply(
    lambda x: dict(sorted(x.items(), key=lambda item: item[1], reverse=True))
)

# Create a figure and subplots
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))

data["Episodes"] = data["Episodes"].apply(lambda x: x.split()[-1]).astype(int)

# Sort the DataFrame based on the converted "Episodes" column
data = data.sort_values(by="Episodes").reset_index().drop(columns="index")
print(data)

# Initialize data for interpolation
current_data = data.iloc[0]
next_data = data.iloc[1]


# Function to update the animated plot
def animate(i):
    global current_data, next_data

    if i < len(data) - 1:
        current_data = data.iloc[i]
        next_data = data.iloc[i + 1]

    # Interpolate between current and next data
    interp_data = {}
    for player, percentage in current_data["Percentages"].items():
        interp_percentage = interp1d(
            [0, 1], [percentage, next_data["Percentages"].get(player, 0)]
        )(i % 1)
        interp_data[player] = interp_percentage

    # Get the top 5 visible players for the left graph
    top_players_left = dict(
        sorted(interp_data.items(), key=lambda item: item[1], reverse=True)[:5]
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
        sorted(top_players_right.items(), key=lambda item: item[1], reverse=True)[:5]
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
    ax_left.set_title(f'Top 5 Visible Players in {current_data["Episodes"]}')

    # Plot the right graph
    ax_right.barh(
        list(top_players_right.keys()), top_players_right.values(), color="red"
    )
    ax_right.set_title(f"Top 5 Players in the Season So Far")


# new_data = (
#     pd.DataFrame(data["Percentages"].to_list(), index=data["Episodes"])
#     .fillna(0)
#     .reset_index()
# )

# Sort the new_data DataFrame based on the "Episodes" column


# Split the elements in the "Episodes" column and convert the last term to an integer
# new_data["Episodes"] = new_data["Episodes"].apply(lambda x: x.split()[-1]).astype(int)

# # Sort the DataFrame based on the converted "Episodes" column
# new_data = new_data.sort_values(by="Episodes").reset_index().drop(columns="index")

# Save the transformed data to a new CSV file
# new_data["Episode"] = new_data["Episode"] + 1
# new_data.to_excel("transformed_data.xlsx", index=False)
# new_data.to_csv("transformed_data.csv", index=False)


# Create the animation
ani = FuncAnimation(
    fig, animate, frames=len(data), repeat=False, interval=300
)  # Adjust interval for smoother animation

# Show the animated plot
plt.show()
