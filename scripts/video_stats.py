import os
import pandas as pd
import matplotlib.pyplot as plt


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
asli_data = df[["Episode", "asli", "Episode Number"]].sort_values(by="Episode Number")

volkan_data = df[["Episode", "volkan", "Episode Number"]].sort_values(
    by="Episode Number"
)
iffet_data = df[["Episode", "iffet", "Episode Number"]].sort_values(by="Episode Number")
tahsin_data = df[["Episode", "tahsin", "Episode Number"]].sort_values(
    by="Episode Number"
)
both_data = asli_data.merge(volkan_data, on="Episode Number")
# print(both_data[["asli", "volkan"]])
# print(both_data)

plt.figure(figsize=(12, 6))
ax = plt.subplot(111)
# Plot "asli" data as bars
ax.bar(df["Episode Number"], df["asli"], width=0.2, label="Asli", color="blue")

# Plot "volkan" data as bars
ax.bar(df["Episode Number"] + 0.2, df["volkan"], width=0.2, label="Volkan", color="red")

ax.bar(df["Episode Number"] + 0.4, df["iffet"], width=0.2, label="Iffet", color="green")
ax.bar(
    df["Episode Number"] + 0.6, df["tahsin"], width=0.2, label="Tahsin", color="orange"
)
plt.title("Percentage of Appearance for Asli and Volkan in Each Episode")
plt.xlabel("Episode Number")
plt.ylabel("Percentage of Appearance")
plt.legend()
# plt.grid(True)
plt.xticks(
    df["Episode Number"],
)  # Show episode names on x-axis
plt.show()
