import os
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk


def list_folders(directory):
    """List all subdirectories in a directory."""
    return [
        d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))
    ]


def load_images_from_folder(folder, num_images=5):
    """Load the first few images from a folder."""
    images = []
    for file_name in os.listdir(folder)[:num_images]:
        if file_name.endswith((".png", ".jpg", ".jpeg")):
            try:
                img_path = os.path.join(folder, file_name)
                img = Image.open(img_path)
                img.thumbnail((100, 100))  # Resize to thumbnail
                images.append(ImageTk.PhotoImage(img))
            except Exception as e:
                print(f"Error loading image {file_name}: {e}")
    return images


def rename_folder_gui(directory):
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    for folder_name in list_folders(directory):
        folder_path = os.path.join(directory, folder_name)
        images = load_images_from_folder(folder_path)

        # Create a new window for each folder
        image_window = tk.Toplevel()
        image_window.title(folder_name)

        # Display images
        for img in images:
            panel = tk.Label(image_window, image=img)
            panel.pack(side="left", fill="both", expand="yes")

        # Ask for new folder name
        new_name = simpledialog.askstring(
            "Rename Folder", "Enter new name for this folder:", parent=image_window
        )

        if new_name:
            new_folder_path = os.path.join(directory, new_name)
            os.rename(folder_path, new_folder_path)
            print(f"Renamed '{folder_name}' to '{new_name}'")

        image_window.destroy()

    root.mainloop()


# Example usage
directory = "/Users/mustafagumustas/screen-time/Avrupa_Yakası_1._Bölüm_|_HD"  # Change this to your directory
rename_folder_gui(directory)
