import os
import shutil

def remove_and_recreate_directory(directory_path):
    # Check if the directory exists
    if os.path.exists(directory_path):
        # Use shutil.rmtree to remove the directory and its contents
        shutil.rmtree(directory_path)
        print(f"Directory '{directory_path}' removed.")

    # Recreate the directory
    os.makedirs(directory_path)
    print(f"Directory '{directory_path}' recreated.")