import os


def remove_first_dirs(path, num_directories):
    # Split the path into directories and filename
    directories, filename = os.path.split(path)

    # Split the directories into separate parts
    directories_parts = directories.split(os.path.sep)

    # Remove the specified number of directories
    new_directories = os.path.sep.join(directories_parts[num_directories:])

    # Reconstruct the new path
    new_path = os.path.join(new_directories, filename)

    return new_path
