import os

def ensure_dir(directory):
    """
    Ensure that a directory exists. If it doesn't, create it.
    
    Args:
        directory (str): Path of the directory to check/create.
    """
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        print(f"❌ Error creating directory {directory}: {e}")

def save_to_file(file_path, data, mode="w"):
    """
    Save data to a file, ensuring the directory exists.
    
    Args:
        file_path (str): Path of the file to save.
        data (str): Data to write to the file.
        mode (str): File open mode ('w' for write, 'a' for append).
    """
    ensure_dir(os.path.dirname(file_path))
    try:
        with open(file_path, mode) as file:
            file.write(data)
        print(f"✅ Successfully saved data to {file_path}")
    except Exception as e:
        print(f"❌ Error writing to file {file_path}: {e}")

def file_exists(file_path):
    """
    Check if a file exists.
    
    Args:
        file_path (str): Path of the file to check.
    
    Returns:
        bool: True if file exists, False otherwise.
    """
    return os.path.exists(file_path)
