## Part 0: Bootstrap File
# You need to execute this at the start of the project. This will install the
# requirements and if it's running in local mode it unpacks the preprocessed 
# dataset included in the repo.

# Install the requirements
#!pip install --progress-bar off -r requirements.txt

import subprocess

def install_requirements(requirements_file_path):
    """
    Install Python packages from requirements.txt file.

    Parameters:
        requirements_file_path (str): The path to the requirements.txt file.

    Returns:
        bool: True if the installation is successful, False otherwise.
    """
    try:
        # Command to run: pip install --progress-bar off -r requirements.txt
        command = ["pip", "install", "--progress-bar", "off", "-r", requirements_file_path]

        # Run the pip install command using subprocess
        subprocess.check_call(command)

        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False

# Usage example
if __name__ == "__main__":
    # Replace 'path/to/requirements.txt' with the actual path to your requirements.txt file
    requirements_file_path = '../requirements.txt'
    success = install_requirements(requirements_file_path)
    if success:
        print("Requirements installed successfully.")
