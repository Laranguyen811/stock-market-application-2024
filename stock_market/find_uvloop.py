import os

def comment_uvloop_in_setup_files(root_dir):
    for root, dirs, files in os.walk(root_dir):
        if 'setup.py' in files:
            setup_path = os.path.join(root, 'setup.py')
            with open(setup_path, 'r') as file:
                lines = file.readlines()

            with open(setup_path, 'w') as file:
                for line in lines:
                    if 'uvloop' in line:
                        file.write('# ' + line)  # Comment out the line containing 'uvloop'
                    else:
                        file.write(line)
            print(f"Processed: {setup_path}")

# Specify the root directory of your project
root_directory = 'D:/Users/laran/PycharmProjects/ASX'
comment_uvloop_in_setup_files(root_directory)


def search_uvloop_in_pyproject(root_dir):
    for root, dirs, files in os.walk(root_dir):
        if 'pyproject.toml' in files:
            pyproject_path = os.path.join(root, 'pyproject.toml')
            with open(pyproject_path, 'r') as file:
                content = file.read()
                if 'uvloop' in content:
                    print(f"'uvloop' found in: {pyproject_path}")
                else:
                    print(f"'uvloop' not found in: {pyproject_path}")

# Specify the root directory of your project
root_directory = 'D:/Users/laran/PycharmProjects/ASX'
search_uvloop_in_pyproject(root_directory)

