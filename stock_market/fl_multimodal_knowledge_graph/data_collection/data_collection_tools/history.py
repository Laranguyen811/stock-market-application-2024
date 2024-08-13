import json
from pathlib import Path

def count_items(file_path):
    '''Takes the file path, open the file and count the number of items in the file.
    Inputs:
        file_path(string): A string of the file path.
    Returns:
        int: The number of items in the file.

    :param file_path:
    :return:
    '''
    try:  # Attempting a block of code before throwing an exception
        with open(file_path,'r') as f:  # Opening a file in a 'read' mode and closing it when done
            data = json.load(f)  # parsing a JSOM file and converting it into a corresponding Python object
            return len(data)

    except FileNotFoundError:  # Throwing an exception error when file not found
        print(f"File not found: {file_path}")
        return 0
    except json.decoder.JSONDecodeError:  # Throwing an exception when error decoding JSON in file
        print(f"Error JSON decoding file: {file_path}")
        return 0

#Define the file paths:
fiLe_paths = {
    'to_fetch': 'history/to_fetch.json',
    'to_extend': 'history/to_extend.json',
    'finished': 'history/finished.json'
}

#Count items in each file and print the results
for key,path in fiLe_paths.items():
    count = count_items(path)
    print(f'{key}: {count}')