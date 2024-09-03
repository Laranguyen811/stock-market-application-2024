import os
import sys
from stock_market.fl_multimodal_knowledge_graph.data_collection.data_collection_tools import transport_tool

def main(name):
    ''' Takes a name of a file, transports the file to a server and clears the file.

    Inputs:
        name(string): A string of a file's name
    Returns:
        None
    Raises:
        Exception: an exception if error executing.
    '''
    try:  # Trying this block of code before throwing an exception
        if not os.path.exists(name + '.tar00'):  # Checking if the .tar00 file (a part of split archive file) does not exist
            if not os.path.exists(name):  # Checking if the original file/directory does not exist
                transport_tool.rename()
            transport_tool.compress()
        transport_tool.transport()
        transport_tool.clear()
    except Exception as e:  # Throwing an exception as varible e
        print(f"An error occurred: {e}")
if __name__ == "__main__":  # Putting the main logic in a separate main function for modularity, reusability, testing simplification, readability and error handling
    if len(sys.argv) == 2:  # Checking if there are one argument exactly in a list of Python with command-like arguments passeed to a script
        name = str(sys.argv[1])  # Assigning the name to be the first argument passed to the script
        main(name)  # Only executing the main function when there is exactly only 1 argument for argument validation, error prevention, user guideline
    else:
        print("Usage: python start_transport_tool.py <name>")


