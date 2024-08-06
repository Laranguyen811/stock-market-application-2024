import subprocess
from typing import List

def run_command(cmd: str):
    ''' Takes a subprocess command and runs it.
    Inputs:
        cmd (string): A string of a subprocess command.
    Returns:
        None
    Raises:
        Exception: an exception if error executing command.
    '''

    ret = subprocess.run(cmd,shell=True)  # Using subprocess module to run command in shell.
    if ret.returncode != 0:  # If the returned code is not 0.
        print(f'Subprocess error: {cmd}')
        raise Exception(f'Error executing command: {cmd}')  # Raise an Exception error notifying error executing command.

def rename(name: str):
    '''

    Takes a name of a file and renames it.
    Inputs:
        name (string): A string of a file's name.
    Returns:
        None
    Raises:
         Exception: an exception if error executing command.
    '''
    cmd = f'mv {name}'
    run_command(cmd)

def compress(name: str):
    ''' Takes a name of a file, splits, packages and compresses it.
    Inputs:
        name(str): A string of a file's name.
    Returns:
        None
    Raises:
        Exception: an exception if error executing the command.
    '''
    cmd = f'tar -cvzf - ./{name} | split -d -b 500m -{name}.tar'
    run_command(cmd)

    cmd = f'rm -rf {name}'
    run_command(cmd)

def transport(name: str):
    ''' Takes a file group and sends it to the server.
    Inputs:
        name(string): A string of a file's name.
    Returns:
        None
    Raises:
     Exception: an exception if error executing the command.
    '''
    path_local = ''
    path_server = '/your/path/here'
    server = 'your.server'
    cmd = f'scp -P 2333 {path_local}{name}.tar* {server}"{path_server}'
    run_command(cmd)



