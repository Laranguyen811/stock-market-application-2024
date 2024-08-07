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
    cmd = f'mv {name}'  # A command to change a file's name.
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
    cmd = f'tar -cvzf - ./{name} | split -d -b 500m -{name}.tar'  # A command to split, package and compress a file.
    run_command(cmd)

    cmd = f'rm -rf {name}'  # A command to remove the original file.
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
    path_local = ''  # A path to local file group
    path_server = '/your/path/here'  # A path to server
    server = 'your.server'  # A server
    cmd = f'scp -P 2333 {path_local}{name}.tar* {server}"{path_server}'  # A command to take a specified file group and send it to a specific server
    run_command(cmd)

def clear(name: str):
    '''Takes a file's name and clears the file
    Inputs:
        name(string): A string of a file's name.
    Returns:
        None
    Raises:
        Exception: an exception if error executing the command.
    '''
    cmd = f'rm -rf {name}*'  # Remove all files and directories whose names start with the value of the name variable and can have any character following it.
    print(cmd)
    run_command(cmd)

def deliver_content(name: str):
    ''' Takes a file's name and compress, transport and remove the file
    Inputs:
        name(string): A string of a file's name.
    Returns:
        None
    Raises:
        Exception: an exception if error executing the command.
    '''
    compress(name)
    transport(name)
    clear(name)

def email(title: str,content: str):
    ''' Takes the title and content to send emails.
    Inputs:
        title(string): A string of the title of the email.
        content(string): A string of the content of the email.
    Returns:
        None
    Raises:
    Exception: an exception if error executing the command.
    '''
    mail_receive_address = ['<EMAIL>']
    for address in mail_receive_address:
        cmd = f'echo {content} | s-nail -s {title} {address}'  # A command to output the content, send an email with a specified title to a target email address.
        run_command(cmd)
