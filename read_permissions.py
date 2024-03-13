import os
stats = os.stat('C:/Program Files')
print(stats.st_mode)
print(oct(stats.st_mode))

os.chmod('C:/Program Files',0o777) #give permissions for everyone to write/execute