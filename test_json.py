import json5
from rfd_conf import *

with open(f'conf/pc1.json', 'r') as file:
    data = json5.load(file)

print(data)

# Print the data
# print(cf)
