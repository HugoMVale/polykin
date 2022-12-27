import sys
import os

here = os.path.dirname(os.path.abspath(__file__))
path_list = ["../src/"]

for path in path_list:
    path_to_add = os.path.join(here, path)
    sys.path.append(path_to_add)
