import yaml
import os
import sys


def print_config(relative_path):
        current_file = __file__
        current_directory = os.path.dirname(current_file)
        with open(f"{current_directory}/{relative_path}") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        return config

if __name__ == "__main__":
    relative_path = sys.argv[1]
    print(print_config(relative_path))