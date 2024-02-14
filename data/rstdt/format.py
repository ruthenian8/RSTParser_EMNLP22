import sys
import json

def load_json_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def main():
    if len(sys.argv) <= 1:
        sys.exit(1)

    filename = sys.argv[1]

    # Load the contents of each file
    objects = load_json_file(filename)

    # Save the original objects to a new file
    with open(filename, 'w') as file:
        json.dump(objects, file, indent=4)

if __name__ == "__main__":
    main()
