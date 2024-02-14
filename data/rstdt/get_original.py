import json
from bisect import bisect_left

def load_json_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def find_object_by_path(objects, path_basename):
    index = bisect_left([obj['path_basename'] for obj in objects], path_basename)
    if index != len(objects) and objects[index]['path_basename'] == path_basename:
        return objects[index]
    return None

def main():
    # Load the contents of each file
    double_objects = load_json_file('double.json')
    test_objects = load_json_file('test.json')
    train_objects = load_json_file('train.json')

    # Concatenate and sort test and train objects by 'path_basename'
    combined_objects = sorted(test_objects + train_objects, key=lambda obj: obj['path_basename'])

    # Find original objects for each entry in double.json
    original_objects = []
    for double_obj in double_objects:
        original_obj = find_object_by_path(combined_objects, double_obj['path_basename'])
        if original_obj:
            original_objects.append(original_obj)

    # Save the original objects to a new file
    with open('original.json', 'w') as file:
        json.dump(original_objects, file, indent=4)

if __name__ == "__main__":
    main()
