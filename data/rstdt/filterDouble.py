import json

# Define the basenames to filter out
basenames_to_filter = {
    "wsj_0627.out",
    "wsj_0684.out",
    "wsj_1129.out",
    "wsj_1365.out",
    "wsj_1387.out"
}

# Read the original json file
with open('double.json', 'r') as file:
    data = json.load(file)

# Filter out the objects
filtered_data = [obj for obj in data if obj.get('path_basename') not in basenames_to_filter]

# Write the filtered data to new json file
with open('new_double.json', 'w') as file:
    json.dump(filtered_data, file, indent=4)

print("Filtered data has been written to new_double.json")
