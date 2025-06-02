import os

# Directory containing the files
source_directory = "/home/edramos/Documents/datasets/blood-cell-cancer-all-4class/Blood cell Cancer [ALL]"

# Get a list of all files in the directory
file_locations = []

for root, _, files in os.walk(source_directory):
    for file in files:
        file_locations.append(os.path.join(root, file))

# Save the file locations to a text file
output_file = "file_locations.txt"
with open(output_file, "w") as f:
    for location in file_locations:
        f.write(location + "\n")

print(f"File locations saved to {output_file}")