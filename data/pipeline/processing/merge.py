import json

# Define the file paths
wlasl_train_path = '../wlasl/wlasl_train.json'
wlasl_val_path = '../wlasl/wlasl_val.json'
wlasl_test_path = '../wlasl/wlasl_test.json'
asl_citizen_train_path = '../ASL_Citizen/asl_citizen_train.json'
asl_citizen_val_path = '../ASL_Citizen/asl_citizen_val.json'
asl_citizen_test_path = '../ASL_Citizen/asl_citizen_test.json'

# Define the output file paths
master_train_path = '../pipeline/aggregated_train.json'
master_val_path = '../pipeline/aggregated_val.json'
master_test_path = '../pipeline/aggregated_test.json'

# Function to load JSON data from a file
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Load the JSON data from the files
wlasl_train_data = load_json_data(wlasl_train_path)
wlasl_val_data = load_json_data(wlasl_val_path)
wlasl_test_data = load_json_data(wlasl_test_path)
asl_citizen_train_data = load_json_data(asl_citizen_train_path)
asl_citizen_val_data = load_json_data(asl_citizen_val_path)
asl_citizen_test_data = load_json_data(asl_citizen_test_path)

# Merge the data sets
master_train_data = wlasl_train_data + asl_citizen_train_data
master_val_data = wlasl_val_data + asl_citizen_val_data
master_test_data = wlasl_test_data + asl_citizen_test_data

# Save the merged data to JSON files
with open(master_train_path, 'w') as file:
    json.dump(master_train_data, file, indent=4)

with open(master_val_path, 'w') as file:
    json.dump(master_val_data, file, indent=4)

with open(master_test_path, 'w') as file:
    json.dump(master_test_data, file, indent=4)

print('Merging complete.')
print(f'Master train set size: {len(master_train_data)}')
print(f'Master validation set size: {len(master_val_data)}')
print(f'Master test set size: {len(master_test_data)}')