import json

# Define the file paths
wlasl_train_path = '/scratch/gpfs/mn4560/final/data/wlasl/main/wlasl_landmark_data_disp_train.json'
wlasl_val_path = '/scratch/gpfs/mn4560/final/data/wlasl/main/wlasl_landmark_data_disp_val.json'
wlasl_test_path = '/scratch/gpfs/mn4560/final/data/wlasl/main/wlasl_landmark_data_disp_test.json'
asl_citizen_train_path = '/scratch/gpfs/mn4560/final/data/asl_citizen/asl_citizen_train_disp.json'
asl_citizen_val_path = '/scratch/gpfs/mn4560/final/data/asl_citizen/asl_citizen_val_disp.json'
asl_citizen_test_path = '/scratch/gpfs/mn4560/final/data/asl_citizen/asl_citizen_test_disp.json'

# Define the output file paths
master_train_path = '/scratch/gpfs/mn4560/final/data/pipeline/train.json'
master_val_path = '/scratch/gpfs/mn4560/final/data/pipeline/val.json'
master_test_path = '/scratch/gpfs/mn4560/final/data/pipeline/test.json'

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

# Combine all the data into unions
wlasl_union = wlasl_train_data + wlasl_val_data + wlasl_test_data
asl_citizen_union = asl_citizen_train_data + asl_citizen_val_data + asl_citizen_test_data

# Function to count the occurrences of words
def word_counts(data):
    counts = {}
    for item in data:
        word = item["word"].lower()
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    return counts

# Count words in both datasets
wlasl_vocab = word_counts(wlasl_union)
asl_citizen_vocab = word_counts(asl_citizen_union)

# Function to merge vocabulary sets based on a threshold
def merge_vocabs(vocab1, vocab2, threshold):
    agreed = {}
    for word in vocab1:
        if word in vocab2:
            total_count = vocab1[word] + vocab2[word]
            if total_count >= threshold:
                agreed[word] = total_count
    return agreed

# Merge the vocabularies using a threshold of 10
total_vocab = merge_vocabs(wlasl_vocab, asl_citizen_vocab, 10)
print('Vocabulary Size: ', len(total_vocab))

# Function to filter and merge datasets based on a vocabulary set
def make_new_sets(set1, set2, vocab_set):
    new_set = []
    for item in set1:
        word = item["word"].lower()
        if word in vocab_set:
            new_set.append(item)
    for item in set2:
        word = item["word"].lower()
        if word in vocab_set:
            new_set.append(item)
    return new_set

# Generate the master datasets by combining the WLASL and ASL Citizen data
master_train_data = make_new_sets(wlasl_train_data, asl_citizen_train_data, total_vocab)
master_val_data = make_new_sets(wlasl_val_data, asl_citizen_val_data, total_vocab)
master_test_data = make_new_sets(wlasl_test_data, asl_citizen_test_data, total_vocab)

# Save the merged data to JSON files
with open(master_train_path, 'w') as file:
    json.dump(master_train_data, file, indent=2)

with open(master_val_path, 'w') as file:
    json.dump(master_val_data, file, indent=2)

with open(master_test_path, 'w') as file:
    json.dump(master_test_data, file, indent=2)

# Output the final dataset sizes
print("Merging complete.")
print(f"Master train set size: {len(master_train_data)}")
print(f"Master validation set size: {len(master_val_data)}")
print(f"Master test set size: {len(master_test_data)}")
