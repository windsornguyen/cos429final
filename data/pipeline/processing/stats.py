# Exploratory data analysis script to help us get a feel for the data

import json
from collections import defaultdict
import matplotlib.pyplot as plt

# Define the file paths
master_train_path = '../pipeline/train.json'
master_val_path = '../pipeline/val.json'
master_test_path = '../pipeline/test.json'

# Function to load JSON data from a file
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Load the JSON data from the files
master_train_data = load_json_data(master_train_path)
master_val_data = load_json_data(master_val_path)
master_test_data = load_json_data(master_test_path)

# Function to calculate word counts in a dataset
def calculate_word_counts(dataset):
    word_counts = defaultdict(int)
    for entry in dataset:
        word = entry['word']
        word_counts[word] += 1
    return word_counts

# Calculate word counts for each dataset
train_word_counts = calculate_word_counts(master_train_data)
val_word_counts = calculate_word_counts(master_val_data)
test_word_counts = calculate_word_counts(master_test_data)

# Calculate average sequence lengths for each dataset
def calculate_avg_sequence_length(dataset):
    total_length = 0
    for entry in dataset:
        left_positions = entry['positions']['leftpositions']
        right_positions = entry['positions']['rightpositions']
        total_length += len(left_positions) + len(right_positions)
    return total_length / len(dataset)

train_avg_seq_length = calculate_avg_sequence_length(master_train_data)
val_avg_seq_length = calculate_avg_sequence_length(master_val_data)
test_avg_seq_length = calculate_avg_sequence_length(master_test_data)

# Find the top 10 most common words in each dataset
def get_top_words(word_counts, top_n=10):
    return sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

train_top_words = get_top_words(train_word_counts)
val_top_words = get_top_words(val_word_counts)
test_top_words = get_top_words(test_word_counts)

# Print EDA results
print('Exploratory Data Analysis (EDA)')
print('--------------------------------')

print(f'Total number of entries in the master train set: {len(master_train_data)}')
print(f'Total number of entries in the master validation set: {len(master_val_data)}')
print(f'Total number of entries in the master test set: {len(master_test_data)}')

print(f'\nAverage sequence length in the master train set: {train_avg_seq_length:.2f}')
print(f'Average sequence length in the master validation set: {val_avg_seq_length:.2f}')
print(f'Average sequence length in the master test set: {test_avg_seq_length:.2f}')

print('\nTop 10 most common words in the master train set:')
for word, count in train_top_words:
    print(f'{word}: {count}')

print('\nTop 10 most common words in the master validation set:')
for word, count in val_top_words:
    print(f'{word}: {count}')

print('\nTop 10 most common words in the master test set:')
for word, count in test_top_words:
    print(f'{word}: {count}')

# Visualize the word distribution in each dataset
def plot_word_distribution(word_counts, title):
    words = list(word_counts.keys())
    counts = list(word_counts.values())
    plt.figure(figsize=(12, 6))
    plt.bar(words, counts)
    plt.xlabel('Words')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_word_distribution(train_word_counts, 'Word Distribution in Master Train Set')
plot_word_distribution(val_word_counts, 'Word Distribution in Master Validation Set')
plot_word_distribution(test_word_counts, 'Word Distribution in Master Test Set')