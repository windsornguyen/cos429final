import json

# Assuming the data is in the form of a JSON string or directly as Python objects
with open('wlasl_train.json', 'r') as file:
    data = json.load(file)


def disp_of_list(positions):
    """Calculate displacements between consecutive positions in a list."""
    displacements = []
    for i in range(1, len(positions)):
        # Compute the displacement as the difference between successive positions
        temp_list = []
        assert len(positions[i]) == len(positions[i - 1])
        for j in range(len(positions[i])):
            # x coordinate computation
            temp_x = 100
            if positions[i][j][0] == -1 and positions[i - 1][j][0] == -1:
                temp_x = -1
            else:
                temp_x = positions[i][j][0] - positions[i - 1][j][0]
            # y coordinate computation
            temp_y = 100
            if positions[i][j][1] == -1 and positions[i - 1][j][1] == -1:
                temp_y = -1
            else:
                temp_y = positions[i][j][1] - positions[i - 1][j][1]
            temp_list.append([temp_x, temp_y])
        displacements.append(temp_list)
    return displacements


def calculate_displacements(data):
    # Iterate over each item in the data list
    for item in data:
        word = item['word']
        positions = item['positions']
        left_displacements = disp_of_list(positions['leftpositions'])
        right_displacements = disp_of_list(positions['rightpositions'])
        positions['leftpositions'] = left_displacements
        positions['rightpositions'] = right_displacements

# Run the function
calculate_displacements(data)

# Save the modified data as a JSON file
with open('editted_wlasl.json', 'w') as json_file:
    json.dump(data, json_file)
