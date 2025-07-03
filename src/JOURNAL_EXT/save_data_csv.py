import json
import csv
import os
import sys

def process_json_to_csv(json_file):
    try:
        # Read the JSON file
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract the lists from the JSON data
        aoi_values = data['aoi_values']
        errors = data['errors']
        
        # Check if the lists have the same length
        if len(aoi_values) != len(errors):
            print(f"Error: Lists have different lengths in {json_file}")
            return
        
        # Generate the CSV file name by replacing the extension
        base, ext = os.path.splitext(json_file)
        csv_file = base + '.csv'
        
        # Write the data to the CSV file
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write the header
            writer.writerow(['AoI', 'Error'])
            # Write the data rows
            for aoi, error in zip(aoi_values, errors):
                writer.writerow([aoi, error])
        
        print(f"Successfully created {csv_file}")
    except Exception as e:
        print(f"Error processing {json_file}: {e}")

if __name__ == "__main__":
    dir_data = '../../data'
    # Get the list of JSON files from command-line arguments
    json_files = [
        f'{dir_data}/detection_results_robot_0_error_function.json',
        f'{dir_data}/detection_results_robot_1_error_function.json',
        f'{dir_data}/detection_results_robot_2_error_function.json',
        f'{dir_data}/detection_results_robot_3_error_function.json',
        f'{dir_data}/detection_results_robot_4_error_function.json',
        f'{dir_data}/detection_results_robot_5_error_function.json',
        f'{dir_data}/detection_results_robot_6_error_function.json',
        f'{dir_data}/detection_results_robot_7_error_function.json',
    ]
    # Process each JSON file
    for json_file in json_files:
        process_json_to_csv(json_file)