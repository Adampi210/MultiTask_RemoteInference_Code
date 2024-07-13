#!/bin/bash

# Put data in desired dir
target_dir="/scratch/gilbreth/apiasecz/data/NGSIM_traffic_data"
mkdir -p "$target_dir"

# Read URLs and names
input_file="./NGSIM_data_URL_list.txt"

# Download all data
while IFS=',' read -r url filename
do
    wget "$url" -O "$target_dir/$filename"
    # Check if successful
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded: $filename to $target_dir"
    else
        echo "Failed to download: $filename"
    fi
done < "$input_file"