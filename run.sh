#!/bin/bash

# Define the range of latent_size values
latent_sizes=(2 4 8 16)

# Loop over each latent_size value and run the Python script
for i in "${latent_sizes[@]}"
do
    echo "Running with latent_size=$i"
    python main.py --latent_size=$i --epochs=300
done
