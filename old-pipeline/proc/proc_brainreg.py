import os
import subprocess

# Directories
input_dir = "/Users/tristan/Dropbox/Neuro/Margrie/hm2p/brains-sorted/"
output_dir = "/Users/tristan/Dropbox/Neuro/Margrie/hm2p/brains-reg/"

# Parameters for brainreg
voxel_sizes = ["25", "25", "25"]
orientation = "psl"
save_original = "--save-original-orientation"

# Find all files with 'green' in the filename
for root, _, files in os.walk(input_dir):
    for file in files:
        if "green" in file.lower():
            input_file = os.path.join(root, file)
            print(f"Processing: {input_file}")

             # Create a subfolder in the output directory for each input file (without extension)
            base_name = os.path.splitext(file)[0]
            output_subdir = os.path.join(output_dir, base_name)
            os.makedirs(output_subdir, exist_ok=True)
            print(f"Output directory: {output_subdir}")

            cmd = [
                "brainreg",
                input_file,
                output_subdir,
                "-v", *voxel_sizes,
                "--orientation", orientation,
                save_original
            ]
        
            subprocess.run(cmd, check=True)