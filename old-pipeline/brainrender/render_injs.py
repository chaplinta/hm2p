import sys
sys.path.append("../")

from brainrender import Scene, settings
from brainrender.actors import Points
from utils import misc as m2putils, metadata as mdutils, db as dbutils
from paths import config
import os
import numpy as np
import nibabel as nib
from pathlib import Path
import tifffile
import pandas as pd

settings.SHOW_AXES = False

# Configuration option: set to True to use CSV summary, False to use TIFF segmentation
USE_CSV_SUMMARY = True  # Change this to switch between modes

# Or manually set some of the ids to run
#expids = ["20220802_15_06_53_1117646"]

cfg = config.M2PConfig()

exps_df = dbutils.get_exp_animals(cfg, exclude=None, primary=None)

# Create a scene with the 25um Allen atlas
scene = Scene(atlas_name="allen_mouse_25um")

# Display a brain region
rspd_region = scene.add_brain_region("RSPd", alpha=0.2, color="green")
scene.add_label(rspd_region, "RSPd")

# Initialize lists to track animals
animals_with_injection_sites = []
animals_missing_injection_sites = []
dirs_not_in_db = []

# Get all animal IDs from database for comparison
db_animal_ids = set()
for index, exp_row in exps_df.iterrows():
    exp_id = exp_row['exp_id']
    animal_id = mdutils.get_animal_id_from_exp_id(exp_id)
    db_animal_ids.add(animal_id)

# Check all directories in brains_reg_path to find animals not in DB
brains_reg_path = cfg.brains_reg_path
for subdir in os.listdir(brains_reg_path):
    full_path = os.path.join(brains_reg_path, subdir)
    if os.path.isdir(full_path):
        # Extract potential animal ID from directory name
        # This assumes animal ID is part of the directory name
        # You may need to adjust this logic based on your naming convention
        potential_animal_id = None
        for db_animal_id in db_animal_ids:
            if db_animal_id in subdir:
                potential_animal_id = db_animal_id
                break
        
        if potential_animal_id is None:
            # This directory doesn't contain any known animal ID
            dirs_not_in_db.append(subdir)

def switch_left_to_right_hemisphere(coords, atlas_name="allen_mouse_25um"):
    """
    Switch coordinates from left hemisphere to right hemisphere.
    
    We need to debug and determine the correct axis and direction for hemisphere switching.
    """
    coords_switched = coords.copy()
    
    print(f"DEBUG: Original coordinate ranges:")
    print(f"  X: {np.min(coords[:, 0]):.1f} to {np.max(coords[:, 0]):.1f}")
    print(f"  Y: {np.min(coords[:, 1]):.1f} to {np.max(coords[:, 1]):.1f}")
    print(f"  Z: {np.min(coords[:, 2]):.1f} to {np.max(coords[:, 2]):.1f}")
    
    # Let's try different hemisphere switching approaches
    # Approach 1: X < 0 is left hemisphere (negative X to positive X)
    left_mask_x_neg = coords[:, 0] < 0
    
    # Approach 2: X > midpoint is left hemisphere (assuming midpoint around 6600 for 25um atlas)
    atlas_width = 13200  # typical width in micrometers for 25um atlas
    midpoint = atlas_width / 2
    left_mask_x_high = coords[:, 0] > midpoint
    
    # Approach 3: Try different axis - maybe Y or Z represents left-right
    left_mask_y_neg = coords[:, 1] < 0
    left_mask_y_high = coords[:, 1] > (8000 / 2)  # assuming Y midpoint
    left_mask_z_neg = coords[:, 2] < 0
    left_mask_z_high = coords[:, 2] > (11400 / 2)  # assuming Z midpoint
    
    print(f"DEBUG: Potential left hemisphere points:")
    print(f"  X < 0: {np.sum(left_mask_x_neg)} points")
    print(f"  X > {midpoint:.0f}: {np.sum(left_mask_x_high)} points")
    print(f"  Y < 0: {np.sum(left_mask_y_neg)} points") 
    print(f"  Y > 4000: {np.sum(left_mask_y_high)} points")
    print(f"  Z < 0: {np.sum(left_mask_z_neg)} points")
    print(f"  Z > 5700: {np.sum(left_mask_z_high)} points")
    
    # For now, let's try the most common approach: mirror across X=0
    # But let's also try mirroring across the actual midpoint
    
    # Method 1: Simple negation for X < 0
    if np.sum(left_mask_x_neg) > 0:
        coords_switched[left_mask_x_neg, 0] = -coords_switched[left_mask_x_neg, 0]
        print(f"Applied Method 1: Switched {np.sum(left_mask_x_neg)} points (X < 0 → -X)")
    
    # If that doesn't work, uncomment one of these alternatives:
    
    # Method 2: Mirror across midpoint
    # if np.sum(left_mask_x_high) > 0:
    #     coords_switched[left_mask_x_high, 0] = atlas_width - coords_switched[left_mask_x_high, 0]
    #     print(f"Applied Method 2: Switched {np.sum(left_mask_x_high)} points (mirror across X midpoint)")
    
    # Method 3: Try Y axis instead
    # if np.sum(left_mask_y_neg) > 0:
    #     coords_switched[left_mask_y_neg, 1] = -coords_switched[left_mask_y_neg, 1]
    #     print(f"Applied Method 3: Switched {np.sum(left_mask_y_neg)} points (Y < 0 → -Y)")
    
    print(f"DEBUG: Final coordinate ranges:")
    print(f"  X: {np.min(coords_switched[:, 0]):.1f} to {np.max(coords_switched[:, 0]):.1f}")
    print(f"  Y: {np.min(coords_switched[:, 1]):.1f} to {np.max(coords_switched[:, 1]):.1f}")
    print(f"  Z: {np.min(coords_switched[:, 2]):.1f} to {np.max(coords_switched[:, 2]):.1f}")
    
    return coords_switched

def switch_left_to_right_hemisphere_single_point(x, y, z):
    """
    Switch a single point from left hemisphere to right hemisphere.
    In BrainGlobe Allen atlas: Z axis is left-to-right.
    - X axis: anterior-posterior 
    - Y axis: superior-inferior (dorsal-ventral)
    - Z axis: left-to-right (left = high Z, right = low Z)
    """
    # For Allen 25um atlas: brain depth ~11.4mm, so midline ~5.7mm in Z direction
    atlas_depth_mm = 11.4
    midline_mm = atlas_depth_mm / 2  # ~5.7mm
    
    # Left hemisphere: Z > midline_mm (higher Z values)
    # Right hemisphere: Z < midline_mm (lower Z values)
    
    if z > midline_mm:
        # This point is in left hemisphere, mirror it to right hemisphere
        # Mirror across midline: new_z = midline - (old_z - midline)
        z = midline_mm - (z - midline_mm)
    
    return x, y, z

# Load the meta animals CSV file - throw error if it doesn't exist
meta_csv_path = cfg.meta_animals_file
if not os.path.exists(meta_csv_path):
    raise FileNotFoundError(f"Meta animals CSV file not found at: {meta_csv_path}")

df_animals = pd.read_csv(meta_csv_path)

# Ensure the required columns exist
required_cols = ['inj_ap', 'inj_ml', 'inj_dv']
for col in required_cols:
    if col not in df_animals.columns:
        df_animals[col] = np.nan

# Dictionary to store injection coordinates for each animal
injection_coords = {}

processed_animal_ids = []
for index, exp_row in exps_df.iterrows():
    exp_id = exp_row['exp_id']
    #print(f"Processing experiment {exp_id}")
    animal_id = mdutils.get_animal_id_from_exp_id(exp_id)
    if animal_id not in processed_animal_ids:
        processed_animal_ids.append(animal_id)
    else:
        continue
    
    # Find the directory brains_reg_path that contains the registered brain for this animal
    # Check all sub dir names, if there is more than one matching throw an exception.
    brains_reg_path = cfg.brains_reg_path
    matching_dirs = []
    for subdir in os.listdir(brains_reg_path):
        full_path = os.path.join(brains_reg_path, subdir)
        if os.path.isdir(full_path) and animal_id in subdir:
            matching_dirs.append(full_path)

    if len(matching_dirs) > 1:
        raise RuntimeError(f"Multiple registered brain directories found for animal_id {animal_id}: {matching_dirs}")

    if len(matching_dirs) == 0:
        animals_missing_injection_sites.append(animal_id)
        continue

    animal_brain_dir = matching_dirs[0]
    is_penk = exp_row["celltype"] == "penk"
    print(animal_id)

    # Find the animals segmentation directory
    brain_reg_path = Path(animal_brain_dir)
    segmentation_dir = brain_reg_path / "segmentation" / "atlas_space" / "regions"
    
    if not segmentation_dir.exists():
        print(f"No segmentation directory found at {segmentation_dir}")
        animals_missing_injection_sites.append(animal_id)
        continue
    
    try:
        injection_site_found = False
        
        if USE_CSV_SUMMARY:
            # Load from CSV summary file
            csv_file = segmentation_dir / "summary.csv"
            
            if not csv_file.exists():
                print(f"No summary.csv file found in {segmentation_dir}")
                animals_missing_injection_sites.append(animal_id)
                continue
            
            print(f"Loading summary from: {csv_file}")
            
            # Read the CSV file
            summary_df = pd.read_csv(csv_file)
            
            if summary_df.empty:
                print(f"Empty summary.csv file: {csv_file}")
                animals_missing_injection_sites.append(animal_id)
                continue
            
            print(f"CSV contains {len(summary_df)} regions")
            injection_site_found = True
            
            # Process each region in the CSV
            for _, row in summary_df.iterrows():
                region_name = row['region']
                volume_mm3 = row['volume_mm3']
                
                # Get center coordinates (convert from micrometers to millimeters for brainrender)
                center_x = row['axis_0_center_um'] / 1000.0  # Convert um to mm
                center_y = row['axis_1_center_um'] / 1000.0
                center_z = row['axis_2_center_um'] / 1000.0
                
                # Switch left hemisphere to right hemisphere
                center_x, center_y, center_z = switch_left_to_right_hemisphere_single_point(center_x, center_y, center_z)
                
                # Store injection coordinates for this animal (keep in millimeters)
                if animal_id not in injection_coords:
                    injection_coords[animal_id] = {
                        'inj_ap': float(center_x),  # Anterior-posterior in mm
                        'inj_ml': float(center_z),  # Medial-lateral in mm (using Z)
                        'inj_dv': float(center_y)   # Dorsal-ventral in mm
                    }
                
                # Calculate sphere radius from volume (assuming spherical injection site)
                # Volume of sphere = (4/3) * π * r³
                # Therefore: r = (3 * Volume / (4 * π))^(1/3)
                radius_mm = ((3 * volume_mm3) / (4 * np.pi)) ** (1/3)
                
                print(f"Region {region_name}: Volume={volume_mm3:.3f} mm³, "
                      f"Center=({center_x:.1f}, {center_y:.1f}, {center_z:.1f}) mm, "
                      f"Radius={radius_mm:.2f} mm")
                
                # Create sphere center coordinates
                center_coords = np.array([[center_x * 1000, center_y * 1000, center_z * 1000]])  # Convert back to micrometers for brainrender
                
                # Choose color based on cell type
                if is_penk:
                    color = 'blue'
                else:
                    color = 'red'
                
                # Create and add sphere to scene
                # Use radius in micrometers (brainrender expects micrometers)
                sphere_radius = radius_mm * 1000  # Convert mm to micrometers
                
                points_actor = Points(
                    center_coords,
                    colors=color,
                    alpha=0.6,
                    radius=sphere_radius
                )
                
                scene.add(points_actor)
                print(f"Added sphere for region {region_name} (animal {animal_id})")
                
        else:
            # Original TIFF loading code
            segmentation_file = segmentation_dir / "region_0.tiff"
            
            if segmentation_file is None or not segmentation_file.exists():
                print(f"No segmentation file found in {segmentation_file}")
                animals_missing_injection_sites.append(animal_id)
                continue
            
            # Check if it's a TIFF file
            if segmentation_file.suffix.lower() in ['.tiff', '.tif']:
                # Load TIFF file with tifffile
                data = tifffile.imread(segmentation_file)
                
                # Get non-zero voxels (segmented regions)
                coords = np.where(data > 0)
                if len(coords[0]) == 0:
                    print(f"No segmented voxels found in {segmentation_file}")
                    animals_missing_injection_sites.append(animal_id)
                    continue
                    
                coords = np.column_stack(coords)
                values = data[data > 0]
                injection_site_found = True
                
                # Transform voxel coordinates to Allen atlas space
                # Since you're using the 25um Allen atlas, each voxel is 25 micrometers
                voxel_size = 25.0  # micrometers per voxel for allen_mouse_25um
                
                # Convert voxel indices to physical coordinates
                real_coords = coords * voxel_size
                
                # Apply offset to center coordinates
                # For Allen atlas, you might need to subtract half the volume size
                atlas_shape = np.array([528, 320, 456])  # Typical Allen 25um atlas shape
                atlas_center = atlas_shape * voxel_size / 2
                real_coords = real_coords - atlas_center
                
                # Switch left hemisphere coordinates to right hemisphere
                real_coords = switch_left_to_right_hemisphere(real_coords)
                
                # Calculate center of mass for injection site coordinates
                center_coords_mm = np.mean(real_coords, axis=0) / 1000.0  # Convert to mm
                
                # Store injection coordinates for this animal (keep in millimeters)
                if animal_id not in injection_coords:
                    injection_coords[animal_id] = {
                        'inj_ap': center_coords_mm[0],  # Anterior-posterior in mm (X)
                        'inj_ml': center_coords_mm[2],  # Medial-lateral in mm (Z)
                        'inj_dv': center_coords_mm[1]   # Dorsal-ventral in mm (Y)
                    }
                
                # Apply transformation matrix if available
                transform_file_patterns = [
                    "transform.txt",
                    "registration_matrix.txt", 
                    "*.mat",
                    "transform.npy"
                ]
                
                transform_matrix = None
                for pattern in transform_file_patterns:
                    if "*" in pattern:
                        transform_files = list(brain_reg_path.glob(pattern))
                        if transform_files:
                            try:
                                if pattern.endswith('.npy'):
                                    transform_matrix = np.load(transform_files[0])
                                elif pattern.endswith('.mat'):
                                    import scipy.io
                                    mat_data = scipy.io.loadmat(transform_files[0])
                                    # Look for common matrix names
                                    for key in ['transform', 'matrix', 'M', 'affine']:
                                        if key in mat_data:
                                            transform_matrix = mat_data[key]
                                            break
                                break
                            except Exception as e:
                                continue
                    else:
                        transform_file = brain_reg_path / pattern
                        if transform_file.exists():
                            try:
                                transform_matrix = np.loadtxt(transform_file)
                                break
                            except Exception as e:
                                continue
                
                # Apply transformation matrix if found
                if transform_matrix is not None:
                    if transform_matrix.shape == (4, 4):
                        # Add homogeneous coordinate
                        coords_homogeneous = np.column_stack([real_coords, np.ones(real_coords.shape[0])])
                        real_coords = coords_homogeneous.dot(transform_matrix.T)[:, :3]
                    elif transform_matrix.shape == (3, 3):
                        real_coords = real_coords.dot(transform_matrix.T)
                    
            else:
                print(f"Unsupported file format: {segmentation_file.suffix}")
                animals_missing_injection_sites.append(animal_id)
                continue
            
            # Choose color based on cell type
            if is_penk:
                color = 'blue'
            else:
                color = 'red'
            
            # Create and add points to scene
            points_actor = Points(
                real_coords,
                colors=color,
                alpha=0.3,
                radius=20,

            )
            
            scene.add(points_actor)
            print(f"Added {len(real_coords)} segmented points for {animal_id}")
        
        # Add to successful list if injection site was found
        if injection_site_found:
            animals_with_injection_sites.append(animal_id)
        
    except Exception as e:
        print(f"Error processing data for {animal_id}: {e}")
        animals_missing_injection_sites.append(animal_id)
        continue

# UPDATE DATAFRAME WITH INJECTION COORDINATES AND SAVE
print("\n" + "="*60)
print("UPDATING INJECTION COORDINATES IN DATAFRAME")
print("="*60)



# Update coordinates for animals that have injection sites
coordinates_updated = 0
for animal_id, coords in injection_coords.items():
    # Find the row index(es) for this animal_id
    animal_mask = df_animals["animal_id"] == int(animal_id)
    matching_rows = df_animals[animal_mask]
    
    if len(matching_rows) == 0:
        print(f"Warning: Animal {animal_id} not found in dataframe")
        continue
    elif len(matching_rows) > 1:
        print(f"Warning: Multiple rows found for animal {animal_id}, updating all")
    
    # Update the coordinates using .loc with boolean indexing
    df_animals.loc[animal_mask, 'inj_ap'] = coords['inj_ap']
    df_animals.loc[animal_mask, 'inj_ml'] = coords['inj_ml'] 
    df_animals.loc[animal_mask, 'inj_dv'] = coords['inj_dv']

    coordinates_updated += 1
    print(f"Updated coordinates for {animal_id}: AP={coords['inj_ap']:.3f}, ML={coords['inj_ml']:.3f}, DV={coords['inj_dv']:.3f}")

print(f"\nUpdated coordinates for {coordinates_updated} animals")


# Save updated dataframe back to CSV
df_animals.to_csv(meta_csv_path, index=False)
print(f"Successfully saved updated coordinates to: {meta_csv_path}")

# Print summary lists
print("\n" + "="*60)
print("SUMMARY REPORT")
print("="*60)

print(f"\nANIMALS WITH INJECTION SITES ({len(animals_with_injection_sites)}):")
for animal_id in sorted(animals_with_injection_sites):
    print(f"  - {animal_id}")

print(f"\nANIMALS MISSING INJECTION SITES ({len(animals_missing_injection_sites)}):")
for animal_id in sorted(animals_missing_injection_sites):
    print(f"  - {animal_id}")

print(f"\nDIRECTORIES NOT IN DATABASE ({len(dirs_not_in_db)}):")
for dir_name in sorted(dirs_not_in_db):
    print(f"  - {dir_name}")

print(f"\nTOTAL COUNTS:")
print(f"  - Animals in database: {len(db_animal_ids)}")
print(f"  - Animals with injection sites: {len(animals_with_injection_sites)}")
print(f"  - Animals missing injection sites: {len(animals_missing_injection_sites)}")
print(f"  - Directories not in database: {len(dirs_not_in_db)}")
print(f"  - Coordinates updated in CSV: {coordinates_updated}")

# Display the figure.
scene.render()

# # Set the scale, which will be used for screenshot resolution.
# # Any value > 1 increases resolution, the default is in brainrender.settings.
# # It is easiest integer scales (non-integer can cause crashes).
# scale = 2

# # Take a screenshot - passing no name uses current time
# # Screenshots can be also created during runtime by pressing "s"
# scene.screenshot(name="RSPd.pdf", scale=scale)

scene.close()