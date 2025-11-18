import numpy as np
import matplotlib.pyplot as plt
import os
import csv
def calculate_distances(file_path, num_beads, num_conformations, target_bead_index, range_start, range_end):
    """
    Calculate distances from the target bead to beads in the specified range across all conformations.
    Returns average distances, standard errors, and x-axis values (index differences).
    """
    # Load and reshape data
    data = np.loadtxt(file_path)
    data = data.reshape(num_conformations, num_beads, 3)
    coordinates = data[:, :, :3]

    # Define the range of indices
    indices_in_range = np.arange(range_start, range_end + 1)

    # Calculate distances for all conformations
    distances = []
    for conf_idx in range(num_conformations):
        target_coords = coordinates[conf_idx, target_bead_index]
        range_coords = coordinates[conf_idx, range_start:range_end + 1]
        distances.append(np.linalg.norm(range_coords - target_coords, axis=1))

    distances = np.array(distances)  # Shape: (num_conformations, len(indices_in_range))

    # Calculate averages and standard errors
    avg_distances = np.mean(distances, axis=0)
    std_errors = np.std(distances, axis=0) / np.sqrt(num_conformations)
    x_values = indices_in_range - target_bead_index  # X-axis: index differences

    return x_values, avg_distances, std_errors, target_bead_index

# Define parameters for the two datasets
dataset1 = {
    "file_path": "/mnt/mydisk/Program/multiscale_polymer_model/new_gene_system_NLbeads/lin28A/analysis/cordinates/Reduced_cordinates",
    "num_beads": 8911,
    "num_conformations": 13000,
    "target_bead_index": 4047,
    "range_start": 1,
    "range_end": 8910
}

dataset2 = {
    "file_path": "/mnt/mydisk/Program/multiscale_polymer_model/Rahul/check_code_chr12/Reduced_cordinates_chr12",
    "num_beads": 9577,
    "num_conformations": 13000,
    "target_bead_index": 1788,
    "range_start": 1,
    "range_end": 9576
}

dataset3 = {
    "file_path": "/mnt/mydisk/Program/multiscale_polymer_model/new_gene_system_NLbeads/hoxa13/analysis/cordinates/Reduced_cordinates",
    "num_beads": 9350,
    "num_conformations": 13000,
    "target_bead_index": 5285,
    "range_start": 1,
    "range_end": 9349
}

dataset4 = {
    "file_path": "/mnt/mydisk/Program/multiscale_polymer_model/Rahul/check_code_chr17/Reduced_cordinates_chr17",
    "num_beads": 8938,
    "num_conformations": 13000,
    "target_bead_index": 3232,
    "range_start": 1,
    "range_end": 8937
}

dataset5 = {
    "file_path": "/mnt/mydisk/Program/multiscale_polymer_model/new_gene_system_NLbeads/hoxa11/analysis/cordinates/Reduced_cordinates",
    "num_beads": 9203,
    "num_conformations": 13000,
    "target_bead_index": 3434,
    "range_start": 1,
    "range_end": 9202
}

# Calculate distances for both datasets
x1, avg1, err1, target_id1 = calculate_distances(**dataset1)
x2, avg2, err2, target_id2 = calculate_distances(**dataset2)
x3, avg3, err3, target_id3 = calculate_distances(**dataset3)
x4, avg4, err4, target_id4 = calculate_distances(**dataset4)
x5, avg5, err5, target_id5 = calculate_distances(**dataset5)



# Function to load custom labels from a file
def load_custom_labels(file_path):
    """
    Loads custom labels from a file.
    Assumes the file contains a single column of integers (0s and 1s).
    Returns a list of integers.
    """
    return np.loadtxt(file_path, dtype=int)

# Define the scaling function
def scale_index(custom_labels):
    """
    Scales index values based on custom_labels.
    custom_labels: List[int] where 0 means +10 and 1 means +20.
    Returns a scaled x-axis array.
    """
    scaling_values = []
    cumulative_value = 0  # Starting value for the scale
    for label in custom_labels:
        if label == 0:
            cumulative_value += 7.35
        elif label == 1:
            cumulative_value += 142
        scaling_values.append(cumulative_value)
    return np.array(scaling_values)

# Load custom labels from file
labels_file1 = "/mnt/mydisk/Program/multiscale_polymer_model/new_gene_system_NLbeads/lin28A/information/binary"  # Replace with your file path
labels_file2 = "/mnt/mydisk/Program/multiscale_polymer_model/Rahul/check_code_chr12/binary"  # Replace with your file path
labels_file3 = "/mnt/mydisk/Program/multiscale_polymer_model/new_gene_system_NLbeads/hoxa13/information/binary"  # Replace with your file path
labels_file4 = "/mnt/mydisk/Program/multiscale_polymer_model/Rahul/check_code_chr17/binary"  # Replace with your file path
labels_file5 = "/mnt/mydisk/Program/multiscale_polymer_model/new_gene_system_NLbeads/hoxa11/information/binary"  # Replace with your file path
custom_labels1 = load_custom_labels(labels_file1)
custom_labels2 = load_custom_labels(labels_file2)
custom_labels3 = load_custom_labels(labels_file3)
custom_labels4 = load_custom_labels(labels_file4)
custom_labels5 = load_custom_labels(labels_file5)
# Scale the x-axis values
scaled_x1 = scale_index(custom_labels1)
scaled_x2 = scale_index(custom_labels2)  # Assuming the same scaling for both datasets
scaled_x3 = scale_index(custom_labels3)
scaled_x4 = scale_index(custom_labels4)  # Assuming the same scaling for both datasets
scaled_x5 = scale_index(custom_labels5)  # Assuming the same scaling for both datasets
#print(scaled_x1)
x1_mod = np.array([index - scaled_x1[target_id1-1] for index in scaled_x1])
x2_mod = np.array([index - scaled_x2[target_id2-1] for index in scaled_x2])
x3_mod = np.array([index - scaled_x3[target_id3-1] for index in scaled_x3])
x4_mod = np.array([index - scaled_x4[target_id4-1] for index in scaled_x4])
x5_mod = np.array([index - scaled_x5[target_id5-1] for index in scaled_x5])

x1_modn=x1_mod[:-1]
x2_modn=x2_mod[:-1]
x3_modn=x3_mod[:-1]
x4_modn=x4_mod[:-1]
x5_modn=x5_mod[:-1]
# Define the output file path
output_file = os.path.join("data", "x1_modn_avg1_data.csv")

# Open the file in write mode
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header (optional)
    writer.writerow(['x1_modn', 'avg1', 'err1'])
    
    # Write the data rows
    for x, y, z in zip(x1_modn, avg1[:len(x1_modn)], err1[:len(x1_modn)]):
        writer.writerow([x, y, z])

print(f"Data saved to {output_file}")

# Define the output file path
output_file = os.path.join("data", "x2_modn_avg2_data.csv")

# Open the file in write mode
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header (optional)
    writer.writerow(['x2_modn', 'avg2', 'err2'])

    # Write the data rows
    for x, y, z in zip(x2_modn, avg2[:len(x2_modn)],err2[:len(x2_modn)]):
        writer.writerow([x, y, z])

print(f"Data saved to {output_file}")

# Define the output file path
output_file = os.path.join("data", "x3_modn_avg3_data.csv")

# Open the file in write mode
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header (optional)
    writer.writerow(['x3_modn', 'avg3', 'err3'])
    
    # Write the data rows
    for x, y, z in zip(x3_modn, avg3[:len(x3_modn)], err3[:len(x3_modn)]):
        writer.writerow([x, y, z])

print(f"Data saved to {output_file}")

# Define the output file path
output_file = os.path.join("data", "x4_modn_avg4_data.csv")

# Open the file in write mode
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header (optional)
    writer.writerow(['x4_modn', 'avg4', 'err4'])

    # Write the data rows
    for x, y, z in zip(x4_modn, avg4[:len(x4_modn)],err4[:len(x4_modn)]):
        writer.writerow([x, y, z])

print(f"Data saved to {output_file}")

# Define the output file path
output_file = os.path.join("data", "x5_modn_avg5_data.csv")

# Open the file in write mode
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header (optional)
    writer.writerow(['x5_modn', 'avg5', 'err5'])

    # Write the data rows
    for x, y, z in zip(x5_modn, avg5[:len(x5_modn)],err5[:len(x5_modn)]):
        writer.writerow([x, y, z])

print(f"Data saved to {output_file}")


# Plot the results
plt.figure(figsize=(12, 6))
#plt.errorbar(x2_modn, avg2[:len(x2_modn)], yerr=err2[:len(x2_modn)], fmt='s', label="chr12 (NANOG)", color='black', ecolor='lightgray', capsize=3, elinewidth=1)  # Adjust elinewidth as needed

#plt.errorbar(x2_modn, avg2[:len(x2_modn)], yerr=err2[:len(x2_modn)], fmt='.', label="Nanog", color='black', ecolor='lightgray', capsize=3, elinewidth=0.1)
#plt.errorbar(x1_modn, avg1[:len(x1_modn)], yerr=err1[:len(x1_modn)], fmt='.', label="Lin28A", color='blue', ecolor='lightgray', capsize=3, elinewidth=0.1)
plt.plot(x1_modn, avg1[:len(x1_modn)], '-s', label="Lin28A", color='blue', markersize=5)
plt.plot(x2_modn, avg2[:len(x2_modn)], '-o', label="Nanog", color='black', markersize=5)
plt.plot(x3_modn, avg3[:len(x3_modn)], '-s', label="Hoxa13", color='red', markersize=5)
plt.plot(x4_modn, avg4[:len(x4_modn)], '-o', label="Hoxb4", color='green', markersize=5)
plt.plot(x5_modn, avg5[:len(x5_modn)], '-o', label="Hoxa11", color='yellow', markersize=5)
#plt.plot(x2_modn, avg2[:len(x2_modn)], '-o', label="Nanog", color='black', markersize=6, linewidth=2)
#plt.plot(x1_modn, avg1[:len(x1_modn)], '-s', label="Lin28A", color='blue', markersize=6, linewidth=2)

#plt.plot(x2_modn, avg2[:len(x2_modn)], marker='.', linestyle='-', color='black', label="chr12 (NANOG)")

#plt.plot(x2_modn, avg2[:len(x2_modn)], fmt='.', label="chr12 (NANOG)", color='black', ecolor='lightgray', capsize=3, elinewidth=0.1)
#plt.errorbar(x3_modn, avg2[1787-75:1787], yerr=err2[1787-75:1787], fmt='s', label="Dataset 2 (chr12 promoter)", color='red', ecolor='lightgray', capsize=3)
#plt.errorbar(x4_modn, avg2[1787+515:1787+523], yerr=err2[1787+515:1787+523], fmt='s', label="Dataset 2 (chr12 enhancer)", color='black', ecolor='lightgray', capsize=3)

# Add labels, title, and legend
# Define font properties
label_font = {
    'family': 'serif',  # Change to 'sans-serif', 'monospace', etc., if desired
    'size': 25,         # Font size
    'weight': 'bold',   # Font weight ('normal', 'bold', 'light', etc.)
    #'style': 'italic'   # Font style ('normal', 'italic', 'oblique')
}
tick_font_size = 25
# Set font size for tick labels

plt.xticks(fontsize=tick_font_size)
plt.yticks(fontsize=tick_font_size)
# Add labels with custom fonts
plt.xlabel("Genomic distance (in kbps)", fontdict=label_font)
plt.ylabel("Average distance (in nm)", fontdict=label_font)
xch = np.arange(-200000, 200001,20000)
scaled_x = xch / 1000  # Divide each x value by 1000 for the labels
# Set custom x-axis ticks and labels
plt.xticks(ticks=xch, labels=scaled_x)
# Define custom y-ticks
x_ticks = np.arange(-40000, 120001,10000 )  # Example range: Adjust as needed
plt.xticks(ticks=x_ticks, fontsize=tick_font_size)

#plt.xlim(-900,1700)
plt.xlim(-40000,120001)

plt.legend(fontsize=25)
#plt.grid(True)
plt.tight_layout()

output_path = os.path.join("Plots", "Enhancer_Promoter.pdf")
plt.savefig(output_path, format="pdf", dpi=600)


plt.show()
