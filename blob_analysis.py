import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from scipy.stats import lognorm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from scipy.stats import mode
import os
import sys

# Function to calculate the center of mass of a blob
def center_of_mass(points):
    return np.mean(points, axis=0)

# Function to calculate the eccentricity of a blob
def eccentricity(points):
    if len(points) < 4:
        return np.nan  # Not enough points for a convex hull
    hull = ConvexHull(points)
    max_distance = max(
        np.linalg.norm(points[simplex[0]] - points[simplex[1]])
        for simplex in hull.simplices
    )
    max_diameter = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    return max_distance / max_diameter if max_diameter > 0 else np.nan

# Function to calculate the radius of gyration of a blob
def radius_of_gyration(points):
    com = center_of_mass(points)
    squared_distances = np.sum((points - com) ** 2, axis=1)
    return np.sqrt(np.mean(squared_distances))

# Function to calculate a power-law fit
def power_law(x, a, b):
    return a * np.power(x, b)

# Function to calculate the end-to-end distance of a blob
def end_to_end_distance(points):
    if len(points) < 2:
        return 0  # Not enough points to calculate distance
    distances = [
        np.linalg.norm(points[i] - points[j])
        for i in range(len(points))
        for j in range(i + 1, len(points))
    ]
    return max(distances)
# Function to calculate ellipsoid surface area
def ellipsoid_surface_area(a, b, c):
    p = 1.6075  # Empirical constant
    return 4 * np.pi * ((a**p * b**p + a**p * c**p + b**p * c**p) / 3) ** (1 / p)
# Function to calculate the maximum pairwise distance in a blob
def calculate_major_axis(points):
    if len(points) < 4:
        return 0  # For blobs with fewer than 2 points
    max_dist = max(
        np.linalg.norm(points[i] - points[j]) for i in range(len(points)) for j in range(i + 1, len(points))
    )
    return max_dist#size_limit   # Major axis is half the maximum distance

# Define the function to calculate score for continuous indices
def calculate_score(indices, values):
    total_score = 0
    start_idx = indices[0]  # Starting index of the first group
    start_val = values[0]  # Starting value of the first group
    count = 1  # Count of continuous indices

    for i in range(1, len(indices)):
        if indices[i] == indices[i-1] + 1:  # Check if continuous
            count += 1  # Increment the count of continuous indices
        else:
            # Calculate score for the previous continuous range
            end_idx = indices[i-1]
            end_val = values[i-1]

            # Calculate the range size and score
            range_size = count
            total_score += (end_val - start_val + 1) - range_size
            total_score += (range_size * 141/7.35)

            # Reset for the new range
            start_idx = indices[i]
            start_val = values[i]
            count = 1

    # For the last continuous segment
    end_idx = indices[-1]
    end_val = values[-1]
    range_size = count
    total_score += (end_val - start_val + 1) - range_size
    total_score += (range_size * 141/7.35)

    return total_score*7.35
# Define the log-normal function
def log_normal(x, mu, sigma, a):
    """
    Log-normal distribution.
    :param x: Input data
    :param mu: Mean of the log
    :param sigma: Standard deviation of the log
    :param a: Scaling factor
    :return: Log-normal function evaluated at x
    """
    return a * (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - mu)**2) / (2 * sigma**2))

# Fit the log-normal distribution
def fit_log_normal(x_data, y_data):
    """
    Fit a log-normal distribution to the data.
    :param x_data: X values (e.g., blob sizes)
    :param y_data: Y values (e.g., end-to-end distances)
    :return: Optimal parameters and covariance
    """
    # Provide an initial guess for the parameters
    #initial_guess = [1,-1]#np.log(np.mean(x_data)), 1.0, np.max(y_data)]
    initial_guess = [np.log(np.mean(x_data)), 1.0, np.max(y_data)]
    #initial_mu = np.log(np.mean(x_data))  # Log of mean of x values
    #initial_sigma = np.std(np.log(x_data))  # Standard deviation of log(x)
    #initial_a = np.max(y_data)  # Maximum of y values

    
    # Fit the log-normal curve to the data
    popt, pcov = curve_fit(log_normal, x_data, y_data, p0=initial_guess, maxfev=5000)
    #popt, pcov = curve_fit(log_normal, x_data, y_data, p0=[initial_mu, initial_sigma, initial_a], maxfev=10000)
    
    return popt, pcov
# Function to calculate the center of mass of a blob
def center_of_mass(points):
    return np.mean(points, axis=0)

# Parameters (User-Defined)
num_snaps_to_process = 13000  # Number of snapshots to process
num_beads_per_snapshot = 998  # Number of beads per snapshot
eps = 24  # DBSCAN parameter
min_samples = 2  # DBSCAN parameter
T1,T2,T3,T4 = 30,5,30,50
HULLPoints = 4

data = np.loadtxt('CORDINATE FILE OF NUCLEOSOMES')
# Load the base-pair positions for the 998 beads
index_values = np.loadtxt('INDEX VALUE OF NUCLEOSOME BEADS')  # Ensure this file aligns with the beads
assert len(index_values) == num_beads_per_snapshot, "Base-pair positions must match number of beads!"

# Validate input data size
total_points = data.size // 3  # Total number of (x, y, z) triplets
available_snaps = total_points // num_beads_per_snapshot

# Adjust `num_snaps_to_process` if it exceeds available snapshots
if num_snaps_to_process > available_snaps:
    print(f"Warning: Requested {num_snaps_to_process} snapshots, but only {available_snaps} are available.")
    num_snaps_to_process = available_snaps

# Calculate the exact number of elements to extract for reshaping
total_elements_needed = num_snaps_to_process * num_beads_per_snapshot * 3

# Slice the data to match the required number of elements
data_subset = data[:total_elements_needed]

# Reshape data: [snap, bead, coords]
coordinates = data_subset.reshape((num_snaps_to_process, num_beads_per_snapshot, 3))

# Prepare lists for storing results
all_blob_sizes = []
all_eccentricities = []
all_radii_of_gyration = []
all_blob_centers = []
all_blob_centers_rdf = []
all_blob_count = []
# Initialize lists to store the properties
all_blob_eccentricities = []
all_blob_majors = []
all_blob_intermediates = []
all_blob_minors = []
all_blob_surface_areas = []
all_blob_vol = []
all_blob_denBPS = []#.append(blob_size/blob_volumn)
all_blob_denNUC = []#.append(len(blob_points)/blob_volumn)
# Prepare a dictionary to store end-to-end distances for each blob size
end_to_end_by_size = {}
# Prepare lists to store a, b, c, and ratios for every eccentricity
semi_axes_data = {"a": [], "b": [], "c": [], "b/a": [], "c/a": [], "b/c": [], "eccentricity": []}
surface_Area=[]#.append(ellipsoid_surface_area(a, b, c))
BLOB_vol=[]#.append(ellipsoid_surface_area(a, b, c))
denNUC=[]
denBPS=[]
# Process each snapshot
print("Chr12")
#with open("chr12_blob_info.txt", "w") as outfile:
for snap in range(num_snaps_to_process):
#for snap in range(Start,End):
    snap_need=snap%130
    simuC=int(snap/130)
    snap_coords = coordinates[snap]
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(snap_coords)
    cntpre=len(all_blob_centers)

    for label in np.unique(labels):
        if label == -1 or snap_need<T1 or snap_need%T2!=0:
            continue  # Skip noise points
        blob_indices = np.where(labels == label)[0]
        # Compute the difference between consecutive elements
        differences = np.diff(blob_indices)
        # Count how many times the difference is greater than 1 (indicating a break)
        num_breaks = np.sum(differences > 1)
        blob_points = snap_coords[blob_indices]
        blob_center = center_of_mass(blob_points)
        print(f"center {snap_need} {simuC} {len(blob_points)} {num_breaks} {blob_center[0]} {blob_center[1]} {blob_center[2]}")
        all_blob_centers.append(blob_center)
        if(snap_need>=T3 and snap_need%T4==0):
            all_blob_centers_rdf.append(blob_center)
        blob_bp_positions = index_values[blob_indices]
        score = calculate_score(blob_indices, blob_bp_positions)
        blob_size = int(score)#len(blob_points)
        if(len(blob_points)< HULLPoints ):
            continue  # Skip if eccentricity cannot be calculated (e.g., fewer than 4 points)
            # Compute the convex hull
        hull = ConvexHull(blob_points)
        hull_points = blob_points[hull.vertices]
        # Compute covariance matrix
        cov_matrix = np.cov(hull_points.T)

        # Compute eigenvalues & eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
        # Sort eigenvalues & corresponding eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Compute axis lengths (2 * sqrt(eigenvalue))
        major_axis_length = 2 * np.sqrt(eigenvalues[0])  # Largest eigenvalue
        second_axis_length = 2 * np.sqrt(eigenvalues[1])  # Middle eigenvalue
        minor_axis_length = 2 * np.sqrt(eigenvalues[2])  # Smallest eigenvalue
        a = major_axis_length
        b = second_axis_length
        c = minor_axis_length
        # Compute eccentricity in 3D
        eccentricity = np.sqrt(1 - (eigenvalues[1] + eigenvalues[2]) / (2 * eigenvalues[0])) 

        surface_area_value = hull.area #ellipsoid_surface_area(a, b, c)
        surface_area_value = surface_area_value / 1e6
        blob_volumn = hull.volume #ellipsoid_surface_area(a, b, c)
        surface_Area.append(surface_area_value)
        BLOB_vol.append(blob_volumn)
        denNUC.append(len(blob_points)/blob_volumn)
        denBPS.append(blob_size/blob_volumn)
        # Perform PCA on the hull points
        pca = PCA()
        pca.fit(hull_points)
        # The explained variance gives the length of the axes
        axis_lengths = np.sqrt(pca.explained_variance_)
        # Major and Minor Axes (the axis with the largest variance is the major axis)
        #major_axis = axis_lengths[0]
        #minor_axes = axis_lengths[1:]
        #print("Major Axis Length:", major_axis)
        #print("Minor Axis Lengths:", minor_axes[0])
        #a=major_axis
        #b=minor_axes[0]
        #c=minor_axes[0]
        # Calculate the eccentricity (assuming 3 axes, this is just a simplification)
        #eccentricity1 = np.sqrt(1 - (minor_axes[0]**2 / major_axis**2))
        #blob_eccentricity=eccentricity1
        blob_eccentricity=eccentricity
        #print(a,b,c,eccentricity1)
        # Store data
        semi_axes_data["a"].append(a)
        semi_axes_data["b"].append(b)
        semi_axes_data["c"].append(c)
        semi_axes_data["b/a"].append(b / a)
        semi_axes_data["c/a"].append(c / a)
        semi_axes_data["b/c"].append(b / c)
        semi_axes_data["eccentricity"].append(blob_eccentricity)


        blob_rg = radius_of_gyration(blob_points)
        all_blob_sizes.append(blob_size)
        all_eccentricities.append(blob_eccentricity)
        all_radii_of_gyration.append(blob_rg)
        e2e_distance = end_to_end_distance(blob_points)

        # Append the properties to their respective lists
        all_blob_eccentricities.append(blob_eccentricity)
        all_blob_majors.append(a)
        all_blob_intermediates.append(b)
        all_blob_minors.append(c)
        all_blob_surface_areas.append(surface_area_value)
        all_blob_vol.append(blob_volumn)
        all_blob_denBPS.append(blob_size/blob_volumn)
        all_blob_denNUC.append(len(blob_points)/blob_volumn)
        #outfile.write(f"{simuC} {snap_need} {major_axis_length} {second_axis_length} {minor_axis_length} {blob_eccentricity} {surface_area_value} {blob_volumn} {len(blob_points)} {blob_size} {blob_rg} {e2e_distance}\n")
        print(f"{simuC} {snap_need} {major_axis_length} {second_axis_length} {minor_axis_length} {blob_eccentricity} {surface_area_value} {blob_volumn} {len(blob_points)} {blob_size} {blob_rg} {e2e_distance} ")
        if blob_size not in end_to_end_by_size:
            end_to_end_by_size[blob_size] = []
        end_to_end_by_size[blob_size].append(e2e_distance)
    if snap_need>=T1 and snap_need%T2==0:
        #print(len(all_blob_centers)-cntpre)
        #print(snap)
        all_blob_count.append(len(all_blob_centers)-cntpre) 
# Plot 1: Blob Size Distribution
all_radii_of_gyration = np.array(all_radii_of_gyration)
filtered_bins = np.linspace(200,4000, 30)
blob_size_hist12, blob_bin_edges12 = np.histogram(all_blob_sizes, bins=filtered_bins, density=True)
max_bin_index = np.argmax(blob_size_hist12)
most_probable_value = (blob_bin_edges12[max_bin_index] + blob_bin_edges12[max_bin_index + 1]) / 2
blob_centers12 = 0.5 * (blob_bin_edges12[:-1] + blob_bin_edges12[1:])
errors_BS12 = np.sqrt(blob_size_hist12 / len(all_blob_sizes))
print("Most probable blob size in base-pairs:", most_probable_value)
# Plot 2: Eccentricity Distribution
# Compute histogram for eccentricity
all_blob_eccentricities12=all_blob_eccentricities
eccentricity_bins = np.linspace(0, 1, 30)
eccentricity_hist12, bin_edges_ED12 = np.histogram(all_blob_eccentricities12, bins=eccentricity_bins, density=True)
eccentricity_centers12 = 0.5 * (bin_edges_ED12[:-1] + bin_edges_ED12[1:])
most_probable_eccentricity_idx = np.argmax(eccentricity_hist12)  # Index of max probability
most_probable_eccentricity = eccentricity_centers12[most_probable_eccentricity_idx]
tolerance = 1e-3  # Small tolerance to account for floating-point precision issues
matching_indices = np.where(np.abs(np.array(all_blob_eccentricities12) - most_probable_eccentricity) < tolerance)[0]
# Compute the average values of a, b, and surface area for these indices
average_rog = np.mean(np.array(all_radii_of_gyration)[matching_indices])
avg_a = np.mean(np.array(all_blob_majors)[matching_indices])
avg_b = np.mean(np.array(all_blob_intermediates)[matching_indices])
avg_c = np.mean(np.array(all_blob_minors)[matching_indices])
avg_area = np.mean(np.array(all_blob_surface_areas)[matching_indices])
avg_vol = np.mean(np.array(all_blob_vol)[matching_indices])  
avg_denBPS = np.mean(np.array(all_blob_denBPS)[matching_indices]) 
avg_denNUC = np.mean(np.array(all_blob_denNUC)[matching_indices]) 
std_rog = np.std(np.array(all_radii_of_gyration)[matching_indices])
std_a = np.std(np.array(all_blob_majors)[matching_indices])
std_b = np.std(np.array(all_blob_intermediates)[matching_indices])
std_c = np.std(np.array(all_blob_minors)[matching_indices])
std_area = np.std(np.array(all_blob_surface_areas)[matching_indices]) 
std_vol = np.std(np.array(all_blob_vol)[matching_indices]) 
std_denBPS = np.std(np.array(all_blob_denBPS)[matching_indices]) 
std_denNUC = np.std(np.array(all_blob_denNUC)[matching_indices]) 
# Print the results
print(f"Most Probable Eccentricity: {most_probable_eccentricity:.4f}")
print(f"Average Major Axis at peak Eccentricity(a): {avg_a:.4f} nm")
print(f"Average Intermediate Axis at peak Eccentricity (b): {avg_b:.4f} nm")
print(f"Average Minor Axis at peak Eccentricity (b): {avg_b:.4f} nm")
print(f"Average Surface Area at peak Eccentricity: {avg_area:.4f} μm²")
print(f"Average BloB Volumn at peak Eccentricity: {avg_vol:.4f} nm3")
print(f"Average denNUC at peak Eccentricity: {avg_denNUC:.4f} nuc per nm3")
print(f"Average denBPS at peak Eccentricity: {avg_denBPS:.4f} nuc per nm3")
print(f"Average ROG at peak Eccentricity: {average_rog:.4f} nm")
print(f"STD Major Axis at peak Eccentricity(a): {std_a:.4f} nm")
print(f"STD Intermediate Axis at peak Eccentricity (b): {std_b:.4f} nm")
print(f"STD Minor Axis at peak Eccentricity (b): {std_c:.4f} nm")
print(f"STD Surface Area at peak Eccentricity: {std_area:.4f} μm²")
print(f"STD BloB Volumn at peak Eccentricity: {std_vol:.4f} nm3")
print(f"STD denNUC at peak Eccentricity: {std_denNUC:.4f} nuc per nm3")
print(f"STD denBPS at peak Eccentricity: {std_denBPS:.4f} nuc per nm3")
print(f"STD ROG at peak Eccentricity: {std_rog:.4f} nm")

mode_result = mode(np.array(all_blob_majors)[matching_indices])
if isinstance(mode_result.mode, np.ndarray):
    most_probable_a = mode_result.mode[0]
else:
    most_probable_a = mode_result.mode  # Directly assign the scalar if it's a single value
mode_result = mode(np.array(all_blob_intermediates)[matching_indices])
if isinstance(mode_result.mode, np.ndarray):
    most_probable_b = mode_result.mode[0]
else:
    most_probable_b = mode_result.mode  # Directly assign the scalar if it's a single value
mode_result = mode(np.array(all_blob_surface_areas)[matching_indices] )
if isinstance(mode_result.mode, np.ndarray):
    most_probable_surface_area = mode_result.mode[0]
else:
    most_probable_surface_area = mode_result.mode  # Directly assign the scalar if it's a single value
mode_result = mode(np.array(all_blob_vol)[matching_indices] )
if isinstance(mode_result.mode, np.ndarray):
    most_probable_vol = mode_result.mode[0]
else:
    most_probable_vol = mode_result.mode  # Directly assign the scalar if it's a single value
mode_result = mode(np.array(all_blob_denNUC)[matching_indices] )
if isinstance(mode_result.mode, np.ndarray):
    most_probable_denNUC = mode_result.mode[0]
else:
    most_probable_denNUC = mode_result.mode  # Directly assign the scalar if it's a single value
mode_result = mode(np.array(all_blob_denBPS)[matching_indices] )
if isinstance(mode_result.mode, np.ndarray):
    most_probable_denBPS = mode_result.mode[0]
else:
    most_probable_denBPS = mode_result.mode  # Directly assign the scalar if it's a single value
mode_result = mode(np.array(all_radii_of_gyration)[matching_indices] )
if isinstance(mode_result.mode, np.ndarray):
    most_probable_rog = mode_result.mode[0]
else:
    most_probable_rog = mode_result.mode  # Directly assign the scalar if it's a single value
print(f"Most probable Major Axis at peak Eccentricity(a): {most_probable_a:.4f} nm")
print(f"Most probable Minor Axis at peak Eccentricity (b): {most_probable_b:.4f} nm")
print(f"Most probabel Surface Area at peak Eccentricity: {most_probable_surface_area:.4f} μm²")
print(f"Most probabel BloB Volume at peak Eccentricity: {most_probable_vol:.4f} nm3")
print(f"Most probabel denNUC at peak Eccentricity: {most_probable_denNUC:.4f} nuc per nm3")
print(f"Most probabel denBPS at peak Eccentricity: {most_probable_denBPS:.4f} bps per nm3")
print(f"Most probabel ROG at peak Eccentricity: {most_probable_rog:.4f} nm")
# Plot 3: Surface Area histogram
# Convert surface areas
surface_areas12 = np.array(surface_Area)
percentile = np.percentile(surface_areas12, 95)
bins_SA12 = np.linspace(0,0.014,30)
hist_values_SA12, bin_edges_SA12 = np.histogram(surface_areas12, bins=bins_SA12)
max_bin_index = np.argmax(hist_values_SA12)
most_probable_value = (bin_edges_SA12[max_bin_index] + bin_edges_SA12[max_bin_index + 1]) / 2
print("Most probable Surface Area in micrometer square :", most_probable_value)
bin_centers_SA12 = (bin_edges_SA12[:-1] + bin_edges_SA12[1:]) / 2  # Compute bin centers
#popt, pcov = fit_log_normal(bin_centers_SA12, hist_values_SA12)
#mu_fit, sigma_fit, a_fit = popt
#fitted_values = log_normal(bin_centers_SA12, mu_fit, sigma_fit, a_fit)
mean_value = np.mean(surface_areas12)
std_value = np.std(surface_areas12)
print(f"Mean of Surface Area distrbution: {mean_value:.4f} μm²\nStd of Surface Area distrbution: {std_value:.4f} μm²\n")
# Plot 3A: Volumn histogram
# Convert Volumn
vol12 = np.array(BLOB_vol) 
percentile = np.percentile(vol12, 95)
print(min(vol12),max(vol12))
bins_V12 = np.linspace(0,100,30)
hist_values_V12, bin_edges_V12 = np.histogram(vol12, bins=bins_V12)
max_bin_index = np.argmax(hist_values_V12)
most_probable_value = (bin_edges_V12[max_bin_index] + bin_edges_V12[max_bin_index + 1]) / 2
print("Most probable BloB Volumn in nmcube :", most_probable_value)
bin_centers_V12 = (bin_edges_V12[:-1] + bin_edges_V12[1:]) / 2  # Compute bin centers
#popt, pcov = fit_log_normal(bin_centers_V12, hist_values_V12)
#mu_fit, sigma_fit, a_fit = popt
#fitted_values = log_normal(bin_centers_V12, mu_fit, sigma_fit, a_fit)
mean_value = np.mean(vol12)
std_value = np.std(vol12)
print(f"Mean of BloB Volumn distrbution: {mean_value:.4f} nm3\nStd of BloB Volumn distrbution: {std_value:.4f} nm3\n")
# Plot 3B: denNUC histogram
# Convert Volumn
denNUC12 = np.array(denNUC) 
percentile = np.percentile(denNUC12, 95)
print(min(denNUC12),max(denNUC12))
bins_denNUC12 = np.linspace(0,0.5,30)
hist_values_denNUC12, bin_edges_denNUC12 = np.histogram(denNUC12, bins=bins_denNUC12)
max_bin_index = np.argmax(hist_values_denNUC12)
most_probable_value = (bin_edges_denNUC12[max_bin_index] + bin_edges_denNUC12[max_bin_index + 1]) / 2
print("Most probable denNUC in nuc per nmcube :", most_probable_value)
bin_centers_denNUC12 = (bin_edges_denNUC12[:-1] + bin_edges_denNUC12[1:]) / 2  # Compute bin centers
#popt, pcov = fit_log_normal(bin_centers_V12, hist_values_V12)
#mu_fit, sigma_fit, a_fit = popt
#fitted_values = log_normal(bin_centers_V12, mu_fit, sigma_fit, a_fit)
mean_value = np.mean(denNUC12)
std_value = np.std(denNUC12)
print(f"Mean of denNUC distrbution: {mean_value:.4f} nuc per nm3\nStd of denNUC distrbution: {std_value:.4f} nuc per nm3\n")
# Plot 3C: denBPS histogram
# Convert Volumn
denBPS12 = np.array(denBPS) 
percentile = np.percentile(denBPS12, 95)
print(min(denBPS12),max(denBPS12))
bins_denBPS12 = np.linspace(0,50,30)
hist_values_denBPS12, bin_edges_denBPS12 = np.histogram(denBPS12, bins=bins_denBPS12)
max_bin_index = np.argmax(hist_values_denBPS12)
most_probable_value = (bin_edges_denBPS12[max_bin_index] + bin_edges_denBPS12[max_bin_index + 1]) / 2
print("Most probable denBPS in bps per nmcube :", most_probable_value)
bin_centers_denBPS12 = (bin_edges_denBPS12[:-1] + bin_edges_denBPS12[1:]) / 2  # Compute bin centers
#popt, pcov = fit_log_normal(bin_centers_V12, hist_values_V12)
#mu_fit, sigma_fit, a_fit = popt
#fitted_values = log_normal(bin_centers_V12, mu_fit, sigma_fit, a_fit)
mean_value = np.mean(denBPS12)
std_value = np.std(denBPS12)
print(f"Mean of denBPS distrbution: {mean_value:.4f} bps per nm3\nStd of denBPS distrbution: {std_value:.4f} bps per nm3\n")

# Plot 3D: AllBlobSize histogram
# Convert Volumn
all_blob_sizes12 = np.array(all_blob_sizes)
percentile = np.percentile(all_blob_sizes12, 95)
print(min(all_blob_sizes12),max(all_blob_sizes12))
bins_all_blob_sizes12 = np.linspace(0,1000,30)
hist_values_all_blob_sizes12, bin_edges_all_blob_sizes12 = np.histogram(all_blob_sizes12, bins=bins_all_blob_sizes12)
max_bin_index = np.argmax(hist_values_all_blob_sizes12)
most_probable_value = (bin_edges_all_blob_sizes12[max_bin_index] + bin_edges_all_blob_sizes12[max_bin_index + 1]) / 2
print("Most probable all_blob_sizes in bps per nmcube :", most_probable_value)
bin_centers_all_blob_sizes12 = (bin_edges_all_blob_sizes12[:-1] + bin_edges_all_blob_sizes12[1:]) / 2  # Compute bin centers
#popt, pcov = fit_log_normal(bin_centers_V12, hist_values_V12)
#mu_fit, sigma_fit, a_fit = popt
#fitted_values = log_normal(bin_centers_V12, mu_fit, sigma_fit, a_fit)
mean_value = np.mean(all_blob_sizes12)
std_value = np.std(all_blob_sizes12)
print(f"Mean of all_blob_sizes distrbution: {mean_value:.4f} bps per nm3\nStd of all_blob_sizes distrbution: {std_value:.4f} bps per nm3\n")

#############################################################################################################################

# Plot 4: ROG
rg_bins12 = np.linspace(6,70, 30)
rg_hist12, rg_bin_edges = np.histogram(all_radii_of_gyration, bins=rg_bins12, density=True)
max_bin_index = np.argmax(rg_hist12)
most_probable_value = (rg_bin_edges[max_bin_index] + rg_bin_edges[max_bin_index + 1]) / 2
print("Most probable ROG in nm:", most_probable_value)
rg_centers12 = 0.5 * (rg_bin_edges[:-1] + rg_bin_edges[1:])
errors_rog12 = np.sqrt(rg_hist12 / len(all_radii_of_gyration))
#popt, pcov = fit_log_normal(rg_centers12, rg_hist12)
#mu_fit, sigma_fit, a_fit = popt
#fitted_values = log_normal(rg_centers12, mu_fit, sigma_fit, a_fit)
mean_value = np.mean(all_radii_of_gyration)
std_value = np.std(all_radii_of_gyration)
print(f"Mean of ROG distrbution: {mean_value:.4f} nm\nStd of ROG distrbution: {std_value:.4f} nm\n")

# Plot 5: END-to-END distances for each blob size
# Calculate the average and standard deviation of end-to-end distances for each blob size
blob_sizes12 = sorted(end_to_end_by_size.keys())
N_values = np.array(blob_sizes12[:])
avg_e2e_distances12 = [np.mean(end_to_end_by_size[size]) for size in blob_sizes12[:]]
std_e2e_distances = [np.std(end_to_end_by_size[size]) for size in blob_sizes12[:]]
blob_sizes_np = np.array(blob_sizes12[:])  # Convert to numpy array
avg_e2e_distances_np = np.array(avg_e2e_distances12)
#smooth_blob_sizes12 = np.linspace(min(blob_sizes12[:]), max(blob_sizes12[:]), 50)
#popt, pcov = fit_log_normal(blob_sizes12[:], avg_e2e_distances12)
#mu_fit, sigma_fit, a_fit = popt
#fitted_values_e2e12 = log_normal(smooth_blob_sizes12, mu_fit, sigma_fit, a_fit)
#print(f"Fitted Log-Normal Parameters: mu={mu_fit:.4f}, sigma={sigma_fit:.4f}, a={a_fit:.4f}")

# Plot 6: RDF
# Convert to NumPy array
all_blob_centers_rdf = np.array(all_blob_centers_rdf, dtype=np.float32)
print("Number of blob centers for rdf:", len(all_blob_centers_rdf))
distances = pdist(all_blob_centers_rdf)
dist_bins12 = np.linspace(0, 1200, 30)
dist_hist12, bin_edges = np.histogram(distances, bins=dist_bins12, density=True)
max_bin_index = np.argmax(dist_hist12)
most_probable_value = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
print("Most probable RDF in nm:", most_probable_value)
mean_value = np.mean(distances)
print("Avg RDF in nm:", mean_value)
std_value = np.std(distances)
print("Std RDF in nm:", std_value)
dist_centers12 = 0.5 * (bin_edges[:-1] + bin_edges[1:])
errors_rdf12 = np.sqrt(dist_hist12 / len(distances))
################################################################################################################################################

# Parameters (User-Defined)
num_beads_per_snapshot = 1036  # Number of beads per snapshot
# Load the file containing x, y, z coordinates
data = np.loadtxt('CORDINATE FILE OF NUCLEOSOMES')
# Load the base-pair positions for the 998 beads
index_values = np.loadtxt('INDEX VALUE OF NUCLEOSOME BEADS')  # Ensure this file aligns with the beads
assert len(index_values) == num_beads_per_snapshot, "Base-pair positions must match number of beads!"

# Validate input data size
total_points = data.size // 3  # Total number of (x, y, z) triplets
available_snaps = total_points // num_beads_per_snapshot

# Adjust `num_snaps_to_process` if it exceeds available snapshots
if num_snaps_to_process > available_snaps:
    print(f"Warning: Requested {num_snaps_to_process} snapshots, but only {available_snaps} are available.")
    num_snaps_to_process = available_snaps

# Calculate the exact number of elements to extract for reshaping
total_elements_needed = num_snaps_to_process * num_beads_per_snapshot * 3

# Slice the data to match the required number of elements
data_subset = data[:total_elements_needed]

# Reshape data: [snap, bead, coords]
coordinates = data_subset.reshape((num_snaps_to_process, num_beads_per_snapshot, 3))

# Prepare lists for storing results
all_blob_sizes = []
all_eccentricities = []
all_radii_of_gyration = []
all_blob_centers = []
all_blob_centers_rdf = []
all_blob_count = []
# Initialize lists to store the properties
all_blob_eccentricities = []
all_blob_majors = []
all_blob_intermediates = []
all_blob_minors = []
all_blob_surface_areas = []
all_blob_vol = []
all_blob_denBPS = []#.append(blob_size/blob_volumn)
all_blob_denNUC = []#.append(len(blob_points)/blob_volumn)
# Prepare a dictionary to store end-to-end distances for each blob size
end_to_end_by_size = {}
# Prepare lists to store a, b, c, and ratios for every eccentricity
semi_axes_data = {"a": [], "b": [], "c": [], "b/a": [], "c/a": [], "b/c": [], "eccentricity": []}
surface_Area=[]#.append(ellipsoid_surface_area(a, b, c))
BLOB_vol=[]#.append(ellipsoid_surface_area(a, b, c))
denNUC=[]
denBPS=[]
# Process each snapshot
print("Chr17")
#with open("chr17_blob_info.txt", "w") as outfile:
for snap in range(num_snaps_to_process):
#for snap in range(Start,End):
    snap_need=snap%130
    simuC=int(snap/130)
    #for simuC in {28,37,45,47,89}:
        #continue
    snap_coords = coordinates[snap]
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(snap_coords)
    cntpre=len(all_blob_centers)

    for label in np.unique(labels):
        if label == -1 or snap_need<T1 or snap_need%T2!=0:
            continue  # Skip noise points
        blob_indices = np.where(labels == label)[0]
        # Compute the difference between consecutive elements
        differences = np.diff(blob_indices)
        # Count how many times the difference is greater than 1 (indicating a break)
        num_breaks = np.sum(differences > 1)
        blob_points = snap_coords[blob_indices]
        blob_center = center_of_mass(blob_points)
        print(f"center {snap_need} {simuC} {len(blob_points)} {num_breaks} {blob_center[0]} {blob_center[1]} {blob_center[2]}")
        all_blob_centers.append(blob_center)
        if(snap_need>=T3 and snap_need%T4==0):
            all_blob_centers_rdf.append(blob_center)
        blob_bp_positions = index_values[blob_indices]
        score = calculate_score(blob_indices, blob_bp_positions)
        blob_size = int(score)#len(blob_points)
        if(len(blob_points)< HULLPoints ):
            continue  # Skip if eccentricity cannot be calculated (e.g., fewer than 4 points)
            # Compute the convex hull
        hull = ConvexHull(blob_points)
        hull_points = blob_points[hull.vertices]
        # Compute covariance matrix
        cov_matrix = np.cov(hull_points.T)

        # Compute eigenvalues & eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues & corresponding eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Compute axis lengths (2 * sqrt(eigenvalue))
        major_axis_length = 2 * np.sqrt(eigenvalues[0])  # Largest eigenvalue
        second_axis_length = 2 * np.sqrt(eigenvalues[1])  # Middle eigenvalue
        minor_axis_length = 2 * np.sqrt(eigenvalues[2])  # Smallest eigenvalue
        a = major_axis_length
        b = second_axis_length
        c = minor_axis_length
        # Compute eccentricity in 3D
        eccentricity = np.sqrt(1 - (eigenvalues[1] + eigenvalues[2]) / (2 * eigenvalues[0]))

        surface_area_value = hull.area #ellipsoid_surface_area(a, b, c)
        surface_area_value = surface_area_value / 1e6
        blob_volumn = hull.volume #ellipsoid_surface_area(a, b, c)
        surface_Area.append(surface_area_value)
        BLOB_vol.append(blob_volumn)
        denNUC.append(len(blob_points)/blob_volumn)
        denBPS.append(blob_size/blob_volumn)
        # Perform PCA on the hull points
        pca = PCA()
        pca.fit(hull_points)
        # The explained variance gives the length of the axes
        axis_lengths = np.sqrt(pca.explained_variance_)
        # Major and Minor Axes (the axis with the largest variance is the major axis)
        #major_axis = axis_lengths[0]
        #minor_axes = axis_lengths[1:]
        #print("Major Axis Length:", major_axis)
        #print("Minor Axis Lengths:", minor_axes[0])
        #a=major_axis
        #b=minor_axes[0]
        #c=minor_axes[0]
        # Calculate the eccentricity (assuming 3 axes, this is just a simplification)
        #eccentricity1 = np.sqrt(1 - (minor_axes[0]**2 / major_axis**2))
        #blob_eccentricity=eccentricity1
        blob_eccentricity=eccentricity
        #print(a,b,c,eccentricity1)
        # Store data
        semi_axes_data["a"].append(a)
        semi_axes_data["b"].append(b)
        semi_axes_data["c"].append(c)
        semi_axes_data["b/a"].append(b / a)
        semi_axes_data["c/a"].append(c / a)
        semi_axes_data["b/c"].append(b / c)
        semi_axes_data["eccentricity"].append(blob_eccentricity)


        blob_rg = radius_of_gyration(blob_points)
        all_blob_sizes.append(blob_size)
        all_eccentricities.append(blob_eccentricity)
        all_radii_of_gyration.append(blob_rg)
        e2e_distance = end_to_end_distance(blob_points)

        # Append the properties to their respective lists
        all_blob_eccentricities.append(blob_eccentricity)
        all_blob_majors.append(a)
        all_blob_intermediates.append(b)
        all_blob_minors.append(c)
        all_blob_surface_areas.append(surface_area_value)
        all_blob_vol.append(blob_volumn)
        all_blob_denBPS.append(blob_size/blob_volumn)
        all_blob_denNUC.append(len(blob_points)/blob_volumn)
        #outfile.write(f"{simuC} {snap_need} {major_axis_length} {second_axis_length} {minor_axis_length} {blob_eccentricity} {surface_area_value} {blob_volumn} {len(blob_points)} {blob_size} {blob_rg} {e2e_distance}\n")
        print(f"{simuC} {snap_need} {major_axis_length} {second_axis_length} {minor_axis_length} {blob_eccentricity} {surface_area_value} {blob_volumn} {len(blob_points)} {blob_size} {blob_rg} {e2e_distance}")
        if blob_size not in end_to_end_by_size:
            end_to_end_by_size[blob_size] = []
        end_to_end_by_size[blob_size].append(e2e_distance)
    if snap_need>=T1 and snap_need%T2==0:
        #print(len(all_blob_centers)-cntpre)
        #print(snap)
        all_blob_count.append(len(all_blob_centers)-cntpre)
#plt.figure(figsize=(10,6))
#plt.plot(all_blob_count)
#plt.show()
#exit()
# Plot 1: Blob Size Distribution
all_radii_of_gyration = np.array(all_radii_of_gyration)
filtered_bins = np.linspace(200,4000, 30)
blob_size_hist17, blob_bin_edges17 = np.histogram(all_blob_sizes, bins=filtered_bins, density=True)
max_bin_index = np.argmax(blob_size_hist17)
most_probable_value = (blob_bin_edges17[max_bin_index] + blob_bin_edges17[max_bin_index + 1]) / 2
blob_centers17 = 0.5 * (blob_bin_edges17[:-1] + blob_bin_edges17[1:])
errors_BS17 = np.sqrt(blob_size_hist17 / len(all_blob_sizes))
print("Most probable blob size in base-pairs:", most_probable_value)
# Plot 2: Eccentricity Distribution
# Compute histogram for eccentricity
all_blob_eccentricities17=all_blob_eccentricities
eccentricity_bins = np.linspace(0, 1, 30)
eccentricity_hist17, bin_edges_ED17 = np.histogram(all_blob_eccentricities17, bins=eccentricity_bins, density=True)
eccentricity_centers17 = 0.5 * (bin_edges_ED17[:-1] + bin_edges_ED17[1:])
most_probable_eccentricity_idx = np.argmax(eccentricity_hist17)  # Index of max probability
most_probable_eccentricity = eccentricity_centers17[most_probable_eccentricity_idx]
tolerance = 1e-3  # Small tolerance to account for floating-point precision issues
matching_indices = np.where(np.abs(np.array(all_blob_eccentricities17) - most_probable_eccentricity) < tolerance)[0]
# Compute the average values of a, b, and surface area for these indices
average_rog = np.mean(np.array(all_radii_of_gyration)[matching_indices])
avg_a = np.mean(np.array(all_blob_majors)[matching_indices])
avg_b = np.mean(np.array(all_blob_intermediates)[matching_indices])
avg_c = np.mean(np.array(all_blob_minors)[matching_indices])
avg_area = np.mean(np.array(all_blob_surface_areas)[matching_indices])
avg_vol = np.mean(np.array(all_blob_vol)[matching_indices])
avg_denBPS = np.mean(np.array(all_blob_denBPS)[matching_indices])
avg_denNUC = np.mean(np.array(all_blob_denNUC)[matching_indices])
std_rog = np.std(np.array(all_radii_of_gyration)[matching_indices])
std_a = np.std(np.array(all_blob_majors)[matching_indices])
std_b = np.std(np.array(all_blob_intermediates)[matching_indices])
std_c = np.std(np.array(all_blob_minors)[matching_indices])
std_area = np.std(np.array(all_blob_surface_areas)[matching_indices])
std_vol = np.std(np.array(all_blob_vol)[matching_indices])
std_denBPS = np.std(np.array(all_blob_denBPS)[matching_indices])
std_denNUC = np.std(np.array(all_blob_denNUC)[matching_indices])
# Print the results
print(f"Most Probable Eccentricity: {most_probable_eccentricity:.4f}")
print(f"Average Major Axis at peak Eccentricity(a): {avg_a:.4f} nm")
print(f"Average Intermediate Axis at peak Eccentricity (b): {avg_b:.4f} nm")
print(f"Average Minor Axis at peak Eccentricity (b): {avg_b:.4f} nm")
print(f"Average Surface Area at peak Eccentricity: {avg_area:.4f} μm²")
print(f"Average BloB Volumn at peak Eccentricity: {avg_vol:.4f} nm3")
print(f"Average denNUC at peak Eccentricity: {avg_denNUC:.4f} nuc per nm3")
print(f"Average denBPS at peak Eccentricity: {avg_denBPS:.4f} nuc per nm3")
print(f"Average ROG at peak Eccentricity: {average_rog:.4f} nm")
print(f"STD Major Axis at peak Eccentricity(a): {std_a:.4f} nm")
print(f"STD Intermediate Axis at peak Eccentricity (b): {std_b:.4f} nm")
print(f"STD Minor Axis at peak Eccentricity (b): {std_c:.4f} nm")
print(f"STD Surface Area at peak Eccentricity: {std_area:.4f} μm²")
print(f"STD BloB Volumn at peak Eccentricity: {std_vol:.4f} nm3")
print(f"STD denNUC at peak Eccentricity: {std_denNUC:.4f} nuc per nm3")
print(f"STD denBPS at peak Eccentricity: {std_denBPS:.4f} nuc per nm3")
print(f"STD ROG at peak Eccentricity: {std_rog:.4f} nm")

mode_result = mode(np.array(all_blob_majors)[matching_indices])
if isinstance(mode_result.mode, np.ndarray):
    most_probable_a = mode_result.mode[0]
else:
    most_probable_a = mode_result.mode  # Directly assign the scalar if it's a single value
mode_result = mode(np.array(all_blob_intermediates)[matching_indices])
if isinstance(mode_result.mode, np.ndarray):
    most_probable_b = mode_result.mode[0]
else:
    most_probable_b = mode_result.mode  # Directly assign the scalar if it's a single value
mode_result = mode(np.array(all_blob_surface_areas)[matching_indices] )
if isinstance(mode_result.mode, np.ndarray):
    most_probable_surface_area = mode_result.mode[0]
else:
    most_probable_surface_area = mode_result.mode  # Directly assign the scalar if it's a single value
mode_result = mode(np.array(all_blob_vol)[matching_indices] )
if isinstance(mode_result.mode, np.ndarray):
    most_probable_vol = mode_result.mode[0]
else:
    most_probable_vol = mode_result.mode  # Directly assign the scalar if it's a single value
mode_result = mode(np.array(all_blob_denNUC)[matching_indices] )
if isinstance(mode_result.mode, np.ndarray):
    most_probable_denNUC = mode_result.mode[0]
else:
    most_probable_denNUC = mode_result.mode  # Directly assign the scalar if it's a single value
mode_result = mode(np.array(all_blob_denBPS)[matching_indices] )
if isinstance(mode_result.mode, np.ndarray):
    most_probable_denBPS = mode_result.mode[0]
else:
    most_probable_denBPS = mode_result.mode  # Directly assign the scalar if it's a single value
mode_result = mode(np.array(all_radii_of_gyration)[matching_indices] )
if isinstance(mode_result.mode, np.ndarray):
    most_probable_rog = mode_result.mode[0]
else:
    most_probable_rog = mode_result.mode  # Directly assign the scalar if it's a single value
print(f"Most probable Major Axis at peak Eccentricity(a): {most_probable_a:.4f} nm")
print(f"Most probable Minor Axis at peak Eccentricity (b): {most_probable_b:.4f} nm")
print(f"Most probabel Surface Area at peak Eccentricity: {most_probable_surface_area:.4f} μm²")
print(f"Most probabel BloB Volume at peak Eccentricity: {most_probable_vol:.4f} nm3")
print(f"Most probabel denNUC at peak Eccentricity: {most_probable_denNUC:.4f} nuc per nm3")
print(f"Most probabel denBPS at peak Eccentricity: {most_probable_denBPS:.4f} bps per nm3")
print(f"Most probabel ROG at peak Eccentricity: {most_probable_rog:.4f} nm")
# Plot 3: Surface Area histogram
# Convert surface areas
surface_areas17 = np.array(surface_Area)
percentile = np.percentile(surface_areas17, 95)
bins_SA17 = np.linspace(0,0.014,30)
hist_values_SA17, bin_edges_SA17 = np.histogram(surface_areas17, bins=bins_SA17)
max_bin_index = np.argmax(hist_values_SA17)
most_probable_value = (bin_edges_SA17[max_bin_index] + bin_edges_SA17[max_bin_index + 1]) / 2
print("Most probable Surface Area in micrometer square :", most_probable_value)
bin_centers_SA17 = (bin_edges_SA17[:-1] + bin_edges_SA17[1:]) / 2  # Compute bin centers
#popt, pcov = fit_log_normal(bin_centers_SA17, hist_values_SA17)
#mu_fit, sigma_fit, a_fit = popt
#fitted_values = log_normal(bin_centers_SA17, mu_fit, sigma_fit, a_fit)
mean_value = np.mean(surface_areas17)
std_value = np.std(surface_areas17)
print(f"Mean of Surface Area distrbution: {mean_value:.4f} μm²\nStd of Surface Area distrbution: {std_value:.4f} μm²\n")
# Plot 3A: Volumn histogram
# Convert Volumn
vol17 = np.array(BLOB_vol)
percentile = np.percentile(vol17, 95)
print(min(vol17),max(vol17))
bins_V17 = np.linspace(0,100,30)
hist_values_V17, bin_edges_V17 = np.histogram(vol17, bins=bins_V17)
max_bin_index = np.argmax(hist_values_V17)
most_probable_value = (bin_edges_V17[max_bin_index] + bin_edges_V17[max_bin_index + 1]) / 2
print("Most probable BloB Volumn in nmcube :", most_probable_value)
bin_centers_V17 = (bin_edges_V17[:-1] + bin_edges_V17[1:]) / 2  # Compute bin centers
#popt, pcov = fit_log_normal(bin_centers_V17, hist_values_V17)
#mu_fit, sigma_fit, a_fit = popt
#fitted_values = log_normal(bin_centers_V17, mu_fit, sigma_fit, a_fit)
mean_value = np.mean(vol17)
std_value = np.std(vol17)
print(f"Mean of BloB Volumn distrbution: {mean_value:.4f} nm3\nStd of BloB Volumn distrbution: {std_value:.4f} nm3\n")
# Plot 3B: denNUC histogram
# Convert Volumn
denNUC17 = np.array(denNUC)
percentile = np.percentile(denNUC17, 95)
print(min(denNUC17),max(denNUC17))
bins_denNUC17 = np.linspace(0,0.5,30)
hist_values_denNUC17, bin_edges_denNUC17 = np.histogram(denNUC17, bins=bins_denNUC17)
max_bin_index = np.argmax(hist_values_denNUC17)
most_probable_value = (bin_edges_denNUC17[max_bin_index] + bin_edges_denNUC17[max_bin_index + 1]) / 2
print("Most probable denNUC in nuc per nmcube :", most_probable_value)
bin_centers_denNUC17 = (bin_edges_denNUC17[:-1] + bin_edges_denNUC17[1:]) / 2  # Compute bin centers
#popt, pcov = fit_log_normal(bin_centers_V17, hist_values_V17)
#mu_fit, sigma_fit, a_fit = popt
#fitted_values = log_normal(bin_centers_V17, mu_fit, sigma_fit, a_fit)
mean_value = np.mean(denNUC17)
std_value = np.std(denNUC17)
print(f"Mean of denNUC distrbution: {mean_value:.4f} nuc per nm3\nStd of denNUC distrbution: {std_value:.4f} nuc per nm3\n")
# Plot 3C: denBPS histogram
# Convert Volumn
denBPS17 = np.array(denBPS)
percentile = np.percentile(denBPS17, 95)
print(min(denBPS17),max(denBPS17))
bins_denBPS17 = np.linspace(0,50,30)
hist_values_denBPS17, bin_edges_denBPS17 = np.histogram(denBPS17, bins=bins_denBPS17)
max_bin_index = np.argmax(hist_values_denBPS17)
most_probable_value = (bin_edges_denBPS17[max_bin_index] + bin_edges_denBPS17[max_bin_index + 1]) / 2
print("Most probable denBPS in bps per nmcube :", most_probable_value)
bin_centers_denBPS17 = (bin_edges_denBPS17[:-1] + bin_edges_denBPS17[1:]) / 2  # Compute bin centers
#popt, pcov = fit_log_normal(bin_centers_V17, hist_values_V17)
#mu_fit, sigma_fit, a_fit = popt
#fitted_values = log_normal(bin_centers_V17, mu_fit, sigma_fit, a_fit)
mean_value = np.mean(denBPS17)
std_value = np.std(denBPS17)
print(f"Mean of denBPS distrbution: {mean_value:.4f} bps per nm3\nStd of denBPS distrbution: {std_value:.4f} bps per nm3\n")

# Plot 3D: AllBlobSize histogram
# Convert Volumn
all_blob_sizes17 = np.array(all_blob_sizes)
percentile = np.percentile(all_blob_sizes17, 95)
print(min(all_blob_sizes17),max(all_blob_sizes17))
bins_all_blob_sizes17 = np.linspace(0,1000,30)
hist_values_all_blob_sizes17, bin_edges_all_blob_sizes17 = np.histogram(all_blob_sizes17, bins=bins_all_blob_sizes17)
max_bin_index = np.argmax(hist_values_all_blob_sizes17)
most_probable_value = (bin_edges_all_blob_sizes17[max_bin_index] + bin_edges_all_blob_sizes17[max_bin_index + 1]) / 2
print("Most probable all_blob_sizes in bps per nmcube :", most_probable_value)
bin_centers_all_blob_sizes17 = (bin_edges_all_blob_sizes17[:-1] + bin_edges_all_blob_sizes17[1:]) / 2  # Compute bin centers
#popt, pcov = fit_log_normal(bin_centers_V17, hist_values_V17)
#mu_fit, sigma_fit, a_fit = popt
#fitted_values = log_normal(bin_centers_V17, mu_fit, sigma_fit, a_fit)
mean_value = np.mean(all_blob_sizes17)
std_value = np.std(all_blob_sizes17)
print(f"Mean of all_blob_sizes distrbution: {mean_value:.4f} bps per nm3\nStd of all_blob_sizes distrbution: {std_value:.4f} bps per nm3\n")

#############################################################################################################################

# Plot 4: ROG
rg_bins17 = np.linspace(6, 70, 30)
rg_hist17, rg_bin_edges = np.histogram(all_radii_of_gyration, bins=rg_bins17, density=True)
max_bin_index = np.argmax(rg_hist17)
most_probable_value = (rg_bin_edges[max_bin_index] + rg_bin_edges[max_bin_index + 1]) / 2
print("Most probable ROG in nm:", most_probable_value)
rg_centers17 = 0.5 * (rg_bin_edges[:-1] + rg_bin_edges[1:])
errors_rog17 = np.sqrt(rg_hist17 / len(all_radii_of_gyration))
#popt, pcov = fit_log_normal(rg_centers17, rg_hist17)
#mu_fit, sigma_fit, a_fit = popt
#fitted_values = log_normal(rg_centers17, mu_fit, sigma_fit, a_fit)
mean_value = np.mean(all_radii_of_gyration)
std_value = np.std(all_radii_of_gyration)
print(f"Mean of ROG distrbution: {mean_value:.4f} nm\nStd of ROG distrbution: {std_value:.4f} nm\n")

# Plot 5: END-to-END distances for each blob size
# Calculate the average and standard deviation of end-to-end distances for each blob size
blob_sizes17 = sorted(end_to_end_by_size.keys())
N_values = np.array(blob_sizes17[:])
avg_e2e_distances17 = [np.mean(end_to_end_by_size[size]) for size in blob_sizes17[:]]
std_e2e_distances = [np.std(end_to_end_by_size[size]) for size in blob_sizes17[:]]
blob_sizes_np = np.array(blob_sizes17[:])  # Convert to numpy array
avg_e2e_distances_np = np.array(avg_e2e_distances17)
#popt, pcov = fit_log_normal(blob_sizes17[:], avg_e2e_distances17)
#mu_fit, sigma_fit, a_fit = popt
#fitted_values_e2e17 = log_normal(smooth_blob_sizes17, mu_fit, sigma_fit, a_fit)
#print(f"Fitted Log-Normal Parameters: mu={mu_fit:.4f}, sigma={sigma_fit:.4f}, a={a_fit:.4f}")
#smooth_blob_sizes17 = np.linspace(min(blob_sizes17[:]), max(blob_sizes17[:]), 50)

# Plot 6: RDF
# Convert to NumPy array
all_blob_centers_rdf = np.array(all_blob_centers_rdf, dtype=np.float32)
print("Number of blob centers for rdf:", len(all_blob_centers_rdf))
distances = pdist(all_blob_centers_rdf)
dist_bins17 = np.linspace(0,1200, 30)
dist_hist17, bin_edges = np.histogram(distances, bins=dist_bins17, density=True)
max_bin_index = np.argmax(dist_hist17)
most_probable_value = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
print("Most probable RDF in nm:", most_probable_value)
mean_value = np.mean(distances)
print("Avg RDF in nm:", mean_value)
std_value = np.std(distances)
print("Std RDF in nm:", std_value)
dist_centers17 = 0.5 * (bin_edges[:-1] + bin_edges[1:])
errors_rdf17 = np.sqrt(dist_hist17 / len(distances))

#Plot 3 A
plt.figure(figsize=(10, 6))
plt.bar(bin_centers_V17, hist_values_V17, width=0.8*(bin_edges_V17[1]-bin_edges_V17[0]), facecolor='none', edgecolor='black', linewidth=2)# facecolor='darkblue', edgecolor='none')#, label="Histogram")
plt.bar(bin_centers_V12, hist_values_V12, width=0.2*(bin_edges_V12[1]-bin_edges_V12[0]), facecolor='darkblue', edgecolor='none')#, label="Histogram")
plt.xticks(fontsize=25)  # Adjust x-axis tick size
plt.yticks(fontsize=25)  # Adjust y-axis tick size
plt.xlabel("Blob Volume ($nm^3$)", fontsize=25, fontweight='bold')
plt.ylabel("Frequency", fontsize=25, fontweight='bold')
#plt.xlim(0,0.0004)
#plt.legend(loc="upper right", fontsize=25)
plt.tight_layout()  # Adjust bottom and top margin
output_path_e2e = os.path.join("Plots_blob_analysis", "Histogram_of_Volumn.pdf")
plt.savefig(output_path_e2e, format="pdf", dpi=600)
#plt.show()
#Plot 3 B
plt.figure(figsize=(10, 6))
plt.bar(bin_centers_denNUC17, hist_values_denNUC17, width=0.8*(bin_edges_denNUC17[1]-bin_edges_denNUC17[0]), facecolor='none', edgecolor='black', linewidth=2)# facecolor='darkblue', edgecolor='none')#, label="Histogram")
plt.bar(bin_centers_denNUC12, hist_values_denNUC12, width=0.2*(bin_edges_denNUC12[1]-bin_edges_denNUC12[0]), facecolor='darkblue', edgecolor='none')#, label="Histogram")
plt.xticks(fontsize=25)  # Adjust x-axis tick size
plt.yticks(fontsize=25)  # Adjust y-axis tick size
plt.xlabel("nucleosome density ($nm^-3$)", fontsize=25, fontweight='bold')
plt.ylabel("Frequency", fontsize=25, fontweight='bold')
#plt.xlim(0,0.0004)
#plt.legend(loc="upper right", fontsize=25)
plt.tight_layout()  # Adjust bottom and top margin
output_path_e2e = os.path.join("Plots_blob_analysis", "Histogram_of_denNUC.pdf")
plt.savefig(output_path_e2e, format="pdf", dpi=600)
#plt.show()
#Plot 3 C
plt.figure(figsize=(10, 6))
plt.bar(bin_centers_denBPS17, hist_values_denBPS17, width=0.8*(bin_edges_denBPS17[1]-bin_edges_denBPS17[0]), facecolor='none', edgecolor='black', linewidth=2)# facecolor='darkblue', edgecolor='none')#, label="Histogram")
plt.bar(bin_centers_denBPS12, hist_values_denBPS12, width=0.2*(bin_edges_denBPS12[1]-bin_edges_denBPS12[0]), facecolor='darkblue', edgecolor='none')#, label="Histogram")
plt.xticks(fontsize=25)  # Adjust x-axis tick size
plt.yticks(fontsize=25)  # Adjust y-axis tick size
plt.xlabel("bps density ($nm^-3$)", fontsize=25, fontweight='bold')
plt.ylabel("Frequency", fontsize=25, fontweight='bold')
#plt.xlim(0,0.0004)
#plt.legend(loc="upper right", fontsize=25)
plt.tight_layout()  # Adjust bottom and top margin
output_path_e2e = os.path.join("Plots_blob_analysis", "Histogram_of_denBPS.pdf")
plt.savefig(output_path_e2e, format="pdf", dpi=600)
#plt.show()
#Plot 3 D
plt.figure(figsize=(10, 6))
plt.bar(bin_centers_all_blob_sizes17, hist_values_all_blob_sizes17, width=0.8*(bin_edges_all_blob_sizes17[1]-bin_edges_all_blob_sizes17[0]), facecolor='none', edgecolor='black', linewidth=2)# facecolor='darkblue', edgecolor='none')#, label="Histogram")
plt.bar(bin_centers_all_blob_sizes12, hist_values_all_blob_sizes12, width=0.2*(bin_edges_all_blob_sizes12[1]-bin_edges_all_blob_sizes12[0]), facecolor='darkblue', edgecolor='none')#, label="Histogram")
plt.xticks(fontsize=25)  # Adjust x-axis tick size
plt.yticks(fontsize=25)  # Adjust y-axis tick size
plt.xlabel("bps density ($nm^-3$)", fontsize=25, fontweight='bold')
plt.ylabel("Frequency", fontsize=25, fontweight='bold')
#plt.xlim(0,0.0004)
#plt.legend(loc="upper right", fontsize=25)
plt.tight_layout()  # Adjust bottom and top margin
output_path_e2e = os.path.join("Plots_blob_analysis", "Histogram_of_all_blob_sizes.pdf")
plt.savefig(output_path_e2e, format="pdf", dpi=600)
#plt.show()


# Plot 1: Blob Size Distribution
plt.figure(figsize=(10, 6))
plt.bar(blob_centers17, blob_size_hist17, width=0.8*(blob_bin_edges17[1] - blob_bin_edges17[0]), facecolor='none', edgecolor='black', linewidth=2)#, label='Blob Size Distribution')
#plt.errorbar(blob_centers17, blob_size_hist17, yerr=errors_BS17, fmt='o', color='black', capsize=3)#, label='Error Bars')
plt.bar(blob_centers12, blob_size_hist12, width=0.2*(blob_bin_edges12[1] - blob_bin_edges12[0]), facecolor='darkblue', edgecolor='none')#, label='Blob Size Distribution')
#plt.errorbar(blob_centers12, blob_size_hist12, yerr=errors_BS12, fmt='o', color='gray', capsize=3)#, label='Error Bars')
plt.xticks(fontsize=25)  # Adjust x-axis tick size
plt.yticks(fontsize=25)  # Adjust y-axis tick size
plt.xlabel('Blob Size', fontsize=25, fontweight='bold')
plt.ylabel('PDF', fontsize=25, fontweight='bold')
#plt.legend(loc="upper right", fontsize=25)
plt.tight_layout()  # Adjust bottom and top margin
output_path = os.path.join("Plots_blob_analysis", "blob_size_distribution.pdf")
plt.savefig(output_path, format="pdf", dpi=600)
#plt.show()

# Plot 2: Eccentricity Distribution
# Plot the histogram with the additional information
plt.figure(figsize=(10,6))
plt.bar(eccentricity_centers17, eccentricity_hist17, width=0.8*(bin_edges_ED17[1]-bin_edges_ED17[0]), facecolor='none', edgecolor='black', linewidth=2)#, label='Eccentricity Distribution')
#plt.errorbar(eccentricity_centers17, eccentricity_hist17, yerr=np.sqrt(eccentricity_hist17 / len(all_blob_eccentricities17)), fmt='o', color='black', capsize=3)
plt.bar(eccentricity_centers12, eccentricity_hist12, width=0.2*(bin_edges_ED12[1]-bin_edges_ED12[0]), facecolor='darkblue', edgecolor='none')#, label='Eccentricity Distribution')
#plt.errorbar(eccentricity_centers12, eccentricity_hist12, yerr=np.sqrt(eccentricity_hist12 / len(all_blob_eccentricities12)), fmt='o', color='gray', capsize=3)
plt.xticks(fontsize=25)  # Adjust x-axis tick size
plt.yticks(fontsize=25)  # Adjust y-axis tick size
plt.xlabel('Eccentricity', fontsize=25, fontweight='bold')
plt.ylabel('PDF', fontsize=25, fontweight='bold')
plt.xlim(0, 1)
#plt.legend(loc="upper left", fontsize=25)
plt.tight_layout()  # Adjust bottom and top margin
# Save and show the plot
output_path = os.path.join("Plots_blob_analysis", "eccentricity_with_properties.pdf")
plt.savefig(output_path, format="pdf", dpi=600)
#plt.show()

# Plot 3: Surface Area histogram
# Plot histogram and fitted curve
'''
plt.figure(figsize=(10, 6))
plt.bar(bin_centers_SA17, hist_values_SA17, width=0.8*(bin_edges_SA17[1]-bin_edges_SA17[0]), facecolor='none', edgecolor='black', linewidth=2)#, label="Histogram")
plt.bar(bin_centers_SA12, hist_values_SA12, width=0.2*(bin_edges_SA17[1]-bin_edges_SA12[0]), facecolor='darkblue', edgecolor='none')#, label="Histogram")
plt.xticks(fontsize=25)  # Adjust x-axis tick size
plt.yticks(fontsize=25)  # Adjust y-axis tick size
plt.xlabel("Surface Area ($\mu m^2$)", fontsize=25, fontweight='bold')
plt.ylabel("Frequency", fontsize=25, fontweight='bold')
plt.xlim(0,0.015)   
#plt.legend(loc="upper right", fontsize=25)
plt.tight_layout()  # Adjust bottom and top margin
output_path_e2e = os.path.join("Plots_blob_analysis", "Histogram_of_surfaceArea.pdf")
plt.savefig(output_path_e2e, format="pdf", dpi=600)
#plt.show()
'''
# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot histogram bars
ax.bar(bin_centers_SA17, hist_values_SA17, width=0.8*(bin_edges_SA17[1]-bin_edges_SA17[0]), facecolor='none', edgecolor='black', linewidth=2)
ax.bar(bin_centers_SA12, hist_values_SA12, width=0.2*(bin_edges_SA17[1]-bin_edges_SA12[0]), facecolor='darkblue', edgecolor='none')

# Adjust tick sizes
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)

# Set x-axis ticks and labels
ax.set_xticks([0, 0.004, 0.008, 0.012])  # Set actual tick positions
ax.set_xticklabels([0, 4, 8, 12])  # Convert them to desired scale
ax.set_yticks([0, 4000, 8000, 12000,16000])  # Set actual tick positions
ax.set_yticklabels([0, 4, 8, 12,16])  # Convert them to desired scale

# Labels and title
ax.set_xlabel("Surface Area (x10$^{-3}$ $\mu m^2$)", fontsize=25, fontweight='normal')
ax.set_ylabel("Frequency (x10$^{3}$)", fontsize=25, fontweight='normal')

# Set x-axis limits
ax.set_xlim(0, 0.013)

# Optional: Add a legend
# ax.legend(loc="upper right", fontsize=25)

# Adjust layout
plt.tight_layout()

# Save the plot
output_path_e2e = os.path.join("Plots_blob_analysis", "Histogram_of_surfaceArea.pdf")
plt.savefig(output_path_e2e, format="pdf", dpi=600)

# Show the plot
#plt.show()


# Plot 4: ROG
plt.figure(figsize=(10, 6))
plt.bar(rg_centers17, rg_hist17, width=0.8*(rg_bins17[1] - rg_bins17[0]), facecolor='none', edgecolor='black', linewidth=2)#, label='Rg Distribution')
#plt.errorbar(rg_centers17, rg_hist17, yerr=errors_rog17, fmt='o', color='black', capsize=3)#, label='Error Bars')
plt.bar(rg_centers12, rg_hist12, width=0.2*(rg_bins12[1] - rg_bins12[0]), facecolor='darkblue', edgecolor='none')#, label='Rg Distribution')
#plt.errorbar(rg_centers12, rg_hist12, yerr=errors_rog12, fmt='o', color='gray', capsize=3)#, label='Error Bars')
plt.xticks(fontsize=25)  # Adjust x-axis tick size
plt.yticks(fontsize=25)  # Adjust y-axis tick size
plt.xlabel('Radius of Gyration (in nm)', fontsize=25, fontweight='bold')
plt.ylabel('PDF', fontsize=25, fontweight='bold')
#plt.legend(loc="upper right", fontsize=25)
plt.tight_layout()  # Adjust bottom and top margin
output_path = os.path.join("Plots_blob_analysis", "ROG.pdf")
plt.savefig(output_path, format="pdf", dpi=600)
#plt.show()

'''
# Plot 5: END-to-END distances for each blob size
plt.figure(figsize=(10, 6))
plt.scatter(blob_sizes17[:], avg_e2e_distances17, color="black", label='HOXB4')#label="Data", color="red")
plt.plot(smooth_blob_sizes17, fitted_values_e2e17, color="black")#label=f"Fit Log-Normal : mu={mu_fit:.4f}, sigma={sigma_fit:.4f}, a={a_fit:.4f}", color="blue")
plt.scatter(blob_sizes12[:], avg_e2e_distances12, color="darkblue", label='NANOG')#label="Data", color="red")
plt.plot(smooth_blob_sizes12, fitted_values_e2e12, color="gray")#label=f"Fit Log-Normal : mu={mu_fit:.4f}, sigma={sigma_fit:.4f}, a={a_fit:.4f}", color="blue")
plt.xticks(fontsize=25)  # Adjust x-axis tick size
plt.yticks(fontsize=25)  # Adjust y-axis tick size
plt.xlabel("Blob Size", fontsize=25, fontweight='bold')
plt.ylabel("End-to-End Distance\n(in nm)", fontsize=25, fontweight='bold')
plt.legend(loc="upper right", fontsize=25)
plt.tight_layout()  # Adjust bottom and top margin
output_path_e2e = os.path.join("Plots_blob_analysis", "blob_size_e2e_distance.pdf")
plt.savefig(output_path_e2e, format="pdf", dpi=600)
plt.show()
'''
# Plot 6: RDF
# Plot RDF with error bars
plt.figure(figsize=(10,6))
plt.bar(dist_centers17, dist_hist17, width=0.8*(dist_bins17[1] - dist_bins17[0]), facecolor='none', edgecolor='black', linewidth=2)#, label='RDF')
#plt.errorbar(dist_centers17, dist_hist17, yerr=errors_rdf17, fmt='o', color='black', capsize=3)#, label='Error Bars')
plt.bar(dist_centers12, dist_hist12, width=0.2*(dist_bins12[1] - dist_bins12[0]), facecolor='darkblue', edgecolor='none')#, label='RDF')
#plt.errorbar(dist_centers12, dist_hist12, yerr=errors_rdf12, fmt='o', color='gray', capsize=3)#, label='Error Bars')
plt.xticks(fontsize=25)  # Adjust x-axis tick size
plt.yticks(fontsize=25)  # Adjust y-axis tick size
plt.xlabel('Distance between Blob Centers (in nm)', fontsize=25, fontweight='bold')
plt.ylabel('RDF', fontsize=25, fontweight='bold')
plt.xlim(0,1250)
plt.ylim(0,0.0035)
#plt.legend(loc="upper right", fontsize=25)
plt.tight_layout()  # Adjust bottom and top margin
output_path = os.path.join("Plots_blob_analysis", "RDF_blobs.pdf")
plt.savefig(output_path, format="pdf", dpi=600)
#plt.show()
