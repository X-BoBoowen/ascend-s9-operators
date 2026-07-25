import random
import math
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def gcd_1_iterations(a_orig, b_orig):
    """
    Replicates the C++ gcd_1 function and returns the number of while loop iterations.
    """
    iterations = 0
    a = abs(a_orig)
    b = abs(b_orig)
    
    tmp_a = a
    a = max(a, b)
    b = min(tmp_a, b)
    
    while b != 0:
        iterations += 1
        if a > b:
            a -= b
        else:
            b -= a
        if a == 1 or b == 1:
            break
    return iterations

# --- Simulation Parameters ---
MIN_VAL = -10000
MAX_VAL = 10000
M = 10000 # abs max value
NUM_TRIALS = 500_000 # Number of random pairs to test. Increased for smoother CDF.

# --- Theoretical Calculation (for reference) ---
E_M_dominant_term = (6 / (math.pi**2)) * (math.log(M)**2)
factor_for_zeros = (4 * M**2) / ((2 * M + 1)**2)
theoretical_expected_iterations = factor_for_zeros * E_M_dominant_term

print(f"--- Theoretical Mean (M={M}) ---")
print(f"Overall theoretical expected iterations: {theoretical_expected_iterations:.4f}")

# --- Simulation for Distribution ---
print(f"\n--- Running Simulation for Distribution & CDF ---")
print(f"Number of trials: {NUM_TRIALS:,}")
print(f"Input range for a and b: [{MIN_VAL}, {MAX_VAL}]")

all_iteration_counts = []
start_time = time.time()

for i in range(NUM_TRIALS):
    a_rand = random.randint(MIN_VAL, MAX_VAL)
    b_rand = random.randint(MIN_VAL, MAX_VAL)
    
    iters = gcd_1_iterations(a_rand, b_rand)
    all_iteration_counts.append(iters)
        
    if (i + 1) % (NUM_TRIALS // 20) == 0 and NUM_TRIALS >=20:
        print(f"Progress: {(i + 1) * 100 / NUM_TRIALS:.0f}% completed...")

end_time = time.time()
simulation_duration = end_time - start_time
print(f"Simulation took: {simulation_duration:.2f} seconds")

empirical_average_iterations = np.mean(all_iteration_counts)
median_iterations = np.median(all_iteration_counts)
max_iterations_observed = np.max(all_iteration_counts)
min_iterations_observed = np.min(all_iteration_counts)

print("\n--- Simulation Statistics ---")
print(f"Empirical average iterations: {empirical_average_iterations:.4f}")
print(f"Median iterations: {median_iterations:.4f}")
print(f"Min iterations observed: {min_iterations_observed}")
print(f"Max iterations observed: {max_iterations_observed}")

# --- Calculate Empirical CDF ---
# Method 1: Sort and iterate (good for plotting steps)
sorted_counts = np.sort(all_iteration_counts)
cdf_x = []
cdf_y = []
# To make the step plot start from 0 probability before the first observation
if sorted_counts[0] > 0:
    cdf_x.append(sorted_counts[0]) # Start step at first observed value
    cdf_y.append(0)                # Probability before this value is 0

# Iterate through sorted unique counts to build the step function
unique_counts, counts_occurrence = np.unique(sorted_counts, return_counts=True)
cumulative_occurrences = np.cumsum(counts_occurrence)
cdf_x_unique = unique_counts
cdf_y_unique = cumulative_occurrences / NUM_TRIALS

# To make the step plot cleaner for plt.step:
# For each unique value, we need a point just before it (same y as previous) and at it (new y)
# And for the very first point, we need to ensure it starts at y=0 if min_val > 0
plot_cdf_x = [min_iterations_observed] # Start at the first observed value
plot_cdf_y = [0.0]                     # Probability is 0 just before the first value is reached

# Use unique counts and their cumulative probabilities
for i in range(len(cdf_x_unique)):
    # Point before the step up (if not the first point)
    if i > 0 and cdf_x_unique[i-1] != cdf_x_unique[i]: # Ensure x value changes
         plot_cdf_x.append(cdf_x_unique[i])
         plot_cdf_y.append(cdf_y_unique[i-1]) # Keep previous y

    # Point at the step up
    plot_cdf_x.append(cdf_x_unique[i])
    plot_cdf_y.append(cdf_y_unique[i])


# Ensure the CDF reaches 1 and stays there
if plot_cdf_x[-1] < max_iterations_observed +1 : # Add a point to extend the line horizontally if needed
    plot_cdf_x.append(max_iterations_observed + 1) # Or some value slightly larger than max
    plot_cdf_y.append(1.0)


# --- Plotting ---
plt.figure(figsize=(18, 7)) # Increased width for 3 plots

# Plot 1: Detailed view of the lower iteration counts (Histogram)
plt.subplot(1, 3, 1)
max_x_display_hist = 300
bins_detail = np.arange(0, max_x_display_hist + 2) - 0.5
counts_hist, bin_edges_hist, _ = plt.hist(all_iteration_counts, bins=bins_detail, density=False, alpha=0.75, label='Frequency')
plt.yscale('log')
plt.title(f'Distribution of Loop Iterations (Detail)\n(N={NUM_TRIALS:,})')
plt.xlabel('Number of Iterations')
plt.ylabel('Frequency (Log Scale)')
plt.axvline(empirical_average_iterations, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {empirical_average_iterations:.2f}')
plt.axvline(median_iterations, color='g', linestyle='dashed', linewidth=1, label=f'Median: {median_iterations:.2f}')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.xlim(-0.5, max_x_display_hist + 0.5)

# Plot 2: Empirical Cumulative Distribution Function (CDF)
plt.subplot(1, 3, 2)
# Use plt.plot with drawstyle for a step function, or plt.step
# plt.step(plot_cdf_x, plot_cdf_y, where='post', label='Empirical CDF')
# Simpler approach for plotting empirical CDF:
# Sort the data. y-values are (1/N, 2/N, ..., 1)
y_ecdf = np.arange(1, NUM_TRIALS + 1) / NUM_TRIALS
plt.plot(sorted_counts, y_ecdf, marker='.', linestyle='none', ms=0.1, alpha=0.5) # Scatter of all points
# For a step plot of the unique values:
unique_sorted_counts = np.unique(sorted_counts)
# Add a point at the beginning if min_iterations_observed > 0 to show P(X < min_iterations_observed) = 0
plot_x_cdf = np.concatenate(([unique_sorted_counts[0]], unique_sorted_counts)) if unique_sorted_counts[0]>0 else unique_sorted_counts
# The y values for the step function. `searchsorted` finds where each unique count would be inserted
# into the sorted full list, giving the cumulative count up to that point.
plot_y_cdf_step = np.searchsorted(sorted_counts, plot_x_cdf, side='right') / NUM_TRIALS
if unique_sorted_counts[0] > 0: # Prepend the (value,0) point for the step if data doesn't start at 0
    plot_y_cdf_step = np.concatenate(([0], plot_y_cdf_step))

# Using the previously calculated unique cdf values for step plot
plt.step(cdf_x_unique, cdf_y_unique, where='post', label='Empirical CDF (Step)')

# For better visualization of the main part of CDF:
max_x_display_cdf = 300 # Cap x-axis for the CDF plot for better readability of the steep rise
plt.xlim(-5, max_x_display_cdf) # Start x from -5 to see the step at 0 clearly

plt.title(f'Empirical CDF of Loop Iterations\n(N={NUM_TRIALS:,})')
plt.xlabel('Number of Iterations (x)')
plt.ylabel('P(Iterations <= x)')
plt.axvline(median_iterations, color='g', linestyle='dashed', linewidth=1, label=f'Median: {median_iterations:.2f} (P <= Median ≈ 0.5)')
plt.axhline(0.5, color='gray', linestyle='dotted', linewidth=1, label='P = 0.5')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)


# Plot 3: Overview of the full range (Histogram)
plt.subplot(1, 3, 3)
num_bins_overview = 100
plt.hist(all_iteration_counts, bins=num_bins_overview, density=False, alpha=0.75, color='skyblue')
plt.yscale('log')
plt.title(f'Distribution (Full Overview)\n(Max obs: {max_iterations_observed})')
plt.xlabel('Number of Iterations')
plt.ylabel('Frequency (Log Scale)')
plt.axvline(empirical_average_iterations, color='r', linestyle='dashed', linewidth=1)
plt.axvline(median_iterations, color='g', linestyle='dashed', linewidth=1)
plt.grid(True, which="both", ls="-", alpha=0.5)


plt.tight_layout()
plt.savefig("gcd_iterations_distribution_and_cdf.png")
print("\nPlot saved as gcd_iterations_distribution_and_cdf.png")
plt.show()