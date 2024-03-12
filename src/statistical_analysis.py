import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance, entropy

def compute_histogram_metrics(hist1, hist2, bin_edges, metrics):
    #metrics=['Bhattacharyya Distance', 'EMD', 'Cosine Distance', 'KL Divergence', 'Chi-Square Distance']
    # Normalize histograms to convert them into probability distributions
    hist1_normalized = hist1 / np.sum(hist1)
    hist2_normalized = hist2 / np.sum(hist2)

    results = {}

    if 'Bhattacharyya Distance' in metrics:
        # Calculate Bhattacharyya distance
        bc = np.sum(np.sqrt(hist1_normalized * hist2_normalized))
        results['Bhattacharyya Distance'] = -np.log(bc)

    if 'EMD' in metrics:
        # Calculate Earth Mover's Distance (EMD)
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        results['EMD'] = wasserstein_distance(bin_midpoints, bin_midpoints, hist1_normalized, hist2_normalized)

    if 'Cosine Distance' in metrics:
        # Calculate Cosine Distance
        results['Cosine Distance'] = cosine(hist1_normalized, hist2_normalized)
    
    if 'KL Divergence' in metrics:
        # Calculate KL Divergence
        # Adding a small value to avoid division by zero or log of zero
        results['KL Divergence'] = entropy(hist1_normalized + 1e-10, hist2_normalized + 1e-10)
    
    if 'Chi-Square Distance' in metrics:
        # Calculate Chi-Square Distance manually
        # Adding a small value to avoid division by zero
        results['Chi-Square Distance'] = np.sum(((hist1_normalized - hist2_normalized) ** 2) / (hist2_normalized + 1e-10))

    return results
# Load the .npz file
data = np.load('outputs/ds19/feature/raw_feature_0-100x0-100.npz')

# List all arrays in the file
print("Arrays in the npz file:", list(data.keys()))
features = data['features']#(10000, 1401)
labels = data['labels'] #(10000,)



def compare_bursts_across_classes(burst_number, class_numbers, show_plots=True):  
    # Determine the global min and max across all classes for the specified burst
    global_min = np.min(features[:, burst_number])
    global_max = np.max(features[:, burst_number])
    
    # Define bin edges based on the global range, ensuring consistent bins across classes
    bins = np.linspace(global_min, global_max, 31)  # 30 bins means 31 edges
    
    histograms = {}  # To store histogram data for each class
    
    for class_number in class_numbers:
        # Filter the features for the current class
        class_features = features[labels == class_number]
        
        # Extract the specified burst element from each vector in the current class
        burst_elements = class_features[:, burst_number]
        
        # Calculate histogram for the current class using the consistent bin edges
        counts, _ = np.histogram(burst_elements, bins=bins)
        histograms[class_number] = (counts, bins)
        
        if show_plots:
            # Plot a histogram for the current class
            plt.hist(burst_elements, bins=bins, alpha=0.5, label=f'Class {class_number}', edgecolor='black')

    if show_plots:
        # Add plot title, labels, and legend
        plt.title(f'Comparison of Histograms for Burst {burst_number} Across Different Classes')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return histograms, bins

def compare_classes_across_bursts(class1, class2, metric_nums):
    
    num_bursts = features.shape[1]
    bhattacharyya_distances = []
    emd_distances = []
    cosine_distances = []
    class_numbers = [class1, class2]
    metrics_dict = {}
    
    for burst_number in range(1,num_bursts):
        histograms, bins =  compare_bursts_across_classes(burst_number, class_numbers, show_plots=False)
        hist1 = histograms[class1][0]
        hist2 = histograms[class2][0]
        metrics=['Bhattacharyya Distance', 'EMD', 'Cosine Distance', 'KL Divergence', 'Chi-Square Distance']
        metrics = compute_histogram_metrics(hist1, hist2, bins, metrics= [metrics[i] for i in metric_nums])
        for key in metrics.keys():
            if key not in metrics_dict.keys():
                metrics_dict[key] = []
            metrics_dict[key].append(metrics[key])
            
    
    # Plotting
    plt.figure(figsize=(14, 6))
    for metric_name, values in metrics_dict.items():
        plt.plot(values, label=metric_name)
    
    plt.xlabel('Burst Number')
    plt.ylabel('Distance')
    plt.title(f'Comparison of Histogram Distances Between Class {class1} and Class {class2}')
    plt.legend()
    plt.show()
burst_number = 400
class_numbers = [10, 100]

compare_classes_across_bursts(10, 99, metric_nums=[1])