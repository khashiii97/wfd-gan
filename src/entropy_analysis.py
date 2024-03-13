from sklearn.decomposition import PCA
import numpy as np
from scipy.stats import gaussian_kde
from scipy import integrate
import matplotlib.pyplot as plt

# Simulate the features array for demonstration (since I don't have the actual data)
# In practice, you would use your actual 'features' array
# Load the .npz file
data = np.load('outputs/ds19/feature/raw_feature_0-100x0-100.npz')

# List all arrays in the file

features = data['features']#(10000, 500)
labels = data['labels'] #(10000,)


# Assuming 'features' is your (10000, 1401) dataset
# Initialize an array to store the entropy for each class
entropies = np.zeros(100)

for class_label in range(100):
    # Extract data for the current class
    class_data = features[class_label*100:(class_label+1)*100, :]
    
    # Apply PCA to reduce dimensionality to 2 for KDE
    pca = PCA(n_components=2)
    class_reduced = pca.fit_transform(class_data)
    
    # Estimate the density with KDE
    kde = gaussian_kde(class_reduced.T)  # KDE requires transposed input for multi-dimensional data
    
    # Generate a grid over which we'll calculate the KDE
    grid_size = 100  # Number of points in each dimension
    xmin, ymin = class_reduced.min(axis=0)
    xmax, ymax = class_reduced.max(axis=0)
    xgrid, ygrid = np.mgrid[xmin:xmax:complex(grid_size), ymin:ymax:complex(grid_size)]
    
    # Calculate the KDE over the grid
    pdf = kde(np.vstack([xgrid.ravel(), ygrid.ravel()]))
    
    # Normalize the PDF
    pdf_normalized = pdf / pdf.sum()
    
    # Compute Shannon Entropy
    entropy = -np.sum(pdf_normalized * np.log2(pdf_normalized, where=(pdf_normalized > 0)))
    
    # Store the entropy
    entropies[class_label] = entropy

# Plot the entropies for all classes
plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), entropies, marker='o')
plt.xlabel('Class Label')
plt.ylabel('Entropy (bits)')
plt.title('Entropy of Each Class')
plt.grid(True)
plt.show()
plt.savefig("outputs/ds19/entropy/pca2.png")
