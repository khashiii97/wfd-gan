import torch
import numpy as np
from model import DF
# Assuming DF is your model class and it's already defined
data = np.load('outputs/ds19/feature/raw_feature_0-100x0-100.npz')

# List all arrays in the file

features = data['features']#(10000, 500)
labels = data['labels'] #(10000,)

model = DF(length=500)  # Initialize your model here
checkpoint_path = 'f_model/df_ds19.ckpt'

# Load the state dict from the checkpoint into the model
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Try directly loading the checkpoint without accessing 'state_dict'
model.load_state_dict(checkpoint)
model.eval()  # Set the model to evaluation mode

outputs_by_class = {}  # Dictionary to store outputs, keyed by class

def extract_outputs_hook(module, input, output):
    return output

# Register the hook
hook = model.layer4.register_forward_hook(extract_outputs_hook)

# Assuming features and labels are already loaded
features_tensor = torch.tensor(features.reshape([10000, 1, -1]), dtype=torch.float)
labels_tensor = torch.tensor(labels, dtype=torch.long)

with torch.no_grad():  # Disable gradient computation
    for i in range(len(features_tensor)):
        feature = features_tensor[i].unsqueeze(0)  # Add batch dimension
        label = labels_tensor[i].item()
        output_layer_4 = model.get_layer4_output(feature)  # This will trigger the hook
        if label not in outputs_by_class:
            outputs_by_class[label] = []
        outputs_by_class[label].append(output_layer_4) # 1, 256, 1
        

# Remove the hook to clean up
hook.remove()

def calculate_entropy(outputs):
    # Convert outputs to a probability distribution (if not already)
    probs = torch.softmax(torch.tensor(outputs), dim=1)
    log_probs = torch.log2(probs)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy.mean().item()

# Calculate entropy for each class
# entropy_by_class = {}

# for label, outputs in outputs_by_class.items():
#     # Step 2: Prepare the Outputs Tensor
#     # This consolidates the list of tensors into a single tensor and removes unnecessary dimensions
#     outputs_tensor = torch.stack([output.squeeze() for output in outputs])
#     # Step 3: Calculate Entropy
#     # Now that you have a single tensor for all outputs of this class, calculate the entropy
#     entropy = calculate_entropy(outputs_tensor)
    
#     # Store the calculated entropy for this class
#     entropy_by_class[label] = entropy

# import matplotlib.pyplot as plt

# # Example entropy values for demonstration purposes

# # Sorting classes by entropy
# classes_sorted = sorted(entropy_by_class, key=entropy_by_class.get)

# # Sorting entropy values
# entropy_sorted = [entropy_by_class[cls] for cls in classes_sorted]

# # Plotting
# plt.figure(figsize=(20, 8))
# plt.bar(range(100), entropy_sorted, tick_label=classes_sorted)
# plt.xlabel('Class Label')
# plt.ylabel('Entropy')
# plt.title('Entropy of Layer4 Outputs by Class')
# plt.xticks(rotation=90)  # Rotate class labels for better readability
# plt.savefig('outputs/ds19/entropy/df.png')
# plt.show()

from sklearn.decomposition import PCA
import numpy as np
from scipy.stats import gaussian_kde
from scipy import integrate
import matplotlib.pyplot as plt

entropies = np.zeros(100)

for class_label in range(100):
    # Extract data for the current class
    outputs = outputs_by_class[class_label]
    class_data = torch.stack([output.squeeze() for output in outputs]).numpy()
    
    
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


plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), entropies, marker='o')
plt.xlabel('Class Label')
plt.ylabel('Entropy (bits)')
plt.title('Entropy of Each Class using df')
plt.grid(True)
#plt.show()
plt.savefig("outputs/ds19/entropy/df.png")
