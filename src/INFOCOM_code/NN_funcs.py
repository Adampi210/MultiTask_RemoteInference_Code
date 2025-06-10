import torch
from segment_anything import sam_model_registry
from PIL import Image
import numpy as np

def get_sample_image():
    # You can replace this with your own image loading logic
    img = Image.new('RGB', (1024, 1024), color='white')
    return np.array(img)

def register_hooks(model):
    activation_sizes = {}
    
    def hook_fn(module, input, output):
        activation_sizes[module] = output.numel()
    
    for name, module in model.named_modules():
        if not list(module.children()):  # leaf module
            module.register_forward_hook(hook_fn)
    
    return activation_sizes

def find_smallest_activation_layer(model, input_image):
    activation_sizes = register_hooks(model)
    
    # Prepare input
    pixel_values = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0).float()
    
    # Forward pass
    with torch.no_grad():
        _ = model(pixel_values)
    
    # Find the layer with the smallest activations
    smallest_layer = min(activation_sizes, key=activation_sizes.get)
    smallest_size = activation_sizes[smallest_layer]
    
    return smallest_layer, smallest_size, activation_sizes

# Load the VIT-L SAM model
sam = sam_model_registry["vit_l"](checkpoint="path/to/sam_vit_l.pth")
sam.to(device="cuda" if torch.cuda.is_available() else "cpu")

# Get a sample image
sample_image = get_sample_image()

# Find the layer with the smallest activations
smallest_layer, smallest_size, all_sizes = find_smallest_activation_layer(sam, sample_image)

print(f"Layer with smallest activations: {smallest_layer}")
print(f"Size of smallest activations: {smallest_size}")

# Print all layers sorted by activation size
sorted_layers = sorted(all_sizes.items(), key=lambda x: x[1])
print("\nAll layers sorted by activation size:")
for layer, size in sorted_layers:
    print(f"{layer}: {size}")