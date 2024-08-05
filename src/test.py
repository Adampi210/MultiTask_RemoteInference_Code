import torch
from segment_anything import sam_model_registry
from PIL import Image
import numpy as np

def get_sample_image():
    img = Image.new('RGB', (1024, 1024), color='white')
    return np.array(img)

def find_optimal_split_point(model, input_image):
    activation_sizes = []
    
    def hook_fn(module, input, output):
        activation_sizes.append((str(module), output.numel()))
    
    hooks = []
    for name, module in model.named_modules():
        if not list(module.children()):  # leaf module
            hooks.append(module.register_forward_hook(hook_fn))
    
    device = next(model.parameters()).device
    pixel_values = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0).float().to(device)
    dummy_point_coords = torch.tensor([[[0, 0]]], dtype=torch.float).to(device)
    dummy_point_labels = torch.tensor([[1]], dtype=torch.float).to(device)
    
    with torch.no_grad():
        image_embeddings = model.image_encoder(pixel_values)
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=(dummy_point_coords, dummy_point_labels),
            boxes=None,
            masks=None,
        )
        _ = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
    
    for hook in hooks:
        hook.remove()
    
    return activation_sizes

# Load the VIT-L SAM model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry["vit_l"](checkpoint="../../../models/sam_vit_l.pth")
sam.to(device)

# Get a sample image
sample_image = get_sample_image()

# Get ordered activation sizes
activation_sizes = find_optimal_split_point(sam, sample_image)

print("ViT-L layers in order of activation:")
for i, (module, size) in enumerate(activation_sizes, 1):
    print(f"{i}. {module}: {size}")

# Find potential split points
def find_split_candidates(sizes, threshold=0.5):
    candidates = []
    for i, (module, size) in enumerate(sizes):
        if i > 0:
            prev_size = sizes[i-1][1]
            if size < prev_size * threshold:
                candidates.append((i, module, size))
    return candidates

split_candidates = find_split_candidates(activation_sizes)

print("\nPotential layer to split:")
for index, module, size in split_candidates:
    print(f"After layer {index}: {module} (size: {size})")

# Find the layer with the smallest activations
smallest_layer, smallest_size = min(activation_sizes, key=lambda x: x[1])
print(f"\nLayer with smallest activations: {smallest_layer}")
print(f"Size of smallest activations: {smallest_size}")