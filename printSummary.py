import torch
from model import *
from parameters import *
from collections import OrderedDict
import numpy as np

def print_model_summary(model, input_size):
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def get_layer_info(module, prefix=''):
        layer_info = []
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if list(child.children()):  # If the module has children
                layer_info.extend(get_layer_info(child, full_name))
            else:  # Leaf module
                layer_info.append({
                    'name': full_name,
                    'type': child.__class__.__name__,
                    'params': sum(p.numel() for p in child.parameters()),
                    'trainable': any(p.requires_grad for p in child.parameters())
                })
        return layer_info

    # Get layer information
    layers = get_layer_info(model)
    
    # Print header
    print("\n" + "=" * 80)
    print("Model Summary")
    print("=" * 80)
    print(f"{'Layer (type)':<40} {'Output Shape':<25} {'Param #':<10}")
    print("=" * 80)
    
    # Print each layer
    total_params = 0
    trainable_params = 0
    
    for layer in layers:
        # Print layer information
        print(f"{layer['name']} ({layer['type']}):")
        print(f"  Parameters: {layer['params']:,}")
        print(f"  Trainable: {layer['trainable']}")
        print("-" * 40)
        
        total_params += layer['params']
        if layer['trainable']:
            trainable_params += layer['params']
    
    # Print summary
    print("=" * 80)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print("=" * 80)

# Create the model
model = VAE(latent_channels=latent_channels).to(device)

# Print the summary
print_model_summary(model, (3, param_height, param_width))

# Print parameter names if needed
# for n, p in model.named_parameters():
#     print(n)




