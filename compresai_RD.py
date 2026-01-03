import os
import torch
import numpy as np
from PIL import Image
import compressai

import sys
sys.path = [p for p in sys.path if 'compressai' not in p]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Print version information
print("CompressAI version:", getattr(compressai, "__version__", "Unknown"))
print("Available modules in compressai:", dir(compressai))

def compute_psnr(a, b):
    """Compute PSNR between tensors a and b."""
    # Convert to float32 to avoid precision issues
    a = a.float()
    b = b.float()
    
    # Calculate MSE
    mse = torch.mean((a - b) ** 2)
    if mse == 0:
        return float('inf')
    
    # For images normalized to [0,1], max_val is 1.0
    max_val = 1.0
    # PSNR formula: PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse.item())
    return psnr

def calculate_bpp(out_enc):
    """Calculate bits per pixel from the compressed representation."""
    total_bits = 0
    
    # Count bits from strings
    for s in out_enc["strings"]:
        if isinstance(s, (list, tuple)):
            total_bits += sum(len(x) * 8 for x in s)
        else:
            total_bits += len(s) * 8
    
    # Add bits from hyperprior if available
    if "hyperprior" in out_enc:
        for s in out_enc["hyperprior"]["strings"]:
            if isinstance(s, (list, tuple)):
                total_bits += sum(len(x) * 8 for x in s)
            else:
                total_bits += len(s) * 8
    
    return total_bits

def load_image(filepath):
    """Load and preprocess image."""
    img = Image.open(filepath).convert('RGB')
    # Convert to float32 and normalize to [0,1]
    img = np.array(img).astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img

# Settings
input_folder = './data_original_81/Pillars_renamed_f001'
quality_levels = [1, 2, 3, 4, 5, 6, 7, 8]
output_root = './data_original_81/compressed_outputs_Pillars'

os.makedirs(output_root, exist_ok=True)
image_filenames = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])

def save_log(logfile, results):
    with open(logfile, 'w') as f:
        f.write("Image,Quality,BPP,PSNR(dB)\n")
        for r in results:
            f.write(f"{r['image']},{r['quality']},{r['bpp']:.4f},{r['psnr']:.2f}\n")

all_results = []
summary_results = []

# First try to examine what's available in compressai
print("\nExamining CompressAI structure...")
try:
    print("compressai.models:", dir(compressai.models))
except:
    print("No compressai.models module")

# Check if 'zoo' module exists
try:
    print("compressai.zoo:", dir(compressai.zoo))
except:
    print("No compressai.zoo module")

# Function to get a model (simplified)
def get_model(quality):
    # Try the most common ways of loading models in CompressAI
    
    # 1. Try using models.bmshj2018_factorized if available
    if hasattr(compressai, 'models'):
        print("Trying models.bmshj2018_factorized...")
        try:
            from compressai.models import bmshj2018_factorized
            model = bmshj2018_factorized(quality=quality, pretrained=True)
            return model.eval().to(device)
        except Exception as e:
            print(f"Failed with models.bmshj2018_factorized: {e}")
    
    # 2. Try zoo.bmshj2018_factorized if available
    if hasattr(compressai, 'zoo'):
        print("Trying zoo.bmshj2018_factorized...")
        try:
            from compressai.zoo import bmshj2018_factorized
            model = bmshj2018_factorized(quality=quality, pretrained=True)
            return model.eval().to(device)
        except Exception as e:
            print(f"Failed with zoo.bmshj2018_factorized: {e}")
    
    # If we get here, no model could be loaded
    raise ValueError(f"Could not load any model for quality level {quality}")

# Try to load and compress with models
for q in quality_levels:
    print(f"\nProcessing quality level: {q}")
    try:
        model = get_model(q)
        print(f"Successfully loaded model for quality {q}")
    except Exception as e:
        print(f"Failed to load model for quality {q}: {e}")
        continue

    quality_folder = os.path.join(output_root, f'q{q}')
    os.makedirs(quality_folder, exist_ok=True)

    bpp_list = []
    psnr_list = []

    for fname in image_filenames:
        image_path = os.path.join(input_folder, fname)
        img = load_image(image_path).to(device)

        with torch.no_grad():
            try:
                out_enc = model.compress(img)
                out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
                rec = out_dec["x_hat"].clamp(0, 1)

                torch.save(out_enc, os.path.join(quality_folder, f"{fname.replace('.png', '')}_compressed.pt"))

                psnr = compute_psnr(img, rec)
                total_bits = calculate_bpp(out_enc)
                H, W = img.shape[2], img.shape[3]
                bpp = total_bits / (H * W)

                print(f" - {fname}: PSNR={psnr:.2f} dB, BPP={bpp:.4f}")

                bpp_list.append(bpp)
                psnr_list.append(psnr)

                all_results.append({
                    'image': fname,
                    'quality': q,
                    'bpp': bpp,
                    'psnr': psnr,
                })
            except Exception as proc_error:
                print(f"Error processing {fname}: {proc_error}")

    if bpp_list:
        avg_bpp = np.mean(bpp_list)
        avg_psnr = np.mean(psnr_list)
        summary_results.append({'quality': q, 'avg_bpp': avg_bpp, 'avg_psnr': avg_psnr})
        print(f"\n▶️ Quality {q}: Avg BPP = {avg_bpp:.4f}, Avg PSNR = {avg_psnr:.2f} dB\n")

# Save logs
if all_results:
    save_log(os.path.join(output_root, 'compression_results.csv'), all_results)

    summary_path = os.path.join(output_root, 'compression_summary.csv')
    with open(summary_path, 'w') as f:
        f.write("Quality,Avg_BPP,Avg_PSNR(dB)\n")
        for s in summary_results:
            f.write(f"{s['quality']},{s['avg_bpp']:.4f},{s['avg_psnr']:.2f}\n")

    print(f"✅ All done. Check:\n- Per-image: {output_root}/compression_results.csv\n- Summary:   {output_root}/compression_summary.csv")
else:
    print("⚠️ No results were generated. Check the CompressAI installation and model availability.")
