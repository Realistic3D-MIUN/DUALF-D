# VAE-based Light Field Image Compression (DUALF-D)

This repository contains the implementation of a VAE-based Light Field Image Compression system. It provides scripts for training, encoding, decoding, and evaluating the performance of the compression model.

## Requirements

To run the code, you need to install the following dependencies:

*   **Python 3.x**
*   **PyTorch** (`torch`, `torchvision`)
*   **CompressAI** (`compressai`)
*   **NumPy** (`numpy`)
*   **Pillow** (`PIL`)
*   **Scikit-Image** (`skimage`)

### Installation

You can install the required packages using pip:

```bash
pip install torch torchvision compressai numpy pillow scikit-image
```

## Checkpoints

We provide 4 pre-trained checkpoints corresponding to different bit rate ranges (controlled by the lambda parameter during training).

The checkpoints are located in the `checkpoint/` directory:

1.  **Low Bitrate**: `vae_model_epoch_147_0001.pth` (Lambda = 0.0001)
2.  **Medium-Low Bitrate**: `vae_model_epoch_149_001.pth` (Lambda = 0.001)
3.  **Medium-High Bitrate**: `vae_model_epoch_149_01.pth` (Lambda = 0.01)
4.  **High Bitrate**: `vae_model_epoch_146_005.pth` (Lambda = 0.005)

## Usage

### Encoding and Decoding

The main script for encoding and decoding images is `predict_full_compressai.py`. This script loads a pre-trained model, compresses images from an input directory, reconstructs them, and calculates performance metrics (PSNR, SSIM, BPP).

**How to run:**

1.  Open `predict_full_compressai.py`.
2.  **Select Checkpoint**: Modify line 201 to point to the desired checkpoint file.
    ```python
    # Example for Medium-High bitrate
    state_dict = torch.load(save_path + "vae_model_epoch_149_01.pth", map_location=device)
    ```
3.  **Set Input/Output**: Update `base_input_folder` (line 220) and `base_output_folder` (line 221) to your data directories.
4.  Run the script:
    ```bash
    python predict_full_compressai.py
    ```

The script will generate compressed/reconstructed images in the output folder and save a results summary file (`compression_results_with_quantization.txt`).

### Training

To train the model from scratch or resume training:

1.  Configure training parameters in `parameters.py` (e.g., `epochs`, `batch_size`, `lr`, `lambda_factor`).
2.  Run the training script:
    ```bash
    python train.py
    ```

### Utility Scripts

*   **`MacPI_To_SAI_Single.py`**: Converts a Macropixel image into Sub-Aperture Images (SAI). You need to specify the `macropixel_image_path` and `output_directory` within the script before running.
*   **`measure_inference_time.py`**: Measures the inference speed of the model.
*   **`PSNR_New.py`**: A standalone script to calculate PSNR metrics between images.
*   **`printSummary.py`**: Prints a summary of the model architecture.

## Directory Structure

*   `checkpoint/`: Stores model checkpoints.
*   `dataset/`: Directory for input datasets.
*   `output/` / `output_test/`: Directories for generated results.
*   `model.py`, `encoder.py`, `decoder.py`: Define the VAE network architecture.
*   `loss.py`: Defines the loss functions (Rate-Distortion Loss).
*   `dataloader.py`: Handles data loading.
*   `parameters.py`: Contains global configuration parameters.
