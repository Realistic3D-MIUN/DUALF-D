import torch

filt_n = 128
filt_n_D = 128
latent_channels = 128
param_height, param_width = 216, 312
epochs = 10000
batch_size = 2
lr = 1e-4
aux_lr = 1e-2
filt_n_final = filt_n

# Set validation_split to 0 to use all data for training
validation_split = 0.0  # Changed from 0.1

save_frequency = 1
#entropy_update_frequency = 50
best_model_path = "./checkpoint/vae_model_best.pth"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
train_dataset_path = './dataset/collected_118_right_9_3x3_macropixels/'
test_dataset_path = './dataset/collected_118_right_9_3x3_macropixels/'
save_path = "./checkpoint/"
#save_path_1 = './checkpoint_64S_64A_10data_NEW/'

resume_training = False
resume_location = save_path+"vae_model_epoch_000.pth"

# Loss parameters
lambda_factor = 0.001  

# Add these parameters
clip_max_norm = 1.0
init_scale_factor = 10
scale_table_levels = 64
update_entropy_frequency = 1  # Update every epoch
