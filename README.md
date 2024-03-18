# AutoEncoded_Image_Transfer
Compressing images using Autoencoders and transferring them over the network

## There are 3 directories

### 1. Image_Transfer_Scripts
#### - Contains the scripts necessary to run the image transfer. By default it uses the 1M parameters architecture. 
#### - The server/client encoder/decoder files facilitate autoencoded transfer, while the base server/client scripts can be used for normal image transfer (for comparison purposes).
#### - There is also a ssim_checker script to check reconstruction quality. Computes SSIM along with a handful of other metrics.

### 2. AutoEncoder_Weights
#### - There are 3 subdirectories
####   |-> Working : Contains the most stable and usable autoencoder weights (Currently for the 1M architecture).
####   |-> Experimental : Contains unstable autoencoder weights (Currently for the 9M architecture). At the moment has highly variable results.
####   |-> Legacy : Contains older versions of the autoencoder weights (Unsupported versions, 1M with less training cycles).

### 3. Training_Notebooks
#### - Contains the notebooks used for training.
