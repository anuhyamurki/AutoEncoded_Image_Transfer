# AutoEncoded_Image_Transfer
Compressing images using Autoencoders and transferring them over the network. Currently supports image transfer at 256x256 image resolution.

![image](https://github.com/05kashyap/AutoEncoded_Image_Transfer/assets/120780494/20b596dd-2682-44e3-b09a-0ba1b7eb16d3)
Source : https://medium.com/@birla.deepak26/autoencoders-76bb49ae6a8f

### Dataset: Stanford Dogs Dataset, Animals-10

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
