# AutoEncoded_Image_Transfer
Compressing images using Autoencoders and transferring them over the network. Currently supports colour image transfer at 256x256 image resolution. The encoder and decoder have been separated after training to deploy on different systems.

### Results: 
- Compression ratio of 12:1
- Image transfer times 10 times faster

![image](https://github.com/05kashyap/AutoEncoded_Image_Transfer/assets/120780494/20b596dd-2682-44e3-b09a-0ba1b7eb16d3)
#### IMG Source : https://medium.com/@birla.deepak26/autoencoders-76bb49ae6a8f

### Training Dataset: ```Stanford Dogs Dataset, Animals-10```

## Directories

### 1. Image_Transfer_Scripts
> Contains 2 sub directories ```With_AE``` which houses the python scripts containing the model along image transfer across network. Also contains directory ```Without_AE``` housing python scripts without containing the model but only image transfer across network.

### 2. AutoEncoder_Weights
#### - There are 3 subdirectories
> - Working : Contains the most stable and usable autoencoder weights (Currently for the 1M architecture).
> - Experimental : Contains unstable autoencoder weights (Currently for the 9M architecture). At the moment has highly variable results.
> - Legacy : Contains older versions of the autoencoder weights (Unsupported versions, 1M with less training cycles).

### 3. Train-Test_Notebooks
> Results notebook for different batch sizes : ```Batch_Tests.ipynb```
> The rest are training notebooks
