# AutoEncoded_Image_Transfer
Compressing images using Autoencoders and transferring them over the network. Currently supports colour image transfer at 256x256 image resolution. The encoder and decoder have been separated after training to deploy on different systems.

### Results:
- Compression ratio of 12:1
- Image transfer times around 10 times faster depending on network congestion.

![image](https://github.com/05kashyap/AutoEncoded_Image_Transfer/assets/120780494/20b596dd-2682-44e3-b09a-0ba1b7eb16d3)
#### IMG Source : https://medium.com/@birla.deepak26/autoencoders-76bb49ae6a8f

### Training Dataset: ```Stanford Dogs Dataset, Animals-10```

## Directories

### Scripts
> Contains 2 sub directories ```With AutoEncoder and Decoder``` which houses the python scripts containing the model along image transfer across network. Also contains directory ```Without AutoEncoder and Decoder``` housing python scripts without containing the model but only image transfer across network.

### Notebooks
> Contains the main notebooks for the AutoEncoder and Decoder files.

### Images
> Contains test images to transfer across the network.

### Tests
> Contains the serialised pytorch state dictionaries basically models which can be used for the simulation.
