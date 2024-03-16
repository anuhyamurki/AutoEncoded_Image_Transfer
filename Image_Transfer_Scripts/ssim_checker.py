from skimage.metrics import structural_similarity as ssim_sk
import matplotlib.pyplot as plt
import numpy as np
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp

def compute_ssim(image1,image2):
    if image1.shape[2] == 4:
        # Convert RGBA image to RGB
        image1 = image1[:, :, :3]

    if image2.shape[2] == 4:
        # Convert RGBA image to RGB
        image2 = image2[:, :, :3]
    ssim_index = ssim_sk(image1, image2, data_range=1.0,multichannel=True, channel_axis=2)
    image1 = (image1 * 255).astype(np.uint8)
    image2 = (image2 * 255).astype(np.uint8)
    print("MSE: ", mse(image1,image2))
    print("RMSE: ", rmse(image1, image2))
    #print("PSNR: ", psnr(image1, image2))
    #print("SSIM: ", ssim(image1, image2))
    print("UQI: ", uqi(image1, image2))
    print("MSSSIM: ", msssim(image1, image2))
    #print("ERGAS: ", ergas(image1, image2))
    #print("SCC: ", scc(image1, image2))
    #print("RASE: ", rase(image1, image2))
    #print("SAM: ", sam(image1, image2))
    print("VIF: ", vifp(image1, image2))
    return ssim_index
def main():
    
    resized = plt.imread('resized_image.png')
    recieved = plt.imread('received_image.png')
    # print(resized.shape)
    # print(recieved.shape)
    return compute_ssim(resized, recieved)

if __name__ == '__main__':
    try:
        print(f"SSIM: {main()}")
    except KeyboardInterrupt:
        print('Interrupted')