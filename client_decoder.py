import socket
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import io

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # Decoder layers
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)


    def forward(self, x_encoded):
        x = F.relu(self.t_conv1(x_encoded))
        x = F.relu(self.t_conv2(x))
        x = F.sigmoid(self.t_conv3(x))

        return x


decoder = Decoder()
decoder.load_state_dict(torch.load('Tests\Scripts\decoder_model.pth',map_location=torch.device('cpu')))
decoder.eval()
print(decoder)

def receive_image_client(server_ip, server_port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, server_port))

    start_time = time.time()

    image_data = b""
    while True:
        chunk = client_socket.recv(1024)
        if not chunk:
            break
        image_data += chunk
    end_time = time.time()
    transfer_time = end_time - start_time
    received_encoded_output = torch.load(io.BytesIO(image_data))
    

    print(f"Image received successfully in {transfer_time} seconds")
    client_socket.close()
    
    display_received_image(received_encoded_output)

def display_received_image(encoded_output):
    #result=torch.Tensor(numpy.frombuffer(encoded_output, dtype=numpy.int32))
    decoded_output = decoder(encoded_output)
    #image = Image.open(BytesIO(decoded_output))
    decoded_output_np = decoded_output.detach().numpy()
    img = np.transpose(decoded_output_np[0], (1, 2, 0))
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    plt.title('recieved image')
    plt.show()
   

if __name__ == "__main__":
    server_ip = '' # Keep the server IP here
    server_port = 55555 # Random port which is free at any time

    receive_image_client(server_ip, server_port)