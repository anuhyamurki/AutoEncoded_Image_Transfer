import socket 

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('localhost',1234))

file = open('client.jpeg','rb')

image_data = file.read(2048)

while image_data:
    client.send(image_data)
    image_data = file.read(2048)

file.close()
client.close()
