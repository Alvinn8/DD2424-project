import asyncio
import json
import websockets
import torchvision
from torchvision import models
import os
import torch

from breed import populate_class_id_to_breed, class_id_to_breed_name

# Set of connected clients
connected_clients = set()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model_path = "models/resnet18_breed_20260520_103350_acc8757.pth"
model = models.resnet18(weights=None).to(device)
model.fc = torch.nn.Linear(model.fc.in_features, 37).to(device)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

async def handle_message(client, message):
    message_bytes = message if isinstance(message, bytes) else message.encode("utf-8")
    encoded_image = torch.frombuffer(memoryview(message_bytes), dtype=torch.uint8)
    image = torchvision.io.decode_image(encoded_image, mode=torchvision.io.ImageReadMode.RGB)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(224),
        #torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std),
    ])
    image = transform(image)

    output = model(image.unsqueeze(0).to(device))
    _, predicted = torch.max(output.data, 1)
    predicted = predicted.item()
    confidence = torch.softmax(output, dim=1)[0][predicted].item()
    breed_name = class_id_to_breed_name.get(predicted, "Unknown")

    await client.send(json.dumps({
        "breed_name": breed_name,
        "confidence": f"{confidence*100:.2f}% confidence",
    }))

    print("Sent prediction to client:", breed_name, confidence)

async def handle_client(websocket):
    print("New client")
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            await handle_message(websocket, message)
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.remove(websocket)

async def main():
    print("Starting...")
    server = await websockets.serve(handle_client, '0.0.0.0', 12345)
    await server.wait_closed()

# Run the server
if __name__ == "__main__":
    populate_class_id_to_breed()
    asyncio.run(main())