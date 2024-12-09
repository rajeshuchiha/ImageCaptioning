import torch
import torchvision.transforms as transforms
from model import CNNtoRNN
from utils import print_examples
from get_loader import get_loader
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose(
    [
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_loader, dataset = get_loader(
    root_folder="flickr8k/images",
    annotation_file="flickr8k/captions.txt",
    transform=transform,
    num_workers=2,
)
    
embed_size = 256
hidden_size = 256
vocab_size = len(dataset.vocab)
num_layers = 1



model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
checkpoint = torch.load("weights\epoch50.pth.tar", weights_only=False)
model.load_state_dict(checkpoint["state_dict"])


print_examples(model, device, dataset)

# # Uncomment the below code for specific single image.

# transform = transforms.Compose(
#     [
#         transforms.Resize((299, 299)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ]
# )

# model.eval()
# # Set the file name to desired test image
# test_img = transform(Image.open("test_examples/dog.jpg").convert("RGB")).unsqueeze(
#     0
# )

# print(
#     "OUTPUT: "
#     + " ".join(model.caption_image(test_img.to(device), dataset.vocab))
# )