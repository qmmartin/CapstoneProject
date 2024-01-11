from diffusers import AutoencoderKL
import torch
import torchvision
from torchvision.datasets.utils import download_and_extract_archive
from torchvision import transforms


num_workers = 4
batch_size = 12
# From https://github.com/fastai/imagenette
IMAGENETTE_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz'

torch.manual_seed(0)
torch.set_grad_enabled(False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float32).to("cuda")

size = 512
image_transform = transforms.Compose([
    transforms.Resize(size),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

root = 'dataset'
# file_name = ('C:\Users\PC\Desktop\VAE Data Set')
download_and_extract_archive(IMAGENETTE_URL, root, None, 'C:\Users\PC\Desktop\VAE Data Set')

dataset = torchvision.datasets.ImageFolder(root, transform=image_transform)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
)

all_latents = []
for image_data, _ in loader:
    image_data = image_data.to(device)
    latents = vae.encode(image_data).latent_dist.sample()
    all_latents.append(latents.cpu())

all_latents_tensor = torch.cat(all_latents)
std = all_latents_tensor.std().item()
normalizer = 1 / std
print(f'{normalizer = }')