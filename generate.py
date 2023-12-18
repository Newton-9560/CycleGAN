import torch
from torchvision import transforms
from PIL import Image
import os
import torchvision.utils as vutils
from tqdm import tqdm 
import argparse


# python generate.py --root './experiments_van/2/' --model_name '62_G_Y2X.pth' --image_dirc 'photo_mine/'
# python generate.py --root './experiments_van/2/' --model_name '62_G_Y2X.pth' --image_dirc 'photo_mine/'
# python generate.py --root './experiments_z2h/1/' --model_name 'best_G_x2Y.pth' --image_dirc 'horse2zebra/trainA/'

def replace_and_remove_chars(input_str):
    """
    Replace '/' with '_' and remove '.' in the given string.
    """
    modified_str = input_str.replace('/', '_').replace('.', '')
    return modified_str

parser = argparse.ArgumentParser(description="Process images using a trained model.")
parser.add_argument('--root', type=str, required=True, help='Root directory path of the experiment settings and model.')
parser.add_argument('--model_name', type=str, required=True, help='Name of the model file.')
parser.add_argument('--image_dirc', type=str, required=True, help='Directory of images to process.')

args = parser.parse_args()

print("Generate image from " + args.image_dirc)

root = args.root
model_name = args.model_name
image_dirc = args.image_dirc

model_path = os.path.join(root, 'models', model_name)
generator = torch.load(model_path).to(torch.device('cuda'))
generator.eval() 

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

image_directory = os.path.join('./datasets', image_dirc)
output_directory = os.path.join(root, 'result', replace_and_remove_chars(image_dirc) + replace_and_remove_chars(model_name))

print("Save results to " + output_directory)

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for filename in tqdm(os.listdir(image_directory), desc="processing"):
    if filename.endswith('.jpg') or filename.endswith('.JPG'):  
        image_path = os.path.join(image_directory, filename)
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(torch.device('cuda'))

        with torch.no_grad():
            output = generator(image)

        output_image = vutils.make_grid(output, padding=2, normalize=True).cpu()
        output_image = transforms.ToPILImage()(output_image)

        output_path = os.path.join(output_directory, filename)
        output_image.save(output_path)
