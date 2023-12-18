import torch
from models.cycle_gan import CycleGAN
from util import data_process
from torchvision import transforms
from dataset.image_dataset import my_dataloader
import matplotlib.pyplot as plt
import argparse
import os
import json
import csv


# python train.py --datatype 'z2h' --train_mode 'ori_gan' --lr_g 0.0001 --lr_d 0.0001 --lamnda_X2Y 10 --lamnda_Y2X 10 --epochs 50 --lr_decay_iters 50
# python train.py --lr_g 0.0001 --lr_d 0.0001 --lamnda_X2Y 10 --lamnda_Y2X 10 --epoch 2 --lr_decay_iters 50
# python train.py --lr_g 0.0001 --lr_d 0.0001 --lamnda_X2Y 10 --lamnda_Y2X 10 --epochs 100 --lr_decay_iters 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

parser = argparse.ArgumentParser(description="Setup parameters for CycleGAN.")
parser.add_argument('--datatype', type=str, default='van', help='The dayatype: van denotes vangogh2photo and z2h denotes zebra to horse')
parser.add_argument('--train_mode', type=str, default='wgan')
parser.add_argument('--g_input_nc', type=int, default=3, help='number of input channels for generator')
parser.add_argument('--g_output_nc', type=int, default=3, help='number of output channels for generator')
parser.add_argument('--d_input_nc', type=int, default=3, help='number of input channels for discriminator')
parser.add_argument('--lr_g', type=float, default=0.0002, help='learning rate for generator')
parser.add_argument('--lr_d', type=float, default=0.0001, help='learning rate for discriminator')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device to train on')
parser.add_argument('--lamnda_X2Y', type=int, default=10, help='lambda for X to Y cycle consistency')
parser.add_argument('--lamnda_Y2X', type=int, default=10, help='lambda for Y to X cycle consistency')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=1, help='number of batch_size')
parser.add_argument('--lr_decay_iters', type=int, default=0, help='epoch at which to start lr decay')
parser.add_argument('--lambda_idt', type=float, default=0.5)

opt = parser.parse_args()

file_paths = data_process.generate_txt(opt.datatype)


base_folder = './experiments_' + opt.datatype

if not os.path.exists(base_folder):
    os.makedirs(base_folder)

existing_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
existing_numbers = [int(folder) for folder in existing_folders if folder.isdigit()]
next_folder_number = max(existing_numbers) + 1 if existing_numbers else 1

folder_name = os.path.join(base_folder, str(next_folder_number))
os.makedirs(folder_name)

with open(os.path.join(folder_name, 'options.json'), 'w') as json_file:
    json.dump(vars(opt), json_file, indent=4)
    

os.makedirs(folder_name + '/' + 'models')
os.makedirs(folder_name + '/' + 'images_result')
print(f"Folder '{folder_name}' created.")

transformations = transforms.Compose([
    transforms.Resize([286,286]), 
    transforms.RandomCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

for file_path in file_paths:
    if 'test' in file_path:
        if 'A' in file_path:
            test_X_dataloader = my_dataloader(file_path, transformations, batch_size = 5, shuffle = False)
        else:
            test_Y_dataloader = my_dataloader(file_path, transformations, batch_size = 5, shuffle = False)
    else:
        if 'A' in file_path:
            train_X_dataloader = my_dataloader(file_path, transformations, batch_size = opt.batch_size)
        else:
            train_Y_dataloader = my_dataloader(file_path, transformations, batch_size = opt.batch_size)


model = CycleGAN(opt)

losses_g = []
losses_d_x = []
losses_d_y = []


test_Y_iter = iter(test_Y_dataloader)
images_Y = next(test_Y_iter).to(opt.device)


test_X_iter = iter(test_X_dataloader)
images_X = next(test_X_iter).to(opt.device)
best_loss_g = 100

for epoch in range(opt.epochs):
    
    best_epoch_loss_g = 100

    for i, data_Y in enumerate(train_Y_dataloader):
        data_X =  next(iter(train_X_dataloader)).to(device)
        loss = model.train(data_X, data_Y.to(device))
        
        if loss[0] < best_loss_g:
            best_loss_g = loss[0]
            model.save_model('Y', folder_name)
            model.save_model('X', folder_name)
            
        if loss[0] < best_epoch_loss_g:
            best_epoch_loss_g = loss[0]
            model.save_model('Y', folder_name, str(epoch))
            model.save_model('X', folder_name, str(epoch))
            
        
        if i%1200 == 0:
            print(f"Epoch [{epoch+1}/{opt.epochs}], Step [{i+1}/{len(train_Y_dataloader)}], "
                  f"Loss: Generator: {loss[0]:.2f}, "
                  f"Discriminator X: {loss[1]:.2f}, Discriminator Y: {loss[2]:.2f}")
            losses_g.append(loss[0])
            losses_d_x.append(loss[1])
            losses_d_y.append(loss[2])

            
    if opt.lr_decay_iters != 0:
        model.update_lr()
        
    generated_X = model.generate_X(images_Y)
    generated_Y = model.generate_Y(images_X)
    data_process.show_generated(images_X, images_Y, generated_X, generated_Y, save = folder_name + '/' + 'images_result' + '/epoch' + str(epoch) + '.png')
    

        




# write CSV
with open(os.path.join(folder_name, 'losses.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Iteration', 'Generator Loss', 'Discriminator X Loss', 'Discriminator Y Loss'])
    for i, (lg, ldx, ldy) in enumerate(zip(losses_g, losses_d_x, losses_d_y)):
        writer.writerow([i, lg, ldx, ldy])

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# First subplot for generator losses
ax1.plot(losses_g, label='Generator Loss')
ax1.set_title('Generator Losses Over Iterations')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Loss')
ax1.legend()

# Second subplot for discriminator losses
ax2.plot(losses_d_x, label='Discriminator X Loss')
ax2.plot(losses_d_y, label='Discriminator Y Loss')
ax2.set_title('Discriminator Losses Over Iterations')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
# plt.show()
plt.savefig(folder_name + '/loss.png', bbox_inches='tight')


