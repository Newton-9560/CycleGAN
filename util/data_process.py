import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

def generate_txt(dataset = 'van'):
    files = []
    if dataset == 'van':
        root = './datasets/vangogh2photo/'
        for folder in os.listdir(root):
            if '.' not in folder:
                txt_file_path = root + folder + '.txt'
                with open(txt_file_path, 'w') as file:
                    for name in os.listdir(root + folder):
                        file.write(root + folder + '/' + name + '\n')

                print(f"Created file: {txt_file_path}, Number of lines: {len(os.listdir(root + folder))}")
                files.append(txt_file_path)
    elif dataset == 'z2h':
        root = './datasets/horse2zebra/'
        for folder in os.listdir(root):
            if '.' not in folder:
                txt_file_path = root + folder + '.txt'
                with open(txt_file_path, 'w') as file:
                    for name in os.listdir(root + folder):
                        file.write(root + folder + '/' + name + '\n')

                print(f"Created file: {txt_file_path}, Number of lines: {len(os.listdir(root + folder))}")
                files.append(txt_file_path)
    else:
        raise ValueError("Currently, dataset {} is not supported!".format(dataset))
    
    return files


def reverse_normalization(img, mean, std):
    img = img.clone()
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    img.mul_(std).add_(mean)
    return img


def show_generated(images_X, images_Y, generated_X, generated_Y, save = None):
    plt.figure(figsize=(15, 4))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)  

    plt.subplot(2, 2, 1)
    plt.axis("off")
    plt.title("Original Y Images")
    plt.imshow(np.transpose(vutils.make_grid(images_Y, padding=2, normalize=True).cpu(), (1, 2, 0)))

    plt.subplot(2, 2, 3)
    plt.axis("off")
    plt.title("Generated X Images")
    plt.imshow(np.transpose(vutils.make_grid(generated_X, padding=2, normalize=True).cpu(), (1, 2, 0)))

    plt.subplot(2, 2, 2)
    plt.axis("off")
    plt.title("Original X Images")
    plt.imshow(np.transpose(vutils.make_grid(images_X, padding=2, normalize=True).cpu(), (1, 2, 0)))

    plt.subplot(2, 2, 4)
    plt.axis("off")
    plt.title("Generated Y Images")
    plt.imshow(np.transpose(vutils.make_grid(generated_Y, padding=2, normalize=True).cpu(), (1, 2, 0)))

    if save is not None:
        plt.savefig(save, bbox_inches='tight')
    else:
        plt.show()