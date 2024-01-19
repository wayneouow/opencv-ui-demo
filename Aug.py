from PIL import Image
import os
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.transforms import functional as TF

folder_path = 'Q5_image\Q5_1'
file_list = os.listdir(folder_path)[:9]

rows = 3
columns = 3
fig, axs = plt.subplots(rows, columns, figsize=(8, 8))

for i in range(rows):
    for j in range(columns):
        image_path = os.path.join(folder_path, file_list[i * columns + j])

        img = Image.open(image_path)
        size = 100
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.6),
            transforms.RandomVerticalFlip(p=0.6),
            transforms.RandomRotation(30)
        ])
        new_img = transform(img)
        
        axs[i, j].imshow(new_img)
# 顯示圖片
plt.show()