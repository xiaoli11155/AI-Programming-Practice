import os
import numpy as np
import torch
from torchvision import datasets
from PIL import Image

def color_grayscale_arr(arr, forground_color, background_color):
    """Converts grayscale image"""
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])
    if background_color == "black":
        if forground_color == "red":
            arr = np.concatenate([arr,
                              np.zeros((h, w, 2), dtype=dtype)], axis=2)
        elif forground_color == "green":
            arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                              arr,
                              np.zeros((h, w, 1), dtype=dtype)], axis=2)
        elif forground_color == "white":
            arr = np.concatenate([arr, arr, arr], axis=2)
    else:
        if forground_color == "yellow":
            arr = np.concatenate([arr, arr, np.zeros((h, w, 1), dtype=dtype)], axis=2)
        else:
            arr = np.concatenate([np.zeros((h, w, 2), dtype=dtype), arr], axis=2)

        c = [255, 255, 255]
        arr[:, :, 0] = (255 - arr[:, :, 0]) / 255 * c[0]
        arr[:, :, 1] = (255 - arr[:, :, 1]) / 255 * c[1]
        arr[:, :, 2] = (255 - arr[:, :, 2]) / 255 * c[2]

    return arr


class ColoredMNIST(datasets.VisionDataset):

    def __init__(self, root='./data', env='train1', transform=None, target_transform=None):
        super(ColoredMNIST, self).__init__(root, transform=transform,
                                           target_transform=target_transform)

        self.prepare_colored_mnist()
        if env in ['train1', 'train2', 'train3', 'test1', 'test2']:
            self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt')
        elif env == 'all_train':
            self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', 'train1.pt')) + \
                                     torch.load(os.path.join(self.root, 'ColoredMNIST', 'train2.pt')) + \
                                     torch.load(os.path.join(self.root, 'ColoredMNIST', 'train3.pt'))
        else:
            raise RuntimeError(f'{env} env unknown. Valid envs are train1, train2, train3, test1, test2, and all_train')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data_label_tuples[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data_label_tuples)

    def prepare_colored_mnist(self):
        colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
        if os.path.exists(os.path.join(colored_mnist_dir, 'train1.pt')) \
                and os.path.exists(os.path.join(colored_mnist_dir, 'train2.pt')) \
                and os.path.exists(os.path.join(colored_mnist_dir, 'train3.pt')) \
                and os.path.exists(os.path.join(colored_mnist_dir, 'test1.pt')) \
                and os.path.exists(os.path.join(colored_mnist_dir, 'test2.pt')):
            print('Colored MNIST dataset already exists')
            return

        print('Preparing Colored MNIST')
        train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

        train1_set = []
        train2_set = []
        train3_set = []
        test1_set, test2_set = [], []
        for idx, (im, label) in enumerate(train_mnist):
            if idx % 10000 == 0:
                print(f'Converting image {idx}/{len(train_mnist)}')
            im_array = np.array(im)
            
            # Assign a binary label y to the image based on the digit
            binary_label = 0 if label < 5 else 1

            # Color the image according to its environment label
            if idx < 10000:
                colored_arr = color_grayscale_arr(im_array, forground_color="red", background_color="black")
                train1_set.append((Image.fromarray(colored_arr), binary_label))
            elif idx < 20000:
                colored_arr = color_grayscale_arr(im_array, forground_color="green", background_color="black")
                train2_set.append((Image.fromarray(colored_arr), binary_label))
            elif idx < 30000:
                colored_arr = color_grayscale_arr(im_array, forground_color="white", background_color="black")
                train3_set.append((Image.fromarray(colored_arr), binary_label))
            elif idx < 45000:
                colored_arr = color_grayscale_arr(im_array, forground_color="yellow", background_color="white")
                test1_set.append((Image.fromarray(colored_arr), binary_label))
            else:
                colored_arr = color_grayscale_arr(im_array, forground_color="blue", background_color="white")
                test2_set.append((Image.fromarray(colored_arr), binary_label))

        if not os.path.exists(colored_mnist_dir):
            os.makedirs(colored_mnist_dir)
        torch.save(train1_set, os.path.join(colored_mnist_dir, 'train1.pt'))
        torch.save(train2_set, os.path.join(colored_mnist_dir, 'train2.pt'))
        torch.save(train3_set, os.path.join(colored_mnist_dir, 'train3.pt'))
        torch.save(test1_set, os.path.join(colored_mnist_dir, 'test1.pt'))
        torch.save(test2_set, os.path.join(colored_mnist_dir, 'test2.pt'))


# Example usage to load the dataset:
if __name__ == '__main__':
    # 1. 定义数据转换
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 2. 加载不同环境的数据集
    print("正在加载数据集...")
    train1_dataset = ColoredMNIST(env='train1', transform=transform)
    train2_dataset = ColoredMNIST(env='train2', transform=transform)
    train3_dataset = ColoredMNIST(env='train3', transform=transform)
    test1_dataset = ColoredMNIST(env='test1', transform=transform)
    test2_dataset = ColoredMNIST(env='test2', transform=transform)
    all_train_dataset = ColoredMNIST(env='all_train', transform=transform)
    
    # 3. 创建DataLoader
    batch_size = 64
    train1_loader = DataLoader(train1_dataset, batch_size=batch_size, shuffle=True)
    train2_loader = DataLoader(train2_dataset, batch_size=batch_size, shuffle=True)
    train3_loader = DataLoader(train3_dataset, batch_size=batch_size, shuffle=True)
    test1_loader = DataLoader(test1_dataset, batch_size=batch_size, shuffle=False)
    test2_loader = DataLoader(test2_dataset, batch_size=batch_size, shuffle=False)
    all_train_loader = DataLoader(all_train_dataset, batch_size=batch_size, shuffle=True)
    
    # 4. 验证数据集加载是否正确
    def check_dataset(loader, name):
        data_iter = iter(loader)
        images, labels = next(data_iter)
        print(f"\n{name}数据集检查:")
        print(f"图像张量形状: {images.shape}")  # 应为 [batch, 3, H, W]
        print(f"标签形状: {labels.shape}")
        print(f"像素值范围: [{images.min().item():.3f}, {images.max().item():.3f}]")
        print(f"标签示例: {labels[:10].tolist()}")
    
    # 检查各个数据集
    check_dataset(train1_loader, "Train1 (红/黑)")
    check_dataset(train2_loader, "Train2 (绿/黑)")
    check_dataset(test1_loader, "Test1 (黄/白)")
    
    # 5. 可视化示例图像（可选）
    import matplotlib.pyplot as plt
    
    def show_images(loader, title):
        data_iter = iter(loader)
        images, labels = next(data_iter)
        fig = plt.figure(figsize=(10, 4))
        for i in range(5):
            ax = fig.add_subplot(1, 5, i+1)
            # 反归一化: img = img * 0.5 + 0.5
            img = images[i].numpy().transpose((1, 2, 0))
            img = np.clip(img * 0.5 + 0.5, 0, 1)  # 将图像从[-1,1]映射到[0,1]
            ax.imshow(img)
            ax.set_title(f"Label: {labels[i].item()}")
            ax.axis('off')
        plt.suptitle(title)
        plt.show()
    
    # 显示不同环境的示例图像
    show_images(train1_loader, "Train1 示例 (红/黑)")
    show_images(test1_loader, "Test1 示例 (黄/白)")
    
    print("\n数据集加载和验证完成！")