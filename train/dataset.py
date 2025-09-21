import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class MetaEpisodeDataset(Dataset):
    """
    Load the image/label data for a single episode
    """
    def __init__(self, episodes, root_dir, label='train', transform=None):
        self.episodes = episodes
        self.root_dir = root_dir
        self.label = label
        self.transform = T.Compose([
            T.Resize((512, 512)),  # Resize to fixed height, width
            T.ToTensor()           # Convert to [1, H, W] and scale to [0.0, 1.0]
        ])

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        task, support_imgs, query_imgs = self.episodes[idx]
        task_path = os.path.join(self.root_dir, task)

        def make_paths(img_list):
            imgs = []
            labels = []
            for img_name in img_list:
                img_path = os.path.join(task_path, f'images/{self.label}', img_name)
                label_path = os.path.join(task_path, f'labels/{self.label}', img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
                # img = Image.open(img_path).convert("L") # Grayscale 
                # img = self.transform(img)  # Should return [1, H, W] tensor
                # with open(label_path, 'r') as f:
                #     label = f.read()

                imgs.append(img_path)
                labels.append(label_path)
            return imgs, labels

        support_imgs, support_labels = make_paths(support_imgs)
        query_imgs, query_labels = make_paths(query_imgs)
        
        return {
            'task': task,
            'support_images': support_imgs, # list of paths
            'support_labels': support_labels,
            'query_images': query_imgs,
            'query_labels': query_labels
        }
