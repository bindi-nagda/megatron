import os
import random
import yaml
from torch.utils.data import Sampler
import random
from collections import defaultdict

def load_datasets_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['datasets']

def create_episode_configs(
    root_dir,
    yaml_config_dir,
    label='train',
    support_size=5,
    query_size=5,
    min_episodes=10,
    max_episodes=50,
    with_replacement=False, # False means don't reuse images 
    seed=None
):
    """
    Create the full list of possible episodes per task.

    Returns an episode config list for the given split
    """
    episodes = []

    if seed is not None:
        random.seed(seed)
    
    config_path = os.path.join(yaml_config_dir, 'datasets_config.yaml')
    datasets = load_datasets_config(config_path)
    allowed_tasks = set(os.path.splitext(ds['filename'])[0] for ds in datasets)
    
    for task in allowed_tasks:
        with_replacement = task in ['Breast-Ultrasound', 'Advanced-MRI-Breast-Lesion']

        task_path = os.path.join(root_dir, task)
        train_img_dir = os.path.join(task_path, 'images/train')
        val_img_dir = os.path.join(task_path, 'images/val')

        if not os.path.exists(train_img_dir):
            print(f"[SKIP] Task '{task}' does not exist in ProcessedData")
            continue

        train_imgs = [f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.png'))]
        total_train_imgs = len(train_imgs)

        val_imgs = [f for f in os.listdir(val_img_dir) if f.endswith(('.jpg', '.png'))]
        total_val_imgs = len(val_imgs)

        if label == 'train' and total_train_imgs < support_size + query_size: 
            print(f"[SKIP] Task '{task}' has too few train images ({total_train_imgs})")
            continue
        elif label == 'val' and total_val_imgs < support_size + query_size:
            print(f"[SKIP] Task '{task}' has too few val images ({total_val_imgs}). ss = {support_size}. qs = {query_size}")
            continue

        random.shuffle(train_imgs)
        random.shuffle(val_imgs)

        def compute_episode_count(pool):
            return min(max(len(pool) // (support_size + query_size), min_episodes), max_episodes)

        def get_label_path(img_filename, task_path, split):
            label_dir = os.path.join(task_path, 'labels', split)
            base_name = os.path.splitext(img_filename)[0] + '.txt'
            return os.path.join(label_dir, base_name)

        def debug_label_coverage(img_pool, task_path, split):
            labeled = 0
            for img_filename in img_pool:
                label_path = get_label_path(img_filename, task_path, split)
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        lines = [line.strip() for line in f if line.strip()]
                        if lines:
                            labeled += 1
            print(f"{labeled} labeled out of {len(img_pool)} total images")

        def sample_episodes(img_pool, target_list, label, episode_count, with_replacement=False):
            """
            Generate episode_count episodes using only labeled images.
            Each episode contains support_size and query_size labeled images.
            """
            # debug_label_coverage(img_pool, task_path, label)

            # Filter to labeled images only
            #  if label=='train':
            labeled_pool = [
                img for img in img_pool
                if os.path.exists(get_label_path(img, task_path, label)) and os.path.getsize(get_label_path(img, task_path, label)) > 0
            ]
            # else:
            #     labeled_pool = img_pool

            usable_per_episode = support_size + query_size
            episode_start = len(target_list)

            if with_replacement:
                for _ in range(episode_count):
                    if len(labeled_pool) < usable_per_episode:
                        break
                    episode_imgs = random.sample(labeled_pool, usable_per_episode)
                    support = episode_imgs[:support_size]
                    query = episode_imgs[support_size:]
                    target_list.append((task, support, query))
            else:
                pool = labeled_pool[:]
                random.shuffle(pool)
                for i in range(0, len(pool) - usable_per_episode + 1, usable_per_episode):
                    episode_imgs = pool[i:i + usable_per_episode]
                    support = episode_imgs[:support_size]
                    query = episode_imgs[support_size:]
                    target_list.append((task, support, query))
                    if len(target_list) - episode_start >= episode_count:
                        break

            episode_end = len(target_list)
            # print(f"[{label.upper()}] {task}: {episode_end - episode_start} episodes from {len(labeled_pool)} labeled images (replacement={with_replacement})")

        if label == 'train':
            sample_episodes(train_imgs, episodes, 'train', compute_episode_count(train_imgs), with_replacement=with_replacement)
        else:
            sample_episodes(val_imgs, episodes, 'val', compute_episode_count(val_imgs), with_replacement=with_replacement) 

    #print(f"\nTotal: {len(train_episodes)} train episodes, {len(val_episodes)} val episodes\n")
    random.shuffle(episodes)
    return episodes

class BalancedEpisodeSampler(Sampler):
    def __init__(self, episodes, episodes_per_task, seed=None):
        """
        Sample a fixed number of episodes per task per epoch, 
        so all tasks are equally represented, regardless of 
        how many episodes they have.
        
        Args:
            episodes: List of (task_name, support, query) tuples.
            episodes_per_task: No. of episodes to sample per task per epoch.
            seed: Optional random seed for reproducibility.
        """
        self.episodes = episodes
        self.episodes_per_task = episodes_per_task
        self.task_to_indices = defaultdict(list)

        for idx, (task, _, _) in enumerate(episodes):
            self.task_to_indices[task].append(idx)

        self.tasks = list(self.task_to_indices.keys())
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)

        selected_indices = []
        for task in self.tasks:
            indices = self.task_to_indices[task]
            if len(indices) < self.episodes_per_task:
                sampled = rng.choices(indices, k=self.episodes_per_task)
            else:
                sampled = rng.sample(indices, k=self.episodes_per_task)
            selected_indices.extend(sampled)

        rng.shuffle(selected_indices)
        return iter(selected_indices)

    def __len__(self):
        return len(self.tasks) * self.episodes_per_task
