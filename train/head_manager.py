import os
import copy
import torch
from ultralytics.nn.modules.head import Detect

class HeadManager:
    def __init__(self, save_dir, save_every=20):
        self.cache = {}  # task → Detect layer
        self.episode_counter = {}  # task → episode count
        self.save_dir = save_dir
        self.save_every = save_every
        self.head_ch = {}  # task → [ch1, ch2, ch3]
        os.makedirs(save_dir, exist_ok=True)

    def _extract_detect_input_channels(self, model):
        detect = model.model.model[-1]  # Detect head
        # Model has Nested Sequential layers (e.g. cv3.0.0)
        ch_list = []
        for m in detect.cv2:
            if isinstance(m, torch.nn.Sequential):
                # Nested conv inside sequential
                conv = next(c for c in m.modules() if isinstance(c, torch.nn.Conv2d))
                ch_list.append(conv.in_channels)
            else:
                ch_list.append(m.conv.in_channels)
        return ch_list


    def get_or_create_head(self, model, task, num_classes, imgsz=512):
        # Run dummy forward pass to ensure Detect head is initialized
        device = next(model.parameters()).device
        model.to(device)
        model.model(torch.zeros(1, 3, imgsz, imgsz).to(device))

        ch = self._extract_detect_input_channels(model)
        cached_head = self.cache.get(task)
        cached_ch = self.head_ch.get(task)

        if cached_head is not None and cached_ch == ch:
            model.model.model[-1] = cached_head
        else:
            print(f"    No cached head for task {task} or channel mismatch. Creating new head.")
            new_head = Detect(num_classes, ch=ch).to(device)

            # Required attributes for YOLO's forward pass
            new_head.f = model.model.model[-1].f  # from layer
            new_head.i = model.model.model[-1].i  # layer index
            
            model.model.model[-1] = new_head
            self.cache[task] = new_head
            self.head_ch[task] = ch


    def update_after_episode(self, model, task):
        # Save updated head weights
        self.cache[task] = copy.deepcopy(model.model.model[-1])
        self.episode_counter[task] = self.episode_counter.get(task, 0) + 1

        if self.save_dir and self.episode_counter[task] % self.save_every == 0:
            path = os.path.join(self.save_dir, f"{task}_{self.episode_counter[task]}.pt")
            torch.save(self.cache[task].state_dict(), path)