import torch
import torchvision as tv
from pathlib import Path
from tqdm.autonotebook import tqdm
import cv2
import os

from pipeline import Preprocess

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, res, device="cuda", aligned_subdir="aligned"):
        self.res = res
        self.root = root
        self.aligned_root = os.path.join(root, aligned_subdir)
        Path(self.aligned_root).mkdir(exist_ok=True)

        self.preprocess_pipeline = Preprocess(self.res)
        self.all_imgs = [img for img in os.listdir(root) if "jpg" in img or "png" in img or "jpeg" in img]
        self.mask_imgs = ["mask_" + img for img in self.all_imgs]

        self.device = torch.device(device)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        #TODO: Decrease number of loads from disk per image to single read

        aligned_path = os.path.join(self.aligned_root, self.all_imgs[idx])
        aligned_mask = os.path.join(self.aligned_root, self.mask_imgs[idx])
        if os.path.exists(aligned_path):
            im = tv.io.read_image(aligned_path) / 255.0
            mask = tv.io.read_image(aligned_mask, mode=tv.io.image.ImageReadMode.GRAY) / 255.0
        else:
            path = os.path.join(self.root, self.all_imgs[idx])
            im, mask = self.preprocess_pipeline.do_pipeline(path)

            tv.io.write_jpeg((im*255).byte(), aligned_path, 90)
            tv.io.write_jpeg((mask*255).byte(), aligned_mask, 90)

        im = im.to(self.device)
        mask = mask.to(self.device)

        return im, mask

class ResumableRandomSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, start_at=0, seed=69):
        self.data_source = data_source
        self.generator = torch.Generator()
        self.generator.manual_seed(69)

        self.perm_index = start_at
        self.perm = torch.randperm(self.num_samples, generator=self.generator)

    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    @property
    def left(self) -> int:
        return self.num_samples - self.perm_index

    def __iter__(self):
        while self.perm_index < len(self.perm):
            yield self.perm[self.perm_index]
            self.perm_index += 1

        if self.perm_index >= len(self.perm):
            self.perm_index = 0
            self.perm = torch.randperm(self.num_samples, generator=self.generator)

    def __len__(self):
        return self.num_samples

    def get_state(self):
        return {"perm_index": self.perm_index, "perm": self.perm,
                "generator": self.generator.get_state()}

    def set_state(self, state):
        self.perm_index = state["perm_index"]
        self.perm = state["perm"]
        self.generator.set_state(state["generator"].cpu())

def get_frames(vid_path, out_dir):
  vidcap = cv2.VideoCapture(vid_path)
  n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
  length = len(str(n_frames))

  success, image = vidcap.read()

  for i in tqdm(range(n_frames)):
    out = f"{out_dir}/{str(i).zfill(length)}.jpg"
    cv2.imwrite(out, image)      
    success, image = vidcap.read()

    if not success:
        return

    
