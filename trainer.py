import torch

import numpy as np
from adabelief_pytorch import AdaBelief
from pytorch_msssim import ssim
from tqdm.notebook import tqdm
import cv2

from arch import Model
from dataset import Dataset, ResumableRandomSampler


class Loss:
    def __init__(self, res):
        self.res = res

    def __call__(self, target, target_mask, pred, pred_mask):

        def dssim(x, y, win_size):
            win_size += 1 - (win_size % 2)
            return (1 - ssim(x, y, data_range=1.0, win_size=win_size))/2
        
        l = 10 * torch.mean((target_mask - pred_mask)**2, dim=(1, 2, 3))
        
        targetm = target * target_mask
        predm = pred * target_mask

        l += 10 * torch.mean((targetm - predm)**2, dim=(1, 2, 3))

        if self.res > 256:
            l += 5*dssim(targetm, predm, self.res/11.6) + 5*dssim(targetm, predm, self.res/23.2)
        else:
            l += 10*dssim(targetm, predm, self.res/11.6)

        return l

def backup_training(model, optimizer, src_dl, dst_dl, curr_iter, path):
    torch.save({
        "model_params": model.get_parameters(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "src_sampler": src_dl.sampler.get_state(),
        "dest_sampler": dst_dl.sampler.get_state(),
        "curr_iter": curr_iter
    }, path)


def write_preview(pred_src, pred_mask_src, pred_dest, pred_mask_dest, batch_dst, model, i, preview_dir, res):
    cv2.imwrite(f"{preview_dir}/src_to_src_{i}.jpg", 
                255*np.array(pred_src[0].detach().cpu().permute(1, 2, 0))[..., [2, 1, 0]])
    cv2.imwrite(f"{preview_dir}/src_to_src_{i}m.jpg", 
                255*np.array(pred_mask_src[0].detach().cpu().view((res, res))))
    cv2.imwrite(f"{preview_dir}/dst_to_dst_{i}.jpg", 
                255*np.array(pred_dest[0].detach().cpu().permute(1, 2, 0))[..., [2, 1, 0]])
    cv2.imwrite(f"{preview_dir}/dst_to_dst_{i}m.jpg", 
                255*np.array(pred_mask_dest[0].detach().cpu().view((res, res))))
    pred_res, pred_mask_res = model(batch_dst[[0]], "SRC")
    cv2.imwrite(f"{preview_dir}/dst_to_src_{i}.jpg", 
                255*np.array(pred_res[0].detach().cpu().permute(1, 2, 0))[..., [2, 1, 0]])
    cv2.imwrite(f"{preview_dir}/dst_to_src_{i}m.jpg", 
                255*np.array(pred_mask_res[0].detach().cpu().view((res, res))))

def train(src_dl, dst_dl, model, optimizer=None, max_iters=200000, 
            backup_every=1000, device="cuda", name="model", curr_iter=0, 
            preview_dir="/content/drive/MyDrive/previews", backup_dir="/content/drive/MyDrive"):

    device = torch.device(device)
    res = model.res

    if optimizer is None:
        optimizer = AdaBelief(model.parameters(), lr=1e-5)
    loss = Loss(res)

    total_iters = curr_iter

    try:
        while True:

            print(f"Current iteration: {total_iters}")

            for (batch_src, masks_src), (batch_dst, masks_dst) in tqdm(zip(src_dl, dst_dl), 
                                                                       initial=min(src_dl.sampler.perm_index//4, dst_dl.sampler.perm_index//4),
                                                                       total=min(src_dl.sampler.left//4, dst_dl.sampler.left//4)):


                pred_src, pred_mask_src = model(batch_src, "SRC")
                pred_dest, pred_mask_dest = model(batch_dst, "DEST")


                if total_iters >= max_iters:
                    backup_training(model, optimizer, src_dl, dst_dl, total_iters, f"{backup_dir}/{name}.pt")
                    
                    write_preview(pred_src, pred_mask_src, pred_dest, pred_mask_dest, 
                                    batch_dst, model, total_iters, preview_dir, res)

                    print("JOB DONE")
                    return

                l_src = loss(batch_src, masks_src, pred_src, pred_mask_src)
                l_dst = loss(batch_dst, masks_dst, pred_dest, pred_mask_dest)

                l = torch.mean(l_src + l_dst)

                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                
                if total_iters % backup_every == 0 and total_iters > curr_iter:
                    print(f"Doing backup at {total_iters}")
                    backup_training(model, optimizer, src_dl, dst_dl, total_iters, f"{backup_dir}/{name}.pt")
                
                    print(f"Creating preview at {preview_dir}")
                    write_preview(pred_src, pred_mask_src, pred_dest, pred_mask_dest, 
                                    batch_dst, model, total_iters, preview_dir, res)

                total_iters += 1

    except KeyboardInterrupt:
        print("FINISHING")
        backup_training(model, optimizer, src_dl, dst_dl, total_iters, f"{backup_dir}/{name}.pt")
        write_preview(pred_src, pred_mask_src, pred_dest, pred_mask_dest, 
                                    batch_dst, model, total_iters, preview_dir, res)
        return

    except:
        backup_training(model, optimizer, src_dl, dst_dl, total_iters, f"{backup_dir}/{name}.pt")
        raise


def start_from_scratch(res, enc_ch, int_ch, dec_ch, dec_mask_ch, src_dir, dst_dir, name, max_iters=500000, batch_size=4, 
        backup_every=1000, device="cuda", preview_dir="/content/drive/MyDrive/previews", backup_dir="/content/drive/MyDrive",
        aligned_subdir="aligned"):

    device = torch.device(device)
    model = Model(res, enc_ch, int_ch, dec_ch, dec_mask_ch)
    model.to(device)
    model.train()

    ds_src = Dataset(src_dir, res, aligned_subdir=aligned_subdir)
    ds_dst = Dataset(dst_dir, res, aligned_subdir=aligned_subdir)

    src_sampler = ResumableRandomSampler(ds_src)
    src_dl = torch.utils.data.DataLoader(ds_src, batch_size=batch_size, 
                    sampler=src_sampler, drop_last=True)

    dst_sampler = ResumableRandomSampler(ds_dst)
    dst_dl = torch.utils.data.DataLoader(ds_dst, batch_size=batch_size, 
                    sampler=dst_sampler, drop_last=True)

    train(src_dl, dst_dl, model, max_iters=max_iters, backup_every=backup_every, 
            device=device, name=name, preview_dir=preview_dir, backup_dir=backup_dir)

def start_from_pretrained(path_to_pretrained, src_dir, dst_dir, name, max_iters=500000, batch_size=4, backup_every=1000, 
        device="cuda", preview_dir="/content/drive/MyDrive/previews", backup_dir="/content/drive/MyDrive",
        aligned_subdir="aligned"):

    device = torch.device(device)
    state = torch.load(path_to_pretrained)

    model = Model(**state["model_params"])
    model.to(device)
    model.load_state_dict(state["model"])
    model.train()

    res = model.res
    ds_src = Dataset(src_dir, model.res, aligned_subdir=aligned_subdir)
    ds_dst = Dataset(dst_dir, model.res, aligned_subdir=aligned_subdir)

    src_sampler = ResumableRandomSampler(ds_src)
    src_dl = torch.utils.data.DataLoader(ds_src, batch_size=batch_size, 
                    sampler=src_sampler, drop_last=True)

    dst_sampler = ResumableRandomSampler(ds_dst)
    dst_dl = torch.utils.data.DataLoader(ds_dst, batch_size=batch_size, 
                    sampler=dst_sampler, drop_last=True)

    print(f"Starting training for at most {max_iters} iterations")

    train(src_dl, dst_dl, model, max_iters=max_iters, backup_every=backup_every, \
            device=device, name=name, preview_dir=preview_dir, backup_dir=backup_dir)



def load_training(path, path_to_dst, path_to_src, name, max_iters=500000, device="cuda", batch_size=4,
                preview_dir="/content/drive/MyDrive/previews", backup_dir="/content/drive/MyDrive",
                aligned_subdir="aligned"):
    device = torch.device("cuda")
    state = torch.load(path, map_location=device)

    model = Model(**state["model_params"])
    model.to(device)

    model.load_state_dict(state["model"])
    optimizer = AdaBelief(model.parameters(), lr=1e-5)
    optimizer.load_state_dict(state["optimizer"])

    dst = Dataset(path_to_dst, model.res, aligned_subdir=aligned_subdir)
    dst_sampler = ResumableRandomSampler(dst)
    dst_sampler.set_state(state["dest_sampler"])
    dst_dl = torch.utils.data.DataLoader(dst, batch_size=batch_size, sampler=dst_sampler, drop_last=True)

    src = Dataset(path_to_src, model.res, aligned_subdir=aligned_subdir)
    src_sampler = ResumableRandomSampler(src)
    src_sampler.set_state(state["dest_sampler"])
    src_dl = torch.utils.data.DataLoader(src, batch_size=batch_size, sampler=src_sampler, drop_last=True)

    curr_iter = state["curr_iter"]

    model.train()

    train(src_dl, dst_dl, model, optimizer=optimizer, max_iters=max_iters, backup_every=1000, 
            device=device, name=name, curr_iter=curr_iter, preview_dir="/content/drive/MyDrive/previews")




