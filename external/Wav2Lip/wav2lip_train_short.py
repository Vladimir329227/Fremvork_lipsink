"""Short finetune run for Wav2Lip (same logic as wav2lip_train.py, bounded steps, Windows-friendly)."""
from __future__ import annotations

from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
from models import Wav2Lip as Wav2Lip
import audio

import torch
from torch import nn
from torch import optim
from torch.utils import data as data_utils
import numpy as np
from glob import glob
import os, random, cv2, argparse
from hparams import hparams, get_image_list

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", required=True, type=str)
parser.add_argument("--checkpoint_dir", required=True, type=str)
parser.add_argument("--syncnet_checkpoint_path", required=True, type=str)
parser.add_argument("--init_wav2lip", default=None, type=str, help="e.g. wav2lip_gan.pth")
parser.add_argument("--max_steps", default=600, type=int)
parser.add_argument("--checkpoint_interval", default=200, type=int)
args = parser.parse_args()

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print("use_cuda:", use_cuda)

syncnet_T = 5
syncnet_mel_step_size = 16


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class Dataset(object):
    def __init__(self, split):
        self.all_videos = get_image_list(args.data_root, split)

    def get_frame_id(self, frame):
        return int(basename(frame).split(".")[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)
        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, "{}.jpg".format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None:
            return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception:
                return None
            window.append(img)
        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80.0 * (start_frame_num / float(hparams.fps)))
        end_idx = start_idx + syncnet_mel_step_size
        return spec[start_idx:end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1
        if start_frame_num - 2 < 0:
            return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)
        return np.asarray(mels)

    def prepare_window(self, window):
        x = np.asarray(window) / 255.0
        x = np.transpose(x, (3, 0, 1, 2))
        return x

    def __len__(self):
        # Stochastic __getitem__; pretend large so DataLoader yields many steps per epoch.
        return max(4096, len(self.all_videos) * 2000)

    def __getitem__(self, idx):
        while 1:
            if not self.all_videos:
                raise RuntimeError("No video folders in filelist")
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, "*.jpg")))
            if len(img_names) <= 3 * syncnet_T:
                continue
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)
            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                continue
            window = self.read_window(window_fnames)
            if window is None:
                continue
            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                continue
            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)
                orig_mel = audio.melspectrogram(wav).T
            except Exception:
                continue
            mel = self.crop_audio_window(orig_mel.copy(), img_name)
            if mel.shape[0] != syncnet_mel_step_size:
                continue
            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            if indiv_mels is None:
                continue
            window = self.prepare_window(window)
            y = window.copy()
            window[:, :, window.shape[2] // 2 :] = 0.0
            wrong_window = self.prepare_window(wrong_window)
            x = np.concatenate([window, wrong_window], axis=0)
            return (
                torch.FloatTensor(x),
                torch.FloatTensor(indiv_mels).unsqueeze(1),
                torch.FloatTensor(mel.T).unsqueeze(0),
                torch.FloatTensor(y),
            )


logloss = nn.BCELoss()


def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    return logloss(d.unsqueeze(1), y)


device = torch.device("cuda" if use_cuda else "cpu")
syncnet = SyncNet().to(device)
for p in syncnet.parameters():
    p.requires_grad = False
recon_loss = nn.L1Loss()


def get_sync_loss(mel, g):
    g = g[:, :, :, g.size(3) // 2 :]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = join(checkpoint_dir, "checkpoint_step{:09d}.pth".format(int(step)))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer_state,
            "global_step": step,
            "global_epoch": epoch,
        },
        checkpoint_path,
    )
    print("Saved checkpoint:", checkpoint_path)


def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step, global_epoch
    print("Load checkpoint from:", path)
    checkpoint = _torch_load(path)
    s = checkpoint["state_dict"]
    new_s = {k.replace("module.", ""): v for k, v in s.items()}
    model.load_state_dict(new_s)
    if not reset_optimizer and optimizer is not None and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint.get("global_step", 0)
        global_epoch = checkpoint.get("global_epoch", 0)
    return model


def train_short(model, optimizer, train_loader, checkpoint_dir):
    global global_step, global_epoch
    max_steps = args.max_steps
    interval = args.checkpoint_interval
    while global_step < max_steps:
        print("Epoch", global_epoch)
        prog = tqdm(train_loader)
        for _, (x, indiv_mels, mel, gt) in enumerate(prog):
            if global_step >= max_steps:
                break
            model.train()
            optimizer.zero_grad()
            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)
            g = model(indiv_mels, x)
            if hparams.syncnet_wt > 0.0:
                sync_loss = get_sync_loss(mel, g)
            else:
                sync_loss = 0.0
            l1loss = recon_loss(g, gt)
            loss = hparams.syncnet_wt * sync_loss + (1.0 - hparams.syncnet_wt) * l1loss
            loss.backward()
            optimizer.step()
            global_step += 1
            prog.set_description(f"L1:{l1loss.item():.4f} step:{global_step}")
            if global_step == 1 or global_step % interval == 0 or global_step >= max_steps:
                save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch)
        global_epoch += 1


if __name__ == "__main__":
    hparams.set_hparam("num_workers", 0)
    hparams.set_hparam("batch_size", 1)
    hparams.set_hparam("checkpoint_interval", args.checkpoint_interval)

    train_dataset = Dataset("train")
    train_loader = data_utils.DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = Wav2Lip().to(device)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=hparams.initial_learning_rate)

    if args.init_wav2lip:
        load_checkpoint(args.init_wav2lip, model, optimizer, reset_optimizer=True, overwrite_global_states=True)
        global_step = 0
        global_epoch = 0

    load_checkpoint(
        args.syncnet_checkpoint_path,
        syncnet,
        None,
        reset_optimizer=True,
        overwrite_global_states=False,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train_short(model, optimizer, train_loader, args.checkpoint_dir)
    print("Done. Last step:", global_step)
