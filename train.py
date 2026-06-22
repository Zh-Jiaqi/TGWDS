# -*- coding: utf-8 -*-
import os
import math
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler

from config import configs
from utils.log import printwrite
from utils.datasets import FixedWindTerrainDataset
from utils.loss import WindLoss
import time

#from models.WSR_model import WindSR_Terrain
from models.model_wo_Dem import WindSR_Terrain


class WarmupCosine:
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr, min_lr=1e-6):
        self.opt = optimizer
        self.warmup = max(1, warmup_steps)
        self.total = max(self.warmup + 1, total_steps)
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.step_id = 0

    def step(self):
        self.step_id += 1
        if self.step_id <= self.warmup:
            lr = self.base_lr * self.step_id / self.warmup
        else:
            t = (self.step_id - self.warmup) / (self.total - self.warmup)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * t))
        for g in self.opt.param_groups:
            g['lr'] = lr
        return lr


# ============================= Trainer =============================
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device

        self.network = WindSR_Terrain(
            upscale=cfg.scale, dim=64,
            group_depth=6, dem_every=2, sample_ids=(1, 3, 5),
            dem_ch=16, psa_kernels=(1, 3, 5), psa_groups=(1, 1, 1),
            enc_res_scale=0.1, decoder_feats=None,
            sft_hidden=64, cross_blocks_per_stage=4,
            stage_res_scale=0.2, dropout_max=0.1
        ).to(self.device)


        self.opt = torch.optim.AdamW(self.network.parameters(),
                                     lr=cfg.lr, weight_decay=1e-4, betas=(0.9, 0.999))
        

        steps_per_epoch = getattr(cfg, "steps_per_epoch_est", 270)
        self.total_steps = cfg.epochs * steps_per_epoch
        self.sched = WarmupCosine(self.opt,
                                  warmup_steps=int(0.05 * self.total_steps),
                                  total_steps=self.total_steps,
                                  base_lr=cfg.lr, min_lr=1e-6)
        
        self.early_stop = cfg.patience

        self.scaler = GradScaler()
        
        self.loss_fn = WindLoss(self.device).to(self.device)

    def _load_stats(self):
        stats_file = os.path.join("exp", self.cfg.name, f"stats_{self.cfg.name}.npy")
        if not os.path.exists(stats_file):
            raise FileNotFoundError(f"Stats file not found: {stats_file}")
    
        stats = np.load(stats_file, allow_pickle=True)
        if isinstance(stats, np.ndarray) and stats.shape == ():
            stats = stats.item()
    
        self.stats = stats
        print(f"Loaded stats from: {stats_file}")
        print(f"Stats keys: {list(stats.keys())}")
        return self.stats


    def train_once(self, lr, hr, gl, gh, epoch):
        lr = lr.float().to(self.device)  # (B,2,H/scale,W/scale)
        hr = hr.float().to(self.device)  # (B,2,H,W)
        gl = gl.float().to(self.device)  # (B,1,H/scale,W/scale)
        gh = gh.float().to(self.device)  # (B,1,H,W)

        pred = self.network(lr, gl, gh)

        loss, info = self.loss_fn(pred, hr, gh, epoch=epoch, ramp_T=10)
        return loss, info


    def test(self, dataset_eval, dataloader_eval):
        self.network.eval()
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for lr, hr, gl, gh in dataloader_eval:
                lr = lr.float().to(self.device)
                hr = hr.float().to(self.device)
                gl = gl.float().to(self.device)
                gh = gh.float().to(self.device)

                pred = self.network(lr, gl, gh)
                loss = self.loss_fn.val_loss(pred, hr)
                total_loss += float(loss.detach().cpu())
                n += 1
        return total_loss / max(1, n)

    def load_model(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt['net'])
        print(f'Loaded model from: {path}')


    def test_and_save(self, dataset_test, dataloader_test, save_dir):
        self.network.eval()
        stats = self._load_stats()
    
        wind_mean = stats["wind_mean"]
        wind_std = stats["wind_std"]
    
        preds = []
        gts = []
    
        abs_err_sum = 0.0
        numel_sum = 0
    
        total_time = 0.0
        count_time = 0
    
        with torch.no_grad():
            for lr, hr, gl, gh in dataloader_test:
                lr = lr.float().to(self.device)
                hr = hr.float().to(self.device)
                gl = gl.float().to(self.device)
                gh = gh.float().to(self.device)
    
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.time()
    
                pred = self.network(lr, gl, gh)
    
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.time()
    
                total_time += (end - start)
                count_time += 1
    
                pred_np = pred.detach().cpu().numpy()
                hr_np = hr.detach().cpu().numpy()
    
                pred_denorm = pred_np * wind_std[None, :, None, None] + wind_mean[None, :, None, None]
                true_denorm = hr_np * wind_std[None, :, None, None] + wind_mean[None, :, None, None]
    
                abs_err = np.abs(pred_denorm - true_denorm)
                abs_err_sum += abs_err.sum()
                numel_sum += abs_err.size
    
                preds.append(pred_denorm)
                gts.append(true_denorm)
    
        avg_mae = abs_err_sum / max(1, numel_sum)
    
        if count_time > 0:
            print("Average inference time: {:.4f} s (≈ {:.2f} ms)".format(
                total_time / count_time, (total_time / count_time) * 1000
            ))
    
        y_pred = np.concatenate(preds, axis=0)
        y_true = np.concatenate(gts, axis=0)
    
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "y_pred.npy"), y_pred)
        np.save(os.path.join(save_dir, "y_true.npy"), y_true)
    
        return avg_mae


    def train(self, dataset_train, dataset_eval, chk_dir):

        log_file = os.path.join(chk_dir, "log.txt")
        best = math.inf
        count = 0

        printwrite(log_file, 'loading train dataloader')
        dl_train = DataLoader(dataset_train, batch_size=self.cfg.batch_size, shuffle=True, drop_last=False)
        printwrite(log_file, 'loading eval dataloader')
        dl_eval = DataLoader(dataset_eval, batch_size=self.cfg.batch_size_val, shuffle=False, drop_last=False)

        self.network.train()
        step_global = 0

        for epoch in range(1, self.cfg.epochs + 1):
            printwrite(log_file, f'\nepoch: {epoch}')
            self.opt.zero_grad()

            for it, (lr, hr, gl, gh) in enumerate(dl_train, start=1):
                self.network.train()
                loss, info = self.train_once(lr, hr, gl, gh, epoch=epoch)

                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                lr_now = self.sched.step()

                step_global += 1
                if (it % self.cfg.display_interval) == 0:
                    printwrite(log_file, 'batch training loss: {:.4f}, MAE={:.4f}, Lhigh={:.4f}, w_high={:.2f}, Phys={:.3f}, w_phys={:.3f}, lr={:.6f}'.format(float(loss.detach().cpu()), info["L1"], info["Lhigh"], info["w_high"], info["Phys"], info["w_phys"], lr_now))

                if (epoch >= 5) and (it % (self.cfg.display_interval * self.cfg.eval_interval) == 0):
                    val = self.test(dataset_eval, dl_eval)
                    printwrite(log_file, f'batch eval loss: {val:.4f}')
                    if val < best:
                        count = 0
                        printwrite(log_file, f'eval loss is reduced from {best:.5f} to {val:.5f}, saving model')
                        self.save_model(os.path.join(chk_dir, f'{self.cfg.name}_best.chk'))
                        best = val


            val = self.test(dataset_eval, dl_eval)
            printwrite(log_file, f'epoch eval loss: {val:.4f}')
            if val < best:
                count = 0
                printwrite(log_file, f'eval loss is reduced from {best:.5f} to {val:.5f}, saving model')
                self.save_model(os.path.join(chk_dir, f'{self.cfg.name}_best.chk'))
                best = val
            else:
                count += 1
                printwrite(log_file, f'eval loss is not reduced for {count} epoch')
                printwrite(log_file, f'best is {best} until now')


            self.save_model(os.path.join(chk_dir, f'{self.cfg.name}_last.chk'))
            

            if count >= self.early_stop:
                printwrite(log_file, f'\nEarly stopping triggered after {count} epochs without improvement.')
                printwrite(log_file, f'Best validation loss: {best:.5f}')
                break 
                

    def save_configs(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.cfg, f)

    def save_model(self, path):
        torch.save({'net': self.network.state_dict()}, path)


# ============================= main =============================
if __name__ == '__main__':
    name = getattr(configs, "name", "WSR")
    

    exp_dir = f"exp/{name}"
    os.makedirs(exp_dir, exist_ok=True)

    log_file = os.path.join(exp_dir, "log.txt")
    printwrite(log_file, 'Configs:\n' + str(configs.__dict__))
    

    stats_file = os.path.join(exp_dir, f"stats_{name}.npy")

    train_path = configs.train_path    
    val_path = configs.val_path          
    scale = configs.scale

    printwrite(log_file, 'processing train set')
    dataset_train = FixedWindTerrainDataset(
        train_path,
        configs.geo_path,
        train=True,
        scale=scale,
        save_stats_path=stats_file
    )

    printwrite(log_file, 'processing val set')
    dataset_eval = FixedWindTerrainDataset(
        val_path,
        configs.geo_path,
        train=False,
        scale=scale,
        save_stats_path=stats_file 
    )

    printwrite(log_file, 'processing test set')
    dataset_test = FixedWindTerrainDataset(
        configs.test_path,
        configs.geo_path,
        train=False,
        scale=scale,
        save_stats_path=stats_file
    )
    
    printwrite(
        log_file,
        f'Dataset loaded: Train={len(dataset_train)}, Val={len(dataset_eval)}, Test={len(dataset_test)}'
    )

    def get_dataset_shape_dict(dataset):
        full_shapes_dict = dataset.GetDataShape()
        current_len = len(dataset)
        new_shapes_dict = {}
        if isinstance(full_shapes_dict, dict):
            for key, shape in full_shapes_dict.items():
                new_shape = (current_len,) + shape[1:]
                new_shapes_dict[key] = new_shape
        else:
            return f"Length: {current_len} (Shape format unknown)"
        return new_shapes_dict

    train_shapes = get_dataset_shape_dict(dataset_train)
    eval_shapes = get_dataset_shape_dict(dataset_eval)
    test_shapes = get_dataset_shape_dict(dataset_test)

    printwrite(log_file, f"Train shapes: {train_shapes}")
    printwrite(log_file, f"Val shapes: {eval_shapes}")
    printwrite(log_file, f"Test shapes: {test_shapes}")

    trainer = Trainer(configs)
    trainer.save_configs(os.path.join(exp_dir, "configs.pkl"))
    trainer.train(dataset_train, dataset_eval, chk_dir=exp_dir)

    best_ckpt = os.path.join(exp_dir, f"{configs.name}_best.chk")
    results_dir = os.path.join(exp_dir, "results")

    printwrite(log_file, 'loading best checkpoint for test')
    trainer.load_model(best_ckpt)

    dl_test = DataLoader(
        dataset_test,
        batch_size=16,
        shuffle=False,
        drop_last=False
    )

    test_loss = trainer.test_and_save(dataset_test, dl_test, results_dir)
    printwrite(log_file, f"test MAE: {test_loss:.6f}")
    printwrite(log_file, f"saved predictions to: {results_dir}")
