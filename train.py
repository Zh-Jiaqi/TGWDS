import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from thop import profile
from thop import clever_format

from config import configs
from utils.log import printwrite
from utils.datasets import FixedWindTerrainDataset

from models.WSR_model import WindSR_Terrain


def load_model(cfg, chk_path, device):
    net = WindSR_Terrain(
        upscale=cfg.scale, dim=64,
        group_depth=6, dem_every=2, sample_ids=(1, 3, 5),
        dem_ch=16, psa_kernels=(1, 3, 5), psa_groups=(1, 1, 1),
        enc_res_scale=0.1, decoder_feats=None,
        sft_hidden=64, cross_blocks_per_stage=4,
        stage_res_scale=0.2, dropout_max=0.1
    ).to(device)

    ckpt = torch.load(chk_path, map_location=device)
    net.load_state_dict(ckpt['net'], strict=True)
    net.eval()
    return net


def evaluate_mae(net, dataloader, device, stats, save_prefix="results"):
    preds, trues = [], []

    abs_err_sum = 0.0
    numel_sum = 0

    wind_mean = stats["wind_mean"]
    wind_std = stats["wind_std"]

    with torch.no_grad():
        total_time = 0.0
        count_time = 0

        for lr, hr, gl, gh in dataloader:
            lr = lr.float().to(device)
            hr = hr.float().to(device)
            gl = gl.float().to(device)
            gh = gh.float().to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()

            pred = net(lr, gl, gh)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()

            total_time += (end - start)
            count_time += 1

            pred_denorm = pred.cpu().numpy() * wind_std[None, :, None, None] + wind_mean[None, :, None, None]
            true_denorm = hr.cpu().numpy() * wind_std[None, :, None, None] + wind_mean[None, :, None, None]

            abs_err = np.abs(pred_denorm - true_denorm)
            abs_err_sum += abs_err.sum()
            numel_sum += abs_err.size

            preds.append(pred_denorm)
            trues.append(true_denorm)

        print("Average inference time: {:.4f} s (≈ {:.2f} ms)".format(
            total_time / count_time, (total_time / count_time) * 1000
        ))

    mae_avg = abs_err_sum / max(1, numel_sum)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    os.makedirs(save_prefix, exist_ok=True)
    np.save(os.path.join(save_prefix, "y_pred.npy"), preds)
    np.save(os.path.join(save_prefix, "y_true.npy"), trues)

    return mae_avg, preds.shape, trues.shape


if __name__ == "__main__":
    name = getattr(configs, "name", "WSR")

    exp_dir = f"exp/{name}"
    os.makedirs(exp_dir, exist_ok=True)

    log_file = os.path.join(exp_dir, "log_test.txt")
    device = configs.device

    printwrite(log_file, "Loading configs and dataset...")

    test_path = configs.test_path
    scale = configs.scale

    stats_file = os.path.join(exp_dir, f"stats_{name}.npy")
    stats = np.load(stats_file, allow_pickle=True).item()

    dataset_eval = FixedWindTerrainDataset(
        test_path,
        configs.geo_path,
        train=False,
        scale=scale,
        save_stats_path=stats_file
    )

    dl_eval = DataLoader(
        dataset_eval,
        batch_size=configs.batch_size_val,
        shuffle=False,
        drop_last=False
    )

    chk_path = os.path.join(exp_dir, f"{name}_best.chk")
    printwrite(log_file, f"Loading checkpoint: {chk_path}")
    net = load_model(configs, chk_path, device)

    sample_lr, sample_hr, sample_gl, sample_gh = dataset_eval[0]
    dummy_lr = sample_lr.unsqueeze(0).to(device)
    dummy_gl = sample_gl.unsqueeze(0).to(device)
    dummy_gh = sample_gh.unsqueeze(0).to(device)

    flops, params = profile(net, inputs=(dummy_lr, dummy_gl, dummy_gh))
    flops, params = clever_format([flops, params], "%.3f")

    print("================================================")
    print("Model Complexity Summary:")
    print(f"Params: {params}")
    print(f"FLOPs : {flops}")
    print("================================================")

    results_dir = os.path.join(exp_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    mae, pred_shape, true_shape = evaluate_mae(
        net, dl_eval, device, stats,
        save_prefix=results_dir
    )

    printwrite(log_file, f"Test MAE: {mae:.6f}")
    printwrite(log_file, f"Pred shape: {pred_shape}, True shape: {true_shape}")

    print(f"Done! Test MAE={mae:.6f}, results saved to {results_dir}/y_pred.npy and y_true.npy")
