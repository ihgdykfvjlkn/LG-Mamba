#!/usr/bin/env python3
"""
Main Script - ULTRA ROBUST CROSS-DATASET EVALUATION (FULLY ALIGNED WITH REFINED demomodel.py)
[ALIGNMENT]: Integrated DFSSMambaNet / IQANet with HighPassFFT + SCIEdge branch,
robust checkpoint parsing, and detailed per-distortion-type profiling.
"""

import os
import sys
# 强制将当前脚本所在的 src 目录以及项目根目录插入 sys.path 最前面
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import time
import argparse
import logging
from collections import OrderedDict

import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

# 引入优化后的模型接口
from .demomodel import IQANet, DFSSMambaNet
from .dataset import IQADataset, SCIDDataset, SIQADDataset
from .utils import AverageMeter, SROCC, PLCC, RMSE

torch.backends.cudnn.benchmark = True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = None


def setup_logger():
    """高可读性终端日志系统"""
    global logger
    if logger is None:
        logger = logging.getLogger("DFSS_IQA_CrossTest")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(console_handler)


def test_cross(test_loader, model, target_dataset_name):
    """
    跨数据集核心测试函数（适配 demomodel.py 的测试模式输出，并按目标数据集细分畸变类型打印）
    """
    srocc = SROCC()
    plcc = PLCC()
    rmse = RMSE()

    # 定义目标数据集的畸变类型映射
    dataset_mapping = {
        'SIQAD': {
            0: 'GN', 1: 'GB', 2: 'JPEG',
            3: 'JP2K', 4: 'LSC', 5: 'CC', 6: 'MAR'
        },
        'SCID': {
            0: 'GN', 1: 'GB', 2: 'MB',
            3: 'CC', 4: 'SC', 5: 'CS',
            6: 'JPEG', 7: 'J2K', 8: 'HEVC'
        }
    }

    dist_names = dataset_mapping.get(target_dataset_name, {})
    dist_scores = {}

    logger.info("\n➡️  Starting Cross-Dataset Evaluation Loop...")
    model.eval()

    pbar = tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        desc="Testing Progress",
        file=sys.stdout,
        leave=True
    )

    with torch.no_grad():
        for i, (img, ref, pref, score, dtype) in pbar:
            img = img.to(device, non_blocking=True)
            ref = ref.to(device, non_blocking=True)
            pref = pref.to(device, non_blocking=True)
            score = score.to(device, non_blocking=True)
            dtype = dtype.to(device, non_blocking=True)

            # 在 istrain=False 时，model(img, ref, pref) 直接返回单张/单批次的预测分数值 Tensor
            outputs = model(img, ref, pref)
            if isinstance(outputs, tuple):
                output = outputs[0]  # 防御性解包
            else:
                output = outputs

            output = output.squeeze()

            # 转成 1D numpy 数组
            pred = output.detach().cpu().numpy().reshape(-1)
            gt = score.detach().cpu().numpy().reshape(-1)
            dtype_np = dtype.cpu().numpy().reshape(-1)

            # 过滤 NaN / Inf，防御异常预测值
            valid = np.isfinite(pred) & np.isfinite(gt)
            if valid.sum() == 0:
                continue

            pred = pred[valid]
            gt = gt[valid]
            dtype_np = dtype_np[valid]

            # 更新全局指标
            srocc.update(gt, pred)
            plcc.update(gt, pred)
            rmse.update(gt, pred)

            # 按畸变类别收集预测分数与真实标签
            for g, p, d in zip(gt, pred, dtype_np):
                d_idx = int(d)
                if d_idx not in dist_scores:
                    dist_scores[d_idx] = {"gt": [], "pred": []}
                dist_scores[d_idx]["gt"].append(g)
                dist_scores[d_idx]["pred"].append(p)

    final_srocc = float(srocc.compute())
    final_plcc = float(plcc.compute())
    final_rmse = float(rmse.compute())

    print(gt[:20])

    print(pred[:20])

    print(dtype_np[:20])

    logger.info("\n" + "=" * 60)
    logger.info(f"🏆 GLOBAL CROSS-DATASET RESULTS (Target Dataset: {target_dataset_name})")
    logger.info("=" * 60)
    logger.info(f"   SROCC: {final_srocc:.6f}")
    logger.info(f"   PLCC : {final_plcc:.6f}")
    logger.info(f"   RMSE : {final_rmse:.6f}")
    logger.info("=" * 60)

    # 打印目标数据集上各细分畸变类型的泛化表现
    logger.info(f"\n📊 PERFORMANCE DETAIL ON TARGET DISTORTIONS")
    logger.info("-" * 60)
    logger.info(f"{'Distortion Type':<15}{'SROCC':<12}{'PLCC':<12}{'Samples'}")
    logger.info("-" * 60)

    for d in sorted(dist_scores.keys()):
        gt_arr = np.array(dist_scores[d]["gt"])
        pred_arr = np.array(dist_scores[d]["pred"])

        if len(gt_arr) < 2:
            continue

        try:
            s = spearmanr(gt_arr, pred_arr)[0]
            p = pearsonr(gt_arr, pred_arr)[0]
        except Exception:
            s = np.nan
            p = np.nan

        logger.info(
            f"{dist_names.get(d, f'Type_{d}'):<15}"
            f"{s:<12.4f}"
            f"{p:<12.4f}"
            f"{len(gt_arr)}"
        )
    logger.info("-" * 60 + "\n")


def main():
    setup_logger()
    args = parse_args()

    # 1. 动态获取与初始化模型结构（测试模式 istrain=False）
    logger.info(f"⚙️  Initializing Model: {'DFSSMambaNet' if args.use_dfss_mamba else 'IQANet'}")
    if args.use_dfss_mamba:
        model = DFSSMambaNet(istrain=False, n_class=args.source_n_dtype, mamba_img_size=args.mamba_img_size)
    else:
        model = IQANet(istrain=False, n_class=args.source_n_dtype)

    model = model.to(device)
    model = nn.DataParallel(model)

    # 2. 精准匹配与加载 Checkpoint
    checkpoint_path = args.resume
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"❌ Checkpoint file not found at: {checkpoint_path}")

    logger.info(f"📂 Loading weights from: '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint['state_dict'] if (isinstance(checkpoint, dict) and 'state_dict' in checkpoint) else checkpoint

    new_state_dict = OrderedDict()
    model_state = model.state_dict()

    for k, v in state_dict.items():
        name = k if k.startswith('module.') else 'module.' + k

        current_param = model_state.get(name, None)
        if current_param is not None and current_param.shape != v.shape:
            logger.warning(
                f" ⚠️ [Shape Mismatch] Layer {name} has mismatched shape: "
                f"Model {current_param.shape} vs Checkpoint {v.shape}. Skipping this layer."
            )
            continue
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    logger.info(" => ✅ Weights successfully aligned and loaded.")

    # 3. 动态配置目标测试数据集
    target_dataset_class = globals().get(args.dataset + 'Dataset', None)
    if target_dataset_class is None:
        raise ValueError(f"❌ Unknown target dataset type: {args.dataset}")

    # 组装划分名字，如 'test_0'
    db_split_name = f"{args.subset}_{args.pro}"
    logger.info(f"📦 Constructing Target Dataset: {args.dataset} ({db_split_name})")

    test_dataset = target_dataset_class(
        args.data_dir,
        db_split_name,
        list_dir=args.list_dir,
        patch_size=args.patch_size,
        n_ptchs=args.n_ptchs_per_img,
        n_class=args.target_n_dtype
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    # 4. 执行测试
    test_cross(test_loader, model, args.dataset)


def parse_args():
    parser = argparse.ArgumentParser(description='Aligned Ultra Robust IQA Cross-Dataset Test Script')

    # --- 运行设备与控制 ---
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--pro', type=int, default=0, help='Split process index (e.g., 0 for test_0)')
    # 【修改点 1】：将默认 subset 改为 'test'，对应 test_0_data.json
    parser.add_argument('--subset', default='test', type=str, help='Evaluation subset (default: test)')

    # --- 目标被测数据集信息 (Target Dataset) ---
    parser.add_argument('--dataset', type=str, default='SCID', choices=['SCID', 'SIQAD'],
                        help='Target dataset you want to TEST on')
    # 【修改点 2】：默认路径设为真实绝对路径，避免上层目录推演报错
    parser.add_argument('-d', '--data-dir', default='/mnt/h/25liuyx/projects/DFSS-IQA-main/datasets/SCID',
                        help='Base dir path where target dataset resides')
    parser.add_argument('-l', '--list-dir', default='sci_scripts/scid-scripts-all/',
                        help='Directory containing test_0_data.json')
    parser.add_argument('--target-n-dtype', type=int, default=9,
                        help='Number of distortions of TARGET dataset (SIQAD: 7, SCID: 9)')

    # --- 源模型和训练权重配置 (Source Model/Checkpoint) ---
    parser.add_argument('--resume', default='../models/siqad/model_best_1.pkl', type=str,
                        help='Path to the BEST checkpoint trained on the SOURCE dataset')
    parser.add_argument('--source-n-dtype', type=int, default=7,
                        help='Number of distortions of SOURCE dataset when trained (SIQAD: 7, SCID: 9)')

    # --- 架构超参数 ---
    parser.add_argument('--use_dfss_mamba', action='store_true', default=False,
                        help='Must match your trained model backbone type')
    parser.add_argument('--mamba_img_size', type=int, default=112)
    parser.add_argument('-n', '--n-ptchs-per-img', type=int, default=32)
    parser.add_argument('-psz', '--patch-size', type=int, default=32)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()



































# # !/usr/bin/env python3
# """
# Main Script - ULTRA ROBUST CROSS-DATASET EVALUATION (FULLY ALIGNED WITH iqaScrach.py)
# [ALIGNMENT]: Integrated DFSSMambaNet, robust checkpoint parsing, and detailed per-distortion-type profiling.
# """
#
# import os
# import sys
# import time
# import argparse
# import logging
# import numpy as np
# from tqdm import tqdm
# from scipy.stats import spearmanr, pearsonr
#
# import torch
# import torch.nn as nn
# import torch.backends.cudnn as cudnn
#
# from demomodel import IQANet, DFSSMambaNet
# from dataset import IQADataset, SCIDDataset, SIQADDataset
# from utils import AverageMeter, SROCC, PLCC, RMSE
#
# torch.backends.cudnn.benchmark = True
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# logger = None
#
#
# def setup_logger():
#   """与训练脚本一致的高可读性日志系统"""
#   global logger
#   if logger is None:
#     logger = logging.getLogger("DFSS_IQA_CrossTest")
#     logger.setLevel(logging.INFO)
#     if not logger.handlers:
#       console_handler = logging.StreamHandler(sys.stdout)
#       console_handler.setFormatter(logging.Formatter('%(message)s'))
#       logger.addHandler(console_handler)
#
#
# def test_cross(test_loader, model, target_dataset_name):
#   """
#   跨数据集核心测试函数（支持过滤 NaN/Inf，并按目标数据集输出细分畸变类型的评估结果）
#   """
#   srocc = SROCC()
#   plcc = PLCC()
#   rmse = RMSE()
#
#   # 定义目标数据集的畸变映射，用于展示跨数据集对具体畸变的泛化能力
#   dataset_mapping = {
#     'SIQAD': {
#       0: 'GN', 1: 'GB', 2: 'JPEG',
#       3: 'JP2K', 4: 'LSC', 5: 'CC', 6: 'MAR'
#     },
#     'SCID': {
#       0: 'GN', 1: 'GB', 2: 'MB',
#       3: 'CC', 4: 'SC', 5: 'CS',
#       6: 'JPEG', 7: 'J2K', 8: 'HEVC'
#     }
#   }
#
#   dist_names = dataset_mapping.get(target_dataset_name, {})
#   dist_scores = {}
#
#   logger.info("\n➡️  Starting Cross-Dataset Evaluation Loop...")
#   model.eval()
#
#   pbar = tqdm(
#     enumerate(test_loader),
#     total=len(test_loader),
#     desc="Testing Progress",
#     file=sys.stdout,
#     leave=True
#   )
#
#   with torch.no_grad():
#     for i, (img, ref, pref, score, dtype) in pbar:
#       img = img.cuda(non_blocking=True)
#       ref = ref.cuda(non_blocking=True)
#       pref = pref.cuda(non_blocking=True)
#       score = score.cuda(non_blocking=True)
#       dtype = dtype.cuda(non_blocking=True)
#
#       # 保持与模型定义前向传播一致
#       # 注意：有些版本的 Mamba 或 IQANet 前向传播返回值数量可能与训练时不同，这里使用通用解包方式
#       outputs = model(img, ref, pref)
#       if isinstance(outputs, tuple):
#         output = outputs[0]  # 第一个返回值通常是主 MOS 预测分数值
#       else:
#         output = outputs
#
#       output = output.squeeze()
#
#       pred = output.detach().cpu().numpy().reshape(-1)
#       gt = score.detach().cpu().numpy().reshape(-1)
#       dtype_np = dtype.cpu().numpy().reshape(-1)
#
#       # 过滤非数/无穷值，保证指标计算安全性
#       valid = np.isfinite(pred) & np.isfinite(gt)
#       if valid.sum() == 0:
#         continue
#
#       pred = pred[valid]
#       gt = gt[valid]
#       dtype_np = dtype_np[valid]
#
#       srocc.update(gt, pred)
#       plcc.update(gt, pred)
#       rmse.update(gt, pred)
#
#       # 收集单类畸变分数
#       for g, p, d in zip(gt, pred, dtype_np):
#         if int(d) not in dist_scores:
#           dist_scores[int(d)] = {"gt": [], "pred": []}
#         dist_scores[int(d)]["gt"].append(g)
#         dist_scores[int(d)]["pred"].append(p)
#
#   final_srocc = float(srocc.compute())
#   final_plcc = float(plcc.compute())
#   final_rmse = float(rmse.compute())
#
#   logger.info("\n" + "=" * 60)
#   logger.info(f"🏆 GLOBAL CROSS-DATASET RESULTS (Target Dataset: {target_dataset_name})")
#   logger.info("=" * 60)
#   logger.info(f"   SROCC: {final_srocc:.6f}")
#   logger.info(f"   PLCC : {final_plcc:.6f}")
#   logger.info(f"   RMSE : {final_rmse:.6f}")
#   logger.info("=" * 60)
#
#   # 打印在目标数据集具体畸变类型上的泛化表现
#   logger.info(f"\n📊 PERFORMANCE DETAIL ON TARGET DISTORTIONS")
#   logger.info("-" * 60)
#   logger.info(f"{'Distortion Type':<15}{'SROCC':<12}{'PLCC':<12}{'Samples'}")
#   logger.info("-" * 60)
#
#   for d in sorted(dist_scores.keys()):
#     gt_arr = np.array(dist_scores[d]["gt"])
#     pred_arr = np.array(dist_scores[d]["pred"])
#
#     if len(gt_arr) < 2:
#       continue
#
#     try:
#       s = spearmanr(gt_arr, pred_arr)[0]
#       p = pearsonr(gt_arr, pred_arr)[0]
#     except Exception:
#       s = np.nan
#       p = np.nan
#
#     logger.info(
#       f"{dist_names.get(d, f'Type_{d}'):<15}"
#       f"{s:<12.4f}"
#       f"{p:<12.4f}"
#       f"{len(gt_arr)}"
#     )
#   logger.info("-" * 60 + "\n")
#
#
# def main():
#   setup_logger()
#   args = parse_args()
#
#   # 1. 动态获取与初始化模型结构（同步支持 Mamba 与传统卷积网络）
#   logger.info(f"⚙️  Initializing Model: {'DFSSMambaNet' if args.use_dfss_mamba else 'IQANet'}")
#   if args.use_dfss_mamba:
#     model = DFSSMambaNet(istrain=False, n_class=args.source_n_dtype, mamba_img_size=args.mamba_img_size)
#   else:
#     model = IQANet(istrain=False, n_class=args.source_n_dtype)
#
#   model = model.cuda()
#   model = nn.DataParallel(model)
#
#   # 2. 精准匹配与加载 Checkpoint（解决 module. 差异，并自适应过滤不兼容层）
#   checkpoint_path = args.resume
#   if not checkpoint_path or not os.path.isfile(checkpoint_path):
#     raise FileNotFoundError(f"❌ Checkpoint file not found at: {checkpoint_path}")
#
#   logger.info(f"📂 Loading weights from: '{checkpoint_path}'")
#   checkpoint = torch.load(checkpoint_path, map_location='cuda')
#
#   state_dict = checkpoint['state_dict'] if (isinstance(checkpoint, dict) and 'state_dict' in checkpoint) else checkpoint
#
#   from collections import OrderedDict
#   new_state_dict = OrderedDict()
#   for k, v in state_dict.items():
#     # 确保 DataParallel 的统一前缀 'module.'
#     name = k if k.startswith('module.') else 'module.' + k
#
#     # 【核心安全设计】：检查最后一层（分类头与回归头）是否与加载的权重匹配
#     # 如果源数据集（权重提供者）与目标数据集在模型创建时传给 n_class 的值不一致，需要跳过或包容
#     current_param = model.state_dict().get(name, None)
#     if current_param is not None and current_param.shape != v.shape:
#       logger.warning(
#         f" ⚠️ [Shape Mismatch] Layer {name} has mismatched shape: Model {current_param.shape} vs Checkpoint {v.shape}. Skipping this layer.")
#       continue
#     new_state_dict[name] = v
#
#   model.load_state_dict(new_state_dict, strict=False)
#   logger.info(" => ✅ Weights successfully aligned and loaded.")
#
#   # 3. 动态配置目标测试数据集
#   target_dataset_class = globals().get(args.dataset + 'Dataset', None)
#   if target_dataset_class is None:
#     raise ValueError(f"❌ Unknown target dataset type: {args.dataset}")
#
#   # 跨数据集推荐验证该数据集的全集（'all_0' 等），由 subset 参数传入决定
#   db_split_name = args.subset + '_' + str(args.pro)
#   logger.info(f"📦 Constructing Target Dataset: {args.dataset} ({db_split_name})")
#
#   test_dataset = target_dataset_class(
#     args.data_dir,
#     db_split_name,
#     list_dir=args.list_dir,
#     patch_size=args.patch_size,
#     n_ptchs=args.n_ptchs_per_img,
#     n_class=args.target_n_dtype
#   )
#
#   test_loader = torch.utils.data.DataLoader(
#     test_dataset,
#     batch_size=1,
#     shuffle=False,
#     num_workers=args.workers,
#     pin_memory=True
#   )
#
#   # 4. 执行测试
#   test_cross(test_loader, model, args.dataset)
#
#
# def parse_args():
#   parser = argparse.ArgumentParser(description='Aligned Ultra Robust IQA Cross-Dataset Test Script')
#
#   # --- 运行设备与控制 ---
#   parser.add_argument('--workers', type=int, default=4)
#   parser.add_argument('--pro', type=int, default=0, help='Split process index (e.g., 0 for all_0)')
#   parser.add_argument('--subset', default='all', type=str, help='Default "all" to evaluate the entire cross dataset')
#
#   # --- 目标被测数据集信息 (Target Dataset) ---
#   parser.add_argument('--dataset', type=str, default='SIQAD', choices=['SCID', 'SIQAD'],
#                       help='Target dataset you want to TEST on')
#   parser.add_argument('-d', '--data-dir', default='../../../datasets',
#                       help='Base dir path where target dataset resides')
#   parser.add_argument('-l', '--list-dir', default='../sci_scripts/siqad-scripts-all/',
#                       help='Txt list directory of the target dataset')
#   parser.add_argument('--target-n-dtype', type=int, default=7,
#                       help='Number of distortions of TARGET dataset (SIQAD: 7, SCID: 9)')
#
#   # --- 源模型和训练权重配置 (Source Model/Checkpoint) ---
#   parser.add_argument('--resume', default='../models/scid/model_best_1.pkl', type=str,
#                       help='Path to the BEST checkpoint trained on the SOURCE dataset')
#   parser.add_argument('--source-n-dtype', type=int, default=9,
#                       help='Number of distortions of SOURCE dataset when trained (Must match checkpoint head!)')
#
#   # --- 架构超参数 (必须与你训练该 Checkpoint 时的模型架构配置完全对齐) ---
#   parser.add_argument('--use_dfss_mamba', action='store_true', default=False,
#                       help='Must match your trained model backbone type')
#   parser.add_argument('--mamba_img_size', type=int, default=112)
#   parser.add_argument('-n', '--n-ptchs-per-img', type=int, default=32)
#   parser.add_argument('-psz', '--patch-size', type=int, default=32)
#
#   args = parser.parse_args()
#   return args
#
#
# if __name__ == '__main__':
#   main()





























# #!/usr/bin/env python3
# """
# Main Script
# """
#
# import sys
# import os
#
# import shutil
# import argparse
#
# import torch
# import torch.backends.cudnn as cudnn
# from torch import nn
#
# from demomodel import IQANet
# from dataset import TID2013Dataset, IQADataset
# from utils import AverageMeter, SROCC, PLCC, RMSE
# from utils import SimpleProgressBar as ProgressBar
#
#
# def test(test_data_loader, model):
#   srocc = SROCC()
#   plcc = PLCC()
#   rmse = RMSE()
#   len_test = len(test_data_loader)
#   pb = ProgressBar(len_test, show_step=True)
#
#   print("Testing")
#
#   model.eval()
#   with torch.no_grad():
#     for i, (img, ref, pref, score, dtype) in enumerate(test_data_loader):
#       img = img.cuda()
#       ref = ref.cuda()
#       pref = pref.cuda()
#       output = model(img, img, img).squeeze()
#
#       output = output.cpu().data.numpy()
#       score = score.data.numpy()
#
#       srocc.update(score, output)
#       plcc.update(score, output)
#       rmse.update(score, output)
#
#       pb.show(i, "Test: [{0:5d}/{1:5d}]\t"
#                  "Score: {2:.4f}\t"
#                  "Label: {3:.4f}"
#               .format(i + 1, len_test, float(output), float(score)))
#
#   print("\n\nSROCC: {0:.4f}\n"
#         "PLCC: {1:.4f}\n"
#         "RMSE: {2:.4f}"
#         .format(srocc.compute(), plcc.compute(), rmse.compute())
#         )
#
#
# def test_iqa(args):
#   batch_size = 1
#   pro = args.pro
#   num_workers = args.workers
#   subset = args.subset
#   data_dir = args.data_dir
#   list_dir = args.list_dir
#   resume = args.resume
#   patch_size = args.patch_size
#   n_ptchs = args.n_ptchs_per_img
#   dtypes = args.n_dtype
#
#   for k, v in args.__dict__.items():
#     print(k, ':', v)
#
#   model = IQANet(istrain=False, n_class=dtypes)
#   model = nn.DataParallel(model)
#   test_loader = torch.utils.data.DataLoader(
#     Dataset(data_dir, 'test_' + str(pro), list_dir=list_dir, patch_size=patch_size,
#             n_ptchs=n_ptchs),
#     batch_size=1, shuffle=False, num_workers=0,
#     pin_memory=True
#   )
#
#   cudnn.benchmark = True
#
#   # Resume from a checkpoint
#   if resume:
#     resume = resume.split('t.')[0] + 't_' + str(pro) + '.pkl'
#     # resume ='../models/checkpoint_latest_0.pkl'
#     if os.path.isfile(resume):
#       print("=> loading checkpoint '{}'".format(resume))
#       checkpoint = torch.load(resume)
#       model.load_state_dict(checkpoint['state_dict'])
#       print("=> loaded checkpoint '{}' (epoch {})"
#             .format(resume, checkpoint['epoch']))
#     else:
#       print("=> no checkpoint found at '{}'".format(resume))
#
#   test(test_loader, model.cuda())
#
#
# # def parse_args():
# #     # Training settings
# #     parser = argparse.ArgumentParser(description='')
# #     parser.add_argument('-cmd', type=str,default='test')
# #     parser.add_argument('-d', '--data-dir', default='../../../datasets/SCID/')
# #     parser.add_argument('-l', '--list-dir', default='../sci_scripts/siqad-scripts-all/',
# #                         help='List dir to look for train_images.txt etc. '
# #                               'It is the same with --data-dir if not set.')
# #     parser.add_argument('-n', '--n-ptchs-per-img', type=int, default=1024, metavar='N',
# #                         help='number of patches for each image (default: 32)')
# #     parser.add_argument('-nd', '--n-dtype', type=int, default=50, metavar='N',
# #                         help='number of distortion types (siqad:7，scid:10)')
# #     parser.add_argument('-psz', '--patch-size', type=int, default=32, metavar='N',
# #                         help='size of cropped patches for each image (default: 64)')
# #     parser.add_argument('--step', type=int, default=200)
# #     parser.add_argument('--batch-size', type=int, default=32, metavar='B',
# #                         help='input batch size for training (default: 64)')
# #     parser.add_argument('--epochs', type=int, default=1000, metavar='NE',
# #                         help='number of epochs to train (default: 1000)')
# #     parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
# #                         help='learning rate (default: 1e-4)')
# #     parser.add_argument('--lr-mode', type=str, default='const')
# #     parser.add_argument('--weight-decay', default=1e-4, type=float,
# #                         metavar='W', help='weight decay (default: 1e-4)')
# #     parser.add_argument('--resume', default='../models/siqad-all/checkpoint_latest.pkl',type=str, metavar='PATH',
# #             help='path to latest checkpoint')
# #     parser.add_argument('--workers', type=int, default=8)
# #     parser.add_argument('--pro', type=int, default=0)
# #     parser.add_argument('--subset', default='test')
# #     parser.add_argument('--evaluate', dest='evaluate',
# #                         action='store_true',
# #                         help='evaluate model on validation set')
# #     parser.add_argument('--weighted',default=True, dest='weighted',
# #             action='store_true')
# #     parser.add_argument('--dump_per', type=int, default=50,
# #                         help='the number of epochs to make a checkpoint')
# #     parser.add_argument('--dataset', type=str, default='IQA')
# #     parser.add_argument('--anew', action='store_true')
#
# #     args = parser.parse_args()
#
# #     return args
#
# def parse_args():
#   # Training settings
#   parser = argparse.ArgumentParser(description='')
#   parser.add_argument('-cmd', type=str, default='test')
#   parser.add_argument('-d', '--data-dir', default='../../../datasets/SIQAD/')
#   parser.add_argument('-l', '--list-dir', default='../sci_scripts/scid-scripts-all/',
#                       help='List dir to look for train_images.txt etc. '
#                            'It is the same with --data-dir if not set.')
#   parser.add_argument('-n', '--n-ptchs-per-img', type=int, default=1024, metavar='N',
#                       help='number of patches for each image (default: 32)')
#   parser.add_argument('-nd', '--n-dtype', type=int, default=46, metavar='N',
#                       help='number of distortion types (siqad:7，scid:10)')
#   parser.add_argument('-psz', '--patch-size', type=int, default=32, metavar='N',
#                       help='size of cropped patches for each image (default: 64)')
#   parser.add_argument('--step', type=int, default=200)
#   parser.add_argument('--batch-size', type=int, default=32, metavar='B',
#                       help='input batch size for training (default: 64)')
#   parser.add_argument('--epochs', type=int, default=1000, metavar='NE',
#                       help='number of epochs to train (default: 1000)')
#   parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
#                       help='learning rate (default: 1e-4)')
#   parser.add_argument('--lr-mode', type=str, default='const')
#   parser.add_argument('--weight-decay', default=1e-4, type=float,
#                       metavar='W', help='weight decay (default: 1e-4)')
#   parser.add_argument('--resume', default='../models/scid-all/checkpoint_latest.pkl', type=str, metavar='PATH',
#                       help='path to latest checkpoint')
#   parser.add_argument('--workers', type=int, default=8)
#   parser.add_argument('--pro', type=int, default=0)
#   parser.add_argument('--subset', default='test')
#   parser.add_argument('--evaluate', dest='evaluate',
#                       action='store_true',
#                       help='evaluate model on validation set')
#   parser.add_argument('--weighted', default=True, dest='weighted',
#                       action='store_true')
#   parser.add_argument('--dump_per', type=int, default=50,
#                       help='the number of epochs to make a checkpoint')
#   parser.add_argument('--dataset', type=str, default='IQA')
#   parser.add_argument('--anew', action='store_true')
#
#   args = parser.parse_args()
#
#   return args
#
#
# def main():
#   args = parse_args()
#   # Choose dataset
#   global Dataset
#   Dataset = globals().get(args.dataset + 'Dataset', None)
#   test_iqa(args)
#
#
# if __name__ == '__main__':
#   main()
