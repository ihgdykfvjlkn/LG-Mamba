# 这是我的主程序
import argparse
import csv
import datetime
import gc
import logging
import math
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from dataset import IQADataset, SCIDDataset, SIQADDataset
from demomodel import DFSSMambaNet, IQANet
from scipy.stats import pearsonr, spearmanr
from src.utils import PLCC, RMSE, SROCC, AverageMeter, MMD_loss
from tqdm import tqdm

# ========================================
# CRITICAL FIX: NO MORE FREEZING ON WINDOWS!
# ========================================
if sys.platform == 'win32':
  import torch.multiprocessing as mp
  try:
    mp.set_start_method('spawn', force=True)
  except:
    pass

torch.backends.cudnn.benchmark = True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def seed_everything(seed=42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


seed_everything(42)

logger = None
log_file_path = None


def setup_logger(project_root, dataset_name=None):
  global logger, log_file_path

  if logger is None:
    logger = logging.getLogger('DFSS_IQA_Scratch')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
      console_handler = logging.StreamHandler(sys.stdout)
      console_handler.setFormatter(logging.Formatter('%(message)s'))
      logger.addHandler(console_handler)

  if dataset_name is not None:
    log_dir = os.path.join(project_root, 'training_logs', dataset_name.lower())
    os.makedirs(log_dir, exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'scratch_run_{timestamp}.log')
    log_file_path = log_file

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)
    return log_file

  return None


class NMSELoss(torch.nn.Module):

  def __init__(self):
    super(NMSELoss, self).__init__()

  def forward(self, x, y):
    return torch.mean((x - y) ** 2) / torch.mean(y**2)


def gaussian_batch(custom_feat):
  custom_feat = custom_feat.view(custom_feat.shape[0], -1)
  dims = custom_feat.shape
  lenth = dims[0] * dims[1]
  inv = torch.normal(mean=0, std=torch.ones(lenth)).to(custom_feat.device)
  return custom_feat, inv.view_as(torch.Tensor(dims[0], dims[1]))


def validate(
    val_loader, model, criterion, show_step=False, dataset_name='SIQAD'
):
  losses = AverageMeter()
  srocc = SROCC()
  plcc = PLCC()
  rmse = RMSE()

  dataset_mapping = {
      'SIQAD': {
          0: 'GN',
          1: 'GB',
          2: 'JPEG',
          3: 'JP2K',
          4: 'LSC',
          5: 'CC',
          6: 'MAR',
      },
      'SCID': {
          0: 'GN',
          1: 'GB',
          2: 'MB',
          3: 'CC',
          4: 'SC',
          5: 'CS',
          6: 'JPEG',
          7: 'J2K',
          8: 'HEVC',
      },
  }

  dist_names = dataset_mapping.get(dataset_name, {})
  dist_scores = {}

  logger.info('Validation...')
  model.eval()

  pbar = tqdm(
      enumerate(val_loader),
      total=len(val_loader),
      desc='Validating',
      file=sys.stdout,
      leave=True,
  )

  with torch.no_grad():
    for i, (img, ref, pref, score, dtype) in pbar:
      img = img.cuda(non_blocking=True)
      ref = ref.cuda(non_blocking=True)
      pref = pref.cuda(non_blocking=True)
      score = score.cuda(non_blocking=True)
      dtype = dtype.cuda(non_blocking=True)

      output, _, _, _, _, _, _, _, _ = model(img, ref, pref)
      loss = criterion(output.squeeze(), score.squeeze())

      losses.update(loss.item(), img.size(0))

      pred = output.detach().cpu().numpy().reshape(-1)
      gt = score.detach().cpu().numpy().reshape(-1)
      dtype_np = dtype.cpu().numpy().reshape(-1)

      valid = np.isfinite(pred) & np.isfinite(gt)
      if valid.sum() == 0:
        continue

      pred = pred[valid]
      gt = gt[valid]
      dtype_np = dtype_np[valid]

      srocc.update(gt, pred)
      plcc.update(gt, pred)
      rmse.update(gt, pred)

      for g, p, d in zip(gt, pred, dtype_np):
        if int(d) not in dist_scores:
          dist_scores[int(d)] = {'gt': [], 'pred': []}
        dist_scores[int(d)]['gt'].append(g)
        dist_scores[int(d)]['pred'].append(p)

      pbar.set_postfix(Loss=f'{losses.avg:.4f}')

  final_srocc = float(srocc.compute())
  final_plcc = float(plcc.compute())
  final_rmse = float(rmse.compute())

  logger.info('\n📊 Validation Complete!')
  logger.info(f'   SROCC: {final_srocc:.6f}')
  logger.info(f'   PLCC : {final_plcc:.6f}')
  logger.info(f'   RMSE : {final_rmse:.6f}')

  logger.info('\n' + '=' * 60)
  logger.info(
      f'📊 PERFORMANCE ON INDIVIDUAL DISTORTION TYPES ({dataset_name})'
  )
  logger.info('=' * 60)
  logger.info(f"{'Type':<15}{'SROCC':<12}{'PLCC':<12}{'Samples'}")
  logger.info('-' * 60)

  for d in sorted(dist_scores.keys()):
    gt = np.array(dist_scores[d]['gt'])
    pred = np.array(dist_scores[d]['pred'])

    if len(gt) < 2:
      continue

    try:
      s = spearmanr(gt, pred)[0]
      p = pearsonr(gt, pred)[0]
    except:
      s = np.nan
      p = np.nan

    logger.info(
        f"{dist_names.get(d, str(d)):<15}{s:<12.4f}{p:<12.4f}{len(gt)}"
    )

  logger.info('=' * 60)
  return final_srocc, final_plcc, final_rmse


def train(train_loader, model, criterion, optimizer, epoch, scaler=None):
  losses1 = AverageMeter()
  losses2 = AverageMeter()
  losses3 = AverageMeter()
  losses4 = AverageMeter()

  logger.info('=' * 70)
  logger.info(f'Training Epoch {epoch} (Scratch)')
  logger.info('=' * 70)

  model.train()
  criterion = criterion.cuda()

  triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2).cuda()
  mse_loss = nn.MSELoss().cuda()
  ce_loss = nn.CrossEntropyLoss().cuda()
  mmdloss = MMD_loss().cuda()

  pbar = tqdm(
      enumerate(train_loader),
      total=len(train_loader),
      desc=f'Epoch {epoch} Train',
      file=sys.stdout,
      leave=True,
  )

  for i, (img, ref, pref, score, dtype) in pbar:
    img = img.cuda(non_blocking=True)
    ref = ref.cuda(non_blocking=True)
    pref = pref.cuda(non_blocking=True)
    score = score.cuda(non_blocking=True)
    dtype = dtype.cuda(non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    use_amp = scaler is not None

    with torch.amp.autocast('cuda', enabled=use_amp):
      (
          pref_mos,
          img_ss,
          ref_ss,
          pref_ss,
          ref_si,
          img_si,
          diff_ref_si,
          diff_pref_si,
          dtype_pred,
      ) = model(img, ref, pref)

      pref_mos = torch.clamp(pref_mos, -100, 100)

      loss1 = criterion(pref_mos.squeeze(), score.squeeze())
      loss2 = triplet_loss(ref_ss, img_ss, pref_ss)

      img_sf_mean = torch.mean(img_si, dim=1, keepdim=True)
      img_sf_std = torch.std(img_si, dim=1, keepdim=True) + 1e-6
      img_sf_std = torch.clamp(img_sf_std, min=1e-4)

      all_sf = (img_si - img_sf_mean) / img_sf_std
      all_sf = torch.clamp(all_sf, -10, 10)

      all_sf_gaus, all_sf = gaussian_batch(all_sf)

      loss3 = 0.0001 * mmdloss(all_sf, all_sf_gaus)
      loss3 += 0.1 * (
          diff_ref_si.pow(2).mean()
          + diff_pref_si.pow(2).mean()
          + mse_loss(diff_ref_si, diff_pref_si)
      )

      loss4 = 0.1 * ce_loss(dtype_pred, dtype)
      loss = loss1 + loss2 + loss3 + loss4

    skip = False
    for name, l in {
        'MOS': loss1,
        'Triplet': loss2,
        'MMD': loss3,
        'DType': loss4,
        'Total': loss,
    }.items():
      if not torch.isfinite(l):
        logger.warning(f'Batch {i}: {name} Loss is NaN/Inf, Skip.')
        skip = True
        break

    if skip:
      optimizer.zero_grad(set_to_none=True)
      continue

    if use_amp:
      scaler.scale(loss).backward()
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      scaler.step(optimizer)
      scaler.update()
    else:
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()

    losses1.update(loss1.item(), img.size(0))
    losses2.update(loss2.item(), img.size(0))
    losses3.update(loss3.item(), img.size(0))
    losses4.update(loss4.item(), img.size(0))

    pbar.set_postfix({
        'MOS': f'{losses1.avg:.4f}',
        'SS': f'{losses2.avg:.4f}',
        'Gaus': f'{losses3.avg:.4f}',
        'DType': f'{losses4.avg:.4f}',
    })

  return losses1.avg



def apply_stage1_freezing(net):
  """Stage 1: 冻结整个 MobileNet 骨干网络（CNN），只训练 Mamba、Attention 及 Head。"""
  logger.info(
      '🔒 [Stage 1] 锁定 MobileNet 全部参数，释放 Mamba/Attention/Head...'
  )
  for name, param in net.named_parameters():
    if 'cnn' in name:
      param.requires_grad = False
    else:
      param.requires_grad = True


def apply_stage2_freezing(net):
  """Stage 2: 精准解冻。冻结 MobileNet 浅层 (features 0-5)，解冻 features 6+ 及后续模块。"""
  logger.info(
      '🔓 [Stage 2] 正在应用精细化解冻 (浅层 features[0:5] 保持冻结)...'
  )
  for name, param in net.named_parameters():
    if 'cnn' in name:
      is_shallow_layer = False
      for i in range(6):
        if f'.features.{i}.' in name or f'features.{i}.' in name:
          is_shallow_layer = True
          break

      if is_shallow_layer:
        param.requires_grad = False
      else:
        param.requires_grad = True
    else:
      param.requires_grad = True


def build_stage2_optimizer(net, weight_decay_val):
  """构建 Stage 2 精细化三段式差分 LR 优化器。"""
  mobilenet_params = []
  mamba_params = []
  head_params = []

  for name, param in net.named_parameters():
    if not param.requires_grad:
      continue

    if 'cnn' in name:
      mobilenet_params.append(param)
    elif any(
        keyword in name for keyword in ['distype_cls', 'regression', 'head']
    ):
      head_params.append(param)
    else:
      mamba_params.append(param)

  param_groups = [
      {'params': mobilenet_params, 'lr': 1e-6, 'name': 'MobileNet_Unfrozen'},
      {'params': mamba_params, 'lr': 5e-6, 'name': 'Mamba_Attention'},
      {'params': head_params, 'lr': 1e-5, 'name': 'Regression_Head'},
  ]

  logger.info('[Stage 2 优化器构建完成] 参数组分布与学习率设置:')
  logger.info(
      f'   🔹 MobileNet (后几层解冻): {len(mobilenet_params)} 个张量 -> LR = 1e-6'
  )
  logger.info(
      f'   🔹 Mamba / Fusion 模块 : {len(mamba_params)} 个张量 -> LR = 5e-6'
  )
  logger.info(
      f'   🔹 Regression / Head 模块 : {len(head_params)} 个张量 -> LR = 1e-5'
  )

  return torch.optim.Adam(
      param_groups, betas=(0.9, 0.999), weight_decay=weight_decay_val
  )


def train_iqa(args):
  global logger, log_file_path

  pro = args.pro
  batch_size = args.batch_size
  num_workers = args.workers
  data_dir = args.data_dir
  list_dir = args.list_dir
  resume = args.resume
  n_ptchs = args.n_ptchs_per_img
  patch_size = args.patch_size
  dtypes = args.n_dtype
  use_dfss_mamba = args.use_dfss_mamba
  mamba_img_size = args.mamba_img_size
  use_amp = args.use_amp
  stage1_epochs = args.stage1_epochs

  gc.collect()
  torch.cuda.empty_cache()

  if use_dfss_mamba:
    print(' Using DFSSMambaNet (Scratch Training)')
    model = DFSSMambaNet(
        istrain=True, n_class=dtypes, mamba_img_size=mamba_img_size
    )
  else:
    model = IQANet(istrain=True, n_class=dtypes)

  model = model.cuda()
  model = nn.DataParallel(model)

  criterion = nn.L1Loss()
  scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

  actual_dataset_class = globals().get(args.dataset + 'Dataset', None)
  if actual_dataset_class is None:
    raise ValueError(f'Unknown dataset: {args.dataset}')

  train_dataset = actual_dataset_class(
      data_dir,
      'train_' + str(pro),
      list_dir=list_dir,
      patch_size=patch_size,
      n_ptchs=n_ptchs,
      n_class=dtypes,
  )


  val_phase_name = f'test_{pro}'
  check_val_path = os.path.join(list_dir, f'{val_phase_name}_data.json')
  if not os.path.exists(check_val_path):
    val_phase_name = f'val_{pro}'

  val_dataset = actual_dataset_class(
      data_dir,
      val_phase_name,
      list_dir=list_dir,
      patch_size=patch_size,
      n_ptchs=n_ptchs,
      n_class=dtypes,
  )

  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
      persistent_workers=True if num_workers > 0 else False,
      prefetch_factor=4 if num_workers > 0 else None,
      drop_last=True,
  )
  val_loader = torch.utils.data.DataLoader(
      val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
  )

  logger.info('\n' + '🔥 ' * 25)
  logger.info('⚙️  EXPERIMENTAL CONFIGURATION SECURE CHECK (SCRATCH)')
  logger.info('🔥 ' * 25)
  logger.info(f'   🎯 Target Dataset:      {args.dataset}')
  logger.info(
      f'   📊 Train Dataset Size:  {len(train_dataset)} samples (Total'
      f' Batches: {len(train_loader)})'
  )
  logger.info(
      f'   🧪 Val Dataset Size:    {len(val_dataset)} samples (Phase:'
      f' {val_phase_name})'
  )
  logger.info(f'   📦 Runtime Batch Size:  {batch_size}')
  logger.info(f'   🧩 Stage 1 Epochs:     {stage1_epochs}')
  logger.info('=' * 50 + '\n')

  start_epoch = 0
  best_srocc = -1.0
  best_plcc = -1.0
  best_rmse = 99.0
  checkpoint = None

  patience = 40
  no_improve = 0

  raw_model = model.module if hasattr(model, 'module') else model

  apply_stage1_freezing(raw_model)
  optimizer = torch.optim.Adam(
      [p for p in raw_model.parameters() if p.requires_grad],
      lr=args.lr,
      betas=(0.9, 0.999),
      weight_decay=args.weight_decay,
  )


  base_models_dir = (
      '/mnt/h/25liuyx/projects/DFSS-IQA-main/DFSS_Release/models'
  )
  default_split_dir = os.path.join(
      base_models_dir, args.dataset.lower(), f'split_{pro}'
  )
  default_split_ckpt = os.path.join(
      default_split_dir, 'checkpoint_latest.pkl'
  )

  resume_file = None
  if not args.anew:
    if resume:
      resume_file = os.path.abspath(os.path.expanduser(resume))
      if not os.path.isfile(resume_file):
        raise FileNotFoundError(
            f'--resume 指定的权重文件不存在: {resume_file}'
        )
    elif os.path.isfile(default_split_ckpt):
      resume_file = default_split_ckpt

  if args.save_dir:
    run_save_dir = os.path.abspath(os.path.expanduser(args.save_dir))
  elif resume_file is not None and resume:
    run_save_dir = os.path.dirname(resume_file)
  else:
    run_save_dir = default_split_dir

  os.makedirs(run_save_dir, exist_ok=True)
  logger.info(f'本次运行的权重保存目录: {run_save_dir}')

  if resume_file is not None:
    if os.path.isfile(resume_file):
      logger.info(f"正在为您加载历史权重: '{resume_file}'")
      checkpoint = torch.load(resume_file, map_location='cuda')

      if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        start_epoch = checkpoint.get(
            'history_base_epoch', checkpoint.get('epoch', 0)
        )
        best_srocc = checkpoint.get(
            'min_loss', checkpoint.get('best_srocc', -1.0)
        )
        best_plcc = checkpoint.get('best_plcc', -1.0)
      else:
        state_dict = checkpoint
        start_epoch = 0

      from collections import OrderedDict

      new_state_dict = OrderedDict()
      for k, v in state_dict.items():
        name = k if k.startswith('module.') else 'module.' + k
        if any(layer in name for layer in ['distype_cls', 'regression']):
          current_param = model.state_dict().get(name, None)
          if current_param is not None and current_param.shape != v.shape:
            logger.info(f'  [跳过不兼容权重] 层 {name} 形状不匹配，已重置')
            continue
        new_state_dict[name] = v

      model.load_state_dict(new_state_dict, strict=False)
      logger.info(f' =>  历史权重接力成功！将从 Epoch {start_epoch} 继续！')

      if 'torch_rng' in checkpoint:
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
      if 'cuda_rng' in checkpoint:
        torch.cuda.set_rng_state(checkpoint['cuda_rng'].cpu())
      if 'numpy_rng' in checkpoint:
        np.random.set_state(checkpoint['numpy_rng'])
      if 'python_rng' in checkpoint:
        random.setstate(checkpoint['python_rng'])
  else:
    logger.info('=>  从零开始训练 (Scratch)，全新初始化网络...')

  if args.evaluate:
    validate(val_loader, model.cuda(), criterion, dataset_name=args.dataset)
    return

  csv_file_path = os.path.join(
      os.path.dirname(log_file_path),
      f"metrics_record_{time.strftime('%Y%m%d_%H%M%S')}.csv",
  )

  if start_epoch == 0:
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as f:
      writer = csv.writer(f)
      writer.writerow([
          'Epoch',
          'Stage',
          'Train_MOS_Loss',
          'Val_SROCC',
          'Val_PLCC',
          'Val_RMSE',
          'Best_SROCC',
          'Best_PLCC',
      ])

  stage2_opt_initialized = False


  for epoch in range(start_epoch, args.epochs):

    gc.collect()
    torch.cuda.empty_cache()


    if epoch < stage1_epochs:
      current_stage = 'Stage1'
      apply_stage1_freezing(raw_model)

      current_lr = args.lr if epoch < 8 else args.lr * 0.5
      for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    else:
      current_stage = 'Stage2'
      if not stage2_opt_initialized:
        apply_stage2_freezing(raw_model)

        old_opt_state = optimizer.state_dict()
        stage2_optimizer = build_stage2_optimizer(
            raw_model, args.weight_decay
        )

        try:
          old_param_map = {
              id(p): p
              for group in optimizer.param_groups
              for p in group['params']
          }
          new_param_map = {
              id(p): p
              for group in stage2_optimizer.param_groups
              for p in group['params']
          }

          migrated_count = 0
          for p_id, p in new_param_map.items():
            if (
                p_id in old_param_map
                and old_param_map[p_id] in old_opt_state['state']
            ):
              stage2_optimizer.state[p] = old_opt_state['state'][
                  old_param_map[p_id]
              ]
              migrated_count += 1
          logger.info(
              f' [动量继承成功] 已平滑迁移 {migrated_count} 个参数的 Adam'
              ' 历史状态！'
          )
        except Exception as e:
          logger.warning(
              f' 动量迁移跳过 (将以全新状态启动 Stage 2 优化器): {e}'
          )

        optimizer = stage2_optimizer
        stage2_opt_initialized = True
      else:
        apply_stage2_freezing(raw_model)

    logger.info(f'\nEpoch: [{epoch}]\tStage: {current_stage}')


    train_loss = train(
        train_loader, model, criterion, optimizer, epoch, scaler
    )

    try:
      logger.info('\n📊 Validating...')
      srocc, plcc, rmse = validate(
          val_loader, model.cuda(), criterion, dataset_name=args.dataset
      )
    except Exception as e:
      logger.error(
          f'🚨 [CRITICAL] Validation crashed at epoch {epoch}: {e}. Skipping'
          ' metrics pass.'
      )
      continue


    is_best = False
    if srocc > best_srocc:
      best_srocc = srocc
      best_plcc = plcc
      best_rmse = rmse
      is_best = True
      no_improve = 0
    else:
      no_improve += 1

    logger.info('\n' + '=' * 80)
    logger.info(f'📊 Epoch {epoch} ({current_stage}) 运行结果:')
    logger.info(
        f'   Current SROCC:  {srocc:.6f} | Current PLCC:   {plcc:.6f} | Current'
        f' RMSE:   {rmse:.6f}'
    )
    logger.info(
        f'    历史最高标准 SROCC: {best_srocc:.6f} (伴随 PLCC: {best_plcc:.6f} |'
        f' RMSE: {best_rmse:.6f})'
    )
    logger.info('=' * 80)


    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as f:
      writer = csv.writer(f)
      writer.writerow([
          epoch,
          current_stage,
          f'{train_loss:.4f}' if train_loss is not None else '0.0000',
          f'{srocc:.6f}',
          f'{plcc:.6f}',
          f'{rmse:.6f}',
          f'{best_srocc:.6f}',
          f'{best_plcc:.6f}',
      ])


    checkpoint_path = os.path.join(run_save_dir, 'checkpoint_latest.pkl')

    save_checkpoint(
        {
            'epoch': epoch + 1,
            'history_base_epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': None,
            'min_loss': best_srocc,
            'best_srocc': best_srocc,
            'best_plcc': best_plcc,
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state(),
            'numpy_rng': np.random.get_state(),
            'python_rng': random.getstate(),
        },
        is_best,
        filename=checkpoint_path,
        pro=args.pro,
        dataset_name=args.dataset.lower(),
    )

    if no_improve >= patience:
      logger.info(
          f'🛑 [EARLY STOPPING] 连续 {patience} 轮无正面 SROCC 效益提升，终止训练。'
      )
      break



def save_checkpoint(
    state,
    is_best,
    filename='checkpoint_latest.pkl',
    pro='1',
    dataset_name='scid',
    exp_name=None,
):
  if exp_name is None:
    if 'args' in globals() and hasattr(globals()['args'], 'exp_name'):
      exp_name = globals()['args'].exp_name
    else:
      exp_name = 'DFSS_Mamba_Scratch_Experiment'


  save_filename = os.path.abspath(os.path.expanduser(filename))
  split_dir = os.path.dirname(save_filename)
  os.makedirs(split_dir, exist_ok=True)

  torch.save(state, save_filename)
  logger.info(f'💾 checkpoint saved: {save_filename}')

  if is_best:
    best_path = os.path.join(split_dir, 'model_best.pkl')
    torch.save(state, best_path)
    logger.info(f'🏆 best model saved: {best_path}')

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_filename = f'{dataset_name.upper()}_{exp_name}_best_{timestamp}.pkl'
    backup_path = os.path.join(split_dir, backup_filename)
    shutil.copyfile(best_path, backup_path)
    logger.info(f'📦 自动标识备份最佳模型至: {backup_path}')


def parse_args():
  parser = argparse.ArgumentParser(
      description='Ultra Robust IQA Scratch Training Script'
  )
  parser.add_argument(
      '-cmd', type=str, default='train', choices=['train', 'test']
  )
  parser.add_argument('--anew', action='store_true', default=False)
  parser.add_argument(
      '--dataset', type=str, default='SCID', choices=['SCID', 'SIQAD']
  )
  args, remaining_args = parser.parse_known_args()
  current_dataset = args.dataset

  release_root_dir = '/mnt/h/25liuyx/projects/DFSS-IQA-main/DFSS_Release'
  main_root_dir = '/mnt/h/25liuyx/projects/DFSS-IQA-main'
  default_data_dir = os.path.join(main_root_dir, 'datasets')

  if current_dataset == 'SCID':
    default_list_dir = os.path.join(
        release_root_dir, 'sci_scripts/scid-scripts-8-2/'
    )
    default_dtype = 9
    default_resume = os.path.join(
        release_root_dir, 'models/scid/checkpoint_latest.pkl'
    )
  else:
    default_list_dir = os.path.join(
        release_root_dir, 'sci_scripts/siqad-scripts-8-2/'
    )
    default_dtype = 7
    default_resume = os.path.join(
        release_root_dir, 'models/siqad/checkpoint_latest.pkl'
    )

  global log_file_path
  log_file_path = setup_logger(release_root_dir, dataset_name=current_dataset)

  logger.info('=' * 60)
  logger.info(
      f'当前目标数据集: {current_dataset} (Scratch Mode)'
  )
  logger.info(f'   数据目录: {default_data_dir}')
  logger.info(f'   脚本目录: {default_list_dir}')
  logger.info(f'   分类头数: {default_dtype} | 默认权重: {default_resume}')
  logger.info('=' * 60)

  parser.add_argument('-d', '--data-dir', default=default_data_dir)
  parser.add_argument('-l', '--list-dir', default=default_list_dir)
  parser.add_argument('-n', '--n-ptchs-per-img', type=int, default=32)
  parser.add_argument('-nd', '--n-dtype', type=int, default=default_dtype)

  parser.add_argument('-psz', '--patch-size', type=int, default=32)
  parser.add_argument('--step', type=int, default=200)
  parser.add_argument('--batch-size', type=int, default=16)
  parser.add_argument('--epochs', type=int, default=101)

  # 默认实验标识加上 Scratch
  parser.add_argument(
      '--exp_name',
      type=str,
      default='DFSS_Mamba_MobileNet_Scratch_Stage2',
      help='实验版本标识，用于自动备份命名',
  )
  parser.add_argument('--stage1_epochs', type=int, default=15)
  parser.add_argument('--lr', type=float, default=1e-4)
  parser.add_argument('--weight-decay', default=1e-4, type=float)
  parser.add_argument(
      '--resume',
      default=None,
      type=str,
      help=(
          '要续训的 checkpoint 完整路径；指定后优先于默认 split '
          '断点'
      ),
  )
  parser.add_argument(
      '--save-dir',
      default=None,
      type=str,
      help=(
          '本次运行的 checkpoint 保存目录。未指定时，显式续训将'
          '保存到 --resume 文件所在目录'
      ),
  )

  parser.add_argument('--pro', type=int, default=1)
  parser.add_argument('--workers', type=int, default=6)
  parser.add_argument('--subset', default='test')
  parser.add_argument('--evaluate', dest='evaluate', action='store_true')
  parser.add_argument('--weighted', default=True, action='store_true')
  parser.add_argument('--dump_per', type=int, default=50)

  parser.add_argument('--use_dfss_mamba', action='store_true', default=True)
  parser.add_argument('--mamba_img_size', type=int, default=112)
  parser.add_argument('--use_amp', action='store_true', default=False)

  args = parser.parse_args()
  return args


def main():
  args = parse_args()
  if args.cmd == 'train':
    train_iqa(args)
