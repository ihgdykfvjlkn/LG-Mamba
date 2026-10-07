"""
Dataset and Transforms - Streamlined for Pure Mamba Model
[SPEEDUP]: Integrated Image RAM Caching and verified parser to fix Windows CPU bottlenecks.
[NR FIX]: Reference and pseudo-reference images are training-only auxiliaries.
"""

import argparse
import json
import os
from os.path import exists, join
import random
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import torch
import torch.utils.data
import torchvision
from torchvision import transforms

# from src.utils import limited_instances, SimpleProgressBar
from src.utils import SimpleProgressBar, limited_instances


class IQADataset(torch.utils.data.Dataset):

  def __init__(

      self,

      data_dir,

      phase,

      patch_size=64,

      n_ptchs=256,

      n_class=46,

      sample_once=False,

      subset='',

      list_dir='',

  ):
    super(IQADataset, self).__init__()

    self.list_dir = data_dir if not list_dir else list_dir
    self.data_dir = data_dir
    self.phase = phase
    self.subset = phase if not subset.strip() else subset
    self.n_ptchs = n_ptchs
    self.img_list = []
    self.ref_list = []
    self.score_list = []
    self.dtype_list = []  # 存失真类型列表
    self.sample_once = sample_once
    self._from_pool = False
    self.patch_size = patch_size
    self.n_class = n_class

    # 👑 提速核心 1：开辟 RAM 全量高速缓存字典，消除每轮 Epoch 重复读取磁盘的噩梦
    self.image_cache = {}

    if n_class == 46:
      self.n_levels = 5
    elif n_class == 7:
      self.n_levels = 1
    elif n_class == 50:
      self.n_levels = 7
    else:
      self.n_levels = 1

    self._read_lists()
    self._aug_lists()

    self.normal = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )

    self.tfs = Transforms()
    if sample_once:

      @limited_instances(self.__len__())
      class IncrementCache:

        def store(self, data):
          self.data = data

      self._pool = IncrementCache
      self._to_pool()
      self._from_pool = True

  def __getitem__(self, index):
    is_train = self.phase.split('_')[0] == 'train'
    img = self._loader(self.img_list[index])

    # NR protocol:
    # - train: ref/pref are retained only for auxiliary losses;
    # - val/test: no reference image is read. Distorted-image copies are returned
    #   as placeholders so the existing five-item DataLoader interface remains
    #   compatible with iqaScrach.py.
    if is_train:
      ref = self._loader(self.ref_list[index])

      lenth = len(self.ref_list)
      pref_id = np.random.randint(0, lenth)
      while self.ref_list[pref_id] == self.ref_list[index]:
        pref_id = np.random.randint(0, lenth)
      pref = self._loader(self.ref_list[pref_id])
    else:
      ref = img
      pref = img

    score = self.score_list[index]
    score = torch.tensor(score).float()

    # 优先解析文件名，如果不成功再降级使用 json 中的 dtype_list
    filename = os.path.basename(self.img_list[index])

    if (
        'SIQAD' in self.list_dir
        or 'siqad' in self.list_dir
        or 'SIQAD' in self.data_dir
        or 'siqad' in self.data_dir
        or self.__class__.__name__ == 'SIQADDataset'
    ):
      try:
        res_split = filename.split('_')
        if len(res_split) > 1:
          parsed_idx = int(res_split[1]) - 1
          if parsed_idx < 0 or parsed_idx >= self.n_class:
            parsed_idx = (
                self.dtype_list[index]
                if index < len(self.dtype_list)
                else 0
            )
          dist_type = torch.tensor(parsed_idx).long()
        else:
          dist_type = torch.tensor(self.dtype_list[index]).long()
      except Exception as e:
        dist_type = torch.tensor(self.dtype_list[index]).long()
    else:
      try:
        res_split = filename.split('_')
        if len(res_split) > 1:
          parsed_idx = int(res_split[1]) - 1
          if parsed_idx < 0 or parsed_idx >= self.n_class:
            parsed_idx = (
                self.dtype_list[index]
                if index < len(self.dtype_list)
                else 0
            )
          dist_type = torch.tensor(parsed_idx).long()
        else:
          dist_type = torch.tensor(self.dtype_list[index]).long()
      except Exception as e:
        dist_type = torch.tensor(self.dtype_list[index]).long()

    # Use a local, sample-specific RNG for reproducible evaluation patches.
    # This does not consume or modify the global training RNG state.
    eval_patch_idx = None
    if not is_train:
      h, w = img.shape[-3:-1]
      n_available = (h // self.patch_size) * (w // self.patch_size)
      n_selected = (
          n_available
          if not self.n_ptchs
          else min(self.n_ptchs, n_available)
      )
      eval_patch_idx = list(range(n_available))
      random.Random(index).shuffle(eval_patch_idx)
      eval_patch_idx = eval_patch_idx[:n_selected]

    if self._from_pool:
      if is_train:
        img, ref = self.tfs.horizontal_flip(img, ref)
        img_ptchs, ref_ptchs = self._to_patch_tensors(img, ref, self.patch_size)
        pref = self.tfs.horizontal_flip(pref)
        pref_ptchs = self._to_patch_tensors_single(pref, self.patch_size)
      else:
        img_ptchs = self._to_patch_tensors_single(
            img, self.patch_size, idx=eval_patch_idx
        )
        ref_ptchs = img_ptchs.clone()
        pref_ptchs = img_ptchs.clone()
    else:
      if is_train:
        img, ref = self.tfs.horizontal_flip(img, ref)
        img_ptchs, ref_ptchs = self._to_patch_tensors(img, ref, self.patch_size)
        pref = self.tfs.horizontal_flip(pref)
        pref_ptchs = self._to_patch_tensors_single(pref, self.patch_size)
      else:
        img_ptchs = self._to_patch_tensors_single(
            img, self.patch_size, idx=eval_patch_idx
        )
        ref_ptchs = img_ptchs.clone()
        pref_ptchs = img_ptchs.clone()

    img_ptchs = torch.stack([self.normal(p) for p in img_ptchs])
    ref_ptchs = torch.stack([self.normal(p) for p in ref_ptchs])
    pref_ptchs = torch.stack([self.normal(p) for p in pref_ptchs])

    return img_ptchs, ref_ptchs, pref_ptchs, score, dist_type

  def __len__(self):
    return len(self.img_list)

  def _loader(self, name):
    # 👑 核心修复：统一将 Windows 风格的 '\' 替换为 Linux 认可的 '/'
    name = str(name).replace('\\', '/')

    # 👑 提速核心 2：命中 RAM 缓存则直接秒回 numpy 矩阵，绕过磁盘 I/O
    if name in self.image_cache:
      return self.image_cache[name]

    if 'datasets' in name:
      name = name.split('datasets')[-1].strip('/')
    if 'references' in name:
      name = name.replace('references', 'ReferenceImages')

    full_path = join(self.data_dir, name)
    full_path = os.path.abspath(full_path)

    img = io.imread(full_path)
    # 写入缓存
    self.image_cache[name] = img
    return img

  def _to_patch_tensors(self, img, ref, patch_size):
    img_ptchs, ref_ptchs = self.tfs.to_patches(
        img, ref, ptch_size=patch_size, n_ptchs=self.n_ptchs
    )
    img_ptchs, ref_ptchs = self.tfs.to_tensor(img_ptchs, ref_ptchs)
    return img_ptchs, ref_ptchs

  def _to_patch_tensors_single(self, img, patch_size, idx=None):
    img_ptchs = self.tfs.to_patches(
        img, ptch_size=patch_size, n_ptchs=self.n_ptchs, idx=idx
    )
    img_ptchs = self.tfs.to_tensor(img_ptchs)
    return img_ptchs

  def _to_pool(self):
    len_data = self.__len__()
    pb = SimpleProgressBar(len_data)
    print('\ninitializing data pool...')
    for index in range(len_data):
      self._pool(index).store(self.__getitem__(index)[0])
      pb.show(index, '[{:d}]/[{:d}] '.format(index + 1, len_data))

  def _aug_lists(self):
    if self.phase.split('_')[0] == 'test':
      return
    len_aug = (
        len(self.ref_list) // 5 if self.phase.split('_')[0] == 'train' else 10
    )
    aug_list = self.ref_list * (len_aug // len(self.ref_list) + 1)
    random.shuffle(aug_list)
    aug_list = aug_list[:len_aug]
    self.img_list.extend(aug_list)
    self.score_list += [0.0] * len_aug
    self.ref_list.extend(aug_list)
    self.dtype_list += [0] * len_aug  # 填充增强部分的 dtype

    if self.phase.split('_')[0] == 'train':
      mul_aug = 16
      self.img_list *= mul_aug
      self.ref_list *= mul_aug
      self.score_list *= mul_aug
      self.dtype_list *= mul_aug  # 👑 关键修复：同步扩增 dtype_list，彻底告别 IndexError！

  def _read_lists(self):
    img_path = join(self.list_dir, self.phase + '_data.json')
    print(img_path)
    assert exists(img_path)

    with open(img_path, 'r') as fp:
      data_dict = json.load(fp)

    self.img_list = data_dict['img']
    self.ref_list = data_dict.get('ref', self.img_list)
    self.score_list = data_dict.get('score', [0.0] * len(self.img_list))
    self.dtype_list = data_dict.get('dtype', [0] * len(self.img_list))


class TID2013Dataset(IQADataset):

  def _read_lists(self):
    super()._read_lists()
    self.score_list = [(9.0 - s) / 9.0 * 100.0 for s in self.score_list]


class SIQADDataset(IQADataset):

  def _aug_lists(self):
    if self.phase.split('_')[0] == 'test':
      return
    if self.phase.split('_')[0] == 'train':
      mul_aug = 16
      self.img_list *= mul_aug
      self.ref_list *= mul_aug
      self.score_list *= mul_aug
      self.dtype_list *= mul_aug  # 同步扩增


class SCIDDataset(IQADataset):

  def _aug_lists(self):
    if self.phase.split('_')[0] == 'test':
      return
    if self.phase.split('_')[0] == 'train':
      mul_aug = 16
      self.img_list *= mul_aug
      self.ref_list *= mul_aug
      self.score_list *= mul_aug
      self.dtype_list *= mul_aug  # 同步扩增


class WaterlooDataset(IQADataset):

  def _read_lists(self):
    super()._read_lists()
    self.score_list = [(1.0 - s) * 100.0 for s in self.score_list]


class Transforms:

  def __init__(self):
    super(Transforms, self).__init__()

  def _pair_deco(tf_func):

    def transform(self, img, ref=None, *args, **kwargs):
      if (ref is not None) and (not isinstance(ref, np.ndarray)):
        args = (ref,) + args
        ref = None
      ret = tf_func(self, img, None, *args, **kwargs)
      assert len(ret) == 2
      if ref is None:
        return ret[0]
      else:
        num_var = tf_func.__code__.co_argcount - 3
        if (len(args) + len(kwargs)) == (num_var - 1):
          var_name = tf_func.__code__.co_varnames[-1]
          kwargs[var_name] = ret[1]
        tf_ref, _ = tf_func(self, ref, None, *args, **kwargs)
        return ret[0], tf_ref

    return transform

  def _horizontal_flip(self, img, flip):
    if flip is None:
      flip = random.random() > 0.5
    return (img[..., ::-1, :] if flip else img), flip

  def _to_tensor(self, img):
    return (
        torch.from_numpy(
            (img.astype(np.float32) / 255).swapaxes(-3, -2).swapaxes(-3, -1)
        ),
        (),
    )

  def _crop_square(self, img, crop_size, pos):
    if pos is None:
      h, w = img.shape[-3:-1]
      assert crop_size <= h and crop_size <= w
      ub = random.randint(0, h - crop_size)
      lb = random.randint(0, w - crop_size)
      pos = (ub, ub + crop_size, lb, lb + crop_size)
    return img[..., pos[0] : pos[1], pos[-2] : pos[-1], :], pos

  def _extract_patches(self, img, ptch_size):
    h, w = img.shape[-3:-1]
    nh, nw = h // ptch_size, w // ptch_size
    assert nh > 0 and nw > 0
    vptchs = np.stack(np.split(img[..., : nh * ptch_size, :, :], nh, axis=-3))
    ptchs = np.concatenate(
        np.split(vptchs[..., : nw * ptch_size, :], nw, axis=-2)
    )
    return ptchs, nh * nw

  def _to_patches(self, img, ptch_size, n_ptchs, idx):
    ptchs, n = self._extract_patches(img, ptch_size)
    if not n_ptchs:
      n_ptchs = n
    elif n_ptchs > n:
      n_ptchs = n
    if idx is None:
      idx = list(range(n))
      random.shuffle(idx)
      idx = idx[:n_ptchs]
    return ptchs[idx], idx

  @_pair_deco
  def horizontal_flip(self, img, ref=None, flip=None):
    return self._horizontal_flip(img, flip=flip)

  @_pair_deco
  def to_tensor(self, img, ref=None):
    return self._to_tensor(img)

  @_pair_deco
  def crop_square(self, img, ref=None, crop_size=64, pos=None):
    return self._crop_square(img, crop_size=crop_size, pos=pos)

  @_pair_deco
  def to_patches(self, img, ref=None, ptch_size=64, n_ptchs=None, idx=None):
    return self._to_patches(img, ptch_size=ptch_size, n_ptchs=n_ptchs, idx=idx)