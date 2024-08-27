import numpy as np
import numpy_indexed as npi
import json

import torch
from torch.autograd import Variable
import torch.nn as nn
from functools import partial

def applyWoodCls(config_file,pcd,model_path,use_cuda=True,progress_bar=None):
    if progress_bar:
        progress_bar.setValue(0)
    try:
        with open(config_file) as json_file:
            configs = json.load(json_file)
    except:
        print(config_file)
        print("cannot load config file")

    nbmat_sz = np.array(configs["model"]["voxel_number_in_block"])
    min_res = np.array(configs["model"]["voxel_resolution_in_meter"])
    patch_size = configs["model"]["patch_size"]
    decoder_dim = configs["model"]["decoder_dim"]
    channel_dims = configs["model"]["channel_dims"]
    SR_ratios = configs["model"]["SR_ratios"]
    num_heads = configs["model"]["num_heads"]
    MLP_ratios = configs["model"]["MLP_ratios"]
    depths = configs["model"]["depths"]
    qkv_bias = configs["model"]["qkv_bias"]
    drop_rate = configs["model"]["drop_rate"]
    drop_path_rate = configs["model"]["drop_path_rate"]

    try:
        from vox3DSegFormer import Segformer
    except:
        from .vox3DSegFormer import Segformer


    model = Segformer(
        pretrained=True,
        block3d_size=nbmat_sz,
        in_chans=1,
        num_classes=3,
        patch_size=patch_size,
        embed_dims=channel_dims,
        num_heads=num_heads,
        mlp_ratios=MLP_ratios,
        qkv_bias=qkv_bias,
        depths=depths,
        sr_ratios=SR_ratios,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        decoder_dim=decoder_dim,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        model_best_save_path=model_path
    )

    if use_cuda:
        model = model.half().cuda()
    model.eval()

    if progress_bar:
        progress_bar.setValue(5)

    nb_tsz = int(nbmat_sz[0] * nbmat_sz[1] * nbmat_sz[2])
    device = "cuda" if use_cuda else "cpu"

    pcd_min = np.min(pcd[:, :3], axis=0)
    block_ijk = np.floor((pcd[:, :3] - pcd_min[:3]) / min_res[:3] / nbmat_sz[:3]).astype(np.int32)

    block_ijk_unq, block_idx_groups = npi.group_by(block_ijk, np.arange(len(block_ijk)))

    nb_idxs = []
    nb_pcd_idxs = []
    nb_inverse_idxs = []


    for iter, idx in enumerate(block_idx_groups):
        columns_sp = pcd[idx, :]
        nb_pcd_idx = idx

        sp_min = np.min(columns_sp[:, :3], axis=0)
        nb_ijk = np.floor((columns_sp[:, :3] - sp_min) / min_res)
        nb_sel = np.all((nb_ijk < nbmat_sz) & (nb_ijk >= 0), axis=1)
        nb_ijk = nb_ijk[nb_sel]
        nb_pcd_idx = nb_pcd_idx[nb_sel]

        nb_idx = np.ravel_multi_index(np.transpose(nb_ijk.astype(np.int32)), nbmat_sz)
        nb_idx_u, nb_idx_uidx, nb_inverse_idx = np.unique(nb_idx, return_index=True, return_inverse=True)

        nb_idxs.append(nb_idx_u)
        nb_inverse_idxs.append(nb_inverse_idx)
        nb_pcd_idxs.append(nb_pcd_idx)

    if progress_bar:
        progress_bar.setValue(15)

    pcd_pred = np.zeros(len(pcd), dtype=np.int32)
    total_nbs=len(nb_idxs)
    for i in range(total_nbs):
        nb_idx = nb_idxs[i]
        nb_pcd_idx = nb_pcd_idxs[i]
        nb_inverse_idx = nb_inverse_idxs[i]

        x = torch.zeros(nb_tsz, 1)
        if len(nb_idx) > 0:
            x[nb_idx, :] = 1.0

        x = torch.moveaxis(x.reshape((1, *nbmat_sz, 1)).float(), -1, 1)
        x = torch.swapaxes(x, -1, 2)

        x = Variable(x)
        h = model(x.to(device))

        h_nonzero = torch.moveaxis(torch.unsqueeze(torch.moveaxis(torch.swapaxes(h, -1, 2), 1, -1).reshape((nb_tsz, 3))[nb_idx, :], 0), -1, 1)
        h_nonzero = torch.argmax(h_nonzero[0], dim=0)

        nb_pred = h_nonzero.cpu().detach().numpy()
        pcd_pred[nb_pcd_idx] = nb_pred[nb_inverse_idx]

        if progress_bar:
            progress_bar.setValue(int(85*i/total_nbs)+15)

    if progress_bar:
        progress_bar.setValue(100)
    return pcd_pred

