import numpy as np
import numpy_indexed as npi
import json

import torch
from torch.autograd import Variable
import torch.nn as nn
from functools import partial

class ProgressBarWrapper:
    def __init__(self, progress_bar=None):
        self.progress_bar = progress_bar

    def set_value(self, value):
        if self.progress_bar:
            self.progress_bar.setValue(value)

def apply_wood_cls(config_file, pcd, model_path, use_cuda=True, progress_bar=None):
    progress = ProgressBarWrapper(progress_bar)
    progress.set_value(0)

    # laod variables from the config file (e.g. woodcls_branch_tls_segformer3D_112_4cm(GPU8GB).json
    try:
        with open(config_file) as json_file:
            configs = json.load(json_file)
    except Exception as e:
        print(config_file)
        print("Cannot load config file:", e)
        return

    nbmat_sz = np.array(configs["model"]["voxel_number_in_block"])
    min_res = np.array(configs["model"]["voxel_resolution_in_meter"])
    patch_size = configs["model"]["patch_size"]
    decoder_dim = configs["model"]["decoder_dim"]
    channel_dims = configs["model"]["channel_dims"]
    sr_ratios = configs["model"]["SR_ratios"]
    num_heads = configs["model"]["num_heads"]
    mlp_ratios = configs["model"]["MLP_ratios"]
    depths = configs["model"]["depths"]
    qkv_bias = configs["model"]["qkv_bias"]
    drop_rate = configs["model"]["drop_rate"]
    drop_path_rate = configs["model"]["drop_path_rate"]

    #define and load the 3D segformer model
    try:
        from vox3DSegFormer import Segformer
    except ImportError:
        from .vox3DSegFormer import Segformer

    model = Segformer(
        pretrained=True,
        block3d_size=nbmat_sz,
        in_chans=1,
        num_classes=3,
        patch_size=patch_size,
        embed_dims=channel_dims,
        num_heads=num_heads,
        mlp_ratios=mlp_ratios,
        qkv_bias=qkv_bias,
        depths=depths,
        sr_ratios=sr_ratios,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        decoder_dim=decoder_dim,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        model_best_save_path=model_path,
    )

    if use_cuda:
        model = model.cuda()
    model.eval()

    progress.set_value(5)

    nb_tsz = int(np.prod(nbmat_sz))
    device = "cuda" if use_cuda else "cpu"

    #voxelize
    pcd_min = np.min(pcd[:, :3], axis=0)
    block_ijk = np.floor(
        (pcd[:, :3] - pcd_min[:3]) / min_res[:3] / nbmat_sz[:3]
    ).astype(np.int32)

    _, block_idx_groups = npi.group_by(block_ijk, np.arange(len(block_ijk)))

    nb_idxs = []
    nb_pcd_idxs = []
    nb_inverse_idxs = []

    for idx in block_idx_groups:
        columns_sp = pcd[idx, :]
        nb_pcd_idx = idx

        sp_min = np.min(columns_sp[:, :3], axis=0)
        nb_ijk = np.floor((columns_sp[:, :3] - sp_min) / min_res)
        nb_sel = np.all((nb_ijk < nbmat_sz) & (nb_ijk >= 0), axis=1)
        nb_ijk = nb_ijk[nb_sel]
        nb_pcd_idx = nb_pcd_idx[nb_sel]

        nb_idx = np.ravel_multi_index(nb_ijk.astype(np.int32).T, nbmat_sz)
        nb_idx_u, _, nb_inverse_idx = np.unique(
            nb_idx, return_index=True, return_inverse=True
        )

        nb_idxs.append(nb_idx_u)#indicies of unique voxels from each block
        nb_inverse_idxs.append(nb_inverse_idx)#indices used to reproject the unique voxels to original order
        nb_pcd_idxs.append(nb_pcd_idx)#within-voxel point indices from the point cloud

    progress.set_value(15)

    #apply the DL model blockwisely
    pcd_pred = np.zeros(len(pcd), dtype=np.int32)
    total_nbs = len(nb_idxs)
    for i in range(total_nbs):
        nb_idx = nb_idxs[i]
        nb_pcd_idx = nb_pcd_idxs[i]
        nb_inverse_idx = nb_inverse_idxs[i]

        x = torch.zeros(nb_tsz, 1)
        if len(nb_idx) > 0:
            x[nb_idx, :] = 1.0

        x = x.reshape((1, *nbmat_sz, 1)).float().moveaxis(-1, 1).swapaxes(-1, -2)
        x = Variable(x)
        h = model(x.to(device))

        h_swapped = h.swapaxes(-1, -2)
        h_moved = h_swapped.moveaxis(1, -1)
        h_reshaped = h_moved.reshape(nb_tsz, 3)
        h_nonzero_data = h_reshaped[nb_idx, :]
        h_unsqueezed = h_nonzero_data.unsqueeze(0)
        h_nonzero = h_unsqueezed.moveaxis(-1, 1)
        h_nonzero = torch.argmax(h_nonzero[0], dim=0)

        nb_pred = h_nonzero.cpu().detach().numpy()
        pcd_pred[nb_pcd_idx] = nb_pred[nb_inverse_idx]

        progress_value = int(85 * i / total_nbs) + 15
        progress.set_value(progress_value)

    progress.set_value(100)
    return pcd_pred

if __name__ == "__main__":
    import os
    import glob
    import laspy


    config_file=os.path.join("woodcls_branch_tls_segformer3D_112_4cm(GPU8GB).json")
    data_dir="../../data"
    pcd_fname = glob.glob(os.path.join(data_dir, "*.la*"))[0]
    model_path="../../models/woodcls_branch_tls_segformer3D_112_4cm(GPU8GB).pth"
    out_path="output"

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    las = laspy.open(pcd_fname).read()
    pcd = np.transpose(np.array([las.x, las.y, las.z]))  # las.point_format.dimension_names
    pcd_pred=apply_wood_cls(config_file,pcd,model_path,use_cuda=True,progress_bar=None)

    las.add_extra_dim(laspy.ExtraBytesParams(name="branchcls", type="int32", description="Predicted branch points"))
    las.branchcls = pcd_pred
    las.write(os.path.join(out_path, f"{os.path.basename(pcd_fname)[:-4]}_branchcls.laz"))
