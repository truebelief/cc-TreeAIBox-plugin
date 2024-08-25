import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_3tuple


class Mlp(nn.Module):
    """Multilayer perceptron with depthwise convolution.

    This module implements a multilayer perceptron with a depthwise convolution,
    activation, and dropout layers.

    Args:
    ----
        in_features (int): Number of input features.
        hidden_features (int, optional): Number of hidden features. Defaults to None.
        out_features (int, optional): Number of output features. Defaults to None.
        act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
        drop (float, optional): Dropout rate. Defaults to 0.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        """Initialize the MLP module.

        Args:
        ----
            in_features (int): Number of input features.
            hidden_features (int, optional): Number of hidden features. If None, set to in_features. Defaults to None.
            out_features (int, optional): Number of output features. If None, set to in_features. Defaults to None.
            act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
            drop (float, optional): Dropout rate. Defaults to 0.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, D, H, W):
        """Forward pass of the MLP module.

        Args:
        ----
            x (torch.Tensor): Input tensor.
            D (int): Depth of the input.
            H (int): Height of the input.
            W (int): Width of the input.

        Returns:
        -------
            torch.Tensor: Output tensor after passing through the MLP.
        """
        x = self.fc1(x)
        x = self.dwconv(x, D, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Attention(nn.Module):
    """Multi-head self-attention mechanism.

    This module implements a multi-head self-attention mechanism with the option
    for spatial reduction.

    Args:
    ----
        dim (int): Input dimension.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to False.
        qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set.
        attn_drop (float, optional): Attention dropout rate. Defaults to 0.
        proj_drop (float, optional): Output dropout rate. Defaults to 0.
        sr_ratio (int, optional): Spatial reduction ratio. Defaults to 1.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        """Initialize the Attention module.

        Args:
        ----
            dim (int): Input dimension.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to False.
            qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            attn_drop (float, optional): Attention dropout rate. Defaults to 0.
            proj_drop (float, optional): Output dropout rate. Defaults to 0.
            sr_ratio (int, optional): Spatial reduction ratio. Defaults to 1.
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, D, H, W):
        """Forward pass of the Attention module.

        Args:
        ----
            x (torch.Tensor): Input tensor.
            D (int): Depth of the input.
            H (int): Height of the input.
            W (int): Width of the input.

        Returns:
        -------
            torch.Tensor: Output tensor after self-attention operation.
        """
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, D, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class Block(nn.Module):
    """Transformer block.

    This module implements a transformer block with self-attention and feedforward network.

    Args:
    ----
        dim (int): Input dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to False.
        qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Defaults to 0.
        attn_drop (float, optional): Attention dropout rate. Defaults to 0.
        drop_path (float, optional): Stochastic depth rate. Defaults to 0.
        act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        sr_ratio (int, optional): Spatial reduction ratio. Defaults to 1.
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        """Initialize the Transformer Block.

        Args:
        ----
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to False.
            qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            drop (float, optional): Dropout rate. Defaults to 0.
            attn_drop (float, optional): Attention dropout rate. Defaults to 0.
            drop_path (float, optional): Stochastic depth rate. Defaults to 0.
            act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
            sr_ratio (int, optional): Spatial reduction ratio. Defaults to 1.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, D, H, W):
        """Forward pass of the Transformer Block.

        Args:
        ----
            x (torch.Tensor): Input tensor.
            D (int): Depth of the input.
            H (int): Height of the input.
            W (int): Width of the input.

        Returns:
        -------
            torch.Tensor: Output tensor after passing through attention and MLP layers.
        """
        x = x + self.drop_path(self.attn(self.norm1(x), D, H, W))

        return x + self.drop_path(self.mlp(self.norm2(x), D, H, W))


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding with overlapping patches.

    This module implements the conversion of an image into a sequence of overlapping patches.

    Args:
    ----
        block3d_size (int or tuple, optional): Size of the 3D input block. Defaults to 224.
        patch_size (int or tuple, optional): Size of each patch. Defaults to 7.
        stride (int, optional): Stride of the convolution. Defaults to 4.
        in_chans (int, optional): Number of input channels. Defaults to 3.
        embed_dim (int, optional): Embedding dimension. Defaults to 768.
    """

    def __init__(self, block3d_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        """Initialize the OverlapPatchEmbed module.

        Args:
        ----
            block3d_size (int or tuple, optional): Size of the 3D input block. Defaults to 224.
            patch_size (int or tuple, optional): Size of each patch. Defaults to 7.
            stride (int, optional): Stride of the convolution. Defaults to 4.
            in_chans (int, optional): Number of input channels. Defaults to 3.
            embed_dim (int, optional): Embedding dimension. Defaults to 768.
        """
        super().__init__()
        block3d_size = to_3tuple(block3d_size)
        patch_size = to_3tuple(patch_size)

        self.block3d_size = block3d_size
        self.patch_size = patch_size
        self.D, self.H, self.W = (
            block3d_size[0] // patch_size[0],
            block3d_size[1] // patch_size[1],
            block3d_size[2] // patch_size[2],
        )
        self.num_patches = self.D * self.H * self.W
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """Forward pass of the OverlapPatchEmbed module.

        Args:
        ----
            x (torch.Tensor): Input tensor.

        Returns:
        -------
            tuple: A tuple containing:
                - torch.Tensor: Embedded patches.
                - int: Depth of the output.
                - int: Height of the output.
                - int: Width of the output.
        """
        x = self.proj(x)
        _, _, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        return self.norm(x), D, H, W


class LinearMLP(nn.Module):
    """Linear Multi-Layer Perceptron for embedding.

    This module implements a simple linear projection for embedding.

    Args:
    ----
        input_dim (int, optional): Input dimension. Defaults to 2048.
        embed_dim (int, optional): Output embedding dimension. Defaults to 768.
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        """Initialize the LinearMLP module.

        Args:
        ----
            input_dim (int, optional): Input dimension. Defaults to 2048.
            embed_dim (int, optional): Output embedding dimension. Defaults to 768.
        """
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        """Forward pass of the LinearMLP.

        Args:
        ----
            x (torch.Tensor): Input tensor.

        Returns:
        -------
            torch.Tensor: Embedded output tensor.
        """
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)


class DWConv(nn.Module):
    """Depthwise Convolution module.

    This module implements a 3D depthwise convolution.

    Args:
    ----
        dim (int, optional): Number of input and output channels. Defaults to 768.
    """

    def __init__(self, dim=768):
        """Initialize the Depthwise Convolution module.

        Args:
        ----
            dim (int, optional): Number of input and output channels. Defaults to 768.
        """
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, D, H, W):
        """Forward pass of the Depthwise Convolution.

        Args:
        ----
            x (torch.Tensor): Input tensor.
            D (int): Depth of the input.
            H (int): Height of the input.
            W (int): Width of the input.

        Returns:
        -------
            torch.Tensor: Output tensor after depthwise convolution.
        """
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, D, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class Segformer(nn.Module):
    """Segformer model for 3D semantic segmentation.

    This module implements the Segformer architecture for 3D semantic segmentation tasks.

    Args:
    ----
        pretrained (bool, optional): If True, load pretrained weights. Defaults to None.
        block3d_size (int, optional): Size of the 3D input block. Defaults to 1024.
        patch_size (int, optional): Size of each patch. Defaults to 3.
        in_chans (int, optional): Number of input channels. Defaults to 3.
        num_classes (int, optional): Number of output classes. Defaults to 19.
        embed_dims (list, optional): Embedding dimensions for each stage. Defaults to [64, 128, 256, 512].
        num_heads (list, optional): Number of attention heads for each stage. Defaults to [1, 2, 5, 8].
        mlp_ratios (list, optional): MLP ratios for each stage. Defaults to [4, 4, 4, 4].
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
        qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
        drop_rate (float, optional): Dropout rate. Defaults to 0.
        attn_drop_rate (float, optional): Attention dropout rate. Defaults to 0.
        drop_path_rate (float, optional): Stochastic depth rate. Defaults to 0.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        depths (list, optional): Depth of each stage. Defaults to [3, 6, 40, 3].
        sr_ratios (list, optional): Spatial reduction ratios for each stage. Defaults to [8, 4, 2, 1].
        decoder_dim (int, optional): Dimension of the decoder. Defaults to 256.
        model_best_save_path (str, optional): Path to save the best model. Defaults to None.
    """

    def __init__(
        self,
        pretrained=None,
        block3d_size=1024,
        patch_size=3,
        in_chans=3,
        num_classes=19,
        embed_dims=(64, 128, 256, 512),
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(4, 4, 4, 4),
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=(3, 6, 40, 3),
        sr_ratios=(8, 4, 2, 1),
        decoder_dim=256,
        model_best_save_path=None,
    ):
        """Initialize the Segformer model.

        Args:
        ----
            pretrained (str, optional): Path to pretrained weights. Defaults to None.
            block3d_size (int, optional): Size of the 3D input block. Defaults to 1024.
            patch_size (int, optional): Size of each patch. Defaults to 3.
            in_chans (int, optional): Number of input channels. Defaults to 3.
            num_classes (int, optional): Number of output classes. Defaults to 19.
            embed_dims (list, optional): Embedding dimensions for each stage. Defaults to [64, 128, 256, 512].
            num_heads (list, optional): Number of attention heads for each stage. Defaults to [1, 2, 5, 8].
            mlp_ratios (list, optional): MLP ratios for each stage. Defaults to [4, 4, 4, 4].
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            drop_rate (float, optional): Dropout rate. Defaults to 0.
            attn_drop_rate (float, optional): Attention dropout rate. Defaults to 0.
            drop_path_rate (float, optional): Stochastic depth rate. Defaults to 0.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
            depths (list, optional): Depth of each stage. Defaults to [3, 6, 40, 3].
            sr_ratios (list, optional): Spatial reduction ratios for each stage. Defaults to [8, 4, 2, 1].
            decoder_dim (int, optional): Dimension of the decoder. Defaults to 256.
            model_best_save_path (str, optional): Path to save the best model. Defaults to None.
        """
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(
            block3d_size=block3d_size,
            # patch_size=7,
            # stride=4,
            patch_size=patch_size,
            stride=2,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
        )
        self.patch_embed2 = OverlapPatchEmbed(
            block3d_size=block3d_size // 4,
            patch_size=patch_size,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
        )
        self.patch_embed3 = OverlapPatchEmbed(
            block3d_size=block3d_size // 8,
            patch_size=patch_size,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            block3d_size=block3d_size // 16,
            patch_size=patch_size,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3],
        )

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
                for i in range(depths[0])
            ]
        )
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[1],
                )
                for i in range(depths[1])
            ]
        )
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratios[2],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[2],
                )
                for i in range(depths[2])
            ]
        )
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratios[3],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[3],
                )
                for i in range(depths[3])
            ]
        )
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # segmentation head
        self.linear_c4 = LinearMLP(input_dim=embed_dims[3], embed_dim=decoder_dim)
        self.linear_c3 = LinearMLP(input_dim=embed_dims[2], embed_dim=decoder_dim)
        self.linear_c2 = LinearMLP(input_dim=embed_dims[1], embed_dim=decoder_dim)
        self.linear_c1 = LinearMLP(input_dim=embed_dims[0], embed_dim=decoder_dim)
        self.linear_fuse = nn.Conv3d(4 * decoder_dim, decoder_dim, 1)
        self.dropout = nn.Dropout3d(drop_rate)
        self.linear_pred = nn.Conv3d(decoder_dim, num_classes, kernel_size=1)

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        if pretrained:
            state_dict = torch.load(model_best_save_path)
            self.load_state_dict(state_dict)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def reset_drop_path(self, drop_path_rate):
        """Reset the drop path rate for all blocks in the network.

        Args:
        ----
            drop_path_rate (float): New drop path rate to be set.
        """
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        """Freeze the patch embedding layer by setting requires_grad to False."""
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        """Specify the parameter names for which weight decay should not be applied.

        Returns
        -------
            set: Set of parameter names to be excluded from weight decay.
        """
        return {"pos_embed1", "pos_embed2", "pos_embed3", "pos_embed4", "cls_token"}  # has pos_embed may be better

    def get_classifier(self):
        """Get the classifier head of the model.

        Returns
        -------
            nn.Module: The classifier head.
        """
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        """Reset the classifier head with a new number of classes.

        Args:
        ----
            num_classes (int): New number of classes for the classifier.
            global_pool (str, optional): Global pooling type. Defaults to "".
        """
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        """Forward pass through the encoder part of the Segformer.

        Args:
        ----
            x (torch.Tensor): Input tensor.

        Returns:
        -------
            list: List of feature maps from each stage of the encoder.
        """
        B = x.shape[0]
        outs = []

        # stage 1
        x, D, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, D, H, W)
        x = self.norm1(x)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x)

        # stage 2
        x, D, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, D, H, W)
        x = self.norm2(x)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x)

        # stage 3
        x, D, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, D, H, W)
        x = self.norm3(x)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x)

        # stage 4
        x, D, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, D, H, W)
        x = self.norm4(x)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        """Forward pass through the entire Segformer model.

        Args:
        ----
            x (torch.Tensor): Input tensor.

        Returns:
        -------
            torch.Tensor: Output tensor with predicted class probabilities.
        """
        d_out, h_out, w_out = x.size()[2], x.size()[3], x.size()[4]

        x = self.forward_features(x)
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, d, h, w = c4.shape
        # d_out, h_out, w_out = c1.size()[2], c1.size()[3], c1.size()[4]

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3], c4.shape[4])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode="trilinear", align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3], c3.shape[4])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode="trilinear", align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3], c2.shape[4])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode="trilinear", align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3], c1.shape[4])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        x = F.interpolate(input=x, size=(d_out, h_out, w_out), mode="trilinear", align_corners=False)

        return x.type(torch.float32)
