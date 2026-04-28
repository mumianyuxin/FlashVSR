from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time


CACHE_T = 2


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias

class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            # print(cache_x.shape, x.shape)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
            # print('cache!')
        x = F.pad(x, padding, mode='replicate') # mode='replicate'
        # print(x[0,0,:,0,0])

        return super().forward(x)
    
class PixelShuffle3d(nn.Module):
    def __init__(self, ff, hh, ww):
        super().__init__()
        self.ff = ff
        self.hh = hh
        self.ww = ww

    def forward(self, x):
        # x: (B, C, F, H, W)
        return rearrange(x, 
                         'b c (f ff) (h hh) (w ww) -> b (c ff hh ww) f h w',
                         ff=self.ff, hh=self.hh, ww=self.ww)

class Buffer_LQ4x_Proj(nn.Module):

    def __init__(self, in_dim, out_dim, layer_num=30):
        super().__init__()
        self.ff = 1
        self.hh = 16
        self.ww = 16
        self.hidden_dim1 = 2048
        self.hidden_dim2 = 3072
        self.layer_num = layer_num

        self.pixel_shuffle = PixelShuffle3d(self.ff, self.hh, self.ww)

        self.conv1 = CausalConv3d(in_dim*self.ff*self.hh*self.ww, self.hidden_dim1, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)) # f -> f/2 h -> h w -> w
        self.norm1 = RMS_norm(self.hidden_dim1, images=False)
        self.act1 = nn.SiLU()

        self.conv2 = CausalConv3d(self.hidden_dim1, self.hidden_dim2, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)) # f -> f/2 h -> h w -> w
        self.norm2 = RMS_norm(self.hidden_dim2, images=False)
        self.act2 = nn.SiLU()

        self.out_dim = out_dim
        self.linear = nn.Linear(self.hidden_dim2, out_dim * layer_num)

        self.clip_idx = 0

    def forward(self, video):
        self.clear_cache()
        # x: (B, C, F, H, W)
        
        t = video.shape[2]
        iter_ = 1 + (t - 1) // 4
        first_frame = video[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
        video = torch.cat([first_frame, video], dim=2)
        # print(video.shape)

        out_x = []
        for i in range(iter_):
            x = self.pixel_shuffle(video[:,:,i*4:(i+1)*4,:,:])
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv1'] = cache1_x
            x = self.conv1(x, self.cache['conv1'])
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv2'] = cache2_x
            if i == 0:
                continue
            x = self.conv2(x, self.cache['conv2'])
            x = self.norm2(x)
            x = self.act2(x)
            out_x.append(x)
        out_x = torch.cat(out_x, dim = 2)
        # print(out_x.shape)
        out_x = rearrange(out_x, 'b c f h w -> b (f h w) c')
        return self._project_outputs(out_x)

    def clear_cache(self):
        self.cache = {}
        self.cache['conv1'] = None
        self.cache['conv2'] = None
        self.clip_idx = 0
    
    def stream_forward(self, video_clip):
        if self.clip_idx == 0:
            # self.clear_cache()
            first_frame = video_clip[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
            video_clip = torch.cat([first_frame, video_clip], dim=2)
            x = self.pixel_shuffle(video_clip)
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv1'] = cache1_x
            x = self.conv1(x, self.cache['conv1'])
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv2'] = cache2_x
            self.clip_idx += 1
            return None
        else:
            x = self.pixel_shuffle(video_clip)
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv1'] = cache1_x
            x = self.conv1(x, self.cache['conv1'])
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv2'] = cache2_x
            x = self.conv2(x, self.cache['conv2'])
            x = self.norm2(x)
            x = self.act2(x)
            out_x = rearrange(x, 'b c f h w -> b (f h w) c')
            self.clip_idx += 1
            return self._project_outputs(out_x)

    def _project_outputs(self, out_x):
        fused = self.linear(out_x)
        fused = fused.view(*out_x.shape[:2], self.layer_num, self.out_dim)
        return [fused[:, :, i, :] for i in range(self.layer_num)]

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        legacy_weight_keys = [f"{prefix}linear_layers.{i}.weight" for i in range(self.layer_num)]
        legacy_bias_keys = [f"{prefix}linear_layers.{i}.bias" for i in range(self.layer_num)]
        fused_weight_key = f"{prefix}linear.weight"
        fused_bias_key = f"{prefix}linear.bias"

        if fused_weight_key not in state_dict and all(key in state_dict for key in legacy_weight_keys):
            state_dict[fused_weight_key] = torch.cat([state_dict[key] for key in legacy_weight_keys], dim=0)
        if fused_bias_key not in state_dict and all(key in state_dict for key in legacy_bias_keys):
            state_dict[fused_bias_key] = torch.cat([state_dict[key] for key in legacy_bias_keys], dim=0)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )

        for key in legacy_weight_keys + legacy_bias_keys:
            if key in unexpected_keys:
                unexpected_keys.remove(key)

class Causal_LQ4x_Proj(nn.Module):

    def __init__(self, in_dim, out_dim, layer_num=30):
        super().__init__()
        self.ff = 1
        self.hh = 16
        self.ww = 16
        self.hidden_dim1 = 2048
        self.hidden_dim2 = 3072
        self.layer_num = layer_num

        self.pixel_shuffle = PixelShuffle3d(self.ff, self.hh, self.ww)

        self.conv1 = CausalConv3d(in_dim*self.ff*self.hh*self.ww, self.hidden_dim1, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)) # f -> f/2 h -> h w -> w
        self.norm1 = RMS_norm(self.hidden_dim1, images=False)
        self.act1 = nn.SiLU()

        self.conv2 = CausalConv3d(self.hidden_dim1, self.hidden_dim2, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)) # f -> f/2 h -> h w -> w
        self.norm2 = RMS_norm(self.hidden_dim2, images=False)
        self.act2 = nn.SiLU()

        self.out_dim = out_dim
        self.linear = nn.Linear(self.hidden_dim2, out_dim * layer_num)

        self.clip_idx = 0

    def forward(self, video):
        self.clear_cache()
        # x: (B, C, F, H, W)
        
        t = video.shape[2]
        iter_ = 1 + (t - 1) // 4
        first_frame = video[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
        video = torch.cat([first_frame, video], dim=2)
        # print(video.shape)

        out_x = []
        for i in range(iter_):
            x = self.pixel_shuffle(video[:,:,i*4:(i+1)*4,:,:])
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv1(x, self.cache['conv1'])
            self.cache['conv1'] = cache1_x
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            if i == 0:
                self.cache['conv2'] = cache2_x
                continue
            x = self.conv2(x, self.cache['conv2'])
            self.cache['conv2'] = cache2_x
            x = self.norm2(x)
            x = self.act2(x)
            out_x.append(x)
        out_x = torch.cat(out_x, dim = 2)
        out_x = rearrange(out_x, 'b c f h w -> b (f h w) c')
        return self._project_outputs(out_x)

    def clear_cache(self):
        self.cache = {}
        self.cache['conv1'] = None
        self.cache['conv2'] = None
        self.clip_idx = 0
    
    def stream_forward(self, video_clip):
        if self.clip_idx == 0:
            # self.clear_cache()
            first_frame = video_clip[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
            video_clip = torch.cat([first_frame, video_clip], dim=2)
            x = self.pixel_shuffle(video_clip)
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv1(x, self.cache['conv1'])
            self.cache['conv1'] = cache1_x
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv2'] = cache2_x
            self.clip_idx += 1
            return None
        else:
            x = self.pixel_shuffle(video_clip)
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv1(x, self.cache['conv1'])
            self.cache['conv1'] = cache1_x
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv2(x, self.cache['conv2'])
            self.cache['conv2'] = cache2_x
            x = self.norm2(x)
            x = self.act2(x)
            out_x = rearrange(x, 'b c f h w -> b (f h w) c')
            self.clip_idx += 1
            return self._project_outputs(out_x)

    def _project_outputs(self, out_x):
        fused = self.linear(out_x)
        fused = fused.view(*out_x.shape[:2], self.layer_num, self.out_dim)
        return [fused[:, :, i, :] for i in range(self.layer_num)]

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        legacy_weight_keys = [f"{prefix}linear_layers.{i}.weight" for i in range(self.layer_num)]
        legacy_bias_keys = [f"{prefix}linear_layers.{i}.bias" for i in range(self.layer_num)]
        fused_weight_key = f"{prefix}linear.weight"
        fused_bias_key = f"{prefix}linear.bias"

        if fused_weight_key not in state_dict and all(key in state_dict for key in legacy_weight_keys):
            state_dict[fused_weight_key] = torch.cat([state_dict[key] for key in legacy_weight_keys], dim=0)
        if fused_bias_key not in state_dict and all(key in state_dict for key in legacy_bias_keys):
            state_dict[fused_bias_key] = torch.cat([state_dict[key] for key in legacy_bias_keys], dim=0)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )

        for key in legacy_weight_keys + legacy_bias_keys:
            if key in unexpected_keys:
                unexpected_keys.remove(key)
