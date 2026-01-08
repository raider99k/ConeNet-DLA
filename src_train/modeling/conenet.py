
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.non_linearity = nn.ReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.non_linearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.non_linearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

class ConeNetHead(nn.Module):
    def __init__(self, in_channels, heads_config):
        super(ConeNetHead, self).__init__()
        # heads_config: {'heatmap': 1, 'offset': 2, 'size': 2} -> Total 5
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        
        # We want to output a single tensor (N, 5, H, W) to avoid Split/Concat overhead if possible,
        # but the prompt says "Channel 0: Heatmap, 1-2: Offset, 3-4: Size".
        # RESEARCH.md Section 5.1: "Questa struttura è puramente convoluzionale (Conv 3x3 -> ReLU -> Conv 1x1)"
        
        total_out_channels = sum(heads_config.values())
        # Atomic-C alignment: DLA works better with multiples of 32. 
        # Even if the final output is 5, we pad to 32 for internal efficiency.
        self.pad_to_32 = (32 - (total_out_channels % 32)) % 32
        self.conv2 = nn.Conv2d(in_channels, total_out_channels + self.pad_to_32, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.pad_to_32 > 0:
            # We only return the relevant channels to the loss/decoder,
            # but the convolution above was executed with 32-aligned filters.
            x = x[:, :x.size(1) - self.pad_to_32, ...]
        return x

class ConeNet(nn.Module):
    def __init__(self, deploy=False):
        super(ConeNet, self).__init__()
        self.deploy = deploy
        
        # Stem: 3x3 s2, 32 ch
        self.stem = RepVGGBlock(in_channels=4, out_channels=32, kernel_size=3, stride=2, padding=1, deploy=deploy)
        
        # Stage 1: 32 -> 64, s2
        self.stage1 = RepVGGBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, deploy=deploy)
        
        # Stage 2: 3 blocks. 64 -> 128. First stride 2.
        self.stage2 = self._make_stage(64, 128, 3, stride=2, deploy=deploy)
        
        # Stage 3: 6 blocks. 128 -> 256. First stride 2.
        self.stage3 = self._make_stage(128, 256, 6, stride=2, deploy=deploy)
        
        # Stage 4: 2 blocks. 256 -> 512. First stride 2.
        self.stage4 = self._make_stage(256, 512, 2, stride=2, deploy=deploy)
        
        # Heads attached to Stage 3 (P3) and Stage 4 (P4)
        # P3 has 256 channels. P4 has 512 channels.
        self.head_p3 = ConeNetHead(256, {'heatmap': 1, 'offset': 2, 'size': 2})
        self.head_p4 = ConeNetHead(512, {'heatmap': 1, 'offset': 2, 'size': 2})

    def _make_stage(self, in_channels, out_channels, num_blocks, stride, deploy):
        layers = []
        # First block handles stride and channel change
        layers.append(RepVGGBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, deploy=deploy))
        # Subsequent blocks are identity-mapped (same channels, stride 1)
        for _ in range(1, num_blocks):
            layers.append(RepVGGBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, deploy=deploy))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input check and padding if needed (Hard Constraint 5)
        # Expected input: (N, 4, H, W) or (N, 3, H, W)
        if x.size(1) == 3:
            # Explicit padding to 4 channels for DLA efficiency (Atomic-C).
            # Using constant pad which is well supported in ONNX/TensorRT.
            x = F.pad(x, (0, 0, 0, 0, 0, 1), "constant", 0)
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        p3_feat = self.stage3(x)
        p4_feat = self.stage4(p3_feat)
        
        out_p3 = self.head_p3(p3_feat)
        out_p4 = self.head_p4(p4_feat)
        
        return [out_p3, out_p4]

    def switch_to_deploy(self):
        if self.deploy:
            return
        
        def fuse_module(m):
            if isinstance(m, RepVGGBlock):
                m.switch_to_deploy()
            
            # Fuse Head Conv-BN-ReLU -> Conv-ReLU (Simulated, BN is absorbed)
            # Actually standard fuse is Conv+BN. 
            # In Head: Conv1 -> BN1 -> ReLU -> Conv2
            # We can fuse Conv1+BN1.
            if isinstance(m, ConeNetHead):
                # Fuse conv1 + bn1
                 kernel, bias = self._fuse_conv_bn_tensor(m.conv1, m.bn1)
                 m.conv1 = nn.Conv2d(m.conv1.in_channels, m.conv1.out_channels, 
                                     m.conv1.kernel_size, m.conv1.stride, m.conv1.padding, bias=True)
                 m.conv1.weight.data = kernel
                 m.conv1.bias.data = bias
                 m.bn1 = nn.Identity()
                 # conv2 is already bias=True, no BN.
        
        self.apply(fuse_module)
        self.deploy = True

    def _fuse_conv_bn_tensor(self, conv, bn):
        # Similar logic to RepVGG
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
