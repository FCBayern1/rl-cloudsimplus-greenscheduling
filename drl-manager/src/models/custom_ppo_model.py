import numpy as np
import gymnasium as gym
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

class CustomPPOModel(TorchModelV2, nn.Module):
    """
    自定义 PPO 模型示例
    
    功能：
    1. 自定义特征提取层 (Feature Extractor)
    2. 自定义 Value Function 分支
    3. 可选：自定义 Action Masking 支持
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # 1. 打印调试信息 (可选)
        # print(f"[CustomModel] Obs Space: {obs_space}")
        # print(f"[CustomModel] Model Config: {model_config}")

        # 2. 构建自定义特征提取网络
        # 这里我们复用 RLlib 的 FullyConnectedNetwork 作为基础
        # 你可以替换成 CNN, LSTM, Attention 等
        self.internal_model = TorchFC(
            obs_space, 
            action_space, 
            num_outputs, 
            model_config, 
            name + "_internal"
        )
        
        # 3. 额外的自定义层 (示例：添加一个 Attention 层或其他逻辑)
        # self.attention = nn.MultiheadAttention(...) 

        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        """
        前向传播
        
        Args:
            input_dict: 包含 "obs", "obs_flat" 等
            state: RNN 状态 (如果使用 LSTM)
            seq_lens: 序列长度
            
        Returns:
            (logits, state)
        """
        # 1. 获取观察值
        obs = input_dict["obs"]
        
        # 2. 通过内部网络 (这里是 FC)
        # model_out 是 action logits
        # self._value_out 会被 internal_model 更新
        model_out, _ = self.internal_model(input_dict, state, seq_lens)
        
        # 3. 获取 value function 输出
        self._value_out = self.internal_model.value_function()
        
        # 4. (可选) 自定义 Action Masking 逻辑
        # 如果环境提供了 action_mask，在这里应用
        if "action_mask" in input_dict["obs"]:
            action_mask = input_dict["obs"]["action_mask"]
            inf_mask = torch.clamp(torch.log(action_mask), min=-1e10)
            model_out = model_out + inf_mask

        return model_out, state

    def value_function(self):
        """返回 Value Function 的估计值 (Critic)"""
        return self._value_out

    def custom_loss(self, policy_loss, loss_inputs):
        """
        [高级] 自定义 Loss 函数
        
        如果你想修改 PPO 的 Loss 计算逻辑，可以在这里添加。
        例如：添加额外的正则化项，或者修改 Value Loss 的权重。
        
        Args:
            policy_loss: RLlib 计算的基础 PPO Loss
            loss_inputs: 包含 batch 数据的字典
            
        Returns:
            List of losses (scalar tensors)
        """
        # 示例：添加 L2 正则化 loss
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param)
            
        # 将自定义 loss 加到原有的 policy loss 上
        # 注意：返回的是 list
        custom_total_loss = [loss + 0.0001 * l2_loss for loss in policy_loss]
        
        return custom_total_loss

