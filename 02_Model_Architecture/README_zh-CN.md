# 阶段 7：PyTorch 张量与 Self-Attention 底层拆解
- **当前进度**：跑通了 $Q \times K^T$ 的张量相乘，验证了除以根号 $d_k$ 防止方差爆炸，以及 Softmax 归一化注意力权重的数学逻辑。