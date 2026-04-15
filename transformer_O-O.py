import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ========== 1. 模型定义（和你刚才的一样）==========
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size=100, d_model=32, nhead=4, num_layers=2, max_len=50):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=128,
            batch_first=True,
            dropout=0.1  # 防止过拟合
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        seq_len = x.shape[1]
        embedded = self.embedding(x)
        embedded = embedded + self.pos_encoding[:, :seq_len, :]
        out = self.transformer(embedded)
        logits = self.fc_out(out)
        return logits

# ========== 2. 准备训练数据 ==========
def create_simple_dataset():
    """创建一个简单的算术序列数据集：学习 [a, b, c] -> [b, c, a+b] 模式"""
    sequences = []
    for i in range(1, 50):
        for j in range(1, 50):
            a, b = i, j
            c = (a + b) % 90 + 1  # 限制在1-90之间
            sequences.append([a, b, c, a+b])
    
    # 转为tensor
    data = torch.tensor(sequences)
    # 输入：前3个数字，输出：后3个数字（偏移1位，做next token prediction）
    x = data[:, :3]
    y = data[:, 1:4]
    return x, y

# ========== 3. 训练配置（这里是你需要调的核心）==========
vocab_size = 100  # 词汇表大小（数字0-99）
model = MiniTransformer(vocab_size=vocab_size, d_model=64, nhead=4, num_layers=3)
criterion = nn.CrossEntropyLoss()  # 分类损失
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # AdamW比Adam更稳
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)  # 学习率衰减

# 准备数据
x, y = create_simple_dataset()
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ========== 4. 训练循环（核心）==========
num_epochs = 2000
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()  # 设置为训练模式
    total_loss = 0
    
    for batch_x, batch_y in dataloader:
        # 前向传播
        predictions = model(batch_x)  # (batch, seq_len, vocab_size)
        
        # 计算损失（需要reshape）
        loss = criterion(predictions.reshape(-1, vocab_size), batch_y.reshape(-1))
        
        # 反向传播（和你线性回归一模一样！）
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸，Transformer训练常用）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    scheduler.step()  # 更新学习率
    
    # 保存最佳模型
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'best_mini_transformer.pth')
    
    # 打印进度
    if epoch % 100 == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:4d} | Loss: {avg_loss:.4f} | LR: {lr:.5f}")

# ========== 5. 测试训练好的模型 ==========
model.eval()
test_x = torch.tensor([[10, 20, 30]])  # 应该预测 20, 30, 40
with torch.no_grad():
    output = model(test_x)
    predicted = output.argmax(dim=-1)
    print(f"\n输入: {test_x[0].tolist()}")
    print(f"预测: {predicted[0].tolist()}")