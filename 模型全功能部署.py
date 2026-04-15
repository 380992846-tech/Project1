# ==================== complete_ai_toolkit.py ====================
# 整合所有功能：文档处理 + 字符级Transformer + 5个扩展功能
# 运行方式：
#   python complete_ai_toolkit.py --mode train        # 训练模型
#   python complete_ai_toolkit.py --mode classify     # 文本分类
#   python complete_ai_toolkit.py --mode generate     # 条件生成
#   python complete_ai_toolkit.py --mode rag          # RAG问答
#   python complete_ai_toolkit.py --mode dashboard    # 启动监控面板
#   python complete_ai_toolkit.py --mode quantize     # 模型量化

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import argparse
import os
from pathlib import Path
import json

# ==================== 1. 文档处理模块 ====================
class DocumentProcessor:
    """处理.docx和.txt文档，构建训练数据"""
    
    @staticmethod
    def read_docx(file_path):
        """读取docx文件（如果安装了python-docx）"""
        try:
            from docx import Document
            doc = Document(file_path)
            full_text = [para.text for para in doc.paragraphs]
            return '\n'.join(full_text)
        except ImportError:
            print("提示：未安装python-docx，跳过docx文件")
            return ""
    
    @staticmethod
    def read_txt(file_path):
        """读取txt文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @classmethod
    def build_corpus(cls, file_paths):
        """从多个文件构建语料库"""
        all_text = []
        for path in file_paths:
            if path.endswith('.docx'):
                text = cls.read_docx(path)
            elif path.endswith('.txt'):
                text = cls.read_txt(path)
            else:
                continue
            
            if text:
                all_text.append(text)
                print(f"✓ 已加载: {path} ({len(text)} 字符)")
        
        combined = '\n\n'.join(all_text)
        
        # 保存为txt备用
        with open('training_data.txt', 'w', encoding='utf-8') as f:
            f.write(combined)
        
        return combined

# ==================== 2. 字符级Transformer模型 ====================
class CharTransformer(nn.Module):
    """支持条件生成、分类、量化的增强版Transformer"""
    
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=4, 
                 max_len=512, dropout=0.1, num_classes=None, num_themes=None):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_themes = num_themes
        
        # 基础组件
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            batch_first=True, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出头（根据任务选择）
        self.fc_out = nn.Linear(d_model, vocab_size)  # 生成任务
        
        # 可选：分类头
        if num_classes:
            self.classifier = nn.Linear(d_model, num_classes)
        
        # 可选：主题嵌入（条件生成）
        if num_themes:
            self.theme_embedding = nn.Embedding(num_themes, d_model)
    
    def forward(self, x, theme_ids=None, task='generate'):
        """
        x: (batch, seq_len)
        theme_ids: (batch,) 可选，用于条件生成
        task: 'generate' 或 'classify'
        """
        seq_len = x.shape[1]
        embedded = self.embedding(x)
        
        # 添加主题向量（条件生成）
        if theme_ids is not None and self.num_themes:
            theme_emb = self.theme_embedding(theme_ids).unsqueeze(1)
            embedded = embedded + theme_emb
        
        # 位置编码
        embedded = embedded + self.pos_encoding[:, :seq_len, :]
        embedded = self.dropout(embedded)
        
        # Transformer
        out = self.transformer(embedded)
        
        if task == 'classify' and self.num_classes:
            # 分类任务：取平均池化
            pooled = out.mean(dim=1)
            logits = self.classifier(pooled)
            return logits
        else:
            # 生成任务
            logits = self.fc_out(out)
            return logits

# ==================== 3. 数据准备模块 ====================
class CharDataset(Dataset):
    """字符级数据集"""
    
    def __init__(self, text, seq_len=128, char_to_idx=None):
        self.seq_len = seq_len
        
        # 构建词汇表
        if char_to_idx is None:
            chars = sorted(list(set(text)))
            self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
            self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
            self.vocab_size = len(chars)
        else:
            self.char_to_idx = char_to_idx
            self.idx_to_char = {i: ch for ch, i in char_to_idx.items()}
            self.vocab_size = len(char_to_idx)
        
        # 转为数字序列
        self.data = torch.tensor([self.char_to_idx.get(ch, 0) for ch in text], dtype=torch.long)
        
        # 创建序列
        self.xs = []
        self.ys = []
        for i in range(0, len(self.data) - seq_len - 1, seq_len//2):  # 重叠采样
            self.xs.append(self.data[i:i+seq_len])
            self.ys.append(self.data[i+1:i+seq_len+1])
    
    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

# ==================== 4. 训练函数 ====================
def train_model(model, train_loader, val_loader, epochs=30, lr=0.001, device='cpu'):
    """通用训练函数"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # 训练
        model.train()
        total_train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            predictions = model(batch_x, task='generate')
            loss = criterion(predictions.reshape(-1, model.vocab_size), batch_y.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # 验证
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions = model(batch_x, task='generate')
                loss = criterion(predictions.reshape(-1, model.vocab_size), batch_y.reshape(-1))
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step()
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    # 保存训练历史
    with open('training_history.json', 'w') as f:
        json.dump(history, f)
    
    return model, history

# ==================== 5. 生成函数 ====================
def generate_text(model, start_str, char_to_idx, idx_to_char, length=300, 
                  temperature=0.8, theme_id=None, device='cpu'):
    """生成文本（支持条件生成）"""
    model.eval()
    
    # 转换起始字符串
    chars = [char_to_idx.get(ch, 0) for ch in start_str]
    input_ids = torch.tensor(chars).unsqueeze(0).to(device)
    generated = start_str
    
    # 主题条件（如果有）
    theme_tensor = None
    if theme_id is not None and hasattr(model, 'num_themes') and model.num_themes:
        theme_tensor = torch.tensor([theme_id]).to(device)
    
    for _ in range(length):
        if input_ids.shape[1] > 512:
            input_ids = input_ids[:, -512:]
        
        with torch.no_grad():
            output = model(input_ids, theme_ids=theme_tensor, task='generate')
            logits = output[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_char_idx = torch.multinomial(probs, 1).item()
        
        next_char = idx_to_char[next_char_idx]
        generated += next_char
        input_ids = torch.cat([input_ids, torch.tensor([[next_char_idx]]).to(device)], dim=1)
    
    return generated

# ==================== 6. 文本分类器 ====================
def train_classifier():
    """训练文本分类器（情感分析）"""
    print("\n=== 训练文本分类器 ===")
    
    # 模拟IMDB数据集（实际使用时替换为真实数据）
    # 这里创建示例数据：正面/负面评论
    texts = [
        "这部电影太棒了，非常好看", "糟糕透顶，浪费时间",
        "精彩绝伦，值得一看", "太差了，不推荐",
        # ... 更多数据
    ]
    labels = [1, 0, 1, 0]  # 1:正面, 0:负面
    
    # 构建词汇表
    all_chars = set(''.join(texts))
    char_to_idx = {ch: i for i, ch in enumerate(sorted(all_chars))}
    vocab_size = len(char_to_idx)
    
    # 转换为数字
    def text_to_tensor(text, max_len=100):
        indices = [char_to_idx.get(ch, 0) for ch in text[:max_len]]
        # 填充
        indices = indices + [0] * (max_len - len(indices))
        return torch.tensor(indices)
    
    X = torch.stack([text_to_tensor(t) for t in texts])
    y = torch.tensor(labels)
    
    # 创建模型
    model = CharTransformer(vocab_size=vocab_size, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    # 训练
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        logits = model(X, task='classify')
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            acc = (logits.argmax(dim=1) == y).float().mean()
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc:.4f}")
    
    # 保存分类器
    torch.save(model.state_dict(), 'classifier_model.pth')
    print("分类器训练完成！")
    return model

# ==================== 7. RAG系统 ====================
class SimpleRAG:
    """检索增强生成系统"""
    
    def __init__(self, generator_model, char_to_idx, idx_to_char, device='cpu'):
        self.generator = generator_model
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.device = device
        self.documents = []
        self.embeddings = None
        
        # 尝试加载sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ 已加载Embedding模型")
        except ImportError:
            print("⚠ 未安装sentence-transformers，将使用简单检索")
            self.encoder = None
    
    def build_index(self, documents):
        """构建检索索引"""
        self.documents = documents
        
        if self.encoder:
            # 使用语义检索
            self.embeddings = self.encoder.encode(documents)
        else:
            # 简单关键词匹配
            self.embeddings = [doc.lower() for doc in documents]
    
    def retrieve(self, query, top_k=3):
        """检索相关文档"""
        if self.encoder:
            query_vec = self.encoder.encode([query])
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_vec, self.embeddings)[0]
            top_indices = similarities.argsort()[-top_k:][::-1]
        else:
            # 简单关键词匹配
            query_words = set(query.lower().split())
            scores = [len(query_words & set(doc.lower().split())) for doc in self.embeddings]
            top_indices = np.argsort(scores)[-top_k:][::-1]
        
        return [self.documents[i] for i in top_indices]
    
    def generate_answer(self, query):
        """生成答案"""
        context = self.retrieve(query)
        prompt = f"根据以下资料回答问题：\n{chr(10).join(context)}\n\n问题：{query}\n答案："
        
        # 生成回复
        answer = generate_text(
            self.generator, prompt, self.char_to_idx, self.idx_to_char,
            length=200, temperature=0.7, device=self.device
        )
        return answer

# ==================== 8. 监控仪表盘 ====================
def launch_dashboard():
    """启动Streamlit监控面板"""
    dashboard_code = '''
import streamlit as st
import torch
import matplotlib.pyplot as plt
import json
from pathlib import Path

st.set_page_config(page_title="Transformer训练监控", layout="wide")
st.title("🤖 Transformer训练监控仪表盘")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 超参数配置")
    learning_rate = st.slider("学习率", 0.0001, 0.01, 0.001, 0.0001)
    num_layers = st.slider("Transformer层数", 1, 8, 4)
    batch_size = st.selectbox("批次大小", [16, 32, 64, 128])
    
    st.header("🎮 生成控制")
    temperature = st.slider("温度参数", 0.1, 2.0, 0.8, 0.05)
    gen_length = st.slider("生成长度", 50, 500, 200)

# 主区域：训练曲线
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 训练曲线")
    if Path("training_history.json").exists():
        with open("training_history.json") as f:
            history = json.load(f)
        
        fig, ax = plt.subplots()
        ax.plot(history['train_loss'], label='Train Loss')
        ax.plot(history['val_loss'], label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("暂无训练数据，请先训练模型")

with col2:
    st.subheader("💾 模型信息")
    if Path("best_model.pth").exists():
        st.success("✅ 模型已训练")
        model_size = Path("best_model.pth").stat().st_size / 1024 / 1024
        st.metric("模型大小", f"{model_size:.2f} MB")
    else:
        st.warning("未找到训练好的模型")
    
    st.subheader("🎨 在线生成")
    prompt = st.text_input("输入提示词", "机器学习")
    if st.button("生成"):
        st.write("生成中...")
        # 调用生成函数
        st.code(f"生成的文本示例: {prompt}相关的回复...")

# 实时损失（如果训练中）
st.subheader("🔄 实时监控")
loss_placeholder = st.empty()
if st.checkbox("显示实时数据"):
    loss_placeholder.info("等待训练数据...")
'''

    # 保存并运行
    with open('dashboard.py', 'w', encoding='utf-8') as f:
        f.write(dashboard_code)
    
    print("启动监控面板...")
    os.system("streamlit run dashboard.py")

# ==================== 9. 模型量化 ====================
def quantize_model(model_path='best_model.pth', vocab_size=100):
    """INT8量化模型"""
    print("\n=== 模型量化 ===")
    
    # 加载模型
    model = CharTransformer(vocab_size=vocab_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 准备校准数据
    calib_data = torch.randint(0, vocab_size, (10, 128))
    
    # 动态量化（最简单）
    model_int8 = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Embedding}, dtype=torch.qint8
    )
    
    # 测试推理速度
    def test_speed(model, test_input, runs=100):
        import time
        model.eval()
        times = []
        with torch.no_grad():
            for _ in range(runs):
                start = time.time()
                _ = model(test_input)
                times.append(time.time() - start)
        return np.mean(times) * 1000  # ms
    
    test_input = torch.randint(0, vocab_size, (1, 128))
    
    fp32_time = test_speed(model, test_input)
    int8_time = test_speed(model_int8, test_input)
    
    print(f"FP32 推理时间: {fp32_time:.2f} ms")
    print(f"INT8 推理时间: {int8_time:.2f} ms")
    print(f"加速比: {fp32_time/int8_time:.2f}x")
    
    # 保存量化模型
    torch.save(model_int8.state_dict(), 'model_int8.pth')
    print("量化模型已保存")
    
    return model_int8

# ==================== 10. 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description='完整AI工具包')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'classify', 'generate', 'rag', 'dashboard', 'quantize'],
                        help='运行模式')
    parser.add_argument('--docs', type=str, nargs='+', 
                        default=['机器学习.docx', '量化.docx', '贵系.docx'],
                        help='文档路径')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if args.mode == 'train':
        # 训练主模型
        print("加载文档...")
        processor = DocumentProcessor()
        text = processor.build_corpus(args.docs)
        
        print("准备数据...")
        dataset = CharDataset(text, seq_len=128)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        print(f"词汇表大小: {dataset.vocab_size}")
        print(f"训练样本: {train_size}, 验证样本: {val_size}")
        
        model = CharTransformer(vocab_size=dataset.vocab_size)
        model, history = train_model(model, train_loader, val_loader, epochs=30, device=device)
        
        # 测试生成
        sample = generate_text(model, "机器学习", dataset.char_to_idx, 
                              dataset.idx_to_char, length=200, device=device)
        print("\n生成样本：")
        print(sample)
    
    elif args.mode == 'classify':
        train_classifier()
    
    elif args.mode == 'generate':
        # 条件生成
        print("条件生成模式")
        # 加载已训练模型
        # 这里需要先有训练好的模型
    
    elif args.mode == 'rag':
        print("RAG问答系统")
        # 需要先加载模型和文档
    
    elif args.mode == 'dashboard':
        launch_dashboard()
    
    elif args.mode == 'quantize':
        quantize_model()

if __name__ == "__main__":
    main()