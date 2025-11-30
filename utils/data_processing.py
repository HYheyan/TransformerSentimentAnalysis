import re
import os
import glob

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def load_aclImdb_dataset(data_path="aclImdb"):
    texts = []
    labels = []
    
    # 检查数据集路径是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集路径 {data_path} 不存在，请确保已正确安装aclImdb数据集")
    
    try:
        # 加载负面评价
        neg_path = os.path.join(data_path, "train", "neg")
        if os.path.exists(neg_path):
            neg_files = glob.glob(os.path.join(neg_path, "*.txt"))
            for file_path in neg_files:
                with open(file_path, encoding='utf-8') as f:
                    text = f.read().strip()
                    texts.append(preprocess_text(text))
                    labels.append(0)  # 负面评价标记为0
        
        # 加载正面评价
        pos_path = os.path.join(data_path, "train", "pos")
        if os.path.exists(pos_path):
            pos_files = glob.glob(os.path.join(pos_path, "*.txt"))
            for file_path in pos_files:
                with open(file_path, encoding='utf-8') as f:
                    text = f.read().strip()
                    texts.append(preprocess_text(text))
                    labels.append(1)  # 正面评价标记为1
                    
        print(f"成功加载 {len(texts)} 条aclImdb数据 ({len([l for l in labels if l == 0])} 条负面评价, {len([l for l in labels if l == 1])} 条正面评价)")
        return texts, labels
        
    except Exception as e:
        raise RuntimeError(f"加载aclImdb数据集时出错: {e}")