import re
import os
import glob

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def load_aclImdb_train_test_dataset(data_path="aclImdb"):
    """
    分别加载训练集和测试集
    返回: (train_texts, train_labels), (test_texts, test_labels)
    """
    train_texts = []
    train_labels = []
    test_texts = []
    test_labels = []
    
    # 检查数据集路径是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集路径 {data_path} 不存在，请确保已正确安装aclImdb数据集")
    
    try:
        # 加载训练集 - 负面评价
        train_neg_path = os.path.join(data_path, "train", "neg")
        if os.path.exists(train_neg_path):
            train_neg_files = glob.glob(os.path.join(train_neg_path, "*.txt"))
            for file_path in train_neg_files:
                with open(file_path, encoding='utf-8') as f:
                    text = f.read().strip()
                    train_texts.append(preprocess_text(text))
                    train_labels.append(0)
        
        # 加载训练集 - 正面评价
        train_pos_path = os.path.join(data_path, "train", "pos")
        if os.path.exists(train_pos_path):
            train_pos_files = glob.glob(os.path.join(train_pos_path, "*.txt"))
            for file_path in train_pos_files:
                with open(file_path, encoding='utf-8') as f:
                    text = f.read().strip()
                    train_texts.append(preprocess_text(text))
                    train_labels.append(1)
        
        # 加载测试集 - 负面评价
        test_neg_path = os.path.join(data_path, "test", "neg")
        if os.path.exists(test_neg_path):
            test_neg_files = glob.glob(os.path.join(test_neg_path, "*.txt"))
            for file_path in test_neg_files:
                with open(file_path, encoding='utf-8') as f:
                    text = f.read().strip()
                    test_texts.append(preprocess_text(text))
                    test_labels.append(0)
        
        # 加载测试集 - 正面评价
        test_pos_path = os.path.join(data_path, "test", "pos")
        if os.path.exists(test_pos_path):
            test_pos_files = glob.glob(os.path.join(test_pos_path, "*.txt"))
            for file_path in test_pos_files:
                with open(file_path, encoding='utf-8') as f:
                    text = f.read().strip()
                    test_texts.append(preprocess_text(text))
                    test_labels.append(1)
        
        print(f"训练集: {len(train_texts)} 条数据 ({len([l for l in train_labels if l == 0])} 条负面评价, {len([l for l in train_labels if l == 1])} 条正面评价)")
        print(f"测试集: {len(test_texts)} 条数据 ({len([l for l in test_labels if l == 0])} 条负面评价, {len([l for l in test_labels if l == 1])} 条正面评价)")
        
        return (train_texts, train_labels), (test_texts, test_labels)
        
    except Exception as e:
        raise RuntimeError(f"加载aclImdb数据集时出错: {e}")