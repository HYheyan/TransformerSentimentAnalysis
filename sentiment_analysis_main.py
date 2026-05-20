import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.transformer_model import TransformerTextClassificationModel, train_transformer_model, \
    evaluate_transformer_model, TransformerDataset
from utils.data_processing import load_aclImdb_train_test_dataset
from utils.simple_tokenizer import SimpleVocab, SimpleTokenizer


def main():
    """主函数"""
    print("=== 使用Transformer模型进行文本情感分析 ===")
    
    # 设置TensorBoard日志目录
    log_dir = "runs/transformer_sentiment_analysis_" + time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir)

    # 直接加载aclImdb数据集（使用train作为训练集，test作为测试集）
    print("正在加载aclImdb数据集")
    (train_texts, train_labels), (test_texts, test_labels) = load_aclImdb_train_test_dataset("aclImdb")
    
    # 创建词汇表和分词器（基于训练集）
    vocab = SimpleVocab()
    vocab.build_from_texts(train_texts)
    tokenizer = SimpleTokenizer(vocab)
    print(f"词汇表大小: {len(vocab)}")

    # 创建训练集和测试集
    train_dataset = TransformerDataset(train_texts, train_labels, tokenizer)
    test_dataset = TransformerDataset(test_texts, test_labels, tokenizer)
    print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")
    batch_size = 64  # 序列缩短后，增大batch size以充分利用GPU/内存
    print(f"使用批次大小: {batch_size}")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型实例
    # 用户自主选择是否启用情感显著性筛选
    print("\n" + "="*60)
    print("情感显著性筛选配置")
    print("="*60)
    print("1. 启用筛选 - 序列长度从128降至32，加速训练")
    print("2. 禁用筛选 - 使用完整序列长度128")
    print("="*60)
    
    # 安全的输入处理
    selective_attention = True  # 默认启用
    top_k = 32
    
    try:
        import sys
        if sys.stdin.isatty():  # 检查是否是交互式终端
            choice = input("请选择 (1/2, 直接回车默认1): ").strip()
            if choice == '2':
                selective_attention = False
                top_k = 128
                print("已选择: 禁用情感显著性筛选")
            else:
                print("已选择: 启用情感显著性筛选（默认）")
        else:
            print("非交互模式，使用默认配置：启用情感显著性筛选")
    except (EOFError, IOError, OSError) as e:
        print(f"输入读取失败 ({type(e).__name__})，使用默认配置：启用情感显著性筛选")
    
    model = TransformerTextClassificationModel(
        vocab_size=len(vocab),
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=512,
        dropout=0.5,
        num_classes=2,
        top_k=top_k,
        selective_attention=selective_attention
    )
    
    print(f"\n配置信息:")
    print(f"  - 情感显著性筛选: {'启用' if selective_attention else '禁用'}")
    if selective_attention:
        print(f"  - 筛选后保留token数: {top_k} (原始序列长度: 128)")
    else:
        print(f"  - 使用完整序列长度: 128")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"模型将使用{device}进行训练")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    # 为aclImdb数据集使用固定的学习率和优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    print("\n开始训练模型")
    epochs = 50  # aclImdb数据集训练50轮
    
    print(f"训练轮数: {epochs}")
    
    start_time = time.time()
    
    # 记录最佳准确率
    best_train_accuracy = 0.0
    best_val_accuracy = 0.0
    
    for epoch in range(epochs):
        try:
            train_loss, train_acc = train_transformer_model(model, train_dataloader, criterion, optimizer, device)
            val_loss, val_acc = evaluate_transformer_model(model, val_dataloader, criterion, device)

            scheduler.step(val_loss) # 更新学习率调度器

            # 记录到TensorBoard
            writer.add_scalar('Training Loss', train_loss, epoch)
            writer.add_scalar('Training Accuracy', train_acc, epoch)
            writer.add_scalar('Validation Loss', val_loss, epoch)
            writer.add_scalar('Validation Accuracy', val_acc, epoch)
            
            display_freq = 5   # 统一每5轮显示一次
            
            # 定期在终端显示
            if (epoch + 1) % display_freq == 0:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
            # 保存最佳模型
            if train_acc > best_train_accuracy:
                best_train_accuracy = train_acc
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(model.state_dict(), 'best_transformer_model.pth')
                
        except Exception as e:
            print(f"在epoch {epoch+1}训练时出现错误: {e}")
            break
    
    print(f"\n训练完成! 总耗时: {time.time() - start_time:.2f} 秒")
    print(f"最佳训练准确率: {best_train_accuracy:.4f}")
    print(f"最佳验证准确率: {best_val_accuracy:.4f}")

    writer.close()
    print(f"\nTensorBoard日志已保存到: {log_dir}")
    print(f"运行tensorboard --logdir={log_dir}启动TensorBoard查看训练过程:")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序已被用户中断")
    except Exception as e:
        print(f"程序执行出错: {e}")