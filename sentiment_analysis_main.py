import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from models.transformer_model import TransformerTextClassificationModel, train_transformer_model, \
    evaluate_transformer_model, TransformerDataset
from utils.data_processing import load_aclImdb_dataset
from utils.simple_tokenizer import SimpleVocab, SimpleTokenizer


def main():
    """主函数"""
    print("=== 使用Transformer模型进行文本情感分析 ===")
    
    # 设置TensorBoard日志目录
    log_dir = "runs/transformer_sentiment_analysis_" + time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir)

    # 直接加载aclImdb数据集
    print("正在加载aclImdb数据集")
    texts, labels = load_aclImdb_dataset("aclImdb")
    print(f"数据集大小: {len(texts)} 条文本")
    
    # 创建词汇表和分词器
    vocab = SimpleVocab()
    vocab.build_from_texts(texts)
    tokenizer = SimpleTokenizer(vocab)
    print(f"词汇表大小: {len(vocab)}")

    # 划分训练集和验证集
    dataset = TransformerDataset(texts, labels, tokenizer)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    batch_size = 16
    print(f"使用批次大小: {batch_size}")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型实例
    model = TransformerTextClassificationModel(
        vocab_size=len(vocab),
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=512,
        dropout=0.5,
        num_classes=2
    )
    
    # 如果有GPU则使用GPU
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
            if val_acc is not None and val_acc > best_val_accuracy:
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