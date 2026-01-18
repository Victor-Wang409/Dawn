import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import soundfile as sf
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from tqdm import tqdm
import os
import argparse
import sys
import logging
from datetime import datetime

from util import split_iemocap

# 回归头（论文指定）
class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits

class AudioDataset(Dataset):
    def __init__(self, csv_file, audio_dir, processor, max_len=250000):
        self.df = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.processor = processor
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row['FileName'])
        
        # 加载音频
        audio, sampling_rate = sf.read(audio_path)
        
        # 确保音频是单声道
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
            
        # 处理音频
        inputs = self.processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
        input_values = inputs.input_values.squeeze(0)
        
        # 裁剪或填充到最大长度
        if input_values.shape[0] > self.max_len:
            input_values = input_values[:self.max_len]
        elif input_values.shape[0] < self.max_len:
            pad_length = self.max_len - input_values.shape[0]
            input_values = torch.nn.functional.pad(input_values, (0, pad_length))
        
        # 获取VAD标签 (并确保顺序为 valence, arousal, dominance)
        vad_values = eval(row['VAD_normalized'])
        labels = torch.tensor(vad_values, dtype=torch.float32)
        
        return input_values, labels

class CCCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, preds, labels):
        pred_mean = torch.mean(preds, dim=0)
        label_mean = torch.mean(labels, dim=0)
        
        pred_var = torch.var(preds, dim=0, unbiased=False)
        label_var = torch.var(labels, dim=0, unbiased=False)
        
        # 计算协方差
        covariance = torch.mean((preds - pred_mean) * (labels - label_mean), dim=0)
        
        # 计算CCC
        ccc = (2 * covariance) / (pred_var + label_var + (pred_mean - label_mean)**2 + 1e-8)
        
        # 返回1-平均CCC作为损失（因为我们要最大化CCC）
        return 1 - torch.mean(ccc)

def compute_ccc(preds, labels):
    """计算CCC值"""
    mean_pred = np.mean(preds, axis=0)
    mean_label = np.mean(labels, axis=0)
    
    var_pred = np.var(preds, axis=0)
    var_label = np.var(labels, axis=0)
    
    covariance = np.mean((preds - mean_pred) * (labels - mean_label), axis=0)
    
    ccc = (2 * covariance) / (var_pred + var_label + (mean_pred - mean_label)**2 + 1e-8)
    return ccc

def train(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{total_epochs} [Train]', leave=False)
    
    for batch in train_loader:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        _, logits = model(inputs)
        
        # 计算损失
        loss = criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.update(1)
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    progress_bar.close()
    return total_loss / len(train_loader)

def evaluate(model, data_loader, criterion, device, desc='Eval'):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(total=len(data_loader), desc=desc, leave=False)
    
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            _, logits = model(inputs)
            
            # 计算损失
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # 收集预测和标签
            all_preds.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            progress_bar.update(1)
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    progress_bar.close()
    
    # 拼接所有预测和标签
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # 计算每个维度的CCC
    ccc_values = compute_ccc(all_preds, all_labels)
    
    return total_loss / len(data_loader), ccc_values

def train_fold(args, fold_idx, folds, dataset, device, processor):
    """训练单个fold"""
    fold_dir = os.path.join(args.output_dir, f'fold{fold_idx+1}')
    os.makedirs(fold_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(fold_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    logging.info(f"{'='*50}")
    logging.info(f"Training Fold {fold_idx+1}/5")
    logging.info(f"{'='*50}")
    
    # 获取当前fold的数据
    current_fold = folds[fold_idx]
    train_idx = current_fold['train_idx']
    eval_idx = current_fold['eval_idx']
    test_idx = current_fold['test_idx']
    
    logging.info(f"Training samples: {len(train_idx)}")
    logging.info(f"Validation samples: {len(eval_idx)}")
    logging.info(f"Test samples: {len(test_idx)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(train_idx),
        num_workers=4
    )
    
    val_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(eval_idx),
        num_workers=4
    )
    
    test_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(test_idx),
        num_workers=4
    )
    
    # 创建模型
    model = EmotionModel.from_pretrained(args.model_name).to(device)
    
    # 设置优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = CCCLoss()
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # 创建metrics跟踪文件
    metrics_file = os.path.join(fold_dir, 'metrics.csv')
    with open(metrics_file, 'w') as f:
        f.write('epoch,train_loss,val_loss,val_ccc_v,val_ccc_a,val_ccc_d,val_ccc_avg\n')
    
    # 训练循环
    best_val_ccc = -float('inf')
    best_model = None
    early_stop_counter = 0
    
    for epoch in range(args.epochs):
        # 训练一个epoch
        train_loss = train(model, train_loader, optimizer, criterion, device, epoch+1, args.epochs)
        
        # 验证
        val_loss, val_ccc = evaluate(model, val_loader, criterion, device, desc=f'Epoch {epoch+1}/{args.epochs} [Val]')
        val_ccc_avg = np.mean(val_ccc)
        
        # 更新学习率
        scheduler.step(val_ccc_avg)
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录指标
        with open(metrics_file, 'a') as f:
            f.write(f'{epoch+1},{train_loss:.4f},{val_loss:.4f},{val_ccc[0]:.4f},{val_ccc[1]:.4f},{val_ccc[2]:.4f},{val_ccc_avg:.4f}\n')
        
        # 记录日志
        logging.info(
            f"Fold {fold_idx+1}, Epoch {epoch+1:3d} | "
            f"LR: {current_lr:.2e} | "
            f"Loss: {train_loss:.4f} | "
            f"Val CCC: V={val_ccc[0]:.3f}, A={val_ccc[1]:.3f}, D={val_ccc[2]:.3f} | "
            f"Avg={val_ccc_avg:.3f}"
        )
        
        # 创建epoch目录
        epoch_dir = os.path.join(fold_dir, f'epoch{epoch+1}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        # 保存当前epoch的模型
        model.save_pretrained(epoch_dir, safe_serialization=False)
        
        # 保存优化器状态
        torch.save(optimizer.state_dict(), os.path.join(epoch_dir, 'optimizer.pt'))
        
        # 更新最佳模型
        if val_ccc_avg > best_val_ccc:
            best_val_ccc = val_ccc_avg
            best_model = model.state_dict()
            
            # 创建并保存最佳模型
            best_model_dir = os.path.join(fold_dir, 'best_model')
            os.makedirs(best_model_dir, exist_ok=True)
            model.save_pretrained(best_model_dir, safe_serialization=False)
            
            logging.info(f"Saved new best model with val_ccc={val_ccc_avg:.3f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # 保存checkpoint以便恢复训练
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_ccc': best_val_ccc
        }
        torch.save(checkpoint, os.path.join(fold_dir, 'checkpoint.pt'))
        
        # 早停检查
        if early_stop_counter >= args.patience:
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # 加载最佳模型进行测试
    model.load_state_dict(best_model)
    test_loss, test_ccc = evaluate(model, test_loader, criterion, device, desc='Testing')
    test_ccc_avg = np.mean(test_ccc)
    
    # 保存测试结果
    with open(os.path.join(fold_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Test CCC:\nValence: {test_ccc[0]:.3f}\nArousal: {test_ccc[1]:.3f}\n"
               f"Dominance: {test_ccc[2]:.3f}\nAverage: {test_ccc_avg:.3f}")
    
    logging.info(
        f"Test Results for Fold {fold_idx+1}:\n"
        f"CCC Valence: {test_ccc[0]:.3f}\n"
        f"CCC Arousal: {test_ccc[1]:.3f}\n"
        f"CCC Dominance: {test_ccc[2]:.3f}\n"
        f"Average CCC: {test_ccc_avg:.3f}"
    )
    
    return test_ccc

def main():
    parser = argparse.ArgumentParser(description='Train Dawn Emotion Model with IEMOCAP dataset')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file with annotations')
    parser.add_argument('--audio_dir', type=str, required=True, help='Directory containing audio files')
    parser.add_argument('--model_name', type=str, default='audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim',
                        help='Pre-trained model name or path')
    parser.add_argument('--output_dir', type=str, default='./model_output', help='Directory to save the model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--max_len', type=int, default=250000, help='Maximum audio length in samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--specific_fold', type=int, default=None, help='Train specific fold only (0-4). None for all folds')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置主日志
    main_log_file = os.path.join(args.output_dir, 'main.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(main_log_file)
        ]
    )
    
    # 记录实验开始时间
    start_time = datetime.now()
    logging.info(f"Experiment started at: {start_time}")
    logging.info(f"Arguments: {args}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # 加载处理器
    logging.info(f"Loading model processor: {args.model_name}")
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    
    # 准备数据
    logging.info(f"Loading data from: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    
    # 使用split_iemocap函数划分数据集
    logging.info("Splitting dataset using split_iemocap function...")
    folds = split_iemocap(df)
    
    # 创建数据集
    dataset = AudioDataset(args.csv_path, args.audio_dir, processor, args.max_len)
    
    # 训练所有folds或指定fold
    fold_results = []
    
    if args.specific_fold is not None:
        # 训练特定fold
        logging.info(f"Training specific fold: {args.specific_fold+1}/5")
        test_ccc = train_fold(args, args.specific_fold, folds, dataset, device, processor)
        fold_results.append(test_ccc)
    else:
        # 训练所有folds
        for fold_idx in range(len(folds)):
            test_ccc = train_fold(args, fold_idx, folds, dataset, device, processor)
            fold_results.append(test_ccc)
    
    # 如果训练了多个fold，计算平均结果
    if len(fold_results) > 1:
        # 计算所有fold的平均结果
        fold_results = np.array(fold_results)
        avg_v = np.mean(fold_results[:, 0])
        avg_a = np.mean(fold_results[:, 1])
        avg_d = np.mean(fold_results[:, 2])
        avg_all = np.mean(fold_results.mean(axis=1))
        
        std_v = np.std(fold_results[:, 0])
        std_a = np.std(fold_results[:, 1])
        std_d = np.std(fold_results[:, 2])
        
        final_results = (
            f"Final Cross-Validation Results\n"
            f"Average CCC ± std:\n"
            f"Valence: {avg_v:.3f} ± {std_v:.3f}\n"
            f"Arousal: {avg_a:.3f} ± {std_a:.3f}\n"
            f"Dominance: {avg_d:.3f} ± {std_d:.3f}\n"
            f"Overall: {avg_all:.3f}"
        )
        
        logging.info(f"\n{'='*50}\n{final_results}\n{'='*50}")
        
        # 保存最终结果
        with open(os.path.join(args.output_dir, 'final_results.txt'), 'w') as f:
            f.write(final_results)
    
    # 记录实验结束时间
    end_time = datetime.now()
    duration = end_time - start_time
    logging.info(f"Experiment completed at: {end_time}")
    logging.info(f"Total duration: {duration}")

if __name__ == "__main__":
    main()