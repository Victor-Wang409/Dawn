import numpy as np

def split_iemocap(df):
    """基于说话人ID划分数据集为5折"""
    # 从文件名中提取session信息(格式如Ses01F_impro01_F000)
    df['session'] = df['FileName'].apply(lambda x: x[:5])  # 提取Ses01这样的前缀
    sessions = sorted(df['session'].unique())  # 获取所有session ['Ses01', 'Ses02', ...]
    # 确保正好有5个session
    assert len(sessions) == 5, f"预期5个session,但找到{len(sessions)}个session"
    # 用于存储5折的结果
    folds = []
    # 为每个session创建一折
    for test_session in sessions:
        # 获取当前session的所有样本索引作为测试集
        test_idx = df[df['session'] == test_session].index.values
        # 获取其他session的样本索引
        other_sessions_idx = df[df['session'] != test_session].index.values
        # 随机打乱其他session的索引
        np.random.shuffle(other_sessions_idx)
        # 计算验证集大小(其他session样本总数的20%)
        eval_size = int(len(other_sessions_idx) * 0.2)
        # 划分验证集和训练集
        eval_idx = other_sessions_idx[:eval_size]
        train_idx = other_sessions_idx[eval_size:]
        # 将当前折的划分结果存储在字典中
        fold_info = {
            'train_idx': train_idx,
            'eval_idx': eval_idx,
            'test_idx': test_idx
        }
        folds.append(fold_info)
        # 打印当前折的详细信息
        print(f"\nFold for test session {test_session}:")
        print(f"Training set: {len(train_idx)} samples")
        print(f"Validation set: {len(eval_idx)} samples")
        print(f"Test set: {len(test_idx)} samples")
        # 打印每个集合中包含的session
        train_sessions = sorted(df.iloc[train_idx]['session'].unique())
        eval_sessions = sorted(df.iloc[eval_idx]['session'].unique())
        test_sessions = sorted(df.iloc[test_idx]['session'].unique())
    
    return folds

def split_msppodcast(df):
    """基于Split_Set列直接划分数据集为训练、验证和测试集"""
    # 根据Split_Set列获取索引
    train_idx = df[df['Split_Set'] == 'Train'].index.values
    eval_idx = df[df['Split_Set'] == 'Validation'].index.values
    test_idx = df[df['Split_Set'] == 'Test'].index.values
    
    # 获取每个集合中唯一的说话人数量
    train_speakers = df[df['Split_Set'] == 'Train']['SpkrID'].nunique()
    eval_speakers = df[df['Split_Set'] == 'Validation']['SpkrID'].nunique()
    test_speakers = df[df['Split_Set'] == 'Test']['SpkrID'].nunique()
    
    # 打印详细统计信息
    print("\nDataset Split Statistics:")
    print(f"Training set: {len(train_idx)} samples ({train_speakers} speakers)")
    print(f"Validation set: {len(eval_idx)} samples ({eval_speakers} speakers)")
    print(f"Test set: {len(test_idx)} samples ({test_speakers} speakers)")
    
    # 计算实际的比例
    total_samples = len(train_idx) + len(eval_idx) + len(test_idx)
    print(f"\nActual split ratio:")
    print(f"Train: {len(train_idx)/total_samples:.1%}")
    print(f"Val: {len(eval_idx)/total_samples:.1%}")
    print(f"Test: {len(test_idx)/total_samples:.1%}")
    
    return [{
        'train_idx': train_idx,
        'eval_idx': eval_idx,
        'test_idx': test_idx
    }]