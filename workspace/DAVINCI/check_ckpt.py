import torch
import os

ckpt_path = '/kwkj-k8s/davinci/LJH/daVinci-MagiHuman2/output_train_ltx_vertical_protected/ckpt_best.pt'
if os.path.exists(ckpt_path):
    size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"=== 最优模型信息 (ckpt_best.pt) ===")
    print(f"文件路径: {ckpt_path}")
    print(f"文件大小: {size_mb:.2f} MB")
    
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        print(f"包含的键值 (Keys): {list(ckpt.keys())}")
        
        if 'step' in ckpt:
            print(f"训练步数 (Step): {ckpt['step']}")
        if 'best_loss' in ckpt:
            print(f"最佳 Loss (Best Loss): {ckpt['best_loss']:.6f}")
        if 'loss' in ckpt:
            print(f"当前 Loss (Loss): {ckpt['loss']:.6f}")
            
        if 'model' in ckpt:
            print(f"微调层数量 (Model Tensor Count): {len(ckpt['model'])}")
    except Exception as e:
        print(f"读取权重信息失败: {e}")
else:
    print(f"未找到文件: {ckpt_path}")
