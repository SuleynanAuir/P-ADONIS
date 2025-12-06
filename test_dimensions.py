#!/usr/bin/env python3
"""
测试TPEM的维度问题
"""
import torch
import yaml
from easydict import EasyDict
from interfaces.super_resolution import TextSR
from dataset.dataset import resizeNormalize
import argparse

def test_tpem_dims():
    # 加载配置
    config = yaml.load(open('config/super_resolution.yaml', 'r'), Loader=yaml.Loader)
    config = EasyDict(config)
    
    # 创建PEAN实例(用于模型初始化)
    args = argparse.Namespace(
        syn=False, syn_ratio=0.5, syn_type='normal', lr=0.001, num_epochs=100,
        batch_size=1, workers=4, beta1=0.9, lr_scheduler=True, srb=8,
        resume=None, test_path=None, dataset_name='TextZoom',
        model_type='pean', evaluate=False, stn=True, tps_outputsize=[32, 128],
        num_fiducial=20, tps_margins=0.05, amp=False, distributed=False, local_rank=0,
        device_id=0, save_interval=500, pretrain='', syn_data_root='./', data_root='./data'
    )
    
    pean = TextSR(config, args)
    
    # 加载模型
    print('Loading PEAN...')
    pean_model = pean.generator_init(resume_this='./ckpt/PEAN_final.pth')['model']
    pean_model.eval()
    
    # 加载PARSeq
    print('Loading PARSeq...')
    parseq = pean.PARSeq_init()
    parseq.eval()
    
    # 加载TPEM
    print('Loading TPEM...')
    pean.diffusion = pean.init_diffusion_model()
    pean.diffusion.load_network()
    
    # 创建假数据
    print('\n' + '='*80)
    print('Creating test data...')
    print('='*80)
    
    # 4通道图像[batch, 4, H, W]
    img_lr = torch.randn(1, 4, 32, 128)
    img_hr = torch.randn(1, 4, 32, 128)
    
    print(f'img_lr shape: {img_lr.shape}')
    print(f'img_hr shape: {img_hr.shape}')
    
    # PARSeq推理
    print('\n' + '='*80)
    print('PARSeq inference...')
    print('='*80)
    pq_in_lr = pean.parse_parseq_data(img_lr[0, :3, :, :])
    print(f'PARSeq input shape: {pq_in_lr.shape}')
    
    with torch.no_grad():
        parseq_out_lr = parseq(pq_in_lr, max_length=25)
        print(f'PARSeq output shape (before softmax): {parseq_out_lr.shape}')
        
        prob_str_lr = parseq_out_lr.softmax(-1)
        print(f'PARSeq softmax shape: {prob_str_lr.shape}')
    
    pq_in_hr = pean.parse_parseq_data(img_hr[0, :3, :, :])
    with torch.no_grad():
        parseq_out_hr = parseq(pq_in_hr, max_length=25)
        prob_str_hr = parseq_out_hr.softmax(-1)
        print(f'PARSeq HR softmax shape: {prob_str_hr.shape}')
    
    # TPEM数据准备
    print('\n' + '='*80)
    print('TPEM data preparation...')
    print('='*80)
    
    weighted_mask = torch.tensor([0]).long()
    text_len = torch.tensor([1]).long()
    predicted_length = torch.ones(prob_str_lr.shape[0]) * prob_str_lr.shape[1]
    
    print(f'weighted_mask shape: {weighted_mask.shape}')
    print(f'text_len shape: {text_len.shape}')
    print(f'predicted_length shape: {predicted_length.shape}')
    print(f'predicted_length value: {predicted_length}')
    
    # Feed TPEM
    print('\n' + '='*80)
    print('Feeding TPEM...')
    print('='*80)
    
    data_diff = {
        "HR": prob_str_hr,
        "SR": prob_str_lr,
        "weighted_mask": weighted_mask,
        "predicted_length": predicted_length,
        "text_len": text_len
    }
    
    print(f"data_diff['HR'] shape: {data_diff['HR'].shape}")
    print(f"data_diff['SR'] shape: {data_diff['SR'].shape}")
    
    try:
        pean.diffusion.feed_data(data_diff)
        print('TPEM feed_data successful')
        
        # TPEM process
        print('\n' + '='*80)
        print('TPEM process...')
        print('='*80)
        _, label_vec_final = pean.diffusion.process()
        print(f'label_vec_final shape: {label_vec_final.shape}')
        print(f'label_vec_final device: {label_vec_final.device}')
        
        # Move to pean device
        label_vec_final = label_vec_final.to(pean.device)
        print(f'After move to device: {label_vec_final.shape}')
        
        # PEAN forward
        print('\n' + '='*80)
        print('PEAN forward...')
        print('='*80)
        print(f'img_lr shape: {img_lr.shape}')
        print(f'label_vec_final shape: {label_vec_final.shape}')
        
        with torch.no_grad():
            img_sr, _ = pean_model(img_lr, label_vec_final)
        print(f'img_sr shape: {img_sr.shape}')
        print('SUCCESS!')
        
    except Exception as e:
        print(f'ERROR: {type(e).__name__}')
        print(f'Message: {str(e)}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_tpem_dims()
