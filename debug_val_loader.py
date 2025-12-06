import sys
sys.path.append('./')
import torch
from dataset.dataset import alignCollate_realWTL, lmdbDataset_real

# Create a simple test
dataset = lmdbDataset_real(root='C:/Users/Aiur/PEAN/data/TextZoom/test/easy', voc_type='lower', max_len=25, test=True)
collate_fn = alignCollate_realWTL(imgH=32, imgW=128, down_sample_scale=4, mask=True)

loader = torch.utils.data.DataLoader(
    dataset, batch_size=2,
    shuffle=False, num_workers=0, pin_memory=False,
    collate_fn=collate_fn,
    drop_last=False
)

print("Testing validation data loader...")
for i, data in enumerate(loader):
    print(f"Batch {i}: type(data) = {type(data)}")
    if isinstance(data, tuple) or isinstance(data, list):
        print(f"  Number of elements: {len(data)}")
        for j, item in enumerate(data):
            if isinstance(item, torch.Tensor):
                print(f"  Element {j}: Tensor with shape {item.shape}")
            elif isinstance(item, (list, tuple)):
                print(f"  Element {j}: {type(item).__name__} with length {len(item)}")
            else:
                print(f"  Element {j}: {type(item).__name__}")
    if i >= 0:  # Only check first batch
        break
