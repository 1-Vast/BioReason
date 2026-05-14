import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.reason import BioReason
import torch
model = BioReason(input_dim=100, num_perts=5, dim=32, hidden=64, steps=2, evidence_dim=128)
x = torch.randn(4, 100)
pert = torch.randint(0, 5, (4,))
with torch.no_grad():
    out = model(x, pert, return_latent=True)
    print('type:', type(out))
    print('keys:', out.keys() if isinstance(out, dict) else 'not dict')
    for k,v in out.items():
        if isinstance(v, torch.Tensor):
            print(f'  {k}: shape={v.shape}')
        else:
            print(f'  {k}: {type(v).__name__} = {v}')
    # Check without return_latent
    out2 = model(x, pert, return_latent=False)
    print('return_latent=False type:', type(out2))
    for k,v in out2.items():
        if isinstance(v, torch.Tensor):
            print(f'  {k}: shape={v.shape}')
