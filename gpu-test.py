import torch
print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)
# expect: 2.4.1+cu121 True 12.1