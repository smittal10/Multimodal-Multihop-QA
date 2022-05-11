import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _ = clip.load("ViT-B/32", device=device)
clip_model = clip_model.visual

model = torch.load('fine_tuned.ckpt')

for p1, p2 in zip(model.parameters(), clip_model.parameters()):
    # p1.data-p2.data>1
    # if p1.data.ne(p2.data).sum() > 0:
    if torch.allclose(p1.data,p2.data,atol=1e-3):
        print("true")
    else:
        print("false")
    # if (p1.data-p2.data>0.5).sum() > 0:
    #     print("false")
    # else:
    #     print("true")
