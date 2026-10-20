import argparse, torch
from PIL import Image
from torchvision import transforms
from model import EcoSortNet
CLASSES=["cardboard","glass","metal","paper","plastic","trash"]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--image",required=True); p.add_argument("--checkpoint",default="outputs/best_model.pth"); a=p.parse_args()
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    model=EcoSortNet(); model.load_state_dict(ck["model_state_dict"]); model.eval()
    tf=transforms.Compose([transforms.Resize((96,96)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    x=tf(Image.open(a.image).convert("RGB")).unsqueeze(0)
    with torch.no_grad(): probs=model(x).softmax(1)[0]
    for i in probs.argsort(descending=True)[:3]:
        print(f"{CLASSES[int(i)]}: {float(probs[i])*100:.2f}%")
if __name__=="__main__": main()
