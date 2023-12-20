import torch
import torchvision
import matplotlib.pyplot as plt
from dataset import CLEVRDataset
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from torchvision import transforms 
import numpy as np
from model import UNet_conditional
from tqdm import tqdm
from evaluator import evaluation_model
from torch.optim import Adam
import os
from torchvision.utils import save_image
from matplotlib.ticker import MaxNLocator
import torch.nn.functional as F
from torchvision import transforms 
from torch.utils.data import DataLoader

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_height=240, img_width=320, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_height = img_height
        self.img_width = img_width
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, conds, num, args, invTrans):
        model.eval()
        gen_images = None
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_height, self.img_width)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, conds)
                if args.cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, args.cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise


                if i in [1, 51, 101, 151, 201, 251, 299]:
                    img = invTrans(x)
                    if gen_images is None:
                        gen_images = (img)
                    else:
                        gen_images = torch.vstack((gen_images, img))
            save_image(gen_images, os.path.join(f'./images/check/process_{num}.png'), nrow=8)
        model.train()
        
        return x

def evaluate(model, diffusion, loader, eval_model, args):
    model.eval()
    avg_acc = 0
    gen_images = None
    invTrans = transforms.Compose([ transforms.Normalize(mean = (0., 0., 0.), std = (1/0.5, 1/0.5, 1/0.5)),
                                    transforms.Normalize(mean = (-0.5, -0.5, -0.5), std = (1., 1., 1.))])
    with torch.no_grad():
        for i, conds in enumerate(loader):
            conds = conds.to(device)

            sampled_images = diffusion.sample(model, n=len(conds), conds=conds, num=i, args=args, invTrans=invTrans)
            
            acc = eval_model.eval(sampled_images, conds)
            avg_acc += acc * conds.shape[0]

            norm_img = invTrans(sampled_images)
            if gen_images is None:
                gen_images = norm_img
            else:
                gen_images = torch.vstack((gen_images, norm_img))
    avg_acc /= len(loader.dataset)
    return avg_acc, gen_images

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    parser = ArgumentParser()
    parser.add_argument('--logdir', default='log/ddpm')
    parser.add_argument('--num_conditions', type=int, default=24)
    parser.add_argument('--time_dim', type=int, default=256)
    parser.add_argument('--img_height', type=int, default=64, help='height of image')
    parser.add_argument('--img_width', type=int, default=64, help='width of image')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--start', type=float, default=0.0001)
    parser.add_argument('--end', type=float, default=0.02)
    parser.add_argument('--exp_name', type=str, default='C')
    parser.add_argument('--noise_steps', type=int, default=300)
    parser.add_argument('--cfg_scale', type=int, default=0)
    args = parser.parse_args()


    data = 'new_test'
    test_dataset = CLEVRDataset(args=args, mode=data)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = UNet_conditional(time_dim=args.time_dim, num_conditions=args.num_conditions).to(device)

    model.load_state_dict(torch.load('./models/C/epoch31-acc0.747.pth'))
    diffusion = Diffusion(beta_start=args.start, beta_end=args.end, noise_steps=args.noise_steps, img_height=args.img_height, img_width=args.img_width)

    eval_model = evaluation_model()

    eval_acc, gen_images = evaluate(model, diffusion, test_loader, eval_model, args)

    
    save_image(gen_images, os.path.join(f'./images/{args.exp_name}', f'inference_{data}.png'), nrow=8)
    print('eval_acc: {0}'.format(eval_acc))