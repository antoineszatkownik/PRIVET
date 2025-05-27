import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import sys
import os
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset, DataLoader
import io
from PIL import Image
# caution: path[0] is reserved for script path (or '' in REPL)
ENV_PATH = "your/path/to/dgm-eval/folder"
sys.path.insert(1, ENV_PATH)

ROOT_PATH = "your/path/to/root"
sys.path.append(ROOT_PATH)  # Replace with your actual path

CACHE_TORCH_PATH = "your/path/to/.cache/torch"
os.environ["TORCH_HOME"] = CACHE_TORCH_PATH

#from privacy.p_utils_images_NN_v2 import *
from p_utils_v6 import *

import time


ngpu=1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)

import os
import random
import torch
from torchvision import transforms
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, ImageFolder
from collections import defaultdict
#from fld.features.DINOv2FeatureExtractor import DINOv2FeatureExtractor
from fld.metrics.FLD import FLD  # if you use FLD later

from fld.metrics.AuthPct import AuthPct
from fld.metrics.CTTest import CTTest


# Downsample synthetic dataset: select 1000 images per class
def downsample_dataset(dataset, n_per_class=1000):
    class_counts = defaultdict(int)
    selected_indices = []
    # dataset.samples is a list of (filepath, class) pairs for ImageFolder.
    for idx, (_, label) in enumerate(dataset.samples):
        if class_counts[label] < n_per_class:
            selected_indices.append(idx)
            class_counts[label] += 1
    return torch.utils.data.Subset(dataset, selected_indices)

def build_mixed_feature_set_determinist(gen_feat, train_feat, beta=0.0):
    """
    Create a combined feature set where the first `alpha` fraction are training features 
    and the remaining are synthetic features (deterministic order).
    
    Parameters:
        gen_feat (Tensor): Features from the synthetic dataset (shape: [N, D]).
        train_feat (Tensor): Features from the training dataset (shape: [M, D]).
        alpha (float): Fraction of the output set to take from training data.
    
    Returns:
        Tensor: Combined feature set of size equal to gen_feat, with training features first.
    """
    N = gen_feat.shape[0]
    n_real = int(beta * N)
    n_syn = N - n_real

    # Take the first n_real training samples and first n_syn synthetic samples
    real_feats = train_feat[:n_real]  # Deterministic: first n_real samples
    syn_feats = gen_feat[:n_syn]      # Deterministic: first n_syn samples
    
    # Concatenate (training features first, synthetic features after)
    combined = torch.cat([real_feats, syn_feats], dim=0)
    return combined  # No shuffling

import pytorchfwd
import os
import pathlib
from typing import List, Tuple

import numpy as np
import torch as th
import torchvision.transforms as tv
from tqdm import tqdm

PATH = "your/path/to/pytorchfwd"
sys.path.append(PATH) 

from freq_math import calculate_frechet_distance, forward_wavelet_packet_transform
from utils import ImagePathDataset, _parse_args

def compute_packet_statistics(
    dataloader, wavelet, max_level, log_scale
):
    """Compute wavelet packet transform across batches.

    Args:
        dataloader (th.utils.data.DataLoader): Torch dataloader.
        wavelet (str): Choice of wavelet.
        max_level (int): Wavelet decomposition level.
        log_scale (bool): Apply log scale.

    Returns:
        Tuple[np.ndarray, ...]: Mean and sigma for each packet.
    """
    packets = []
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    for img_batch in tqdm(dataloader):
        if isinstance(img_batch, list):
            img_batch = img_batch[0]
        img_batch = img_batch.to(device)
        packets.append(
            forward_wavelet_packet_transform(
                img_batch, wavelet, max_level, log_scale
            ).cpu()
        )
    packet_tensor = torch.cat(packets, dim=0)
    print(packet_tensor.shape)
    packet_tensor = torch.permute(packet_tensor, (1, 0, 2, 3, 4))
    P, BS, C, H, W = packet_tensor.shape
    packet_tensor = torch.reshape(packet_tensor, (P, BS, C * H * W))

    return packet_tensor

wavelet = "sym5"
max_level = 3
log_scale = False

class JPEGQuality(object):
    def __init__(self, quality=75):
        """
        Args:
            quality (int): JPEG quality, 1 (worst) to 95 (best). 75 is a good default.
        """
        assert 1 <= quality <= 95, "quality must be between 1 and 95"
        self.quality = quality

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.
        Returns:
            PIL Image: Re-encoded JPEG at self.quality.
        """
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer)

jpg75 = T.Compose([
    JPEGQuality(quality=75),
    transforms.ToTensor(),  # First convert to [0, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Then normalize to [-1, 1]
])

crop28 = T.Compose([
    T.CenterCrop(28),
    T.Pad(2),
    transforms.ToTensor(),  # First convert to [0, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Then normalize to [-1, 1]
])

class Posterize(object):
    def __init__(self, bits=5):
        self.bits = bits
    def __call__(self, img):
        return T.functional.posterize(img, self.bits)

posterize = T.Compose([
    Posterize(bits=5),
    transforms.ToTensor(),  # First convert to [0, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Then normalize to [-1, 1]
])

elastic = T.Compose([
    T.ElasticTransform(),
    transforms.ToTensor(),  # First convert to [0, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Then normalize to [-1, 1]
])

transforms_lst = [posterize, crop28,  jpg75, elastic] #jpg75

transform = transforms.Compose([
    transforms.ToTensor(),  # First convert to [0, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Then normalize to [-1, 1]
])

train_loader = DataLoader(
    datasets.CIFAR10("data", train=True, download=True,
                    transform=transform),batch_size=64, shuffle=True)

test_loader = DataLoader(
    datasets.CIFAR10("data", train=False, download=True,
                    transform=transform),batch_size=64, shuffle=True)

SYNTH_DATA_PATH = "your/path/to/CIFAR10-PFGMPP"
synthetic_full = ImageFolder(root=SYNTH_DATA_PATH,
                             transform=transform)

synthetic_dataset = downsample_dataset(synthetic_full, n_per_class=1000)

synthetic_loader = DataLoader(
    synthetic_dataset,         # your ImageFolder dataset
    batch_size=64,          # pick whatever batch-size you like
    shuffle=True,           # shuffle each epoch?
)

del synthetic_full

train_feat = compute_packet_statistics(train_loader, wavelet, max_level, log_scale)
print("AFTER")
print(train_feat.shape)
train_feat = train_feat.permute(1,0,2)
train_feat = train_feat.reshape(-1,train_feat.shape[1]*train_feat.shape[2])
print(train_feat.shape)

test_feat = compute_packet_statistics(test_loader, wavelet, max_level, log_scale)
print("AFTER")
print(test_feat.shape)
test_feat = test_feat.permute(1,0,2)
test_feat = test_feat.reshape(-1,test_feat.shape[1]*test_feat.shape[2])
print(test_feat.shape)

############################
## COMPUTE 1-NN distances ##
############################
#Train-Train
start = time.time()
dist_NN_tr_tr = gpu_nearest_neighbors(train_feat, k=1,distance='standard_euclidean',chunk_size=256,device=device.type, verbose=False)
p_tr_tr_NN_dist, p_tr_tr_NN_idx = sorting(dist_NN_tr_tr)
end = time.time()
print(end-start)

############################
## FIT CDF on Train-Train ##
############################

N_train = train_feat.shape[0]
partition_start = 0.001 #this is a hyperparameter of PRIVET
partition_end = 0.05 #this is a hyperparameter of PRIVET
#default would be to fit on 1% to 20%. but depending on shape of CDF (if there are multiple modes), 
#we want to fit the first mode, because we are interested in lower tail
#to be 100% sure on the partition of the fit, visual inspection of the CDF of 1-NN between train-train is recommended

start = int(np.ceil(partition_start*N_train)) #int(0.01*N) # if N is small start = 0 --> problem with log
end = int(partition_end*N_train)

# Fit parameters (adjust start/end indices to avoid extremes)
intercept, alpha, std_err_intercept, std_err_alpha, sigma_Y_pred = fit_nearest_neighbor_cdf(p_tr_tr_NN_dist.numpy().reshape(-1,), start_idx=start, end_idx=end)

print(f"Estimated intercept = {intercept:.2f} ± {std_err_intercept:.2f}")
print(f"Estimated alpha = {alpha:.2f} ± {std_err_alpha:.2f}")

transforms_lst = [posterize, crop28, jpg75, elastic]
transform_names = ['Posterize', 'Center crop 28', 'JPG 75', 'Elastic transform']


# metrics storage
beta_values = [0.0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
metrics = {
    name: {
        'NPL':   [],
        'FLD':   [],
        'Auth':  [],
        'CT':    []
    } for name in transform_names
}

time_PRIVET, time_FLD, time_AUTH, time_CT, time_PQM = [], [], [], [], []


torch.manual_seed(42)
for transform_fn, name in zip(transforms_lst, transform_names):
    print(name)
    # extract features once per transform
    train_dataset_trans = DataLoader(
        datasets.CIFAR10("data", train=True, download=True,
                        transform=transform_fn),batch_size=64, shuffle=True)
    train_feat_trans = compute_packet_statistics(train_dataset_trans, wavelet, max_level, log_scale)
    train_feat_trans = train_feat_trans.permute(1,0,2)
    train_feat_trans = train_feat_trans.reshape(-1,train_feat_trans.shape[1]*train_feat_trans.shape[2])
    del train_dataset_trans

    synthetic_full = ImageFolder(root=SYNTH_DATA_PATH,
                                 transform=transform_fn)
    synthetic_dataset = downsample_dataset(synthetic_full, n_per_class=1000)
    synthetic_loader = DataLoader(
        synthetic_dataset,         # your ImageFolder dataset
        batch_size=64,          # pick whatever batch-size you like
        shuffle=True,           # shuffle each epoch?
    )
    del synthetic_full, synthetic_dataset
    gen_feat = compute_packet_statistics(synthetic_loader, wavelet, max_level, log_scale)
    gen_feat = gen_feat.permute(1,0,2)
    gen_feat = gen_feat.reshape(-1,gen_feat.shape[1]*gen_feat.shape[2])

    shuffled_indices = torch.randperm(train_feat_trans.shape[0], generator=torch.Generator().manual_seed(42))
    train_feat_shuf = train_feat_trans[shuffled_indices]
    del train_feat_trans

    shuffled_indices = torch.randperm(gen_feat.shape[0], generator=torch.Generator().manual_seed(42))
    gen_feat = gen_feat[shuffled_indices]

    for beta in beta_values:
        mixed_feat = build_mixed_feature_set_determinist(
            gen_feat, train_feat_shuf, beta=beta
        )

        start = time.time()
        # PRIVET → NPL
        store, p_tr, p_te = PRIVET(
            train_feat, test_feat, mixed_feat,
            intercept, alpha,
            renormalization=5**(1/20),
            distance='standard_euclidean',
            device=device.type
        )
        end = time.time()
        NPL = (store[:,0] <= threshold).sum()
        metrics[name]['NPL'].append(NPL)
        time_PRIVET.append(end-start)

        # FLD generalization gap
        torch.cuda.empty_cache()
        start = time.time()
        fld_gap = FLD(eval_feat="gap").compute_metric(train_feat, test_feat, mixed_feat)
        end = time.time()
        time_FLD.append(end-start)
        metrics[name]['FLD'].append(fld_gap)
        torch.cuda.empty_cache()

        start = time.time()
        # AuthPct
        # code from https://github.com/marcojira/fld/blob/main/fld/metrics/AuthPct.py was changed at line 41
        # return (100 * torch.sum(authen) / len(authen)).item() - 50
        # became
        # return (100 * torch.sum(authen) / len(authen)).item() - 50, authen
        # authen is a list of booleans, one per synthetic, True if sample is deemed authentic, False otherwise
        auth_mask = AuthPct().compute_metric(train_feat, test_feat, mixed_feat)[1]
        end = time.time()
        time_AUTH.append(end-start)
        auth_score = (~auth_mask.detach().cpu().numpy()).sum()
        metrics[name]['Auth'].append(auth_score)

        start = time.time()
        # C_T
        ct_val = CTTest().compute_metric(train_feat, test_feat, mixed_feat)
        end = time.time()
        time_CT.append(end-start)
        metrics[name]['CT'].append(ct_val)

import pickle
with open('v6_copycat_expe_comparison_CIFAR10_wavelet_features_not_dinov2_CLEAN_METRICS_DIC.pickle', 'wb') as handle:
    pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"time elapsed PRIVET mean = {np.mean(time_PRIVET)}, std = {np.std(time_PRIVET)}")
print(f"time elapsed FLD mean = {np.mean(time_FLD)}, std = {np.std(time_FLD)}")
print(f"time elapsed AUTH mean = {np.mean(time_AUTH)}, std = {np.std(time_AUTH)}")
print(f"time elapsed CT mean = {np.mean(time_CT)}, std = {np.std(time_CT)}")

plt.rcParams.update({
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15
})

styles = {
    'Posterize': {'color':'orange', 'marker':'s'},
    'Center crop 28': {'color':'green',  'marker':'.'},
    'JPG 75':     {'color':'violet',   'marker':'D'},
    'Elastic transform':     {'color':'blue',   'marker':'x'}
}

# 'JPG 75':     {'color':'violet',   'marker':'D'},

#transform_names = ['Posterize', 'Center crop 28', 'Elastic transform']

fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(16,4))

x = [b * 100 for b in beta_values]   # percent copied

# 1) NPL (PRIVET)
ax = axs[0]
for name in transform_names:
    ax.plot(x, 100*np.array(metrics[name]['NPL'])/mixed_feat.shape[0],
            label=name,
            color=styles[name]['color'],
            marker=styles[name]['marker'],
            linestyle='-')
ax.plot(x,x, color="black", linestyle="dashed",label="groundtruth")
ax.set_xlabel("% copied samples")
ax.set_ylabel("Number of Privacy Leaks (%)")
ax.set_title("PRIVET")
#ax.axhline(0, color='red', linestyle='--', label="Overfit line (below is overfit)")
#ax.legend(fontsize=12)
ax.grid(True)

# 2) Auth
ax = axs[1]
for name in transform_names:
    ax.plot(x, 100*np.array(metrics[name]['Auth'])/mixed_feat.shape[0],
            label=name,
            color=styles[name]['color'],
            marker=styles[name]['marker'],
            linestyle='-')
ax.plot(x,x, color="black", linestyle="dashed",label="groundtruth")
ax.set_xlabel("% copied samples")
ax.set_ylabel("Number of Privacy Leaks (%)")
ax.set_title("In-Authenticity Score")
#ax.axhline(0, color='red', linestyle='--', label="Overfit line (below is overfit)")
ax.grid(True)

# 2) Generalization Gap FLD
ax = axs[2]
for name in transform_names:
    ax.plot(x, metrics[name]['FLD'],
            label=name,
            color=styles[name]['color'],
            marker=styles[name]['marker'],
            linestyle='-')
ax.set_xlabel("% copied samples")
ax.set_ylabel("Generalization Gap FLD")
#ax.set_title("Generalization Gap FLD")
ax.axhline(0, color='red', linestyle='--', label="Overfit line (below is overfit)")
ax.grid(True)


# 4) $C_T$
ax = axs[3]
for name in transform_names:
    ax.plot(x, metrics[name]['CT'],
            label=name,
            color=styles[name]['color'],
            marker=styles[name]['marker'],
            linestyle='-')
ax.set_xlabel("% copied samples")
ax.set_ylabel("$C_T$")
#ax.set_title("$C_T$")
ax.axhline(0, color='red', linestyle='--', label="Overfit line (below is overfit)")
ax.grid(True)

tmp = axs[2].get_legend_handles_labels()
tmp[1].insert(3,"Groundtruth")
tmp[0].insert(3,axs[1].get_legend_handles_labels()[0][-1])


fig.legend(
    *tmp,
    loc='upper center',
    bbox_to_anchor=[0.5, 1.1],
    ncol=5,
    fontsize=12
)


fig.tight_layout()

FIG_PATH = "your/path/to/FIG/folder"
fig.savefig(f"{FIG_PATH}/copycat_cifar10_comparison_v6_wavelet.png", 
            dpi=200, bbox_inches='tight')

plt.show()