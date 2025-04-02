import os

import torch
from torch.utils.data import DataLoader, Dataset
import open_clip
from tqdm import tqdm

from misc.constants import PIXELS_INPUT, PIXELS_PHI, PIXELS_PHI_SEQ, TEXT_INPUT, CACHE_DIR, INPUT_ID


class ConvertToRGB:
    def __init__(self):
        super().__init__()

    def __call__(self, image):
        return image.convert('RGB')

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"



class OpenFlamingoVisualFeaturesCache(Dataset):
    def __init__(self, original_dataset, cache_path, batch_size=128, device="cuda"):
        self.original_dataset = DictWrapper(original_dataset)
        self.batch_size = batch_size
        self.device = device
        self.cache_path = cache_path

        if not os.path.exists(self.cache_path):
            print("Cached representations are not found!")
            self._precompute_features()
        else:
            print("Cached representations are found!")
            self._load_cache()

    def _load_cache(self):
        print("Loading cached representations...")
        cache = torch.load(self.cache_path)
        self.phi = cache[PIXELS_PHI]
        self.phi_seq = cache[PIXELS_PHI_SEQ]
        print(f"PHI {self.phi.shape}")
        print(f"PHI_SEQ {self.phi_seq.shape}")

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        torch.save(
            {
                PIXELS_PHI: self.phi,
                PIXELS_PHI_SEQ: self.phi_seq,
            },
            self.cache_path
        )

    def _precompute_features(self):
        # Here I assume that self.original_dataset has correct transforms for ViT-L-14
        # It is enough for now, since we only plan to use ViT-L-14 linear task encoder

        # prepare model
        vision_encoder, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="openai",
            cache_dir=CACHE_DIR,
        )
        vision_encoder = vision_encoder.visual
        vision_encoder.output_tokens = True

        vision_encoder = vision_encoder.to(self.device)

        # prepare dataloader
        loader = DataLoader(
            self.original_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=10,
            pin_memory=True
        )

        print("Precomputing Features...")
        self.phi = []
        self.phi_seq = []
        with torch.no_grad():
            for input_dict, gt_labels in tqdm(loader):
                img = input_dict[PIXELS_INPUT]
                cur_phi, cur_phi_seq = vision_encoder(img.to(self.device))
                self.phi.append(cur_phi.cpu())
                self.phi_seq.append(cur_phi_seq.cpu())

        self.phi = torch.cat(self.phi, dim=0)
        self.phi_seq = torch.cat(self.phi_seq, dim=0)
        assert len(self.phi) == len(self.original_dataset)
        assert len(self.phi_seq) == len(self.original_dataset)
        print("Features precomputed")
        print(f"PHI {self.phi.shape}")
        print(f"PHI_SEQ {self.phi_seq.shape}")
        self._save_cache()

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        input_dict, label = self.original_dataset[idx]
        input_dict.update(
            {PIXELS_PHI: self.phi[idx], PIXELS_PHI_SEQ: self.phi_seq[idx], INPUT_ID: idx}
        )
        return input_dict, label

class DictWrapper(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        output = self.original_dataset[idx]
        assert (len(output) == 2) or (len(output) == 3)
        # output can be
        # (img, label) or (img, question, label)
        input_dict = {
            PIXELS_INPUT: output[0]
        }
        if len(output) == 3:
            # VQA
            input_dict.update({
                TEXT_INPUT: output[1]
            })
        label = output[-1]
        input_dict.update({
            INPUT_ID: idx
        })
        return input_dict, label


if __name__ == "__main__":
    # 10 classes
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as trans

    transform = trans.Compose([
        trans.Resize(224, interpolation=3, antialias="warn"),
        trans.CenterCrop(224),
        ConvertToRGB(),
        trans.ToTensor(),
        trans.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    DATA_DIR = "/mlbio_scratch/gadetski/datasets"
    dataset = CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    dataset = OpenFlamingoVisualFeaturesCache(dataset, os.path.join(DATA_DIR, "representations", "cifar10.pth"))
    




