"""
Feature extractor wrapper for the trained metric learning model.
Supports both ResNet50 and DINOv2 backbones — reads config from checkpoint.
"""
import os
import sys
import torch
from PIL import Image
from torchvision import transforms

# Allow imports from the parent 'metric learning' directory
import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import ResNet50_Embedder, DINOv2_Embedder
from config import MODEL_CONFIG


class MetricFeatureExtractor:
    """Extracts L2-normalised embeddings using the trained metric learning model."""

    def __init__(self, checkpoint_path: str = None, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        if checkpoint_path is None:
            backbone = MODEL_CONFIG.get('backbone', 'resnet50')
            checkpoint_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'checkpoints',
                f'{backbone}_metric_best.pth'
            )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint    = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg           = checkpoint.get('config') or checkpoint.get('params', {})
        embedding_dim = cfg.get('embedding_dim', 512)
        ckpt_backbone = cfg.get('backbone', 'resnet50')

        if ckpt_backbone == 'resnet50':
            self.model = ResNet50_Embedder(embedding_dim=embedding_dim)
        else:
            self.model = DINOv2_Embedder(
                embedding_dim=embedding_dim,
                model_name=ckpt_backbone,
                dropout=0.0
            )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        self._embedding_dim = embedding_dim
        self._backbone      = ckpt_backbone

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print(f"Model loaded  |  backbone: {ckpt_backbone}  |  "
              f"embedding dim: {embedding_dim}  |  device: {self.device}")

    def get_embedding_dim(self) -> int:
        return self._embedding_dim

    def extract(self, image: Image.Image) -> torch.Tensor:
        """Return a 1-D L2-normalised embedding (CPU tensor)."""
        with torch.no_grad():
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            emb    = self.model(tensor)   # (1, dim)
            return emb.squeeze().cpu()    # (dim,)
