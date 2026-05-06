"""
Predictor classes for three classification models:
  1. CLIPPredictor        — Zero-shot with OpenAI CLIP ViT-B-16
  2. HierarchicalPredictor — Two-stage ResNet50 (group → species)
  3. FlatPredictor         — Single ResNet50 (100-class)
"""

import json
import os
import zipfile
import tempfile
from collections import defaultdict

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ═══════════════════════════════════════════════════════════════════════════════
# Shared constants
# ═══════════════════════════════════════════════════════════════════════════════

ORDERED_CLASS_NAMES = [
    # butterflies (0-29)
    "CAIRNS BIRDWING", "BLUE SPOTTED CROW", "AMERICAN SNOOT", "BECKERS WHITE",
    "BLACK HAIRSTREAK", "BANDED PEACOCK", "BROWN SIPROETA", "BIRD CHERRY ERMINE MOTH",
    "CABBAGE WHITE", "BANDED TIGER MOTH", "APPOLLO", "CLEARWING MOTH",
    "ATALA", "CLEOPATRA", "AFRICAN GIANT SWALLOWTAIL", "CLOUDED SULPHUR",
    "ATLAS MOTH", "ADONIS", "BANDED ORANGE HELICONIAN", "ARCIGERA FLOWER MOTH",
    "BROOKES BIRDWING", "CHECQUERED SKIPPER", "CLODIUS PARNASSIAN", "CINNABAR MOTH",
    "COMET MOTH", "CHALK HILL BLUE", "BLUE MORPHO", "CHESTNUT", "AN 88", "BROWN ARGUS",
    # mammals (30-74)
    "warthog", "vampire_bat", "highland_cattle", "seal", "badger",
    "baboon", "horse", "rhinoceros", "arctic_fox", "wildebeest",
    "opossum", "orangutan", "polar_bear", "blue_whale", "jackal",
    "vicuna", "manatee", "otter", "mountain_goat", "yak",
    "squirrel", "giraffe", "porcupine", "weasel", "dolphin",
    "brown_bear", "zebra", "camel", "tapir", "alpaca",
    "snow_leopard", "sugar_glider", "kangaroo", "sea_lion", "red_panda",
    "african_elephant", "walrus", "american_bison", "koala", "mongoose",
    "wombat", "groundhog", "armadillo", "anteater", "water_buffalo",
    # birds (75-99)
    "Forest Wagtail", "Northern Lapwing", "Rufous Treepie", "Cattle Egret",
    "Common Kingfisher", "Gray Wagtail", "Indian Peacock", "House Crow",
    "White-Breasted Waterhen", "Indian Pitta", "Red-Wattled Lapwing",
    "White-Breasted Kingfisher", "Indian Roller", "Common Tailorbird",
    "White Wagtail", "Common Rosefinch", "Jungle Babbler", "Coppersmith Barbet",
    "Hoopoe", "Sarus Crane", "Common Myna", "Brown-Headed Barbet",
    "Ruddy Shelduck", "Indian Grey Hornbill", "Asian Green Bee-Eater",
]

GROUP_MAP = {
    **{i: "butterfly" for i in range(0, 30)},
    **{i: "mammals"   for i in range(30, 75)},
    **{i: "birds"     for i in range(75, 100)},
}

GROUP_TO_IDX = {"birds": 0, "butterfly": 1, "mammals": 2}
IDX_TO_GROUP = {v: k for k, v in GROUP_TO_IDX.items()}

# Per-group local species lists (preserves training order)
_group_species = defaultdict(list)
for _sp_idx in range(len(ORDERED_CLASS_NAMES)):
    _group_species[GROUP_MAP[_sp_idx]].append(_sp_idx)

_local_to_global = {}
for _group, _sp_list in _group_species.items():
    for _local_idx, _global_idx in enumerate(_sp_list):
        _local_to_global[(_group, _local_idx)] = _global_idx

# Standard ImageNet normalisation
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CLIP Zero-Shot Predictor
# ═══════════════════════════════════════════════════════════════════════════════

class CLIPPredictor:
    """Zero-shot species classification using CLIP or BioCLIP."""

    def __init__(self, model_name="ViT-B-16", pretrained="openai"):
        import open_clip

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # hf-hub models (e.g. BioCLIP) don't use a separate pretrained arg
        if model_name.startswith("hf-hub:"):
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
            self.tokenizer = open_clip.get_tokenizer(model_name)
        else:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model = self.model.to(self.device).eval()

        # Pre-compute text embeddings for all 100 species
        prompts = [f"a photo of a {name}" for name in ORDERED_CLASS_NAMES]
        tokens = self.tokenizer(prompts).to(self.device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(tokens)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        img_tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)

        image_features = self.model.encode_image(img_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Cosine similarity → softmax
        similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)[0]

        # Top prediction
        top_idx = similarity.argmax().item()
        species = ORDERED_CLASS_NAMES[top_idx]
        group = GROUP_MAP[top_idx]

        # Group probabilities (aggregate per group)
        group_probs = {}
        for g in ["birds", "butterfly", "mammals"]:
            indices = [i for i, gm in GROUP_MAP.items() if gm == g]
            group_probs[g] = round(similarity[indices].sum().item() * 100, 2)

        # Top 5 species
        top5 = similarity.topk(5)
        top3_species = [
            {
                "species": ORDERED_CLASS_NAMES[idx.item()],
                "confidence": round(prob.item() * 100, 2),
            }
            for prob, idx in zip(top5.values, top5.indices)
        ]

        return {
            "group": group,
            "group_confidence": group_probs[group],
            "group_probs": group_probs,
            "species": species,
            "species_confidence": round(similarity[top_idx].item() * 100, 2),
            "top3_species": top3_species,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 1b. CLIP + KNN Predictor
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_label(label):
    """Parse 'category_species' label → (group, species_name)."""
    group, species = label.split("_", 1)
    return group, species


class CLIPKNNPredictor:
    """Classifies by encoding the image with CLIP then running KNN on embeddings."""

    def __init__(self, model_name="ViT-B-16", pretrained="openai", knn_model_path=""):
        import open_clip
        import joblib

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load CLIP encoder
        if model_name.startswith("hf-hub:"):
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        else:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
        self.model = self.model.to(self.device).eval()

        # Load pre-trained KNN
        self.knn = joblib.load(knn_model_path)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        import numpy as np

        img_tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        features = self.model.encode_image(img_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        feat_np = features.cpu().numpy()

        # KNN prediction + probabilities
        pred_label = self.knn.predict(feat_np)[0]
        proba = self.knn.predict_proba(feat_np)[0]
        classes = self.knn.classes_

        group, species = _parse_label(pred_label)

        # Group probabilities
        group_probs = {"birds": 0.0, "butterfly": 0.0, "mammals": 0.0}
        for cls, prob in zip(classes, proba):
            g, _ = _parse_label(cls)
            if g in group_probs:
                group_probs[g] += prob
        group_probs = {k: round(v * 100, 2) for k, v in group_probs.items()}

        # Top 5 species
        sorted_idx = np.argsort(proba)[::-1][:5]
        top3_species = [
            {
                "species": _parse_label(classes[i])[1],
                "confidence": round(proba[i] * 100, 2),
            }
            for i in sorted_idx
        ]

        pred_conf = round(proba[list(classes).index(pred_label)] * 100, 2)

        return {
            "group": group,
            "group_confidence": group_probs.get(group, 0.0),
            "group_probs": group_probs,
            "species": species,
            "species_confidence": pred_conf,
            "top3_species": top3_species,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 1c. CLIP + SVM Predictor
# ═══════════════════════════════════════════════════════════════════════════════

class CLIPSVMPredictor:
    """Classifies by encoding the image with CLIP then running a linear SVM."""

    def __init__(self, model_name="ViT-B-16", pretrained="openai", svm_model_path=""):
        import open_clip
        import joblib

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load CLIP encoder
        if model_name.startswith("hf-hub:"):
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        else:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
        self.model = self.model.to(self.device).eval()

        # Load pre-trained SVM
        self.svm = joblib.load(svm_model_path)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        import numpy as np

        img_tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        features = self.model.encode_image(img_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        feat_np = features.cpu().numpy()

        # SVM prediction + probabilities
        pred_label = self.svm.predict(feat_np)[0]
        proba = self.svm.predict_proba(feat_np)[0]
        classes = self.svm.classes_

        group, species = _parse_label(pred_label)

        # Group probabilities
        group_probs = {"birds": 0.0, "butterfly": 0.0, "mammals": 0.0}
        for cls, prob in zip(classes, proba):
            g, _ = _parse_label(cls)
            if g in group_probs:
                group_probs[g] += prob
        group_probs = {k: round(v * 100, 2) for k, v in group_probs.items()}

        # Top 5 species
        sorted_idx = np.argsort(proba)[::-1][:5]
        top3_species = [
            {
                "species": _parse_label(classes[i])[1],
                "confidence": round(proba[i] * 100, 2),
            }
            for i in sorted_idx
        ]

        pred_conf = round(proba[list(classes).index(pred_label)] * 100, 2)

        return {
            "group": group,
            "group_confidence": group_probs.get(group, 0.0),
            "group_probs": group_probs,
            "species": species,
            "species_confidence": pred_conf,
            "top3_species": top3_species,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Hierarchical Fine-tune Predictor (ResNet50 × 4)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_resnet(num_classes):
    """Build a ResNet50 with a Dropout+Linear head."""
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(m.fc.in_features, num_classes),
    )
    return m


class HierarchicalPredictor:
    """
    Stage 1: group classification (birds / butterfly / mammals)
    Stage 2: species classification within the predicted group
    """

    def __init__(self, stage1_path: str, stage2_paths: dict, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Stage 1
        self.s1 = _build_resnet(3)
        self.s1.load_state_dict(torch.load(stage1_path, map_location=self.device, weights_only=False))
        self.s1.to(self.device).eval()

        # Stage 2 (one model per group)
        n_species = {"birds": 25, "butterfly": 30, "mammals": 45}
        self.s2 = {}
        for group, path in stage2_paths.items():
            m = _build_resnet(n_species[group])
            m.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
            self.s2[group] = m.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        img_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        # Stage 1 — group
        s1_logits = self.s1(img_tensor)
        s1_probs = torch.softmax(s1_logits, dim=1)[0]
        group_idx = s1_probs.argmax().item()
        group = IDX_TO_GROUP[group_idx]

        # Stage 2 — species within predicted group
        s2_logits = self.s2[group](img_tensor)
        s2_probs = torch.softmax(s2_logits, dim=1)[0]
        local_idx = s2_probs.argmax().item()
        global_idx = _local_to_global[(group, local_idx)]

        # Top-5 species within the group
        top5_local = s2_probs.topk(min(5, len(s2_probs)))
        top3_species = [
            {
                "species": ORDERED_CLASS_NAMES[_local_to_global[(group, i.item())]],
                "confidence": round(p.item() * 100, 2),
            }
            for p, i in zip(top5_local.values, top5_local.indices)
        ]

        return {
            "group": group,
            "group_confidence": round(s1_probs[group_idx].item() * 100, 2),
            "group_probs": {
                IDX_TO_GROUP[i]: round(s1_probs[i].item() * 100, 2)
                for i in range(3)
            },
            "species": ORDERED_CLASS_NAMES[global_idx],
            "species_confidence": round(s2_probs[local_idx].item() * 100, 2),
            "top3_species": top3_species,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Flat Fine-tune Predictor (single ResNet50, 100 classes)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_directory_checkpoint(model_dir, device):
    """
    Load a PyTorch state dict that was saved in the directory-based
    serialization format (torch >= 2.6). Reconstructs a temporary zip
    and loads normally.
    """
    # Check if it's a directory-format save (has data.pkl + data/)
    data_pkl = os.path.join(model_dir, "data.pkl")
    if not os.path.exists(data_pkl):
        raise FileNotFoundError(f"Not a directory-format checkpoint: {model_dir}")

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with zipfile.ZipFile(tmp_path, "w") as zf:
            base_name = os.path.basename(model_dir)
            for root, _, files in os.walk(model_dir):
                for f in files:
                    if f == ".DS_Store":
                        continue
                    full = os.path.join(root, f)
                    arc = os.path.join(base_name, os.path.relpath(full, model_dir))
                    info = zipfile.ZipInfo(arc, date_time=(2026, 1, 1, 0, 0, 0))
                    with open(full, "rb") as fh:
                        zf.writestr(info, fh.read())

        state_dict = torch.load(tmp_path, map_location=device, weights_only=False)
    finally:
        os.unlink(tmp_path)

    return state_dict


class FlatPredictor:
    """Single ResNet50 fine-tuned on all 100 species at once."""

    def __init__(self, model_dir: str, label_map_path: str, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load label map
        with open(label_map_path) as f:
            self.label_map = json.load(f)  # {species_name: idx}
        self.idx_to_species = {v: k for k, v in self.label_map.items()}
        num_classes = len(self.label_map)

        # Build model
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.model.fc.in_features, num_classes),
        )

        # Load weights (directory-format checkpoint)
        state_dict = _load_directory_checkpoint(model_dir, self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        # Build group mapping from label_map species names
        self._species_to_group = {}
        for name in self.label_map:
            # Determine group from the master ORDERED_CLASS_NAMES list
            if name in ORDERED_CLASS_NAMES:
                idx_in_ordered = ORDERED_CLASS_NAMES.index(name)
                self._species_to_group[name] = GROUP_MAP[idx_in_ordered]
            else:
                # Fallback heuristic: butterflies are UPPERCASE, birds are Title Case,
                # mammals are lowercase
                if name.isupper() or name.upper() == name:
                    self._species_to_group[name] = "butterfly"
                elif name[0].isupper() and not name.isupper():
                    self._species_to_group[name] = "birds"
                else:
                    self._species_to_group[name] = "mammals"

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        img_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        logits = self.model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]

        top_idx = probs.argmax().item()
        species = self.idx_to_species[top_idx]
        group = self._species_to_group.get(species, "mammals")

        # Group probabilities (aggregate)
        group_probs = {"birds": 0.0, "butterfly": 0.0, "mammals": 0.0}
        for idx_val in range(len(probs)):
            sp = self.idx_to_species[idx_val]
            g = self._species_to_group.get(sp, "mammals")
            group_probs[g] += probs[idx_val].item()
        group_probs = {k: round(v * 100, 2) for k, v in group_probs.items()}

        # Top 5 species
        top5 = probs.topk(5)
        top3_species = [
            {
                "species": self.idx_to_species[i.item()],
                "confidence": round(p.item() * 100, 2),
            }
            for p, i in zip(top5.values, top5.indices)
        ]

        return {
            "group": group,
            "group_confidence": group_probs[group],
            "group_probs": group_probs,
            "species": species,
            "species_confidence": round(probs[top_idx].item() * 100, 2),
            "top3_species": top3_species,
        }
