import os
import random
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy

import torch
import torchvision
from torch import nn

try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"

@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)

class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        clip_model_pretrained = os.environ.get("OPENCLIP_PRETRAINED", self.config.clip_model_pretrained)
        if clip_model_pretrained != self.config.clip_model_pretrained:
            self.config.clip_model_pretrained = clip_model_pretrained
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=clip_model_pretrained,  # e.g., laion2b_s34b_b88k or a local checkpoint path
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives    
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        pretrained_name = Path(self.config.clip_model_pretrained).name
        return "openclip_{}_{}".format(self.config.clip_model_type, pretrained_name)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def gui_cb(self,element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)


MASK_LEVELS = ("default", "s", "m", "l")
FAST_MASK_NMS_RESOLUTION = 0


def create(image_list, data_list, save_folder, dataset_path, fail_fast=False):
    assert image_list is not None, "image_list must be provided to generate features"
    timer = 0
    mask_generator.predictor.model.to('cuda')

    for i, img in tqdm(enumerate(image_list), desc="Embedding images", leave=False):
        save_path = os.path.join(save_folder, data_list[i].split('.')[0])
        if os.path.exists(save_path + '_f.npy') and os.path.exists(save_path + '_s.npy'):
            continue

        timer += 1
        try:
            img_embed, seg_map = _embed_clip_sam_tiles(img.unsqueeze(0), sam_encoder)
        except Exception as error:
            if fail_fast:
                raise RuntimeError(f"Failed to build multiscale features for {data_list[i]}") from error
            print(f"[WARN] Failed to build features for {data_list[i]}: {error}")
            curr = {
                'feature': torch.zeros((0, 512), dtype=torch.float16),
                'seg_maps': -torch.ones((4, *image_list[0].shape[1:]), dtype=torch.float32)
            }
            sava_numpy(save_path, curr)
            continue

        feature_chunks = []
        seg_map_tensor = []
        offset = 0
        height, width = image_list[0].shape[1:]
        for level in MASK_LEVELS:
            if level not in img_embed or level not in seg_map:
                seg_map_tensor.append(-torch.ones((height, width), dtype=torch.int32))
                continue
            features = img_embed[level]
            level_map = np.asarray(seg_map[level]).copy()
            if level_map.shape != (height, width):
                raise ValueError(f"Unexpected {level} segmentation shape {level_map.shape}")
            if level_map.size and int(level_map.max()) >= 0:
                if int(level_map.max()) >= features.shape[0]:
                    raise ValueError(
                        f"{level} segmentation IDs exceed feature count "
                        f"({level_map.max()} >= {features.shape[0]})"
                    )
                level_map[level_map >= 0] += offset
            feature_chunks.append(features)
            offset += int(features.shape[0])
            seg_map_tensor.append(torch.from_numpy(level_map.astype(np.int32, copy=False)))

        if not feature_chunks:
            raise ValueError("SAM did not produce usable masks at any hierarchy level")
        img_embed = torch.cat(feature_chunks, dim=0)
        seg_map = torch.stack(seg_map_tensor, dim=0)
        assert offset == int(img_embed.shape[0])

        curr = {
            'feature': img_embed,
            'seg_maps': seg_map
        }
        sava_numpy(save_path, curr)

    # mask_generator.predictor.model.to('cpu')

def sava_numpy(save_path, data):
    save_path_s = save_path + '_s.npy'
    save_path_f = save_path + '_f.npy'
    np.save(save_path_s, data['seg_maps'].numpy())
    np.save(save_path_f, data['feature'].numpy())

def _embed_clip_sam_tiles(image, sam_encoder):
    aug_imgs = torch.cat([image])
    seg_images, seg_map = sam_encoder(aug_imgs)

    clip_embeds = {}
    for mode in MASK_LEVELS:
        if mode not in seg_images or mode not in seg_map:
            continue
        tiles = seg_images[mode]
        tiles = tiles.to("cuda")
        with torch.no_grad():
            # clip_embed = model.encode_image(tiles)[0]
            clip_embed = model.encode_image(tiles)
        clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
        clip_embeds[mode] = clip_embed.detach().cpu().half()
    
    return clip_embeds, {mode: seg_map[mode] for mode in clip_embeds}

def get_seg_img(mask, image):
    image = image.copy()
    image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8)
    x,y,w,h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    if FAST_MASK_NMS_RESOLUTION > 0:
        return fast_mask_nms(
            masks_ord,
            idx,
            scores,
            iou_thr=iou_thr,
            score_thr=score_thr,
            inner_thr=inner_thr,
            resolution=FAST_MASK_NMS_RESOLUTION,
        )

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx


def fast_mask_nms(masks_ord, sorted_indices, sorted_scores, iou_thr, score_thr, inner_thr, resolution):
    """Approximate the legacy mask NMS with downsampled vectorized intersections."""
    if masks_ord.numel() == 0:
        return sorted_indices
    device = "cuda" if torch.cuda.is_available() else masks_ord.device
    step_h = max(1, int(np.ceil(masks_ord.shape[-2] / resolution)))
    step_w = max(1, int(np.ceil(masks_ord.shape[-1] / resolution)))
    masks = masks_ord[:, ::step_h, ::step_w].to(device=device, dtype=torch.float32)
    flat = masks.flatten(1)
    areas = flat.sum(dim=1).clamp_min(1.0)
    intersections = flat @ flat.T
    fractions = intersections / areas[:, None]
    iou = intersections / (areas[:, None] + areas[None, :] - intersections).clamp_min(1.0)
    upper = torch.triu(torch.ones_like(iou, dtype=torch.bool), diagonal=1)
    iou_max = torch.where(upper, iou, torch.zeros_like(iou)).max(dim=0).values

    first_fraction = fractions
    second_fraction = fractions.T
    inner_value = 1.0 - first_fraction * second_fraction
    first_contains_second = (first_fraction < 0.5) & (second_fraction >= 0.85)
    first_inside_second = (first_fraction >= 0.85) & (second_fraction < 0.5)
    inner_upper = torch.where(
        upper & first_contains_second, inner_value, torch.zeros_like(inner_value)
    )
    inner_lower = torch.where(
        upper & first_inside_second, inner_value, torch.zeros_like(inner_value)
    ).T
    inner_upper_max = inner_upper.max(dim=0).values
    inner_lower_max = inner_lower.max(dim=0).values
    keep = (
        (iou_max <= iou_thr)
        & (sorted_scores.to(device) > score_thr)
        & (inner_upper_max <= 1.0 - inner_thr)
        & (inner_lower_max <= 1.0 - inner_thr)
    )
    return sorted_indices[keep.detach().cpu()]

def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new

def sam_encoder(image):
    image = cv2.cvtColor(image[0].permute(1,2,0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
    # pre-compute masks
    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
    # pre-compute postprocess
    masks_default, masks_s, masks_m, masks_l = \
        masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
    
    def mask2segmap(masks, image):
        seg_img_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        for i in range(len(masks)):
            mask = masks[i]
            seg_img = get_seg_img(mask, image)
            pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))
            seg_img_list.append(pad_seg_img)

            seg_map[masks[i]['segmentation']] = i
        seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
        seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to('cuda')

        return seg_imgs, seg_map

    seg_images, seg_maps = {}, {}
    seg_images['default'], seg_maps['default'] = mask2segmap(masks_default, image)
    if len(masks_s) != 0:
        seg_images['s'], seg_maps['s'] = mask2segmap(masks_s, image)
    if len(masks_m) != 0:
        seg_images['m'], seg_maps['m'] = mask2segmap(masks_m, image)
    if len(masks_l) != 0:
        seg_images['l'], seg_maps['l'] = mask2segmap(masks_l, image)
    
    # 0:default 1:s 2:m 3:l
    return seg_images, seg_maps

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    seed_num = 42
    seed_everything(seed_num)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument(
        '--output_feature_dir',
        default=None,
        help='Destination feature directory; relative paths are resolved below dataset_path.',
    )
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--sam_ckpt_path', type=str, default="ckpts/sam_vit_h_4b8939.pth")
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--shard_index', type=int, default=0)
    parser.add_argument(
        '--only_missing',
        action='store_true',
        help='Only enqueue images whose *_f.npy or *_s.npy feature files are missing.',
    )
    parser.add_argument(
        '--fail_fast',
        action='store_true',
        help='Raise feature-generation errors instead of writing empty feature files.',
    )
    parser.add_argument(
        '--fast_mask_nms',
        action='store_true',
        help='Use downsampled vectorized NMS for newly generated multiscale features.',
    )
    args = parser.parse_args()
    if args.fast_mask_nms:
        FAST_MASK_NMS_RESOLUTION = 32
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    sam_ckpt_path = args.sam_ckpt_path
    img_folder = os.path.join(dataset_path, 'images')
    data_list = os.listdir(img_folder)
    data_list.sort()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")

    if args.output_feature_dir:
        save_folder = args.output_feature_dir
        if not os.path.isabs(save_folder):
            save_folder = os.path.join(dataset_path, save_folder)
    else:
        save_folder = os.path.join(dataset_path, 'language_features')
    os.makedirs(save_folder, exist_ok=True)
    if args.only_missing:
        filtered_data_list = []
        for data_path in data_list:
            save_path = os.path.join(save_folder, os.path.splitext(data_path)[0])
            if not (os.path.exists(save_path + '_f.npy') and os.path.exists(save_path + '_s.npy')):
                filtered_data_list.append(data_path)
        data_list = filtered_data_list
    if args.num_shards > 1:
        data_list = [data_path for i, data_path in enumerate(data_list) if i % args.num_shards == args.shard_index]
    print(
        f"Preprocessing {len(data_list)} images "
        f"(shard {args.shard_index}/{args.num_shards}, only_missing={args.only_missing})"
    )
    if not data_list:
        print("No images to preprocess; all requested feature files already exist.")
        sys.exit(0)

    model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
    # model, preprocess_for_tensor = clip.load("./CLIP/pretrain_models/ViT-B-16.pt", #"./CLIP/pretrain_models/RN50x64.pt", #"./CLIP/pretrain_models/ViT-B-16.pt", #"./CLIP/pretrain_models/RN50x64.pt", #"./CLIP/pretrain_models/ViT-L-14.pt",
    #                                                     device="cuda",
    #                                                     download_root='./CLIP/pretrain_models/',
    #                                                     if_transform_tensor=True)
    for name, param in model.named_parameters():
        param.requires_grad = False
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )

    img_list = []
    WARNED = False
    for data_path in data_list:
        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)

        orig_w, orig_h = image.shape[1], image.shape[0]
        if args.resolution == -1:
            if orig_h > 1080:
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_h / 1080
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
            
        scale = float(global_down)
        resolution = (int( orig_w  / scale), int(orig_h / scale))
        
        image = cv2.resize(image, resolution)
        image = torch.from_numpy(image)
        img_list.append(image)
    images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
    imgs = torch.cat(images)

    create(imgs, data_list, save_folder, dataset_path, fail_fast=args.fail_fast)
