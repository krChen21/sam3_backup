#!/usr/bin/env python3
import os

# 强制离线，避免任何 Hugging Face 访问
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


DEFAULT_CHECKPOINT = "/home/ubuntu/sam3/checkpoints/sam3/sam3.pt"
DEFAULT_BPE = "/home/ubuntu/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"


def xyxy_to_norm_cxcywh(box_xyxy, width, height):
    x0, y0, x1, y1 = box_xyxy

    x0 = max(0.0, min(float(x0), width - 1))
    x1 = max(0.0, min(float(x1), width - 1))
    y0 = max(0.0, min(float(y0), height - 1))
    y1 = max(0.0, min(float(y1), height - 1))

    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    cx = x0 + bw / 2.0
    cy = y0 + bh / 2.0

    return [
        cx / float(width),
        cy / float(height),
        bw / float(width),
        bh / float(height),
    ]


def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach()
        if x.is_floating_point():
            x = x.float()
        return x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, list):
        return np.array([to_numpy(v) for v in x])
    return np.array(x)


def normalize_masks(masks):
    masks = to_numpy(masks)

    if masks is None:
        return np.zeros((0, 1, 1), dtype=bool)

    # SAM3 原生 masks shape 通常为 (N, 1, H, W)
    if masks.ndim == 4:
        if masks.shape[1] == 1:
            masks = masks[:, 0]
        elif masks.shape[0] == 1:
            masks = masks[0]

    if masks.ndim == 2:
        masks = masks[None, :, :]

    if masks.dtype != bool:
        masks = masks > 0

    return masks.astype(bool)


def normalize_boxes(boxes, n):
    boxes = to_numpy(boxes)
    if boxes is None:
        return np.zeros((n, 4), dtype=float)

    boxes = np.asarray(boxes)
    if boxes.ndim == 1 and boxes.size == 4:
        boxes = boxes[None, :]

    if boxes.shape[0] < n:
        pad = np.zeros((n - boxes.shape[0], 4), dtype=float)
        boxes = np.concatenate([boxes, pad], axis=0)

    return boxes[:n].astype(float)


def normalize_scores(scores, n):
    scores = to_numpy(scores)
    if scores is None:
        return np.ones((n,), dtype=float)

    scores = np.asarray(scores).reshape(-1)

    if scores.shape[0] < n:
        pad = np.ones((n - scores.shape[0],), dtype=float)
        scores = np.concatenate([scores, pad], axis=0)

    return scores[:n].astype(float)


def select_box_with_matplotlib(image, title="SAM3 Box Select"):
    """
    用 OpenCV selectROI 交互式框选。
    鼠标拖拽矩形框，Enter/Space 确认，c 取消。
    返回 xyxy 像素坐标。
    """
    img_rgb = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    win_name = title
    print(f"OpenCV 框选窗口: {win_name}")
    print("请用鼠标拖拽框选，按 Enter 或 Space 确认，按 c 取消。")

    roi = cv2.selectROI(
        win_name,
        img_bgr,
        showCrosshair=True,
        fromCenter=False,
    )
    try:
        cv2.destroyWindow(win_name)
    except cv2.error:
        cv2.destroyAllWindows()

    x, y, w, h = roi

    if w <= 0 or h <= 0:
        raise RuntimeError("No box selected or selection cancelled.")

    box_xyxy = [float(x), float(y), float(x + w), float(y + h)]
    print(f"Selected box xyxy: {box_xyxy}")
    return box_xyxy


def _yes_no(prompt, default_no=True):
    answer = input(prompt).strip().lower()
    if default_no:
        return answer in ("y", "yes")
    return answer not in ("n", "no", "")


def interactive_select_positive_boxes(image):
    """交互式选择至少一个正框，并可继续添加。"""
    positive_boxes = []
    box = select_box_with_matplotlib(
        image,
        title="Select POSITIVE box - Enter/Space confirm, c cancel",
    )
    positive_boxes.append(box)
    while _yes_no("是否继续添加正框？[y/N]: "):
        box = select_box_with_matplotlib(
            image,
            title="Select POSITIVE box - Enter/Space confirm, c cancel",
        )
        positive_boxes.append(box)
    return positive_boxes


def interactive_select_negative_boxes(image):
    """交互式选择负框，可添加多个。"""
    if not _yes_no("是否添加负框来排除背景/其他物体？[y/N]: "):
        return []
    negative_boxes = []
    box = select_box_with_matplotlib(
        image,
        title="Select NEGATIVE box - Enter/Space confirm, c cancel",
    )
    negative_boxes.append(box)
    while _yes_no("是否继续添加负框？[y/N]: "):
        box = select_box_with_matplotlib(
            image,
            title="Select NEGATIVE box - Enter/Space confirm, c cancel",
        )
        negative_boxes.append(box)
    return negative_boxes


def print_prompt_and_boxes(prompt, positive_boxes_xyxy, negative_boxes_xyxy):
    print(f"Prompt: {prompt if prompt is not None else 'None'}")
    print("Positive boxes:")
    if positive_boxes_xyxy:
        for i, box in enumerate(positive_boxes_xyxy):
            print(f"  pos{i}: {box}")
    else:
        print("  (none)")
    print("Negative boxes:")
    if negative_boxes_xyxy:
        for i, box in enumerate(negative_boxes_xyxy):
            print(f"  neg{i}: {box}")
    else:
        print("  (none)")


def color_for_index(i):
    palette = [
        (255, 64, 64),
        (64, 255, 64),
        (64, 128, 255),
        (255, 192, 64),
        (192, 64, 255),
        (64, 255, 255),
        (255, 64, 192),
        (160, 255, 64),
    ]
    return palette[i % len(palette)]


def save_overlay(image_rgba, masks, boxes, scores, positive_boxes_xyxy, negative_boxes_xyxy, out_path):
    overlay_arr = np.array(image_rgba).astype(np.float32)

    for i, mask in enumerate(masks):
        color = np.array(color_for_index(i), dtype=np.float32)
        alpha = 0.45
        overlay_arr[mask, :3] = overlay_arr[mask, :3] * (1 - alpha) + color * alpha
        overlay_arr[mask, 3] = 255

    overlay = Image.fromarray(np.clip(overlay_arr, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(overlay)

    # 画 SAM3 返回 box
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = [float(v) for v in box]
        color = color_for_index(i)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0, max(0, y0 - 16)), f"mask {i}: {scores[i]:.3f}", fill=color)

    # 画正向输入 box（白色）
    for i, box in enumerate(positive_boxes_xyxy or []):
        x0, y0, x1, y1 = [float(v) for v in box]
        draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255), width=2)
        draw.text((x0, max(0, y0 - 16)), f"pos{i}", fill=(255, 255, 255))

    # 画负向输入 box（黑色）
    for i, box in enumerate(negative_boxes_xyxy or []):
        x0, y0, x1, y1 = [float(v) for v in box]
        draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=2)
        draw.text((x0, min(image_rgba.height - 18, y1 + 4)), f"neg{i}", fill=(0, 0, 0))

    overlay.save(out_path)


def save_cutout(image_rgba, mask, out_path):
    arr = np.array(image_rgba).copy()
    arr[..., 3] = (mask.astype(np.uint8) * 255)
    Image.fromarray(arr).save(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 local image segmentation with text + box prompts."
    )
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--prompt", type=str, default=None, help="文本提示，例如 mechanical fixture")
    parser.add_argument("--box", nargs=4, type=float, default=None, metavar=("X0", "Y0", "X1", "Y1"),
                        help="可选：单个正向 xyxy 像素框（兼容旧用法）")
    parser.add_argument("--pos-box", nargs=4, type=float, action="append", default=[],
                        metavar=("X0", "Y0", "X1", "Y1"),
                        help="正向 xyxy 像素框，可重复多次")
    parser.add_argument("--neg-box", nargs=4, type=float, action="append", default=[],
                        metavar=("X0", "Y0", "X1", "Y1"),
                        help="负向 xyxy 像素框，可重复多次")
    parser.add_argument("--text-only", action="store_true",
                        help="只使用文本提示，不进入框选界面")
    parser.add_argument("--no-neg-interactive", action="store_true",
                        help="交互模式下不询问负框")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="本地 sam3.pt 权重路径")
    parser.add_argument("--bpe", default=DEFAULT_BPE, help="BPE 词表路径")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="运行设备")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="SAM3 processor 置信度阈值")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    out_dir = Path(args.out).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    bpe_path = Path(args.bpe).resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"输入图片不存在: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAM3 权重不存在: {checkpoint_path}")
    if not bpe_path.is_file():
        raise FileNotFoundError(f"BPE 词表不存在: {bpe_path}")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA 不可用，自动切换到 CPU")
        args.device = "cpu"

    masks_dir = out_dir / "masks"
    cutouts_dir = out_dir / "cutouts"
    masks_dir.mkdir(parents=True, exist_ok=True)
    cutouts_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    image_rgba = image.convert("RGBA")
    width, height = image.size

    print("Image:", image_path)
    print("Image size:", width, height)
    print("Checkpoint:", checkpoint_path)
    print("BPE:", bpe_path)
    print("Device:", args.device)
    print("Output:", out_dir)

    if args.prompt is None:
        user_prompt = input("请输入文本提示 prompt，可留空直接回车表示不用文本提示：").strip()
        if user_prompt == "":
            args.prompt = None
        else:
            args.prompt = user_prompt

    positive_boxes_xyxy = []
    negative_boxes_xyxy = []

    if args.box is not None:
        positive_boxes_xyxy.append(list(args.box))
    if args.pos_box:
        positive_boxes_xyxy.extend([list(b) for b in args.pos_box])
    if args.neg_box:
        negative_boxes_xyxy.extend([list(b) for b in args.neg_box])

    if len(positive_boxes_xyxy) == 0 and not args.text_only:
        print("进入交互式正框选择（至少选择一个正框）...")
        positive_boxes_xyxy.extend(interactive_select_positive_boxes(image))

    if len(positive_boxes_xyxy) == 0 and args.text_only and args.prompt is None:
        raise RuntimeError("text-only 模式必须提供 --prompt 或手动输入非空 prompt。")

    if not args.no_neg_interactive:
        negative_boxes_xyxy.extend(interactive_select_negative_boxes(image))

    positive_boxes_norm = [
        xyxy_to_norm_cxcywh(box, width, height) for box in positive_boxes_xyxy
    ]
    negative_boxes_norm = [
        xyxy_to_norm_cxcywh(box, width, height) for box in negative_boxes_xyxy
    ]

    print("")
    print_prompt_and_boxes(args.prompt, positive_boxes_xyxy, negative_boxes_xyxy)
    print("")
    print("Loading SAM3 image model from local checkpoint...")
    model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        device=args.device,
        checkpoint_path=str(checkpoint_path),
        load_from_HF=False,
    )

    processor = Sam3Processor(
        model,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )

    print("Running SAM3 segmentation...")
    if args.device == "cuda":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext()

    with torch.inference_mode(), autocast_ctx:
        state = processor.set_image(image)
        if args.prompt is not None:
            state = processor.set_text_prompt(state=state, prompt=args.prompt)
        for norm_box in positive_boxes_norm:
            state = processor.add_geometric_prompt(
                state=state,
                box=norm_box,
                label=True,
            )
        for norm_box in negative_boxes_norm:
            state = processor.add_geometric_prompt(
                state=state,
                box=norm_box,
                label=False,
            )

    masks = normalize_masks(state.get("masks"))
    n = masks.shape[0]
    boxes = normalize_boxes(state.get("boxes"), n)
    scores = normalize_scores(state.get("scores"), n)

    detections = {
        "image": str(image_path),
        "prompt": args.prompt,
        "checkpoint": str(checkpoint_path),
        "bpe": str(bpe_path),
        "positive_boxes_xyxy": [[float(v) for v in b] for b in positive_boxes_xyxy],
        "negative_boxes_xyxy": [[float(v) for v in b] for b in negative_boxes_xyxy],
        "positive_boxes_norm_cxcywh": [[float(v) for v in b] for b in positive_boxes_norm],
        "negative_boxes_norm_cxcywh": [[float(v) for v in b] for b in negative_boxes_norm],
        "num_masks": int(n),
        "detections": [],
    }

    combined = np.zeros((height, width), dtype=bool)

    for i, mask in enumerate(masks):
        if mask.shape[0] != height or mask.shape[1] != width:
            mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
            mask_img = mask_img.resize((width, height), resample=Image.NEAREST)
            mask = np.array(mask_img) > 0
            masks[i] = mask

        combined |= mask

        mask_path = masks_dir / f"mask_{i:03d}.png"
        cutout_path = cutouts_dir / f"cutout_{i:03d}.png"

        Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path)
        save_cutout(image_rgba, mask, cutout_path)

        box = [float(v) for v in boxes[i].tolist()]
        score = float(scores[i])

        detections["detections"].append(
            {
                "index": i,
                "score": score,
                "box_xyxy": box,
                "mask": str(mask_path),
                "cutout": str(cutout_path),
            }
        )

        print(f"[{i}] score={score:.4f}, box_xyxy={box}")
        print(f"    mask: {mask_path}")
        print(f"    cutout: {cutout_path}")

    Image.fromarray((combined.astype(np.uint8) * 255)).save(out_dir / "combined_mask.png")

    if n > 0:
        save_overlay(
            image_rgba,
            masks,
            boxes,
            scores,
            positive_boxes_xyxy,
            negative_boxes_xyxy,
            out_dir / "overlay.png",
        )
    else:
        image_rgba.save(out_dir / "overlay.png")
        print("WARNING: SAM3 没有输出 mask，overlay.png 保存为原图。")

    with open(out_dir / "detections.json", "w", encoding="utf-8") as f:
        json.dump(detections, f, ensure_ascii=False, indent=2)

    print("")
    print("Done.")
    print("Mask count:", n)
    print("Output dir:", out_dir)
    print("Overlay:", out_dir / "overlay.png")
    print("Combined mask:", out_dir / "combined_mask.png")
    print("Detections:", out_dir / "detections.json")


if __name__ == "__main__":
    main()
