"""
04_pruning.py — Structured Pruning on SpeechBrain CRDNN-VAD
Usage:
    python 04_pruning.py --audio_dir data/LibriParty/dataset/eval
    python 04_pruning.py --audio_dir data/LibriParty/dataset/eval --skip_inference
"""

import argparse
import copy
import csv
import json
import os
import time

import torch
import torch.nn.utils.prune as prune
import torchaudio
from speechbrain.inference.VAD import VAD

# ── helpers ───────────────────────────────────────────────────────────────────

def get_model_size_mb(model):
    """
    Accurate size: only count actual non-zero parameter bytes.
    torch.save(state_dict) keeps zero-weights so size doesn't shrink there.
    Instead we sum up bytes of all parameter tensors directly.
    """
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 * 1024)

def get_disk_size_mb(model):
    """Disk size via torch.save — useful to show the 'on-disk' number too."""
    path = "_tmp_model_size.pt"
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / (1024 * 1024)
    os.remove(path)
    return size

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_nonzero(model):
    return sum(p.nonzero().size(0) for p in model.parameters())

def measure_latency(vad_model, n_runs=20):
    dummy = torch.randn(1, 16000 * 5)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            vad_model.get_speech_prob_chunk(dummy)
            times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times)

def apply_pruning(mods, amount):
    """
    Structured L1-norm pruning on all torch.nn.Linear layers (dim=0 = output neurons).
    GRU/Conv layers are skipped:
      - GRU: PyTorch structured pruning doesn't support recurrent weight matrices natively.
      - Conv2d: structured channel pruning requires downstream layer reshaping (future work).
    After pruning, prune.remove() makes the mask permanent so the zeroed rows
    are frozen in the weight tensor.
    """
    pruned = copy.deepcopy(mods)
    count = 0
    for name, module in pruned.named_modules():
        # Linear layers: structured pruning (row-level)
        if isinstance(module, torch.nn.Linear):
            n_to_prune = int(module.out_features * amount)
            if n_to_prune == 0 or n_to_prune >= module.out_features:
                continue
            prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)
            prune.remove(module, 'weight')
            count += 1
            print(f"  [Linear-structured] {name}  out={module.out_features}")

        # GRU layers: unstructured L1 pruning on all weight matrices
        elif isinstance(module, torch.nn.GRU):
            for param_name in [n for n, _ in module.named_parameters() if 'weight' in n]:
                prune.l1_unstructured(module, name=param_name, amount=amount)
                prune.remove(module, param_name)
                count += 1
                print(f"  [GRU-unstructured]  {name}.{param_name}")

        # Conv2d: unstructured L1 pruning
        elif isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
            count += 1
            print(f"  [Conv-unstructured]  {name}")

    print(f"  → {count} layers pruned at {int(amount*100)}% sparsity")
    return pruned

# ── F1 evaluation ─────────────────────────────────────────────────────────────

def load_libriparty_labels(label_dir):
    """
    Parse LibriParty ground-truth JSON files.
    Actual format:
      {
        "speaker_id": [
          {"start": 2.18, "stop": 13.67, ...},
          ...
        ],
        ...
      }
    We collect ALL segments across ALL speakers → union of speech intervals.
    Returns: dict { session_stem -> list of (start, end) tuples }
    """
    labels = {}
    eval_json = os.path.join(label_dir, "eval.json")
    if not os.path.isfile(eval_json):
        print(f"  [WARN] eval.json not found: {eval_json}")
        return labels

    with open(eval_json) as f:
        data = json.load(f)

    for session_name, session_data in data.items():
        segs = []
        for key, value in session_data.items():
            if key in ("noises", "background"):
                continue
            if isinstance(value, list):
                for seg in value:
                    if isinstance(seg, dict) and "start" in seg and "stop" in seg:
                        segs.append((float(seg["start"]), float(seg["stop"])))
        segs.sort()
        labels[session_name] = segs

    print(f"  Loaded labels for {len(labels)} sessions")
    return labels

def frame_level_f1(pred_boundaries, gt_boundaries, duration, frame_shift=0.01):
    """
    Convert segment lists to binary frame arrays, then compute F1/Precision/Recall.
    frame_shift: seconds per frame (10 ms default).
    """
    n_frames = int(duration / frame_shift) + 1

    def segs_to_frames(segs):
        arr = torch.zeros(n_frames, dtype=torch.bool)
        for s, e in segs:
            s_idx = max(0, int(s / frame_shift))
            e_idx = min(n_frames - 1, int(e / frame_shift))
            arr[s_idx:e_idx] = True
        return arr

    pred = segs_to_frames(pred_boundaries)
    gt   = segs_to_frames(gt_boundaries)

    tp = (pred & gt).sum().item()
    fp = (pred & ~gt).sum().item()
    fn = (~pred & gt).sum().item()
    tn = (~pred & ~gt).sum().item()

    precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1         = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
    miss_rate  = fn / (tp + fn) if (tp + fn) > 0 else 0.0   # = 1 - recall
    false_alarm= fp / (tn + fp) if (tn + fp) > 0 else 0.0

    return dict(precision=precision, recall=recall, f1=f1,
                miss_rate=miss_rate, false_alarm=false_alarm)

def evaluate_vad(vad_model, audio_dir, label_dir=None):
    """
    Run inference on all wav files.
    Returns avg speech_ratio, avg RTF, and (if labels provided) avg F1 metrics.
    """
    wav_files = sorted([
        os.path.join(root, f)
        for root, _, files in os.walk(audio_dir)
        for f in files if f.endswith(".wav")
    ])
    if not wav_files:
        print(f"  [WARN] No .wav files in {audio_dir}")
        return None

    gt_labels = load_libriparty_labels(label_dir) if label_dir else {}
    has_labels = bool(gt_labels)

    ratios, rtfs = [], []
    metrics_list = []

    for wav_path in wav_files:
        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        duration = waveform.shape[-1] / 16000

        t0 = time.perf_counter()
        with torch.no_grad():
            boundaries = vad_model.get_speech_segments(wav_path)
        elapsed = time.perf_counter() - t0

        pred_segs = [(s.item(), e.item()) for s, e in boundaries] if len(boundaries) else []
        speech_dur = sum(e - s for s, e in pred_segs)
        ratios.append(speech_dur / duration)
        rtfs.append(elapsed / duration)

        if has_labels:
            stem = os.path.splitext(os.path.basename(wav_path))[0].replace("_mixture", "")
            if stem in gt_labels:
                m = frame_level_f1(pred_segs, gt_labels[stem], duration)
                metrics_list.append(m)

    avg = dict(
        speech_ratio=sum(ratios) / len(ratios),
        rtf=sum(rtfs) / len(rtfs),
    )
    if metrics_list:
        for key in ["f1", "precision", "recall", "miss_rate", "false_alarm"]:
            avg[key] = sum(m[key] for m in metrics_list) / len(metrics_list)

    return avg

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir",  default="data/LibriParty/dataset/eval")
    parser.add_argument("--label_dir",  default="data/LibriParty/dataset/metadata",
                        help="Dir with per-file JSON ground-truth labels")
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip slow per-file inference (latency-only mode)")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  Structured Pruning — CRDNN VAD")
    print("="*65)

    print("\nLoading pretrained CRDNN from HuggingFace...")
    vad = VAD.from_hparams(
        source="speechbrain/vad-crdnn-libriparty",
        savedir="pretrained_models/vad-crdnn-libriparty"
    )
    print("Model loaded.\n")

    # ── Baseline ──────────────────────────────────────────────────
    print("── Baseline (FP32) ──")
    base_params   = count_parameters(vad.mods)
    base_nonzero  = count_nonzero(vad.mods)
    base_size_mem = get_model_size_mb(vad.mods)
    base_size_dsk = get_disk_size_mb(vad.mods)
    base_latency  = measure_latency(vad)
    print(f"  Params     : {base_params:,}")
    print(f"  Size (mem) : {base_size_mem:.4f} MB  |  disk: {base_size_dsk:.4f} MB")
    print(f"  Latency    : {base_latency:.2f} ms  (dummy 5-sec clip, avg 20 runs)")

    base_eval = {}
    if not args.skip_inference:
        print("  Running inference for baseline…")
        base_eval = evaluate_vad(vad, args.audio_dir, args.label_dir) or {}
        if base_eval:
            print(f"  Speech ratio : {base_eval['speech_ratio']:.1%}")
            print(f"  RTF          : {base_eval['rtf']:.4f}")
            if "f1" in base_eval:
                print(f"  F1           : {base_eval['f1']:.4f}")
                print(f"  Precision    : {base_eval['precision']:.4f}")
                print(f"  Recall       : {base_eval['recall']:.4f}")
                print(f"  Miss Rate    : {base_eval['miss_rate']:.4f}")
                print(f"  False Alarm  : {base_eval['false_alarm']:.4f}")

    results = [{
        "model":           "FP32 Baseline",
        "sparsity_target": "0%",
        "size_mem_mb":     round(base_size_mem, 4),
        "size_disk_mb":    round(base_size_dsk, 4),
        "params":          base_params,
        "nonzero":         base_nonzero,
        "sparsity_actual": "0.00%",
        "latency_ms":      round(base_latency, 2),
        **{k: round(v, 4) for k, v in base_eval.items()},
    }]

    # ── Pruning sweep ──────────────────────────────────────────────
    for sparsity in [0.1, 0.3, 0.5]:
        label = f"Pruned {int(sparsity*100)}%"
        print(f"\n── {label} ──")

        pruned_mods = apply_pruning(vad.mods, sparsity)

        params   = count_parameters(pruned_mods)
        nonzero  = count_nonzero(pruned_mods)
        size_mem = get_model_size_mb(pruned_mods)
        size_dsk = get_disk_size_mb(pruned_mods)
        actual_sparsity = 1 - nonzero / params if params > 0 else 0

        original_mods = vad.mods
        vad.mods = pruned_mods
        latency = measure_latency(vad)

        pruned_eval = {}
        if not args.skip_inference:
            print(f"  Running inference for {label}…")
            pruned_eval = evaluate_vad(vad, args.audio_dir, args.label_dir) or {}

        vad.mods = original_mods

        print(f"  Size (mem) : {size_mem:.4f} MB  (baseline: {base_size_mem:.4f} MB)")
        print(f"  Params     : {params:,}  |  Non-zero: {nonzero:,}")
        print(f"  Actual sparsity: {actual_sparsity:.2%}")
        print(f"  Latency    : {latency:.2f} ms")
        if pruned_eval:
            print(f"  Speech ratio : {pruned_eval.get('speech_ratio',0):.1%}")
            if "f1" in pruned_eval:
                print(f"  F1 / Miss / FA : "
                      f"{pruned_eval['f1']:.4f} / "
                      f"{pruned_eval['miss_rate']:.4f} / "
                      f"{pruned_eval['false_alarm']:.4f}")

        results.append({
            "model":           label,
            "sparsity_target": f"{int(sparsity*100)}%",
            "size_mem_mb":     round(size_mem, 4),
            "size_disk_mb":    round(size_dsk, 4),
            "params":          params,
            "nonzero":         nonzero,
            "sparsity_actual": f"{actual_sparsity:.2%}",
            "latency_ms":      round(latency, 2),
            **{k: round(v, 4) for k, v in pruned_eval.items()},
        })

    # ── Summary table ──────────────────────────────────────────────
    print("\n")
    print("="*80)
    print(f"{'Model':<18} {'Tgt':>5} {'MemMB':>7} {'DskMB':>7} "
          f"{'Nonzero':>10} {'Actual%':>8} {'Lat ms':>8}"
          + ("  F1    MissR  FA" if "f1" in results[0] else ""))
    print("="*80)
    for r in results:
        line = (f"{r['model']:<18} {r['sparsity_target']:>5} "
                f"{r['size_mem_mb']:>7} {r['size_disk_mb']:>7} "
                f"{r['nonzero']:>10,} {r['sparsity_actual']:>8} "
                f"{r['latency_ms']:>8}")
        if "f1" in r:
            line += f"  {r['f1']:.4f} {r['miss_rate']:.4f} {r['false_alarm']:.4f}"
        print(line)
    print("="*80)

    # ── Save CSV ───────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    csv_path = "results/pruning_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✅ Results saved to {csv_path}")
    if "f1" in results[0]:
        plot_results(results)

def plot_results(results):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    os.makedirs("results", exist_ok=True)

    sparsity_labels = [r["sparsity_target"] for r in results]
    x = range(len(results))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Structured Pruning Results — CRDNN VAD", fontsize=14, fontweight='bold')

    # F1
    axes[0,0].plot(x, [r["f1"] for r in results], marker='o', color='steelblue')
    axes[0,0].set_title("F1 Score vs Sparsity")
    axes[0,0].set_xticks(x); axes[0,0].set_xticklabels(sparsity_labels)
    axes[0,0].set_ylim(0, 1); axes[0,0].set_ylabel("F1")
    axes[0,0].grid(True)

    # Miss Rate & False Alarm
    axes[0,1].plot(x, [r["miss_rate"] for r in results], marker='o', color='tomato', label='Miss Rate')
    axes[0,1].plot(x, [r["false_alarm"] for r in results], marker='s', color='orange', label='False Alarm')
    axes[0,1].set_title("Miss Rate & False Alarm vs Sparsity")
    axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(sparsity_labels)
    axes[0,1].set_ylim(0, 1); axes[0,1].set_ylabel("Rate")
    axes[0,1].legend(); axes[0,1].grid(True)

    # Latency
    axes[1,0].plot(x, [r["latency_ms"] for r in results], marker='o', color='seagreen')
    axes[1,0].set_title("Latency vs Sparsity")
    axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(sparsity_labels)
    axes[1,0].set_ylabel("Latency (ms)"); axes[1,0].grid(True)

    # Nonzero params
    axes[1,1].plot(x, [r["nonzero"] for r in results], marker='o', color='mediumpurple')
    axes[1,1].set_title("Non-zero Parameters vs Sparsity")
    axes[1,1].set_xticks(x); axes[1,1].set_xticklabels(sparsity_labels)
    axes[1,1].set_ylabel("Non-zero Params"); axes[1,1].grid(True)

    plt.tight_layout()
    path = "results/pruning_plots.png"
    plt.savefig(path, dpi=150)
    print(f"\n📊 Plots saved to {path}")
    plt.close()

if __name__ == "__main__":
    main()