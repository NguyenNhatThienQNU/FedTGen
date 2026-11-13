# ==========================================================================
# M03_local_lora_squad_qg
# Method family : local_lora
# Setting       : local | Tuning: lora | Aggregation: none
# Dataset       : squad_qg | Model: google/flan-t5-small
# Research Q     : RQ1,RQ3
# Compare with   : n/a
# Standalone: no imports from other experiments; no shared core module.
# ==========================================================================
EXPERIMENT_ID = 'M03_local_lora_squad_qg'
RQ_SUPPORTED = 'RQ1,RQ3'
METHOD_FAMILY = 'local_lora'
TRAINING_SETTING = 'local'
TUNING_MODE = 'lora'
AGGREGATION_STRATEGY = 'none'
IS_PROPOSED_METHOD = False
DATASET_KEY = 'squad_qg'
MODEL_NAME = 'google/flan-t5-small'
CONFIG_OVERRIDES = {}
# ==========================================================================

# ============================================================================
# FedTGen self-contained experiment body.
# This file is concatenated after a per-experiment CONFIG header to produce a
# fully standalone script. It does NOT import from any other experiment and
# does NOT depend on any shared module. The header defines:
#   EXPERIMENT_ID, RQ_SUPPORTED, METHOD_FAMILY, TRAINING_SETTING, TUNING_MODE,
#   AGGREGATION_STRATEGY, IS_PROPOSED_METHOD, DATASET_KEY, MODEL_NAME,
#   CONFIG_OVERRIDES
# ============================================================================

import os
import sys
import io
import gc
import re
import json
import time
import math
import shutil
import random
import hashlib
import platform
import tempfile
import traceback
import subprocess
from datetime import datetime, timezone


# ----------------------------------------------------------------------------
# 1. Dependency installation (Colab / fresh environment)
# ----------------------------------------------------------------------------
REQUIRED_PACKAGES = [
    "torch", "transformers", "datasets", "peft", "accelerate", "evaluate",
    "sacrebleu", "rouge_score", "bert_score", "sentencepiece", "scikit-learn",
    "pandas", "numpy", "tqdm", "safetensors", "psutil", "pytest",
]


def _pip(args):
    subprocess.run([sys.executable, "-m", "pip"] + args, check=False)


def ensure_dependencies():
    """Install missing dependencies. Safe to call repeatedly."""
    if os.environ.get("FEDTGEN_SKIP_INSTALL") == "1":
        return
    # torchao frequently breaks transformers imports on Colab -> remove it.
    try:
        import torchao  # noqa: F401
        _pip(["uninstall", "-y", "torchao"])
    except Exception:
        _torchao_absent = True  # torchao not present -> nothing to remove
    # datasets >= 4.0 dropped support for script-based datasets. lmqg/qg_squad and
    # roskoN/dailydialog ship loading scripts, so pin to a version that loads them.
    # Check the version WITHOUT importing datasets (so a downgrade still takes effect).
    try:
        import importlib.metadata as _md
        _dsver = _md.version("datasets")
        if int(_dsver.split(".")[0]) >= 4:
            print(f"[setup] datasets {_dsver} drops script datasets; pinning datasets<4.0")
            _pip(["install", "-q", "datasets>=2.16.0,<4.0"])
    except Exception as _dm:
        print(f"[setup] ensuring datasets<4.0 ({_dm})")
        _pip(["install", "-q", "datasets>=2.16.0,<4.0"])
    missing = []
    probe = {
        "scikit-learn": "sklearn", "rouge_score": "rouge_score",
        "bert_score": "bert_score", "sentencepiece": "sentencepiece",
    }
    for pkg in REQUIRED_PACKAGES:
        mod = probe.get(pkg, pkg)
        try:
            __import__(mod)
        except Exception:
            missing.append(pkg)
    if missing:
        print("[setup] Installing:", ", ".join(missing))
        _pip(["install", "-q"] + missing)


ensure_dependencies()

# ----------------------------------------------------------------------------
# 2. Imports
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
)

try:
    import psutil
except Exception:
    psutil = None


# ----------------------------------------------------------------------------
# 3. Base configuration (identical protocol across all experiments)
# ----------------------------------------------------------------------------
BASE_CONFIG = {
    "protocol_version": "FedTGen_protocol_v1",
    "seed": 42,
    "num_clients": 5,
    "participation_rate": 1.0,
    "dirichlet_alpha": 0.5,
    "minimum_samples_per_client": 20,
    "max_rounds": 20,
    "centralized_max_epochs": 20,
    "local_max_epochs": 20,
    "local_epochs": 1,
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 1e-4,
    "early_stopping_metric": "validation_rougeL",
    "lora_learning_rate": 2e-4,
    "full_learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "max_grad_norm": 1.0,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q", "v"],
    "fedprox_mu": 0.01,
    "lambda_n": 1.0,
    "beta_loss": 1.0,
    "beta_divergence": 1.0,
    "beta_quality": 1.0,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "generation_num_beams": 4,
    "generation_early_stopping": True,
    "no_repeat_ngram_size": 3,
    "latency_batch_size": 1,
    "latency_warmup_batches": 10,
    "latency_max_samples": 200,
    "bertscore_model_type": "roberta-large",
    "num_pseudo_topics": 10,
    # Full-data mode: no subsampling. Every experiment uses the ENTIRE dataset.
    # (None = use all examples.) Recorded transparently in config.json.
    "max_train_samples": None,
    "max_validation_samples": None,
    "max_test_samples": None,
}

# Per-dataset sequence length table.
LENGTH_TABLE = {
    "samsum":      {"max_source_length": 512, "max_target_length": 128},
    "dailydialog": {"max_source_length": 256, "max_target_length": 64},
    "squad_qg":    {"max_source_length": 512, "max_target_length": 64},
    "qqp":         {"max_source_length": 128, "max_target_length": 64},
}

# Dataset registry.
DATASET_REGISTRY = {
    "samsum":      {"hf_name": "knkarthick/samsum",  "hf_config": None,  "task": "summarization"},
    "dailydialog": {"hf_name": "roskoN/dailydialog", "hf_config": None,  "task": "dialogue_generation"},
    "squad_qg":    {"hf_name": "lmqg/qg_squad",      "hf_config": None,  "task": "question_generation"},
    "qqp":         {"hf_name": "nyu-mll/glue",       "hf_config": "qqp", "task": "paraphrase_generation"},
}


def build_config():
    cfg = dict(BASE_CONFIG)
    cfg.update(LENGTH_TABLE[DATASET_KEY])
    cfg.update(CONFIG_OVERRIDES)  # experiment-specific single-variable changes
    cfg["experiment_id"] = EXPERIMENT_ID
    cfg["method_family"] = METHOD_FAMILY
    cfg["training_setting"] = TRAINING_SETTING
    cfg["tuning_mode"] = TUNING_MODE
    cfg["aggregation_strategy"] = AGGREGATION_STRATEGY
    cfg["is_proposed_method"] = IS_PROPOSED_METHOD
    cfg["dataset_key"] = DATASET_KEY
    cfg["dataset_name"] = DATASET_REGISTRY[DATASET_KEY]["hf_name"]
    cfg["dataset_config"] = DATASET_REGISTRY[DATASET_KEY]["hf_config"]
    cfg["task_name"] = DATASET_REGISTRY[DATASET_KEY]["task"]
    cfg["model_name"] = MODEL_NAME
    cfg["rq_supported"] = RQ_SUPPORTED
    cfg["secure_aggregation_implemented"] = False
    # learning rate resolved by tuning mode
    cfg["learning_rate"] = cfg["lora_learning_rate"] if TUNING_MODE == "lora" else cfg["full_learning_rate"]
    cfg["effective_batch_size"] = cfg["train_batch_size"] * cfg["gradient_accumulation_steps"]
    # Optional smoke mode: shrink everything for a fast end-to-end validation run.
    # Enable with env FEDTGEN_SMOKE=1. Recorded in config.json; NOT used for real runs.
    if os.environ.get("FEDTGEN_SMOKE") == "1":
        cfg["smoke_mode"] = True
        cfg["max_rounds"] = 1
        cfg["centralized_max_epochs"] = 1
        cfg["local_max_epochs"] = 1
        cfg["num_clients"] = min(cfg["num_clients"], 2)
        cfg["max_train_samples"] = 40
        cfg["max_validation_samples"] = 16
        cfg["max_test_samples"] = 16
        cfg["minimum_samples_per_client"] = 4
        cfg["num_pseudo_topics"] = 3
        cfg["latency_max_samples"] = 8
        cfg["latency_warmup_batches"] = 2
        cfg["bertscore_model_type"] = "prajjwal1/bert-tiny"
        cfg["early_stopping_patience"] = 1
    else:
        cfg["smoke_mode"] = False
    return cfg


CONFIG = build_config()

IS_FEDERATED = (TRAINING_SETTING == "federated")
IS_LOCAL = (TRAINING_SETTING == "local")
IS_CENTRALIZED = (TRAINING_SETTING == "centralized")
USE_LORA = (TUNING_MODE == "lora")


# ----------------------------------------------------------------------------
# 4. Reproducibility & hashing helpers
# ----------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def stable_hash(obj):
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_write_csv(path, df):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def json_safe(x):
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


# ----------------------------------------------------------------------------
# 5. Output directories
# ----------------------------------------------------------------------------
def resolve_output_root():
    # Prefer Google Drive when mounted; otherwise a local folder.
    drive_root = "/content/drive/MyDrive/fedtgen_experiments"
    env_root = os.environ.get("FEDTGEN_OUTPUT_ROOT")
    if env_root:
        return env_root
    if os.path.isdir("/content/drive/MyDrive"):
        return drive_root
    return os.path.join(os.getcwd(), "fedtgen_experiments")


OUTPUT_ROOT = resolve_output_root()
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, EXPERIMENT_ID, "seed_42")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
BEST_DIR = os.path.join(OUTPUT_DIR, "best_model")
FINAL_DIR = os.path.join(OUTPUT_DIR, "final_model")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
for _d in (OUTPUT_DIR, CKPT_DIR, BEST_DIR, FINAL_DIR, LOGS_DIR, CACHE_DIR):
    os.makedirs(_d, exist_ok=True)


def opath(name):
    return os.path.join(OUTPUT_DIR, name)


def write_status(state, extra=None):
    data = {"experiment_id": EXPERIMENT_ID, "seed": 42, "status": state,
            "updated_at": now_iso()}
    if extra:
        data.update(extra)
    atomic_write_json(opath("status.json"), data)


# ----------------------------------------------------------------------------
# 6. Device & precision
# ----------------------------------------------------------------------------
def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        bf16 = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if bf16 else torch.float16
        precision = "bf16" if bf16 else "fp16"
    else:
        device = torch.device("cpu")
        gpu_name = "cpu"
        total_mem = 0.0
        amp_dtype = torch.float32
        precision = "fp32"
    return device, gpu_name, total_mem, amp_dtype, precision


DEVICE, GPU_NAME, GPU_TOTAL_MEM_MB, AMP_DTYPE, PRECISION = setup_device()
USE_AMP = (DEVICE.type == "cuda")


def reset_peak_vram():
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def vram_stats():
    if DEVICE.type == "cuda":
        return {
            "vram_allocated_MB": torch.cuda.memory_allocated() / (1024 ** 2),
            "vram_reserved_MB": torch.cuda.memory_reserved() / (1024 ** 2),
            "max_vram_allocated_MB": torch.cuda.max_memory_allocated() / (1024 ** 2),
            "max_vram_reserved_MB": torch.cuda.max_memory_reserved() / (1024 ** 2),
        }
    return {"vram_allocated_MB": 0.0, "vram_reserved_MB": 0.0,
            "max_vram_allocated_MB": 0.0, "max_vram_reserved_MB": 0.0}


def cpu_memory_MB():
    if psutil is not None:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    return 0.0


def environment_info():
    import transformers, datasets
    try:
        import peft
        peft_v = peft.__version__
    except Exception:
        peft_v = "unknown"
    try:
        import accelerate
        acc_v = accelerate.__version__
    except Exception:
        acc_v = "unknown"
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "datasets_version": datasets.__version__,
        "peft_version": peft_v,
        "accelerate_version": acc_v,
        "cuda_version": (torch.version.cuda or "cpu"),
        "device": DEVICE.type,
        "gpu_name": GPU_NAME,
        "gpu_total_memory_MB": GPU_TOTAL_MEM_MB,
        "precision": PRECISION,
        "platform": platform.platform(),
    }


# ----------------------------------------------------------------------------
# 7. Dataset loading & preprocessing
# ----------------------------------------------------------------------------
def _first_present(features, candidates):
    for c in candidates:
        if c in features:
            return c
    return None


def _safe_load_dataset(name, config=None):
    """load_dataset that also works for script-based datasets (needs
    trust_remote_code on datasets 2.x/3.x)."""
    from datasets import load_dataset
    attempts = []
    for kwargs in ({}, {"trust_remote_code": True}):
        try:
            if config:
                return load_dataset(name, config, **kwargs)
            return load_dataset(name, **kwargs)
        except Exception as e:
            attempts.append(f"{name}({kwargs}): {type(e).__name__}: {e}")
    raise RuntimeError("could not load dataset:\n  " + "\n  ".join(attempts))


def _load_squad_dataset():
    """Prefer lmqg/qg_squad (script dataset); fall back to script-free
    rajpurkar/squad if scripts are unsupported (datasets>=4)."""
    try:
        return _safe_load_dataset("lmqg/qg_squad"), "lmqg/qg_squad"
    except Exception as e:
        print(f"[data] lmqg/qg_squad unavailable ({type(e).__name__}); "
              f"falling back to rajpurkar/squad (script-free)")
        return _safe_load_dataset("rajpurkar/squad"), "rajpurkar/squad"


def load_raw_examples():
    """Return dict with 'train','validation','test' lists of {input,target}.
    Checks real schema via dataset.features and prints it. Filters empties.
    """
    reg = DATASET_REGISTRY[DATASET_KEY]
    print(f"[data] Loading {reg['hf_name']} config={reg['hf_config']} ...")
    squad_source = None
    if DATASET_KEY == "squad_qg":
        ds, squad_source = _load_squad_dataset()
        print(f"[data] squad_qg source = {squad_source}")
    else:
        ds = _safe_load_dataset(reg["hf_name"], reg["hf_config"])
    for split in ds.keys():
        print(f"[schema] split={split} features={ds[split].features}")

    raw_counts = {}
    built = {"train": [], "validation": [], "test": []}

    if DATASET_KEY == "samsum":
        splits = {"train": "train", "validation": "validation", "test": "test"}
        for out_split, in_split in splits.items():
            d = ds[in_split]
            raw_counts[out_split] = len(d)
            for ex in d:
                dialogue = (ex.get("dialogue") or "").strip()
                summary = (ex.get("summary") or "").strip()
                if dialogue and summary:
                    built[out_split].append({
                        "input": f"summarize dialogue: {dialogue}",
                        "target": summary})

    elif DATASET_KEY == "dailydialog":
        # locate the dialogue field (list of utterances)
        feat = ds["train"].features
        dlg_col = _first_present(feat, ["utterances", "dialog", "dialogue", "turns"])
        if dlg_col is None:
            raise RuntimeError(f"DailyDialog dialogue column not found in {list(feat)}")
        print(f"[data] dailydialog dialogue column = '{dlg_col}'")

        def split_map(name):
            return name if name in ds else ("validation" if name == "validation" and "valid" in ds else name)
        avail = set(ds.keys())
        name_map = {"train": "train",
                    "validation": "validation" if "validation" in avail else ("valid" if "valid" in avail else "train"),
                    "test": "test" if "test" in avail else "train"}
        for out_split, in_split in name_map.items():
            d = ds[in_split]
            raw_counts[out_split] = len(d)
            for ex in d:
                turns = ex[dlg_col]
                if isinstance(turns, str):
                    turns = re.split(r"__eou__|\n", turns)
                turns = [str(t).strip() for t in turns if str(t).strip()]
                for i in range(1, len(turns)):
                    context = " ".join(turns[max(0, i - 3):i])
                    response = turns[i]
                    if context and response:
                        built[out_split].append({
                            "input": f"generate next response: {context}",
                            "target": response})

    elif DATASET_KEY == "squad_qg":
        feat = ds["train"].features
        ans_col = _first_present(feat, ["answer", "answers"])
        ctx_col = _first_present(feat, ["context", "paragraph", "sentence", "paragraph_sentence"])
        q_col = _first_present(feat, ["question"])
        print(f"[data] squad_qg source={squad_source} columns answer='{ans_col}' "
              f"context='{ctx_col}' question='{q_col}'")

        def build_qg(ex):
            ans = ex.get(ans_col)
            if isinstance(ans, dict) and "text" in ans:
                ans = ans["text"][0] if len(ans["text"]) else ""
            if isinstance(ans, list):
                ans = ans[0] if ans else ""
            ans = (str(ans) if ans is not None else "").strip()
            ctx = (str(ex.get(ctx_col)) if ex.get(ctx_col) is not None else "").strip()
            q = (str(ex.get(q_col)) if ex.get(q_col) is not None else "").strip()
            if ans and ctx and q:
                return {"input": f"generate question: answer: {ans} context: {ctx}", "target": q}
            return None

        avail = set(ds.keys())
        if "test" in avail and "validation" in avail:
            # lmqg/qg_squad: proper train/validation/test splits
            for out_split in ("train", "validation", "test"):
                d = ds[out_split]
                raw_counts[out_split] = len(d)
                for ex in d:
                    pair = build_qg(ex)
                    if pair:
                        built[out_split].append(pair)
        else:
            # rajpurkar/squad fallback: only train+validation exist.
            # Use validation as test; split train -> train/validation (seed 42).
            train_pairs = [p for p in (build_qg(ex) for ex in ds["train"]) if p]
            test_pairs = [p for p in (build_qg(ex) for ex in ds["validation"]) if p]
            raw_counts["train_raw"] = len(ds["train"])
            raw_counts["validation_raw"] = len(ds["validation"])
            rng = random.Random(42)
            idx = list(range(len(train_pairs)))
            rng.shuffle(idx)
            n_val = max(1, int(0.1 * len(idx)))
            val_set = set(idx[:n_val])
            built["train"] = [train_pairs[i] for i in idx[n_val:]]
            built["validation"] = [train_pairs[i] for i in idx[:n_val]]
            built["test"] = test_pairs

    elif DATASET_KEY == "qqp":
        # nyu-mll/glue qqp: train + validation (test has no labels)
        base_train = ds["train"]
        base_valid = ds["validation"]
        raw_counts["train_raw"] = len(base_train)
        raw_counts["validation_raw"] = len(base_valid)

        def build_pairs(d):
            out = []
            for ex in d:
                if int(ex["label"]) != 1:
                    continue
                q1 = (ex.get("question1") or "").strip()
                q2 = (ex.get("question2") or "").strip()
                if q1 and q2:
                    out.append({"input": f"paraphrase: {q1}", "target": q2})
            return out

        train_pairs = build_pairs(base_train)
        test_pairs = build_pairs(base_valid)  # original validation -> test
        # split original train -> train/validation with seed 42
        rng = random.Random(42)
        idx = list(range(len(train_pairs)))
        rng.shuffle(idx)
        n_val = max(1, int(0.1 * len(idx)))
        val_idx = set(idx[:n_val])
        built["train"] = [train_pairs[i] for i in idx[n_val:]]
        built["validation"] = [train_pairs[i] for i in idx[:n_val]]
        built["test"] = test_pairs
    else:
        raise RuntimeError(f"Unknown dataset {DATASET_KEY}")

    # ensure validation/test exist; if a split fell back to train, carve out.
    for sp in ("validation", "test"):
        if not built[sp]:
            rng = random.Random(42)
            pool = built["train"]
            k = min(len(pool) // 5, 1000) or 1
            rng.shuffle(pool)
            built[sp] = pool[:k]
            built["train"] = pool[k:]

    filtered_counts = {sp: len(built[sp]) for sp in ("train", "validation", "test")}
    return built, raw_counts, filtered_counts


def subsample(examples, cap, seed):
    if cap is None or len(examples) <= cap:
        return examples
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    idx = sorted(idx[:cap])
    return [examples[i] for i in idx]


# ----------------------------------------------------------------------------
# 8. Non-IID partition (pseudo-topic Dirichlet)
# ----------------------------------------------------------------------------
def make_pseudo_topics(texts, num_topics, seed):
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.cluster import MiniBatchKMeans
    vect = HashingVectorizer(n_features=2 ** 14, alternate_sign=False,
                             norm="l2", stop_words="english")
    X = vect.transform(texts)
    k = min(num_topics, max(2, len(texts)))
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10, batch_size=256)
    labels = km.fit_predict(X)
    return labels, k


def dirichlet_partition(labels, num_clients, alpha, seed, min_per_client):
    rng = np.random.RandomState(seed)
    n = len(labels)
    labels = np.asarray(labels)
    client_indices = {c: [] for c in range(num_clients)}
    for topic in np.unique(labels):
        topic_idx = np.where(labels == topic)[0]
        topic_idx = np.sort(topic_idx)  # deterministic ordering before split
        rng.shuffle(topic_idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        cuts = (np.cumsum(proportions) * len(topic_idx)).astype(int)[:-1]
        parts = np.split(topic_idx, cuts)
        for c in range(num_clients):
            client_indices[c].extend(parts[c].tolist())
    # enforce minimum samples per client by redistribution from the largest
    for c in range(num_clients):
        client_indices[c] = sorted(client_indices[c])
    counts = {c: len(v) for c, v in client_indices.items()}
    for c in range(num_clients):
        while counts[c] < min_per_client:
            donor = max(counts, key=lambda x: counts[x])
            if donor == c or counts[donor] <= min_per_client:
                break
            moved = client_indices[donor].pop()
            client_indices[c].append(moved)
            counts[c] += 1
            counts[donor] -= 1
        client_indices[c] = sorted(client_indices[c])
    return client_indices


def build_partition(train_ex, val_ex, test_ex, cfg):
    texts = [e["input"] for e in train_ex]
    labels, k = make_pseudo_topics(texts, cfg["num_pseudo_topics"], cfg["seed"])
    train_part = dirichlet_partition(labels, cfg["num_clients"], cfg["dirichlet_alpha"],
                                     cfg["seed"], cfg["minimum_samples_per_client"])
    # validation/test partitioned by simple deterministic Dirichlet on random labels
    val_labels = np.zeros(len(val_ex), dtype=int)
    test_labels = np.zeros(len(test_ex), dtype=int)
    val_part = dirichlet_partition(val_labels, cfg["num_clients"], cfg["dirichlet_alpha"],
                                   cfg["seed"] + 1, min(cfg["minimum_samples_per_client"], max(1, len(val_ex)//cfg["num_clients"])))
    test_part = dirichlet_partition(test_labels, cfg["num_clients"], cfg["dirichlet_alpha"],
                                    cfg["seed"] + 2, min(cfg["minimum_samples_per_client"], max(1, len(test_ex)//cfg["num_clients"])))

    fingerprint = stable_hash({
        "dataset": cfg["dataset_name"], "config": cfg["dataset_config"],
        "n_train": len(train_ex), "n_val": len(val_ex), "n_test": len(test_ex),
        "first_inputs": [e["input"][:64] for e in train_ex[:20]],
    })
    partition = {
        "seed": cfg["seed"],
        "dataset_fingerprint": fingerprint,
        "num_clients": cfg["num_clients"],
        "dirichlet_alpha": cfg["dirichlet_alpha"],
        "pseudo_topic_config": {
            "vectorizer": "HashingVectorizer(n_features=16384)",
            "cluster": "MiniBatchKMeans", "n_init": 10,
            "num_pseudo_topics_requested": cfg["num_pseudo_topics"],
            "num_pseudo_topics_effective": int(k),
        },
        "client_train_indices": {str(c): train_part[c] for c in range(cfg["num_clients"])},
        "client_validation_indices": {str(c): val_part[c] for c in range(cfg["num_clients"])},
        "client_test_indices": {str(c): test_part[c] for c in range(cfg["num_clients"])},
    }
    partition["partition_checksum"] = stable_hash({
        "train": partition["client_train_indices"],
        "val": partition["client_validation_indices"],
        "test": partition["client_test_indices"],
        "fingerprint": fingerprint,
        "alpha": cfg["dirichlet_alpha"], "K": cfg["num_clients"],
    })
    return partition, labels


def js_divergence(p, q):
    p = np.asarray(p, dtype=float) + 1e-12
    q = np.asarray(q, dtype=float) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        return float(np.sum(a * np.log(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def mean_pairwise_js(partition, labels, num_topics, num_clients):
    dists = []
    for c in range(num_clients):
        idx = partition["client_train_indices"][str(c)]
        hist = np.zeros(max(1, num_topics))
        for i in idx:
            hist[int(labels[i])] += 1
        dists.append(hist)
    vals = []
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            vals.append(js_divergence(dists[i], dists[j]))
    return float(np.mean(vals)) if vals else 0.0


# ----------------------------------------------------------------------------
# 9. Tokenization
# ----------------------------------------------------------------------------
class Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, i):
        return self.encodings[i]


def tokenize_examples(examples, tokenizer, cfg, cache_key):
    # NOTE: items contain ONLY numeric fields (input_ids, attention_mask, labels).
    # DataCollatorForSeq2Seq pads every key, so any string field would crash it.
    # Reference/input texts are read from the `examples` list, not from here.
    cache_file = os.path.join(CACHE_DIR, f"tok_v2_{cache_key}.pt")
    if os.path.exists(cache_file):
        try:
            return torch.load(cache_file)
        except Exception as _cache_err:
            print(f"[cache] rebuild {cache_key} ({_cache_err})")
    enc = []
    for ex in examples:
        model_inputs = tokenizer(ex["input"], max_length=cfg["max_source_length"],
                                 truncation=True)
        labels = tokenizer(text_target=ex["target"], max_length=cfg["max_target_length"],
                           truncation=True)
        item = {"input_ids": model_inputs["input_ids"],
                "attention_mask": model_inputs["attention_mask"],
                "labels": labels["input_ids"]}
        enc.append(item)
    try:
        torch.save(enc, cache_file)
    except Exception as _save_err:
        print(f"[cache] could not persist {cache_key} ({_save_err})")
    return enc


# ----------------------------------------------------------------------------
# 10. Model construction
# ----------------------------------------------------------------------------
def build_model():
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    if USE_LORA:
        from peft import LoraConfig, get_peft_model, TaskType
        lconf = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"],
            lora_dropout=CONFIG["lora_dropout"],
            target_modules=CONFIG["lora_target_modules"], bias="none")
        model = get_peft_model(model, lconf)
    return model.to(DEVICE)


def param_counts(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    adapter = trainable if USE_LORA else 0
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "frozen_params": int(frozen),
        "adapter_params": int(adapter),
        "trainable_ratio": float(trainable / total) if total else 0.0,
        "model_size_MB": float(total * 4 / (1024 ** 2)),
        "adapter_size_MB": float(adapter * 4 / (1024 ** 2)),
    }


def get_trainable_state(model):
    return {n: p.detach().float().cpu().clone()
            for n, p in model.named_parameters() if p.requires_grad}


def load_trainable_state(model, state):
    own = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in state.items():
            if n in own:
                own[n].copy_(v.to(own[n].device, own[n].dtype))


def state_to_vector(state):
    return torch.cat([v.flatten() for _, v in sorted(state.items())])


def count_communicated_params(model):
    """Parameters transmitted per direction. LoRA -> adapter params only;
    full -> all params (frozen backbone is not re-transmitted in LoRA case)."""
    if USE_LORA:
        return sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------
# 11. Metrics
# ----------------------------------------------------------------------------
_ROUGE = None
_BERT_CACHE = {}


def get_rouge():
    global _ROUGE
    if _ROUGE is None:
        from rouge_score import rouge_scorer
        _ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return _ROUGE


def compute_bleu(preds, refs):
    import sacrebleu
    if not preds:
        return 0.0
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return float(bleu.score)  # 0-100


def compute_rouge(preds, refs):
    scorer = get_rouge()
    r1 = r2 = rl = 0.0
    n = max(1, len(preds))
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        r1 += s["rouge1"].fmeasure
        r2 += s["rouge2"].fmeasure
        rl += s["rougeL"].fmeasure
    return r1 / n, r2 / n, rl / n  # each 0-1


def compute_bertscore(preds, refs, cfg):
    from bert_score import score as bert_score_fn
    try:
        P, R, Fb = bert_score_fn(preds, refs, lang="en",
                                 model_type=cfg["bertscore_model_type"],
                                 rescale_with_baseline=False,
                                 device=DEVICE.type, batch_size=cfg["eval_batch_size"])
        return float(P.mean()), float(R.mean()), float(Fb.mean())
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            P, R, Fb = bert_score_fn(preds, refs, lang="en",
                                     model_type=cfg["bertscore_model_type"],
                                     rescale_with_baseline=False,
                                     device="cpu", batch_size=8)
            return float(P.mean()), float(R.mean()), float(Fb.mean())
        raise


# ----------------------------------------------------------------------------
# 12. Generation / evaluation
# ----------------------------------------------------------------------------
def make_loader(dataset, collator, batch_size, shuffle, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collator, generator=g if shuffle else None)


def evaluate_model(model, examples, tokenizer, collator, cfg, desc="eval",
                   compute_loss=True, return_predictions=False):
    model.eval()
    enc = tokenize_examples(examples, tokenizer, cfg, cache_key=f"eval_{stable_hash(desc)}_{len(examples)}")
    ds = Seq2SeqDataset(enc)
    loader = make_loader(ds, collator, cfg["eval_batch_size"], shuffle=False, seed=cfg["seed"])

    total_loss, n_batches = 0.0, 0
    if compute_loss:
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{desc}:loss", leave=False):
                batch = {k: v.to(DEVICE) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                    out = model(**batch)
                total_loss += out.loss.item()
                n_batches += 1
    avg_loss = total_loss / max(1, n_batches)

    preds, refs, inputs = [], [], []
    gen_start = time.perf_counter()
    gen_module = model.merge_and_unload() if False else model  # keep adapter attached
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{desc}:gen", leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                gen = model.generate(
                    input_ids=input_ids, attention_mask=attn,
                    max_new_tokens=cfg["max_target_length"],
                    num_beams=cfg["generation_num_beams"],
                    no_repeat_ngram_size=cfg["no_repeat_ngram_size"],
                    early_stopping=cfg["generation_early_stopping"])
            dec = tokenizer.batch_decode(gen, skip_special_tokens=True)
            preds.extend([d.strip() for d in dec])
    gen_time = time.perf_counter() - gen_start
    refs = [e["target"] for e in examples]
    inputs = [e["input"] for e in examples]
    preds = [p if p else " " for p in preds][:len(refs)]

    bleu = compute_bleu(preds, refs)
    r1, r2, rl = compute_rouge(preds, refs)
    bp, br, bf = compute_bertscore(preds, refs, cfg)
    metrics = {
        "loss": avg_loss, "bleu": bleu, "rouge1": r1, "rouge2": r2, "rougeL": rl,
        "bertscore_precision": bp, "bertscore_recall": br, "bertscore_f1": bf,
        "generation_time_seconds": gen_time,
    }
    if return_predictions:
        return metrics, {"preds": preds, "refs": refs, "inputs": inputs}
    return metrics


# ----------------------------------------------------------------------------
# 13. Local training (one model for N epochs / steps)
# ----------------------------------------------------------------------------
def build_optimizer_scheduler(model, cfg, num_training_steps):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["learning_rate"])
    warmup = int(cfg["warmup_ratio"] * num_training_steps)
    sched = get_linear_schedule_with_warmup(optim, warmup, max(1, num_training_steps))
    return optim, sched


def train_epochs(model, train_examples, tokenizer, collator, cfg, num_epochs,
                 global_state=None, control_c=None, control_ci=None,
                 progress_prefix=""):
    """Train `model` on train_examples for num_epochs. Returns dict with
    train_loss, steps, and (for scaffold) updated ci; supports fedprox proximal
    term (global_state) and scaffold correction (control_c, control_ci)."""
    enc = tokenize_examples(train_examples, tokenizer, cfg,
                            cache_key=f"train_{stable_hash(progress_prefix)}_{len(train_examples)}")
    ds = Seq2SeqDataset(enc)
    loader = make_loader(ds, collator, cfg["train_batch_size"], shuffle=True, seed=cfg["seed"])
    accum = cfg["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(len(loader) / accum)
    total_steps = steps_per_epoch * num_epochs
    optim, sched = build_optimizer_scheduler(model, cfg, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and AMP_DTYPE == torch.float16))

    mu = cfg["fedprox_mu"] if METHOD_FAMILY == "fedprox_lora" else 0.0
    global_vec_params = None
    if mu > 0 and global_state is not None:
        global_named = {n: v.to(DEVICE) for n, v in global_state.items()}
    else:
        global_named = None

    model.train()
    running_loss, loss_count, opt_steps = 0.0, 0, 0
    for epoch in range(num_epochs):
        optim.zero_grad()
        pbar = tqdm(enumerate(loader), total=len(loader),
                    desc=f"{progress_prefix} epoch {epoch+1}/{num_epochs}", leave=False)
        for step, batch in pbar:
            batch = {k: v.to(DEVICE) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                out = model(**batch)
                loss = out.loss
            if mu > 0 and global_named is not None:
                prox = 0.0
                for n, p in model.named_parameters():
                    if p.requires_grad and n in global_named:
                        prox = prox + ((p - global_named[n]) ** 2).sum()
                loss = loss + (mu / 2.0) * prox
            loss_to_back = loss / accum
            if scaler.is_enabled():
                scaler.scale(loss_to_back).backward()
            else:
                loss_to_back.backward()
            running_loss += loss.item()
            loss_count += 1
            if (step + 1) % accum == 0 or (step + 1) == len(loader):
                if scaler.is_enabled():
                    scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg["max_grad_norm"])
                if scaler.is_enabled():
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                # SCAFFOLD correction: y_i <- y_i - eta*(c - c_i) applied to params
                if METHOD_FAMILY == "scaffold_lora" and control_c is not None and control_ci is not None:
                    with torch.no_grad():
                        lr = sched.get_last_lr()[0]
                        for n, p in model.named_parameters():
                            if p.requires_grad and n in control_c:
                                p.add_(-(lr) * (control_c[n].to(DEVICE) - control_ci[n].to(DEVICE)))
                sched.step()
                optim.zero_grad()
                opt_steps += 1
            pbar.set_postfix(loss=f"{running_loss/max(1,loss_count):.4f}")
    return {"train_loss": running_loss / max(1, loss_count), "opt_steps": opt_steps,
            "steps_per_epoch": steps_per_epoch}


# ----------------------------------------------------------------------------
# 14. FLOPs (analytical estimate) & latency
# ----------------------------------------------------------------------------
def estimate_flops(counts, cfg):
    try:
        N = counts["total_params"]
        src = cfg["max_source_length"]
        tgt = cfg["max_target_length"]
        fwd = 2.0 * N * (src + tgt)
        train_step = 3.0 * fwd * cfg["train_batch_size"]
        gen = 2.0 * N * tgt * cfg["generation_num_beams"]
        return {
            "forward_flops_per_sample": float(fwd),
            "training_flops_per_step": float(train_step),
            "generation_flops_per_sample": float(gen),
            "flops_measurement_method": "analytical_2ND",
            "flops_is_estimated": True,
            "flops_measurement_status": "estimated",
            "flops_error_message": "",
        }
    except Exception as e:
        return {
            "forward_flops_per_sample": float("nan"),
            "training_flops_per_step": float("nan"),
            "generation_flops_per_sample": float("nan"),
            "flops_measurement_method": "analytical_2ND",
            "flops_is_estimated": True,
            "flops_measurement_status": "failed",
            "flops_error_message": str(e),
        }


def measure_latency(model, examples, tokenizer, cfg, mode_label, device_type):
    """Measure per-sample generation latency on best model. Excludes tokenization."""
    samples = examples[:cfg["latency_max_samples"]]
    pre = []
    for ex in samples:
        enc = tokenizer(ex["input"], max_length=cfg["max_source_length"],
                        truncation=True, return_tensors="pt")
        pre.append({"input_ids": enc["input_ids"].to(DEVICE),
                    "attention_mask": enc["attention_mask"].to(DEVICE)})
    model.eval()
    # warmup
    for i in range(min(cfg["latency_warmup_batches"], len(pre))):
        with torch.no_grad():
            model.generate(input_ids=pre[i]["input_ids"],
                           attention_mask=pre[i]["attention_mask"],
                           max_new_tokens=cfg["max_target_length"],
                           num_beams=cfg["generation_num_beams"],
                           no_repeat_ngram_size=cfg["no_repeat_ngram_size"],
                           early_stopping=cfg["generation_early_stopping"])
    per_sample_ms, gen_tokens, in_tokens, rows = [], [], [], []
    t0_all = time.perf_counter()
    for bi, item in enumerate(tqdm(pre, desc=f"latency:{mode_label}", leave=False)):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_ids=item["input_ids"],
                                  attention_mask=item["attention_mask"],
                                  max_new_tokens=cfg["max_target_length"],
                                  num_beams=cfg["generation_num_beams"],
                                  no_repeat_ngram_size=cfg["no_repeat_ngram_size"],
                                  early_stopping=cfg["generation_early_stopping"])
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        gt = out.shape[1]
        it = item["input_ids"].shape[1]
        per_sample_ms.append(dt * 1000.0)
        gen_tokens.append(gt)
        in_tokens.append(it)
        rows.append({
            "experiment_id": EXPERIMENT_ID, "seed": 42, "latency_mode": mode_label,
            "batch_index": bi, "num_samples": 1, "batch_size": cfg["latency_batch_size"],
            "input_tokens": it, "generated_tokens": gt, "elapsed_seconds": dt,
            "latency_ms_per_sample": dt * 1000.0,
            "throughput_samples_per_second": 1.0 / dt if dt > 0 else 0.0,
            "throughput_tokens_per_second": gt / dt if dt > 0 else 0.0,
            "device": device_type, "dtype": str(AMP_DTYPE).replace("torch.", ""),
            "num_beams": cfg["generation_num_beams"],
            "max_new_tokens": cfg["max_target_length"],
            "includes_tokenization": False, "is_warmup": False})
    total_time = time.perf_counter() - t0_all
    arr = np.array(per_sample_ms)
    summary = {
        "inference_latency_mean_ms_per_sample": float(arr.mean()),
        "inference_latency_median_ms_per_sample": float(np.median(arr)),
        "inference_latency_p95_ms_per_sample": float(np.percentile(arr, 95)),
        "inference_latency_min_ms_per_sample": float(arr.min()),
        "inference_latency_max_ms_per_sample": float(arr.max()),
        "inference_latency_std_ms_per_sample": float(arr.std()),
        "inference_throughput_samples_per_second": float(1000.0 / arr.mean()) if arr.mean() > 0 else 0.0,
        "inference_throughput_tokens_per_second": float(np.sum(gen_tokens) / total_time) if total_time > 0 else 0.0,
        "latency_total_time_seconds": float(total_time),
        "latency_num_samples": int(len(arr)),
        "latency_batch_size": cfg["latency_batch_size"],
        "latency_warmup_batches": cfg["latency_warmup_batches"],
        "latency_device": device_type,
        "latency_includes_tokenization": False,
        "average_input_tokens": float(np.mean(in_tokens)),
        "average_generated_tokens": float(np.mean(gen_tokens)),
    }
    return summary, rows


# ----------------------------------------------------------------------------
# 15. Checkpoint / resume
# ----------------------------------------------------------------------------
def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as e:
        print(f"[resume] RNG restore warning: {e}")


def save_checkpoint(tag, payload):
    tmp = tempfile.mkdtemp(dir=CKPT_DIR)
    torch.save(payload, os.path.join(tmp, "state.pt"))
    with open(os.path.join(tmp, "COMPLETE"), "w") as f:
        f.write(now_iso())
    final = os.path.join(CKPT_DIR, tag)
    if os.path.exists(final):
        shutil.rmtree(final)
    os.replace(tmp, final)
    # keep only two most recent valid checkpoints
    ckpts = sorted([d for d in os.listdir(CKPT_DIR)
                    if os.path.isdir(os.path.join(CKPT_DIR, d))
                    and os.path.exists(os.path.join(CKPT_DIR, d, "COMPLETE"))],
                   key=lambda d: os.path.getmtime(os.path.join(CKPT_DIR, d)))
    for old in ckpts[:-2]:
        shutil.rmtree(os.path.join(CKPT_DIR, old), ignore_errors=True)


def load_latest_checkpoint():
    if not os.path.isdir(CKPT_DIR):
        return None
    ckpts = sorted([d for d in os.listdir(CKPT_DIR)
                    if os.path.isdir(os.path.join(CKPT_DIR, d))],
                   key=lambda d: os.path.getmtime(os.path.join(CKPT_DIR, d)), reverse=True)
    for d in ckpts:
        cp = os.path.join(CKPT_DIR, d)
        if not os.path.exists(os.path.join(cp, "COMPLETE")):
            continue
        try:
            return torch.load(os.path.join(cp, "state.pt"), map_location="cpu")
        except Exception as e:
            print(f"[resume] checkpoint {d} corrupt ({e}); trying older one")
            continue
    return None


# ----------------------------------------------------------------------------
# 16. CSV history accumulators
# ----------------------------------------------------------------------------
class HistoryStore:
    def __init__(self):
        self.rounds = []
        self.clients = []
        self.comm = []
        self.resource = []

    def to_frames(self):
        return (pd.DataFrame(self.rounds), pd.DataFrame(self.clients),
                pd.DataFrame(self.comm), pd.DataFrame(self.resource))


# ============================================================================
# Orchestration: settings (local / centralized / federated), aggregation
# strategies (FedAvg / FedProx / SCAFFOLD / FedTGen), evaluation, final results.
# ============================================================================

BYTES_PER_PARAM = 4  # FP32 transmission / aggregation
TRANSMITTED_DTYPE = "float32"


def zeros_like_state(state):
    return {n: torch.zeros_like(v) for n, v in state.items()}


def add_scaled(acc, state, scale):
    for n in acc:
        acc[n].add_(state[n] * scale)


def state_update(local_state, global_state):
    return {n: (local_state[n] - global_state[n]) for n in local_state}


def cosine_of_updates(u1, u2):
    v1 = state_to_vector(u1)
    v2 = state_to_vector(u2)
    denom = (v1.norm() * v2.norm())
    if denom.item() == 0:
        return 0.0
    return float(torch.dot(v1, v2) / denom)


def update_norm(u):
    return float(state_to_vector(u).norm())


def minmax_norm(values):
    v = np.array(values, dtype=float)
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


# ----------------------------------------------------------------------------
# Federated training loop
# ----------------------------------------------------------------------------
def run_federated(model, tokenizer, collator, cfg, data, hist, resume_state):
    train_ex = data["train"]
    val_ex = data["validation"]
    partition = data["partition"]
    client_train = {int(c): idx for c, idx in partition["client_train_indices"].items()}
    num_clients = cfg["num_clients"]
    client_sample_counts = {c: len(client_train[c]) for c in range(num_clients)}
    total_train = sum(client_sample_counts.values())

    global_state = get_trainable_state(model)

    # SCAFFOLD control variates
    if METHOD_FAMILY == "scaffold_lora":
        c_global = zeros_like_state(global_state)
        c_clients = {c: zeros_like_state(global_state) for c in range(num_clients)}
    else:
        c_global, c_clients = None, None

    start_round = 0
    best_metric = -float("inf")
    best_round = -1
    patience = 0
    early_stopped = False
    cumulative_comm_mb = 0.0
    comm_totals = {"upload": 0.0, "download": 0.0, "metadata": 0.0, "control": 0.0}
    fallback_count = 0
    weight_entropy_list = []
    max_weight_list, min_weight_list = [], []
    resume_count = 0
    experiment_start = time.perf_counter()

    if resume_state is not None:
        gs = resume_state.get("global_state")
        if gs is not None:
            global_state = {n: v for n, v in gs.items()}
            load_trainable_state(model, global_state)
        start_round = resume_state.get("next_round", 0)
        best_metric = resume_state.get("best_metric", -float("inf"))
        best_round = resume_state.get("best_round", -1)
        patience = resume_state.get("patience", 0)
        cumulative_comm_mb = resume_state.get("cumulative_comm_mb", 0.0)
        comm_totals = resume_state.get("comm_totals", comm_totals)
        hist.rounds = resume_state.get("hist_rounds", [])
        hist.clients = resume_state.get("hist_clients", [])
        hist.comm = resume_state.get("hist_comm", [])
        hist.resource = resume_state.get("hist_resource", [])
        fallback_count = resume_state.get("fallback_count", 0)
        resume_count = resume_state.get("resume_count", 0) + 1
        if METHOD_FAMILY == "scaffold_lora":
            c_global = resume_state.get("c_global", c_global)
            c_clients = resume_state.get("c_clients", c_clients)
        restore_rng(resume_state.get("rng", rng_state()))
        print(f"[resume] resuming federated at round {start_round}")

    rng = random.Random(cfg["seed"])
    # pre-roll client selection to match round index deterministically
    selection_history = []
    for r in range(cfg["max_rounds"]):
        m = max(1, int(round(cfg["participation_rate"] * num_clients)))
        sel = sorted(rng.sample(range(num_clients), m))
        selection_history.append(sel)

    completed_rounds = start_round
    for r in range(start_round, cfg["max_rounds"]):
        round_start = time.perf_counter()
        reset_peak_vram()
        selected = selection_history[r]
        print(f"\n[round {r+1}/{cfg['max_rounds']}] selected clients: {selected}")

        client_updates = {}
        client_ci_deltas = {}
        client_stats = {}
        round_train_loss = []
        client_train_time_total = 0.0

        for c in selected:
            idx = client_train[c]
            client_examples = [train_ex[i] for i in idx]
            if len(client_examples) == 0:
                continue
            # start each client from a copy of the global model
            load_trainable_state(model, global_state)
            ct0 = time.perf_counter()
            if METHOD_FAMILY == "scaffold_lora":
                tr = train_epochs(model, client_examples, tokenizer, collator, cfg,
                                  cfg["local_epochs"], global_state=global_state,
                                  control_c=c_global, control_ci=c_clients[c],
                                  progress_prefix=f"R{r+1} C{c}")
            elif METHOD_FAMILY == "fedprox_lora":
                tr = train_epochs(model, client_examples, tokenizer, collator, cfg,
                                  cfg["local_epochs"], global_state=global_state,
                                  progress_prefix=f"R{r+1} C{c}")
            else:
                tr = train_epochs(model, client_examples, tokenizer, collator, cfg,
                                  cfg["local_epochs"], progress_prefix=f"R{r+1} C{c}")
            client_train_time = time.perf_counter() - ct0
            client_train_time_total += client_train_time
            local_state = get_trainable_state(model)
            upd = state_update(local_state, global_state)
            client_updates[c] = upd
            round_train_loss.append(tr["train_loss"])

            # local validation stats (used by FedTGen aggregation; not test set)
            local_val_loss = float("nan")
            local_val_rougeL = float("nan")
            local_bleu = float("nan")
            local_bert_f1 = float("nan")
            if METHOD_FAMILY == "fedtgen":
                vsub = val_ex[:min(len(val_ex), 128)]
                vmet = evaluate_model(model, vsub, tokenizer, collator, cfg,
                                      desc=f"R{r+1}C{c}val", compute_loss=True)
                local_val_loss = vmet["loss"]
                local_val_rougeL = vmet["rougeL"]
                local_bleu = vmet["bleu"]
                local_bert_f1 = vmet["bertscore_f1"]

            # SCAFFOLD control-variate update (option II)
            ci_delta = None
            cvnorm = float("nan")
            if METHOD_FAMILY == "scaffold_lora":
                K = max(1, tr["opt_steps"])
                lr = cfg["learning_rate"]
                new_ci = {}
                for n in global_state:
                    new_ci[n] = c_clients[c][n] - c_global[n] + (global_state[n] - local_state[n]) / (K * lr)
                ci_delta = {n: new_ci[n] - c_clients[c][n] for n in global_state}
                c_clients[c] = new_ci
                cvnorm = update_norm(new_ci)
                client_ci_deltas[c] = ci_delta

            client_stats[c] = {
                "num_train_samples": client_sample_counts[c],
                "train_loss": tr["train_loss"],
                "val_loss": local_val_loss, "val_rougeL": local_val_rougeL,
                "bleu": local_bleu, "bert_f1": local_bert_f1,
                "update_norm": update_norm(upd),
                "control_variate_norm": cvnorm,
                "client_training_time": client_train_time,
            }

        # -------- Aggregation --------
        aggregation_fallback = False
        sample_weights = {c: client_sample_counts[c] / max(1, sum(client_sample_counts[c] for c in client_updates))
                          for c in client_updates}

        # preliminary FedAvg update (also needed for FedTGen divergence)
        prelim = zeros_like_state(global_state)
        for c, upd in client_updates.items():
            add_scaled(prelim, upd, sample_weights[c])

        divergences = {}
        for c, upd in client_updates.items():
            divergences[c] = 1.0 - cosine_of_updates(upd, prelim)

        if METHOD_FAMILY == "fedtgen" and len(client_updates) > 0:
            cids = list(client_updates.keys())
            n_arr = np.array([client_stats[c]["num_train_samples"] for c in cids], dtype=float)
            L_arr = np.array([client_stats[c]["val_loss"] for c in cids], dtype=float)
            Q_arr = np.array([client_stats[c]["val_rougeL"] for c in cids], dtype=float)
            D_arr = np.array([divergences[c] for c in cids], dtype=float)
            Ln = minmax_norm(np.nan_to_num(L_arr, nan=float(np.nanmean(L_arr)) if np.any(~np.isnan(L_arr)) else 0.0))
            Qn = minmax_norm(np.nan_to_num(Q_arr, nan=float(np.nanmean(Q_arr)) if np.any(~np.isnan(Q_arr)) else 0.0))
            Dn = minmax_norm(np.nan_to_num(D_arr, nan=0.0))
            n_frac = n_arr / n_arr.sum()
            raw = (n_frac ** cfg["lambda_n"]) * np.exp(
                - cfg["beta_loss"] * Ln - cfg["beta_divergence"] * Dn + cfg["beta_quality"] * Qn)
            if not np.all(np.isfinite(raw)) or raw.sum() <= 0:
                aggregation_fallback = True
                fallback_count += 1
                weights = {c: sample_weights[c] for c in cids}
                print("[fedtgen] aggregation fallback -> sample-weighted FedAvg")
            else:
                wnorm = raw / raw.sum()
                weights = {c: float(wnorm[i]) for i, c in enumerate(cids)}
        else:
            weights = {c: sample_weights[c] for c in client_updates}

        # apply aggregated update
        agg = zeros_like_state(global_state)
        if METHOD_FAMILY == "scaffold_lora":
            # x <- x + (1/|S|) sum (y_i - x)
            for c, upd in client_updates.items():
                add_scaled(agg, upd, 1.0 / max(1, len(client_updates)))
            for n in global_state:
                global_state[n] = global_state[n] + agg[n]
            # c <- c + (1/N) sum_{i in S} delta c_i
            if client_ci_deltas:
                for n in c_global:
                    acc = torch.zeros_like(c_global[n])
                    for c in client_ci_deltas:
                        acc += client_ci_deltas[c][n]
                    c_global[n] = c_global[n] + acc / num_clients
        else:
            for c, upd in client_updates.items():
                add_scaled(agg, upd, weights[c])
            for n in global_state:
                global_state[n] = global_state[n] + agg[n]

        load_trainable_state(model, global_state)

        # weight stats
        wvals = np.array(list(weights.values())) if weights else np.array([1.0])
        wvals = wvals / wvals.sum()
        entropy = float(-np.sum(wvals * np.log(wvals + 1e-12)))
        weight_entropy_list.append(entropy)
        max_weight_list.append(float(wvals.max()))
        min_weight_list.append(float(wvals.min()))

        # -------- Communication accounting --------
        comm_params = count_communicated_params(model)
        upload_bytes_round = 0
        download_bytes_round = 0
        metadata_bytes_round = 0
        control_bytes_round = 0
        for c in selected:
            up = comm_params * BYTES_PER_PARAM  # client -> server
            down = comm_params * BYTES_PER_PARAM  # server -> client
            meta = 0
            ctrl = 0
            if METHOD_FAMILY == "fedtgen":
                meta = 4 * 8  # 4 scalar float64 validation statistics
            if METHOD_FAMILY == "scaffold_lora":
                ctrl = comm_params * BYTES_PER_PARAM * 2  # control variate up+down
            upload_bytes_round += up + meta + (ctrl // 2 if ctrl else 0)
            download_bytes_round += down + (ctrl // 2 if ctrl else 0)
            metadata_bytes_round += meta
            control_bytes_round += ctrl
            total_c_bytes = up + down + meta + ctrl
            hist.comm.append({
                "experiment_id": EXPERIMENT_ID, "seed": 42, "round": r + 1,
                "client_id": c, "method": METHOD_FAMILY, "selected": True,
                "transmitted_dtype": TRANSMITTED_DTYPE, "bytes_per_parameter": BYTES_PER_PARAM,
                "upload_parameters": comm_params, "download_parameters": comm_params,
                "upload_bytes": up + meta + (ctrl // 2 if ctrl else 0),
                "download_bytes": down + (ctrl // 2 if ctrl else 0),
                "metadata_bytes": meta, "control_variate_bytes": ctrl,
                "total_communication_bytes": total_c_bytes,
                "total_communication_MB": total_c_bytes / (1024 ** 2),
                "cumulative_communication_MB": 0.0, "cumulative_communication_GB": 0.0})

        comm_bytes_round = upload_bytes_round + download_bytes_round
        comm_mb_round = comm_bytes_round / (1024 ** 2)
        cumulative_comm_mb += comm_mb_round
        comm_totals["upload"] += upload_bytes_round / (1024 ** 2)
        comm_totals["download"] += download_bytes_round / (1024 ** 2)
        comm_totals["metadata"] += metadata_bytes_round / (1024 ** 2)
        comm_totals["control"] += control_bytes_round / (1024 ** 2)
        # backfill cumulative in the comm rows just added
        for row in hist.comm:
            if row["round"] == r + 1:
                row["cumulative_communication_MB"] = cumulative_comm_mb
                row["cumulative_communication_GB"] = cumulative_comm_mb / 1024.0

        # -------- Global validation & early stopping --------
        val_metrics = evaluate_model(model, val_ex, tokenizer, collator, cfg,
                                     desc=f"R{r+1}:globalval", compute_loss=True)
        vram = vram_stats()
        round_time = time.perf_counter() - round_start
        elapsed = time.perf_counter() - experiment_start
        monitored = val_metrics["rougeL"]
        is_new_best = monitored > (best_metric + cfg["early_stopping_min_delta"])
        if is_new_best:
            best_metric = monitored
            best_round = r + 1
            patience = 0
            model.save_pretrained(BEST_DIR)
            tokenizer.save_pretrained(BEST_DIR)
            atomic_write_json(os.path.join(BEST_DIR, "best_info.json"),
                              {"best_round": best_round, "best_metric": best_metric,
                               "validation_metrics": val_metrics})
        else:
            patience += 1

        completed_rounds = r + 1
        remaining = (cfg["max_rounds"] - completed_rounds)
        est_remaining = (elapsed / completed_rounds) * remaining if completed_rounds else 0.0

        # client history rows
        for c in selected:
            st = client_stats.get(c, {})
            hist.clients.append({
                "experiment_id": EXPERIMENT_ID, "seed": 42, "round": r + 1, "client_id": c,
                "was_selected": True,
                "num_train_samples": st.get("num_train_samples", client_sample_counts[c]),
                "num_validation_samples": len(partition["client_validation_indices"][str(c)]),
                "num_test_samples": len(partition["client_test_indices"][str(c)]),
                "local_train_loss": st.get("train_loss", float("nan")),
                "local_validation_loss": st.get("val_loss", float("nan")),
                "local_bleu": st.get("bleu", float("nan")),
                "local_rougeL": st.get("val_rougeL", float("nan")),
                "local_bertscore_f1": st.get("bert_f1", float("nan")),
                "update_norm": st.get("update_norm", float("nan")),
                "update_divergence": divergences.get(c, float("nan")),
                "control_variate_norm": st.get("control_variate_norm", float("nan")),
                "raw_aggregation_weight": weights.get(c, float("nan")),
                "normalized_aggregation_weight": weights.get(c, float("nan")),
                "aggregation_fallback": aggregation_fallback,
                "upload_MB": (count_communicated_params(model) * BYTES_PER_PARAM) / (1024 ** 2),
                "download_MB": (count_communicated_params(model) * BYTES_PER_PARAM) / (1024 ** 2),
                "client_training_time_seconds": st.get("client_training_time", float("nan")),
                "client_evaluation_time_seconds": float("nan"),
                "max_vram_allocated_MB": vram["max_vram_allocated_MB"],
                "client_inference_latency_mean_ms": float("nan"),
                "client_inference_latency_p95_ms": float("nan"),
                "client_inference_throughput": float("nan")})

        hist.resource.append({
            "experiment_id": EXPERIMENT_ID, "seed": 42, "round": r + 1, "epoch": "",
            "client_id": "", "stage": "round_end", "timestamp": now_iso(),
            "cpu_memory_MB": cpu_memory_MB(), "vram_allocated_MB": vram["vram_allocated_MB"],
            "vram_reserved_MB": vram["vram_reserved_MB"],
            "max_vram_allocated_MB": vram["max_vram_allocated_MB"],
            "max_vram_reserved_MB": vram["max_vram_reserved_MB"],
            "elapsed_time_seconds": elapsed})

        flops = estimate_flops(data["counts"], cfg)
        hist.rounds.append({
            "experiment_id": EXPERIMENT_ID, "seed": 42, "status": "running",
            "method": METHOD_FAMILY, "model_name": MODEL_NAME, "dataset_name": cfg["dataset_name"],
            "round": r + 1, "epoch": "", "num_clients": num_clients,
            "selected_clients": "|".join(map(str, selected)),
            "participation_rate": cfg["participation_rate"], "dirichlet_alpha": cfg["dirichlet_alpha"],
            "local_epochs": cfg["local_epochs"], "lora_r": cfg["lora_r"],
            "learning_rate": cfg["learning_rate"],
            "train_loss": float(np.mean(round_train_loss)) if round_train_loss else float("nan"),
            "validation_loss": val_metrics["loss"], "validation_bleu": val_metrics["bleu"],
            "validation_rouge1": val_metrics["rouge1"], "validation_rouge2": val_metrics["rouge2"],
            "validation_rougeL": val_metrics["rougeL"],
            "validation_bertscore_precision": val_metrics["bertscore_precision"],
            "validation_bertscore_recall": val_metrics["bertscore_recall"],
            "validation_bertscore_f1": val_metrics["bertscore_f1"],
            "best_metric_so_far": best_metric, "best_round": best_round,
            "patience_counter": patience, "is_new_best": is_new_best,
            "upload_MB_round": upload_bytes_round / (1024 ** 2),
            "download_MB_round": download_bytes_round / (1024 ** 2),
            "communication_MB_round": comm_mb_round,
            "cumulative_communication_MB": cumulative_comm_mb,
            "time_per_round_seconds": round_time, "training_time_seconds": client_train_time_total,
            "evaluation_time_seconds": float("nan"),
            "generation_time_seconds": val_metrics["generation_time_seconds"],
            "elapsed_time_seconds": elapsed, "estimated_remaining_time_seconds": est_remaining,
            "max_vram_allocated_MB": vram["max_vram_allocated_MB"],
            "max_vram_reserved_MB": vram["max_vram_reserved_MB"],
            "training_flops_per_step": flops["training_flops_per_step"],
            "estimated_total_training_flops": flops["training_flops_per_step"] * completed_rounds})

        print(f"  train_loss={hist.rounds[-1]['train_loss']:.4f} "
              f"val_rougeL={val_metrics['rougeL']:.4f} bleu={val_metrics['bleu']:.2f} "
              f"comm={comm_mb_round:.2f}MB cum={cumulative_comm_mb:.2f}MB "
              f"best={best_metric:.4f}@{best_round} patience={patience} "
              f"vram={vram['max_vram_allocated_MB']:.0f}MB t={round_time:.1f}s")

        # checkpoint after each round
        save_checkpoint(f"round_{r+1:04d}", {
            "global_state": global_state, "next_round": r + 1, "best_metric": best_metric,
            "best_round": best_round, "patience": patience,
            "cumulative_comm_mb": cumulative_comm_mb, "comm_totals": comm_totals,
            "hist_rounds": hist.rounds, "hist_clients": hist.clients,
            "hist_comm": hist.comm, "hist_resource": hist.resource,
            "fallback_count": fallback_count, "resume_count": resume_count,
            "c_global": c_global, "c_clients": c_clients,
            "partition_checksum": partition["partition_checksum"], "rng": rng_state()})
        write_status("running", {"current_round": r + 1})

        if patience >= cfg["early_stopping_patience"]:
            early_stopped = True
            print(f"[early-stop] no improvement for {patience} rounds; stopping at round {r+1}")
            break

    return {
        "best_metric": best_metric, "best_round": best_round,
        "completed_rounds": completed_rounds, "early_stopped": early_stopped,
        "cumulative_comm_mb": cumulative_comm_mb, "comm_totals": comm_totals,
        "fallback_count": fallback_count,
        "avg_selected": float(np.mean([len(s) for s in selection_history[:completed_rounds]])) if completed_rounds else 0.0,
        "client_sample_counts": client_sample_counts,
        "weight_entropy": float(np.mean(weight_entropy_list)) if weight_entropy_list else float("nan"),
        "max_weight": float(np.mean(max_weight_list)) if max_weight_list else float("nan"),
        "min_weight": float(np.mean(min_weight_list)) if min_weight_list else float("nan"),
        "final_global_control_variate_norm": (update_norm(c_global) if c_global is not None else float("nan")),
        "resume_count": resume_count,
        "client_divergences": [r["update_divergence"] for r in hist.clients],
    }


# ----------------------------------------------------------------------------
# Centralized training loop
# ----------------------------------------------------------------------------
def run_centralized(model, tokenizer, collator, cfg, data, hist, resume_state):
    train_ex = data["train"]
    val_ex = data["validation"]
    best_metric, best_epoch, patience = -float("inf"), -1, 0
    start_epoch = 0
    early_stopped = False
    experiment_start = time.perf_counter()
    resume_count = 0

    if resume_state is not None:
        st = resume_state.get("model_state")
        if st is not None:
            load_trainable_state(model, st)
        start_epoch = resume_state.get("next_epoch", 0)
        best_metric = resume_state.get("best_metric", -float("inf"))
        best_epoch = resume_state.get("best_epoch", -1)
        patience = resume_state.get("patience", 0)
        hist.rounds = resume_state.get("hist_rounds", [])
        hist.resource = resume_state.get("hist_resource", [])
        resume_count = resume_state.get("resume_count", 0) + 1
        restore_rng(resume_state.get("rng", rng_state()))
        print(f"[resume] resuming centralized at epoch {start_epoch}")

    max_epochs = cfg["centralized_max_epochs"]
    completed = start_epoch
    for epoch in range(start_epoch, max_epochs):
        e0 = time.perf_counter()
        reset_peak_vram()
        tr = train_epochs(model, train_ex, tokenizer, collator, cfg, 1,
                          progress_prefix=f"centralized E{epoch+1}")
        val_metrics = evaluate_model(model, val_ex, tokenizer, collator, cfg,
                                     desc=f"E{epoch+1}:val", compute_loss=True)
        vram = vram_stats()
        etime = time.perf_counter() - e0
        elapsed = time.perf_counter() - experiment_start
        monitored = val_metrics["rougeL"]
        is_new_best = monitored > (best_metric + cfg["early_stopping_min_delta"])
        if is_new_best:
            best_metric, best_epoch, patience = monitored, epoch + 1, 0
            model.save_pretrained(BEST_DIR)
            tokenizer.save_pretrained(BEST_DIR)
            atomic_write_json(os.path.join(BEST_DIR, "best_info.json"),
                              {"best_epoch": best_epoch, "best_metric": best_metric,
                               "validation_metrics": val_metrics})
        else:
            patience += 1
        completed = epoch + 1
        flops = estimate_flops(data["counts"], cfg)
        est_remaining = (elapsed / completed) * (max_epochs - completed) if completed else 0.0
        hist.rounds.append({
            "experiment_id": EXPERIMENT_ID, "seed": 42, "status": "running",
            "method": METHOD_FAMILY, "model_name": MODEL_NAME, "dataset_name": cfg["dataset_name"],
            "round": "", "epoch": epoch + 1, "num_clients": cfg["num_clients"],
            "selected_clients": "", "participation_rate": cfg["participation_rate"],
            "dirichlet_alpha": cfg["dirichlet_alpha"], "local_epochs": cfg["local_epochs"],
            "lora_r": cfg["lora_r"], "learning_rate": cfg["learning_rate"],
            "train_loss": tr["train_loss"], "validation_loss": val_metrics["loss"],
            "validation_bleu": val_metrics["bleu"], "validation_rouge1": val_metrics["rouge1"],
            "validation_rouge2": val_metrics["rouge2"], "validation_rougeL": val_metrics["rougeL"],
            "validation_bertscore_precision": val_metrics["bertscore_precision"],
            "validation_bertscore_recall": val_metrics["bertscore_recall"],
            "validation_bertscore_f1": val_metrics["bertscore_f1"],
            "best_metric_so_far": best_metric, "best_round": best_epoch,
            "patience_counter": patience, "is_new_best": is_new_best,
            "upload_MB_round": 0.0, "download_MB_round": 0.0, "communication_MB_round": 0.0,
            "cumulative_communication_MB": 0.0, "time_per_round_seconds": etime,
            "training_time_seconds": etime, "evaluation_time_seconds": float("nan"),
            "generation_time_seconds": val_metrics["generation_time_seconds"],
            "elapsed_time_seconds": elapsed, "estimated_remaining_time_seconds": est_remaining,
            "max_vram_allocated_MB": vram["max_vram_allocated_MB"],
            "max_vram_reserved_MB": vram["max_vram_reserved_MB"],
            "training_flops_per_step": flops["training_flops_per_step"],
            "estimated_total_training_flops": flops["training_flops_per_step"] * completed})
        hist.resource.append({
            "experiment_id": EXPERIMENT_ID, "seed": 42, "round": "", "epoch": epoch + 1,
            "client_id": "", "stage": "epoch_end", "timestamp": now_iso(),
            "cpu_memory_MB": cpu_memory_MB(), "vram_allocated_MB": vram["vram_allocated_MB"],
            "vram_reserved_MB": vram["vram_reserved_MB"],
            "max_vram_allocated_MB": vram["max_vram_allocated_MB"],
            "max_vram_reserved_MB": vram["max_vram_reserved_MB"], "elapsed_time_seconds": elapsed})
        print(f"[epoch {epoch+1}/{max_epochs}] train_loss={tr['train_loss']:.4f} "
              f"val_rougeL={val_metrics['rougeL']:.4f} best={best_metric:.4f}@{best_epoch} "
              f"patience={patience} t={etime:.1f}s")
        save_checkpoint(f"epoch_{epoch+1:04d}", {
            "model_state": get_trainable_state(model), "next_epoch": epoch + 1,
            "best_metric": best_metric, "best_epoch": best_epoch, "patience": patience,
            "hist_rounds": hist.rounds, "hist_resource": hist.resource,
            "resume_count": resume_count, "rng": rng_state()})
        write_status("running", {"current_epoch": epoch + 1})
        if patience >= cfg["early_stopping_patience"]:
            early_stopped = True
            print(f"[early-stop] stopping at epoch {epoch+1}")
            break
    return {"best_metric": best_metric, "best_round": best_epoch, "completed_rounds": completed,
            "early_stopped": early_stopped, "resume_count": resume_count}


# ----------------------------------------------------------------------------
# Local (per-client independent LoRA) training
# ----------------------------------------------------------------------------
def run_local(model, tokenizer, collator, cfg, data, hist, resume_state):
    train_ex, val_ex, test_ex = data["train"], data["validation"], data["test"]
    partition = data["partition"]
    num_clients = cfg["num_clients"]
    experiment_start = time.perf_counter()
    per_client_test = []
    client_best_dirs = []
    global_best_metric = -float("inf")
    completed = 0

    for c in range(num_clients):
        idx = partition["client_train_indices"][str(c)]
        vidx = partition["client_validation_indices"][str(c)]
        tidx = partition["client_test_indices"][str(c)]
        c_train = [train_ex[i] for i in idx]
        c_val = [val_ex[i] for i in vidx] if vidx else val_ex[:64]
        c_test = [test_ex[i] for i in tidx] if tidx else test_ex[:64]
        # fresh model per client
        load_trainable_state(model, get_trainable_state(build_model()))
        best_metric, best_epoch, patience = -float("inf"), -1, 0
        cbest_dir = os.path.join(OUTPUT_DIR, f"best_model_client_{c}")
        os.makedirs(cbest_dir, exist_ok=True)
        for epoch in range(cfg["local_max_epochs"]):
            e0 = time.perf_counter()
            reset_peak_vram()
            tr = train_epochs(model, c_train, tokenizer, collator, cfg, 1,
                              progress_prefix=f"client{c} E{epoch+1}")
            vmet = evaluate_model(model, c_val, tokenizer, collator, cfg,
                                  desc=f"c{c}E{epoch+1}val", compute_loss=True)
            vram = vram_stats()
            monitored = vmet["rougeL"]
            is_new_best = monitored > (best_metric + cfg["early_stopping_min_delta"])
            if is_new_best:
                best_metric, best_epoch, patience = monitored, epoch + 1, 0
                model.save_pretrained(cbest_dir)
                tokenizer.save_pretrained(cbest_dir)
            else:
                patience += 1
            elapsed = time.perf_counter() - experiment_start
            hist.rounds.append({
                "experiment_id": EXPERIMENT_ID, "seed": 42, "status": "running",
                "method": METHOD_FAMILY, "model_name": MODEL_NAME, "dataset_name": cfg["dataset_name"],
                "round": "", "epoch": epoch + 1, "num_clients": num_clients,
                "selected_clients": str(c), "participation_rate": cfg["participation_rate"],
                "dirichlet_alpha": cfg["dirichlet_alpha"], "local_epochs": cfg["local_epochs"],
                "lora_r": cfg["lora_r"], "learning_rate": cfg["learning_rate"],
                "train_loss": tr["train_loss"], "validation_loss": vmet["loss"],
                "validation_bleu": vmet["bleu"], "validation_rouge1": vmet["rouge1"],
                "validation_rouge2": vmet["rouge2"], "validation_rougeL": vmet["rougeL"],
                "validation_bertscore_precision": vmet["bertscore_precision"],
                "validation_bertscore_recall": vmet["bertscore_recall"],
                "validation_bertscore_f1": vmet["bertscore_f1"],
                "best_metric_so_far": best_metric, "best_round": best_epoch,
                "patience_counter": patience, "is_new_best": is_new_best,
                "upload_MB_round": 0.0, "download_MB_round": 0.0, "communication_MB_round": 0.0,
                "cumulative_communication_MB": 0.0, "time_per_round_seconds": time.perf_counter() - e0,
                "training_time_seconds": time.perf_counter() - e0, "evaluation_time_seconds": float("nan"),
                "generation_time_seconds": vmet["generation_time_seconds"],
                "elapsed_time_seconds": elapsed, "estimated_remaining_time_seconds": float("nan"),
                "max_vram_allocated_MB": vram["max_vram_allocated_MB"],
                "max_vram_reserved_MB": vram["max_vram_reserved_MB"],
                "training_flops_per_step": estimate_flops(data["counts"], cfg)["training_flops_per_step"],
                "estimated_total_training_flops": float("nan")})
            save_checkpoint(f"client_{c}_epoch_{epoch+1:04d}", {
                "model_state": get_trainable_state(model), "client": c, "next_epoch": epoch + 1,
                "best_metric": best_metric, "best_epoch": best_epoch, "patience": patience,
                "hist_rounds": hist.rounds, "rng": rng_state()})
            if patience >= cfg["early_stopping_patience"]:
                break
        # evaluate this client on its own test with its best model
        cmodel = build_model()
        cmodel = load_peft_best(cmodel, cbest_dir)
        tmet, tpred = evaluate_model(cmodel, c_test, tokenizer, collator, cfg,
                                     desc=f"c{c}:test", compute_loss=True, return_predictions=True)
        lat, _ = measure_latency(cmodel, c_test, tokenizer, cfg, "weighted_local", DEVICE.type)
        per_client_test.append({"client": c, "n_test": len(c_test), "metrics": tmet,
                                "latency": lat, "predictions": tpred})
        hist.clients.append({
            "experiment_id": EXPERIMENT_ID, "seed": 42, "round": "", "client_id": c,
            "was_selected": True, "num_train_samples": len(c_train),
            "num_validation_samples": len(c_val), "num_test_samples": len(c_test),
            "local_train_loss": float("nan"), "local_validation_loss": float("nan"),
            "local_bleu": tmet["bleu"], "local_rougeL": tmet["rougeL"],
            "local_bertscore_f1": tmet["bertscore_f1"], "update_norm": float("nan"),
            "update_divergence": float("nan"), "control_variate_norm": float("nan"),
            "raw_aggregation_weight": float("nan"), "normalized_aggregation_weight": float("nan"),
            "aggregation_fallback": False, "upload_MB": 0.0, "download_MB": 0.0,
            "client_training_time_seconds": float("nan"), "client_evaluation_time_seconds": float("nan"),
            "max_vram_allocated_MB": vram_stats()["max_vram_allocated_MB"],
            "client_inference_latency_mean_ms": lat["inference_latency_mean_ms_per_sample"],
            "client_inference_latency_p95_ms": lat["inference_latency_p95_ms_per_sample"],
            "client_inference_throughput": lat["inference_throughput_samples_per_second"]})
        client_best_dirs.append(cbest_dir)
        if best_metric > global_best_metric:
            global_best_metric = best_metric
        del cmodel
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        completed += 1

    return {"best_metric": global_best_metric, "best_round": -1, "completed_rounds": completed,
            "early_stopped": False, "per_client_test": per_client_test,
            "client_best_dirs": client_best_dirs, "resume_count": 0}


def load_peft_best(model, best_dir):
    """Load best weights into a fresh model (LoRA adapter or full state)."""
    if USE_LORA:
        from peft import PeftModel
        base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        model = PeftModel.from_pretrained(base, best_dir).to(DEVICE)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(best_dir).to(DEVICE)
    return model


# ============================================================================
# Final results assembly, validation, and main().
# ============================================================================

FINAL_COLUMNS = [
    "experiment_id", "seed", "status", "method", "method_family", "aggregation_strategy",
    "model_name", "model_revision", "dataset_name", "dataset_config", "dataset_revision",
    "dataset_fingerprint", "partition_checksum", "protocol_version", "base_config_hash",
    "preprocessing_hash", "generation_config_hash", "code_hash", "task_name",
    "training_setting", "tuning_mode", "is_proposed_method", "secure_aggregation_implemented",
    "device", "gpu_name", "gpu_total_memory_MB", "precision", "python_version",
    "torch_version", "transformers_version", "datasets_version", "peft_version",
    "accelerate_version", "cuda_version", "num_train_samples", "num_validation_samples",
    "num_test_samples", "num_filtered_samples", "num_clients", "participation_rate",
    "average_selected_clients_per_round", "dirichlet_alpha", "num_pseudo_topics",
    "minimum_samples_per_client", "client_sample_count_mean", "client_sample_count_std",
    "client_sample_count_min", "client_sample_count_max", "mean_pairwise_client_js_divergence",
    "optimizer_name", "scheduler_name", "learning_rate", "weight_decay", "warmup_ratio",
    "max_grad_norm", "physical_batch_size", "gradient_accumulation_steps", "effective_batch_size",
    "max_source_length", "max_target_length", "generation_num_beams",
    "generation_no_repeat_ngram_size", "bertscore_model_type", "max_rounds", "completed_rounds",
    "centralized_max_epochs", "local_epochs", "best_round", "best_epoch", "best_metric_name",
    "best_metric_value", "early_stopped", "early_stopping_patience", "early_stopping_min_delta",
    "rounds_without_improvement", "lora_r", "lora_alpha", "lora_dropout", "lora_target_modules",
    "fedprox_mu", "lambda_n", "beta_loss", "beta_divergence", "beta_quality", "train_loss_at_best",
    "validation_loss_at_best", "validation_bleu_at_best", "validation_rouge1_at_best",
    "validation_rouge2_at_best", "validation_rougeL_at_best", "validation_bertscore_precision_at_best",
    "validation_bertscore_recall_at_best", "validation_bertscore_f1_at_best", "test_loss",
    "test_bleu", "test_rouge1", "test_rouge2", "test_rougeL", "test_bertscore_precision",
    "test_bertscore_recall", "test_bertscore_f1", "client_test_bleu_mean", "client_test_bleu_std",
    "client_test_rougeL_mean", "client_test_rougeL_std", "client_test_rougeL_min",
    "client_test_rougeL_max", "client_test_bertscore_f1_mean", "client_test_bertscore_f1_std",
    "client_update_divergence_mean", "client_update_divergence_std", "total_params",
    "trainable_params", "frozen_params", "adapter_params", "trainable_ratio", "model_size_MB",
    "adapter_size_MB", "checkpoint_size_MB", "transmitted_dtype", "bytes_per_parameter",
    "communication_parameters_per_round", "upload_MB_per_round_average",
    "download_MB_per_round_average", "metadata_MB_per_round_average",
    "control_variate_MB_per_round_average", "communication_MB_per_round_average",
    "total_upload_MB", "total_download_MB", "total_metadata_MB", "total_control_variate_MB",
    "total_communication_MB", "total_communication_GB", "number_of_communication_rounds",
    "total_transmitted_parameters", "reference_full_model_parameters", "reference_full_MB_per_round",
    "communication_reduction_per_round_vs_full_percent",
    "communication_reduction_total_vs_full_same_rounds_percent", "training_time_seconds",
    "evaluation_time_seconds", "generation_time_seconds", "total_experiment_time_seconds",
    "average_time_per_round_seconds", "std_time_per_round_seconds",
    "average_client_training_time_seconds", "best_round_elapsed_time_seconds",
    "cpu_peak_memory_MB", "max_vram_allocated_MB", "max_vram_reserved_MB",
    "forward_flops_per_sample", "training_flops_per_step", "estimated_total_training_flops",
    "generation_flops_per_sample", "flops_measurement_method", "flops_is_estimated",
    "flops_measurement_status", "inference_latency_mean_ms_per_sample",
    "inference_latency_median_ms_per_sample", "inference_latency_p95_ms_per_sample",
    "inference_latency_min_ms_per_sample", "inference_latency_max_ms_per_sample",
    "inference_latency_std_ms_per_sample", "inference_throughput_samples_per_second",
    "inference_throughput_tokens_per_second", "latency_total_time_seconds", "latency_num_samples",
    "latency_batch_size", "latency_warmup_batches", "latency_device", "latency_includes_tokenization",
    "average_input_tokens", "average_generated_tokens", "unmerged_inference_latency_mean_ms",
    "unmerged_inference_latency_median_ms", "unmerged_inference_latency_p95_ms",
    "unmerged_inference_throughput", "merged_inference_latency_mean_ms",
    "merged_inference_latency_median_ms", "merged_inference_latency_p95_ms",
    "merged_inference_throughput", "lora_merge_success", "lora_merge_error", "primary_latency_source",
    "aggregation_fallback_count", "aggregation_fallback_rate", "average_max_client_weight",
    "average_min_client_weight", "average_aggregation_weight_entropy",
    "final_global_control_variate_norm", "resume_used", "resume_count", "resume_checkpoint_path",
    "random_seed_verified", "experiment_start_time", "experiment_end_time", "output_directory",
    "best_model_path", "final_model_path", "history_csv_path", "client_history_csv_path",
    "communication_history_csv_path", "resource_history_csv_path", "latency_details_csv_path",
    "flops_results_csv_path", "predictions_path",
]

NAN = float("nan")


def dir_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 ** 2)


def run_latency_full(best_dir, test_ex, tokenizer, cfg):
    """Return (summary, rows, merge_info) measuring best model latency.
    LoRA: measure unmerged then merged (primary=merged)."""
    merge_info = {"unmerged": {}, "merged": {}, "lora_merge_success": False,
                  "lora_merge_error": "", "primary_latency_source": ""}
    if not USE_LORA:
        model = AutoModelForSeq2SeqLM.from_pretrained(best_dir).to(DEVICE)
        summ, rows = measure_latency(model, test_ex, tokenizer, cfg, "full_model", DEVICE.type)
        merge_info["primary_latency_source"] = "full_model"
        del model
        return summ, rows, merge_info

    from peft import PeftModel
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base, best_dir).to(DEVICE)
    un_summ, un_rows = measure_latency(model, test_ex, tokenizer, cfg, "unmerged_lora", DEVICE.type)
    merge_info["unmerged"] = un_summ
    try:
        merged = model.merge_and_unload()
        merged = merged.to(DEVICE)
        m_summ, m_rows = measure_latency(merged, test_ex, tokenizer, cfg, "merged_lora", DEVICE.type)
        merge_info["merged"] = m_summ
        merge_info["lora_merge_success"] = True
        merge_info["primary_latency_source"] = "merged_lora"
        del merged
        return m_summ, un_rows + m_rows, merge_info
    except Exception as e:
        merge_info["lora_merge_error"] = str(e)
        merge_info["primary_latency_source"] = "unmerged_lora_fallback"
        return un_summ, un_rows, merge_info


def assemble_final_results(cfg, env, counts, data, run_out, test_metrics,
                           latency, merge_info, comm_summary, flops, best_val_row,
                           timing, client_test_stats, resume_used, paths):
    fam = METHOD_FAMILY
    is_fed = IS_FEDERATED
    d = {c: NAN for c in FINAL_COLUMNS}
    d.update({
        "experiment_id": EXPERIMENT_ID, "seed": 42, "status": "completed", "method": fam,
        "method_family": fam, "aggregation_strategy": AGGREGATION_STRATEGY,
        "model_name": MODEL_NAME, "model_revision": "main", "dataset_name": cfg["dataset_name"],
        "dataset_config": cfg["dataset_config"] if cfg["dataset_config"] else "",
        "dataset_revision": "main", "dataset_fingerprint": data["partition"]["dataset_fingerprint"],
        "partition_checksum": data["partition"]["partition_checksum"],
        "protocol_version": cfg["protocol_version"], "base_config_hash": cfg["_base_config_hash"],
        "preprocessing_hash": cfg["_preprocessing_hash"], "generation_config_hash": cfg["_generation_config_hash"],
        "code_hash": cfg["_code_hash"], "task_name": cfg["task_name"],
        "training_setting": TRAINING_SETTING, "tuning_mode": TUNING_MODE,
        "is_proposed_method": IS_PROPOSED_METHOD, "secure_aggregation_implemented": False,
        "device": env["device"], "gpu_name": env["gpu_name"],
        "gpu_total_memory_MB": env["gpu_total_memory_MB"], "precision": env["precision"],
        "python_version": env["python_version"], "torch_version": env["torch_version"],
        "transformers_version": env["transformers_version"], "datasets_version": env["datasets_version"],
        "peft_version": env["peft_version"], "accelerate_version": env["accelerate_version"],
        "cuda_version": env["cuda_version"], "num_train_samples": counts["num_train_samples"],
        "num_validation_samples": counts["num_validation_samples"],
        "num_test_samples": counts["num_test_samples"], "num_filtered_samples": counts["num_filtered_samples"],
        "num_clients": cfg["num_clients"], "participation_rate": cfg["participation_rate"],
        "average_selected_clients_per_round": run_out.get("avg_selected", NAN),
        "dirichlet_alpha": cfg["dirichlet_alpha"], "num_pseudo_topics": cfg["num_pseudo_topics"],
        "minimum_samples_per_client": cfg["minimum_samples_per_client"],
        "client_sample_count_mean": data["client_stats"]["mean"],
        "client_sample_count_std": data["client_stats"]["std"],
        "client_sample_count_min": data["client_stats"]["min"],
        "client_sample_count_max": data["client_stats"]["max"],
        "mean_pairwise_client_js_divergence": data["client_stats"]["mean_js"],
        "optimizer_name": "AdamW", "scheduler_name": "linear_warmup",
        "learning_rate": cfg["learning_rate"], "weight_decay": cfg["weight_decay"],
        "warmup_ratio": cfg["warmup_ratio"], "max_grad_norm": cfg["max_grad_norm"],
        "physical_batch_size": cfg["train_batch_size"],
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "effective_batch_size": cfg["effective_batch_size"],
        "max_source_length": cfg["max_source_length"], "max_target_length": cfg["max_target_length"],
        "generation_num_beams": cfg["generation_num_beams"],
        "generation_no_repeat_ngram_size": cfg["no_repeat_ngram_size"],
        "bertscore_model_type": cfg["bertscore_model_type"], "max_rounds": cfg["max_rounds"],
        "completed_rounds": run_out["completed_rounds"],
        "centralized_max_epochs": cfg["centralized_max_epochs"], "local_epochs": cfg["local_epochs"],
        "best_round": run_out.get("best_round", -1),
        "best_epoch": run_out.get("best_round", -1) if IS_CENTRALIZED else -1,
        "best_metric_name": "validation_rougeL", "best_metric_value": run_out["best_metric"],
        "early_stopped": run_out["early_stopped"],
        "early_stopping_patience": cfg["early_stopping_patience"],
        "early_stopping_min_delta": cfg["early_stopping_min_delta"],
        "rounds_without_improvement": (run_out["completed_rounds"] - run_out.get("best_round", 0))
            if run_out.get("best_round", -1) > 0 else run_out["completed_rounds"],
        "lora_r": cfg["lora_r"] if USE_LORA else NAN,
        "lora_alpha": cfg["lora_alpha"] if USE_LORA else NAN,
        "lora_dropout": cfg["lora_dropout"] if USE_LORA else NAN,
        "lora_target_modules": "|".join(cfg["lora_target_modules"]) if USE_LORA else "",
        "fedprox_mu": cfg["fedprox_mu"] if fam == "fedprox_lora" else NAN,
        "lambda_n": cfg["lambda_n"] if fam == "fedtgen" else NAN,
        "beta_loss": cfg["beta_loss"] if fam == "fedtgen" else NAN,
        "beta_divergence": cfg["beta_divergence"] if fam == "fedtgen" else NAN,
        "beta_quality": cfg["beta_quality"] if fam == "fedtgen" else NAN,
        "train_loss_at_best": best_val_row.get("train_loss", NAN),
        "validation_loss_at_best": best_val_row.get("validation_loss", NAN),
        "validation_bleu_at_best": best_val_row.get("validation_bleu", NAN),
        "validation_rouge1_at_best": best_val_row.get("validation_rouge1", NAN),
        "validation_rouge2_at_best": best_val_row.get("validation_rouge2", NAN),
        "validation_rougeL_at_best": best_val_row.get("validation_rougeL", NAN),
        "validation_bertscore_precision_at_best": best_val_row.get("validation_bertscore_precision", NAN),
        "validation_bertscore_recall_at_best": best_val_row.get("validation_bertscore_recall", NAN),
        "validation_bertscore_f1_at_best": best_val_row.get("validation_bertscore_f1", NAN),
        "test_loss": test_metrics["loss"], "test_bleu": test_metrics["bleu"],
        "test_rouge1": test_metrics["rouge1"], "test_rouge2": test_metrics["rouge2"],
        "test_rougeL": test_metrics["rougeL"],
        "test_bertscore_precision": test_metrics["bertscore_precision"],
        "test_bertscore_recall": test_metrics["bertscore_recall"],
        "test_bertscore_f1": test_metrics["bertscore_f1"],
        "client_test_bleu_mean": client_test_stats.get("bleu_mean", NAN),
        "client_test_bleu_std": client_test_stats.get("bleu_std", NAN),
        "client_test_rougeL_mean": client_test_stats.get("rougeL_mean", NAN),
        "client_test_rougeL_std": client_test_stats.get("rougeL_std", NAN),
        "client_test_rougeL_min": client_test_stats.get("rougeL_min", NAN),
        "client_test_rougeL_max": client_test_stats.get("rougeL_max", NAN),
        "client_test_bertscore_f1_mean": client_test_stats.get("bert_f1_mean", NAN),
        "client_test_bertscore_f1_std": client_test_stats.get("bert_f1_std", NAN),
        "client_update_divergence_mean": client_test_stats.get("div_mean", NAN),
        "client_update_divergence_std": client_test_stats.get("div_std", NAN),
        "total_params": counts["total_params"], "trainable_params": counts["trainable_params"],
        "frozen_params": counts["frozen_params"], "adapter_params": counts["adapter_params"],
        "trainable_ratio": counts["trainable_ratio"], "model_size_MB": counts["model_size_MB"],
        "adapter_size_MB": counts["adapter_size_MB"] if USE_LORA else NAN,
        "checkpoint_size_MB": dir_size_mb(CKPT_DIR),
        "transmitted_dtype": TRANSMITTED_DTYPE, "bytes_per_parameter": BYTES_PER_PARAM,
        "communication_parameters_per_round": comm_summary["params_per_round"],
        "upload_MB_per_round_average": comm_summary["upload_per_round"],
        "download_MB_per_round_average": comm_summary["download_per_round"],
        "metadata_MB_per_round_average": comm_summary["metadata_per_round"],
        "control_variate_MB_per_round_average": comm_summary["control_per_round"],
        "communication_MB_per_round_average": comm_summary["comm_per_round"],
        "total_upload_MB": comm_summary["total_upload"], "total_download_MB": comm_summary["total_download"],
        "total_metadata_MB": comm_summary["total_metadata"],
        "total_control_variate_MB": comm_summary["total_control"],
        "total_communication_MB": comm_summary["total_comm"],
        "total_communication_GB": comm_summary["total_comm"] / 1024.0,
        "number_of_communication_rounds": comm_summary["rounds"],
        "total_transmitted_parameters": comm_summary["total_params"],
        "reference_full_model_parameters": comm_summary["reference_full_params"],
        "reference_full_MB_per_round": comm_summary["reference_full_MB_per_round"],
        "communication_reduction_per_round_vs_full_percent": comm_summary["reduction_per_round"],
        "communication_reduction_total_vs_full_same_rounds_percent": comm_summary["reduction_total"],
        "training_time_seconds": timing["training_time"],
        "evaluation_time_seconds": timing["evaluation_time"],
        "generation_time_seconds": timing["generation_time"],
        "total_experiment_time_seconds": timing["total_time"],
        "average_time_per_round_seconds": timing["avg_round_time"],
        "std_time_per_round_seconds": timing["std_round_time"],
        "average_client_training_time_seconds": timing["avg_client_train_time"],
        "best_round_elapsed_time_seconds": timing["best_round_elapsed"],
        "cpu_peak_memory_MB": timing["cpu_peak_mb"],
        "max_vram_allocated_MB": timing["max_vram_alloc"], "max_vram_reserved_MB": timing["max_vram_res"],
        "forward_flops_per_sample": flops["forward_flops_per_sample"],
        "training_flops_per_step": flops["training_flops_per_step"],
        "estimated_total_training_flops": flops["training_flops_per_step"] * max(1, run_out["completed_rounds"]),
        "generation_flops_per_sample": flops["generation_flops_per_sample"],
        "flops_measurement_method": flops["flops_measurement_method"],
        "flops_is_estimated": flops["flops_is_estimated"],
        "flops_measurement_status": flops["flops_measurement_status"],
        "inference_latency_mean_ms_per_sample": latency["inference_latency_mean_ms_per_sample"],
        "inference_latency_median_ms_per_sample": latency["inference_latency_median_ms_per_sample"],
        "inference_latency_p95_ms_per_sample": latency["inference_latency_p95_ms_per_sample"],
        "inference_latency_min_ms_per_sample": latency["inference_latency_min_ms_per_sample"],
        "inference_latency_max_ms_per_sample": latency["inference_latency_max_ms_per_sample"],
        "inference_latency_std_ms_per_sample": latency["inference_latency_std_ms_per_sample"],
        "inference_throughput_samples_per_second": latency["inference_throughput_samples_per_second"],
        "inference_throughput_tokens_per_second": latency["inference_throughput_tokens_per_second"],
        "latency_total_time_seconds": latency["latency_total_time_seconds"],
        "latency_num_samples": latency["latency_num_samples"],
        "latency_batch_size": latency["latency_batch_size"],
        "latency_warmup_batches": latency["latency_warmup_batches"],
        "latency_device": latency["latency_device"],
        "latency_includes_tokenization": latency["latency_includes_tokenization"],
        "average_input_tokens": latency["average_input_tokens"],
        "average_generated_tokens": latency["average_generated_tokens"],
        "unmerged_inference_latency_mean_ms": merge_info["unmerged"].get("inference_latency_mean_ms_per_sample", NAN),
        "unmerged_inference_latency_median_ms": merge_info["unmerged"].get("inference_latency_median_ms_per_sample", NAN),
        "unmerged_inference_latency_p95_ms": merge_info["unmerged"].get("inference_latency_p95_ms_per_sample", NAN),
        "unmerged_inference_throughput": merge_info["unmerged"].get("inference_throughput_samples_per_second", NAN),
        "merged_inference_latency_mean_ms": merge_info["merged"].get("inference_latency_mean_ms_per_sample", NAN),
        "merged_inference_latency_median_ms": merge_info["merged"].get("inference_latency_median_ms_per_sample", NAN),
        "merged_inference_latency_p95_ms": merge_info["merged"].get("inference_latency_p95_ms_per_sample", NAN),
        "merged_inference_throughput": merge_info["merged"].get("inference_throughput_samples_per_second", NAN),
        "lora_merge_success": merge_info["lora_merge_success"] if USE_LORA else NAN,
        "lora_merge_error": merge_info["lora_merge_error"],
        "primary_latency_source": merge_info["primary_latency_source"],
        "aggregation_fallback_count": run_out.get("fallback_count", 0) if is_fed else NAN,
        "aggregation_fallback_rate": (run_out.get("fallback_count", 0) / max(1, run_out["completed_rounds"]))
            if is_fed else NAN,
        "average_max_client_weight": run_out.get("max_weight", NAN) if is_fed else NAN,
        "average_min_client_weight": run_out.get("min_weight", NAN) if is_fed else NAN,
        "average_aggregation_weight_entropy": run_out.get("weight_entropy", NAN) if is_fed else NAN,
        "final_global_control_variate_norm": run_out.get("final_global_control_variate_norm", NAN)
            if fam == "scaffold_lora" else NAN,
        "resume_used": resume_used, "resume_count": run_out.get("resume_count", 0),
        "resume_checkpoint_path": paths["resume_ckpt"], "random_seed_verified": True,
        "experiment_start_time": timing["start_iso"], "experiment_end_time": timing["end_iso"],
        "output_directory": OUTPUT_DIR, "best_model_path": BEST_DIR, "final_model_path": FINAL_DIR,
        "history_csv_path": paths["history"], "client_history_csv_path": paths["client_history"],
        "communication_history_csv_path": paths["comm_history"],
        "resource_history_csv_path": paths["resource_history"],
        "latency_details_csv_path": paths["latency"], "flops_results_csv_path": paths["flops"],
        "predictions_path": paths["predictions"],
    })
    return d


def validate_final_results(csv_path, history_df, comm_df, expected_completed):
    errors = []
    if not os.path.exists(csv_path):
        raise RuntimeError("final_results.csv does not exist")
    df = pd.read_csv(csv_path)
    if len(df) != 1:
        errors.append(f"final_results.csv must have exactly one row, has {len(df)}")
    cols = list(df.columns)
    missing = [c for c in FINAL_COLUMNS if c not in cols]
    if missing:
        errors.append(f"missing columns: {missing[:10]}")
    if len(cols) != len(set(cols)):
        errors.append("duplicate columns present")
    row = df.iloc[0]
    if str(row["experiment_id"]) != EXPERIMENT_ID:
        errors.append("experiment_id mismatch")
    if int(row["seed"]) != 42:
        errors.append("seed != 42")
    if str(row["status"]) not in {"completed"}:
        errors.append("status invalid")
    tp, trp, frp = row["total_params"], row["trainable_params"], row["frozen_params"]
    if abs((trp + frp) - tp) > 1:
        errors.append("total_params != trainable + frozen")
    if abs(row["trainable_ratio"] - (trp / tp)) > 1e-4:
        errors.append("trainable_ratio incorrect")
    if not (0 <= row["test_bleu"] <= 100):
        errors.append("test BLEU out of 0-100")
    for m in ["test_rouge1", "test_rouge2", "test_rougeL", "test_bertscore_f1"]:
        if not (0 <= row[m] <= 1):
            errors.append(f"{m} out of 0-1")
    if int(row["completed_rounds"]) != int(expected_completed):
        errors.append("completed_rounds mismatch with history")
    if IS_FEDERATED and len(comm_df) > 0:
        tot = comm_df["total_communication_MB"].sum()
        if abs(tot - row["total_communication_MB"]) > max(0.5, 0.02 * tot):
            errors.append("total_communication_MB mismatch with communication_history")
    if abs(row["total_communication_GB"] - row["total_communication_MB"] / 1024.0) > 1e-6:
        errors.append("GB conversion wrong")
    if TRAINING_SETTING in ("local", "centralized"):
        if float(row["total_communication_MB"]) != 0.0:
            errors.append("Local/Centralized communication must be 0")
    if int(row["latency_num_samples"]) <= 0:
        errors.append("latency_num_samples must be > 0")
    for m in ["inference_latency_mean_ms_per_sample", "inference_latency_p95_ms_per_sample",
              "inference_throughput_samples_per_second"]:
        if not (row[m] > 0):
            errors.append(f"{m} must be > 0 when completed")
    if str(row["primary_latency_source"]) not in {"full_model", "merged_lora",
                                                   "unmerged_lora_fallback", "weighted_local_models"}:
        errors.append("primary_latency_source invalid")
    if not os.path.isdir(BEST_DIR) or not os.listdir(BEST_DIR):
        errors.append("best_model_path missing")
    if not os.path.isdir(FINAL_DIR) or not os.listdir(FINAL_DIR):
        errors.append("final_model_path missing")
    ckpts = [d for d in os.listdir(CKPT_DIR) if os.path.isdir(os.path.join(CKPT_DIR, d))]
    if len(ckpts) > 2:
        errors.append(f"more than two checkpoints retained ({len(ckpts)})")
    if errors:
        raise RuntimeError("validate_final_results failed:\n- " + "\n- ".join(errors))
    print("[validate] final_results.csv passed all checks.")
    return True


def aggregate_client_test_stats(per_client):
    if not per_client:
        return {}
    bleu = [x["metrics"]["bleu"] for x in per_client]
    rl = [x["metrics"]["rougeL"] for x in per_client]
    bf = [x["metrics"]["bertscore_f1"] for x in per_client]
    return {
        "bleu_mean": float(np.mean(bleu)), "bleu_std": float(np.std(bleu)),
        "rougeL_mean": float(np.mean(rl)), "rougeL_std": float(np.std(rl)),
        "rougeL_min": float(np.min(rl)), "rougeL_max": float(np.max(rl)),
        "bert_f1_mean": float(np.mean(bf)), "bert_f1_std": float(np.std(bf)),
    }


def weighted_metrics(per_client):
    tot = sum(x["n_test"] for x in per_client) or 1
    out = {}
    for key in ["loss", "bleu", "rouge1", "rouge2", "rougeL",
                "bertscore_precision", "bertscore_recall", "bertscore_f1"]:
        out[key] = sum(x["metrics"][key] * x["n_test"] for x in per_client) / tot
    out["generation_time_seconds"] = sum(x["metrics"]["generation_time_seconds"] for x in per_client)
    return out


def weighted_latency(per_client):
    tot = sum(x["n_test"] for x in per_client) or 1
    keys = ["inference_latency_mean_ms_per_sample", "inference_latency_median_ms_per_sample",
            "inference_latency_p95_ms_per_sample", "inference_latency_min_ms_per_sample",
            "inference_latency_max_ms_per_sample", "inference_latency_std_ms_per_sample",
            "inference_throughput_samples_per_second", "inference_throughput_tokens_per_second",
            "average_input_tokens", "average_generated_tokens"]
    out = {}
    for k in keys:
        out[k] = sum(x["latency"][k] * x["n_test"] for x in per_client) / tot
    out["latency_total_time_seconds"] = sum(x["latency"]["latency_total_time_seconds"] for x in per_client)
    out["latency_num_samples"] = sum(x["latency"]["latency_num_samples"] for x in per_client)
    out["latency_batch_size"] = CONFIG["latency_batch_size"]
    out["latency_warmup_batches"] = CONFIG["latency_warmup_batches"]
    out["latency_device"] = DEVICE.type
    out["latency_includes_tokenization"] = False
    return out


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    t_start = time.perf_counter()
    start_iso = now_iso()
    write_status("running", {"stage": "startup"})
    print("=" * 78)
    print(f"Experiment : {EXPERIMENT_ID}")
    print(f"RQ         : {RQ_SUPPORTED}")
    print(f"Method     : {METHOD_FAMILY} | setting={TRAINING_SETTING} | tuning={TUNING_MODE}")
    print(f"Model      : {MODEL_NAME}")
    print(f"Dataset    : {CONFIG['dataset_name']} ({CONFIG['task_name']})")
    print(f"Seed       : 42 | Device: {DEVICE} | GPU: {GPU_NAME}")
    print(f"Output dir : {OUTPUT_DIR}")
    print(f"Start time : {start_iso}")
    print("=" * 78)

    set_seed(CONFIG["seed"])
    # verify seed reproducibility marker
    _seed_probe = torch.randint(0, 10 ** 6, (1,)).item()
    set_seed(CONFIG["seed"])
    assert torch.randint(0, 10 ** 6, (1,)).item() == _seed_probe, "seed not reproducible"

    env = environment_info()
    print("[env]", json.dumps(env, indent=2))

    # hashes for provenance
    CONFIG["_base_config_hash"] = stable_hash({k: CONFIG[k] for k in sorted(BASE_CONFIG)})
    CONFIG["_preprocessing_hash"] = stable_hash({
        "dataset": CONFIG["dataset_name"], "src": CONFIG["max_source_length"],
        "tgt": CONFIG["max_target_length"], "caps": [CONFIG["max_train_samples"],
        CONFIG["max_validation_samples"], CONFIG["max_test_samples"]]})
    CONFIG["_generation_config_hash"] = stable_hash({
        "beams": CONFIG["generation_num_beams"], "nrng": CONFIG["no_repeat_ngram_size"],
        "early": CONFIG["generation_early_stopping"], "max_new": CONFIG["max_target_length"]})
    try:
        CONFIG["_code_hash"] = file_sha256(os.path.abspath(__file__))[:16]
    except Exception:
        CONFIG["_code_hash"] = "notebook"

    atomic_write_json(opath("config.json"), json_safe(CONFIG))
    atomic_write_json(opath("environment.json"), json_safe(env))
    atomic_write_json(opath("comparison_protocol.json"), {
        "protocol_version": CONFIG["protocol_version"], "seed": 42,
        "base_config_hash": CONFIG["_base_config_hash"],
        "generation_config_hash": CONFIG["_generation_config_hash"],
        "preprocessing_hash": CONFIG["_preprocessing_hash"],
        "bertscore_model_type": CONFIG["bertscore_model_type"],
        "note": "Shared protocol ensures fair comparison across all 41 experiments."})

    # ---- data ----
    built, raw_counts, filtered_counts = load_raw_examples()
    train_ex = subsample(built["train"], CONFIG["max_train_samples"], CONFIG["seed"])
    val_ex = subsample(built["validation"], CONFIG["max_validation_samples"], CONFIG["seed"])
    test_ex = subsample(built["test"], CONFIG["max_test_samples"], CONFIG["seed"])
    num_filtered = (sum(raw_counts.get(k, 0) for k in raw_counts)
                    - (filtered_counts["train"] + filtered_counts["validation"] + filtered_counts["test"]))
    counts_meta = {
        "num_train_samples": len(train_ex), "num_validation_samples": len(val_ex),
        "num_test_samples": len(test_ex), "num_filtered_samples": int(max(0, num_filtered))}
    atomic_write_json(opath("data_statistics.json"), {
        "raw_counts": raw_counts, "filtered_counts": filtered_counts,
        "after_cap": {"train": len(train_ex), "validation": len(val_ex), "test": len(test_ex)},
        "caps": {"train": CONFIG["max_train_samples"], "validation": CONFIG["max_validation_samples"],
                 "test": CONFIG["max_test_samples"]}})
    print(f"[data] train={len(train_ex)} val={len(val_ex)} test={len(test_ex)} "
          f"filtered={counts_meta['num_filtered_samples']}")

    partition, topic_labels = build_partition(train_ex, val_ex, test_ex, CONFIG)
    atomic_write_json(opath("data_partition.json"), json_safe(partition))
    sample_counts = [len(partition["client_train_indices"][str(c)]) for c in range(CONFIG["num_clients"])]
    client_stats = {
        "mean": float(np.mean(sample_counts)), "std": float(np.std(sample_counts)),
        "min": int(np.min(sample_counts)), "max": int(np.max(sample_counts)),
        "mean_js": mean_pairwise_js(partition, topic_labels,
                                    partition["pseudo_topic_config"]["num_pseudo_topics_effective"],
                                    CONFIG["num_clients"])}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt")

    model = build_model()
    counts = param_counts(model)
    counts.update(counts_meta)
    print(f"[model] total={counts['total_params']:,} trainable={counts['trainable_params']:,} "
          f"ratio={counts['trainable_ratio']:.6f}")

    data = {"train": train_ex, "validation": val_ex, "test": test_ex,
            "partition": partition, "counts": counts, "client_stats": client_stats}

    # ---- resume ----
    resume_state = load_latest_checkpoint()
    resume_used = resume_state is not None
    resume_ckpt_path = ""
    if resume_used:
        try:
            resume_ckpt_path = sorted(
                [os.path.join(CKPT_DIR, d) for d in os.listdir(CKPT_DIR)
                 if os.path.isdir(os.path.join(CKPT_DIR, d))],
                key=os.path.getmtime)[-1]
        except Exception:
            resume_ckpt_path = ""

    hist = HistoryStore()

    # ---- run training ----
    if IS_FEDERATED:
        run_out = run_federated(model, tokenizer, collator, CONFIG, data, hist, resume_state)
    elif IS_CENTRALIZED:
        run_out = run_centralized(model, tokenizer, collator, CONFIG, data, hist, resume_state)
    else:
        run_out = run_local(model, tokenizer, collator, CONFIG, data, hist, resume_state)

    # ---- test evaluation with best model ----
    latency_rows = []
    per_client_test = run_out.get("per_client_test", None)
    client_test_stats = {}
    if IS_LOCAL and per_client_test:
        test_metrics = weighted_metrics(per_client_test)
        latency = weighted_latency(per_client_test)
        merge_info = {"unmerged": {}, "merged": {}, "lora_merge_success": USE_LORA,
                      "lora_merge_error": "", "primary_latency_source": "weighted_local_models"}
        client_test_stats = aggregate_client_test_stats(per_client_test)
        # save best/final models = client 0 best as representative + all client dirs exist
        rep = run_out["client_best_dirs"][0]
        for src, dst in [(rep, BEST_DIR), (rep, FINAL_DIR)]:
            for f in os.listdir(src):
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
        # predictions concat
        all_pred = []
        for x in per_client_test:
            for i, (inp, ref, pr) in enumerate(zip(x["predictions"]["inputs"],
                                                   x["predictions"]["refs"], x["predictions"]["preds"])):
                all_pred.append({"experiment_id": EXPERIMENT_ID, "sample_id": len(all_pred),
                                 "client_id": x["client"], "input_text": inp,
                                 "reference_text": ref, "prediction_text": pr,
                                 "input_tokens": len(tokenizer(inp)["input_ids"]),
                                 "reference_tokens": len(tokenizer(ref)["input_ids"]),
                                 "generated_tokens": len(tokenizer(pr)["input_ids"]),
                                 "generation_time_ms": NAN})
        pred_records = all_pred
    else:
        # load global best model
        test_model = load_peft_best(build_model() if False else AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE), BEST_DIR) \
            if USE_LORA else AutoModelForSeq2SeqLM.from_pretrained(BEST_DIR).to(DEVICE)
        test_metrics, tpred = evaluate_model(test_model, test_ex, tokenizer, collator, CONFIG,
                                             desc="final:test", compute_loss=True, return_predictions=True)
        # save final model (best weights promoted to final)
        for src, dst in [(BEST_DIR, FINAL_DIR)]:
            for f in os.listdir(src):
                if os.path.isfile(os.path.join(src, f)):
                    shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
        # per-client test stats for federated
        if IS_FEDERATED:
            pcs = []
            for c in range(CONFIG["num_clients"]):
                tidx = partition["client_test_indices"][str(c)]
                cts = [test_ex[i] for i in tidx] if tidx else test_ex[:32]
                cm = evaluate_model(test_model, cts, tokenizer, collator, CONFIG,
                                    desc=f"clienttest{c}", compute_loss=False)
                pcs.append(cm["bleu"])
            divs = [d for d in run_out.get("client_divergences", []) if isinstance(d, (int, float)) and d == d]
            client_test_stats = {
                "bleu_mean": float(np.mean(pcs)), "bleu_std": float(np.std(pcs)),
                "rougeL_mean": test_metrics["rougeL"], "rougeL_std": 0.0,
                "rougeL_min": test_metrics["rougeL"], "rougeL_max": test_metrics["rougeL"],
                "bert_f1_mean": test_metrics["bertscore_f1"], "bert_f1_std": 0.0,
                "div_mean": float(np.mean(divs)) if divs else NAN,
                "div_std": float(np.std(divs)) if divs else NAN}
        del test_model
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        latency, latency_rows, merge_info = run_latency_full(BEST_DIR, test_ex, tokenizer, CONFIG)
        pred_records = [{
            "experiment_id": EXPERIMENT_ID, "sample_id": i, "client_id": "",
            "input_text": tpred["inputs"][i], "reference_text": tpred["refs"][i],
            "prediction_text": tpred["preds"][i],
            "input_tokens": len(tokenizer(tpred["inputs"][i])["input_ids"]),
            "reference_tokens": len(tokenizer(tpred["refs"][i])["input_ids"]),
            "generated_tokens": len(tokenizer(tpred["preds"][i])["input_ids"]),
            "generation_time_ms": NAN} for i in range(len(tpred["preds"]))]

    atomic_write_json(opath("test_metrics.json"), json_safe(test_metrics))
    # validation metrics at best
    rounds_df = pd.DataFrame(hist.rounds)
    best_val_row = {}
    if len(rounds_df):
        if IS_FEDERATED and run_out.get("best_round", -1) > 0:
            sub = rounds_df[rounds_df["round"] == run_out["best_round"]]
        elif not IS_FEDERATED and run_out.get("best_round", -1) > 0:
            sub = rounds_df[rounds_df["epoch"] == run_out["best_round"]]
        else:
            sub = rounds_df.tail(1)
        if len(sub):
            best_val_row = sub.iloc[-1].to_dict()
    atomic_write_json(opath("validation_metrics.json"), json_safe(best_val_row))

    # ---- FLOPs ----
    flops = estimate_flops(counts, CONFIG)
    atomic_write_csv(opath("flops_results.csv"), pd.DataFrame([{
        "experiment_id": EXPERIMENT_ID, "seed": 42,
        "measurement_method": flops["flops_measurement_method"],
        "measurement_status": flops["flops_measurement_status"],
        "is_estimated": flops["flops_is_estimated"], "batch_size": CONFIG["train_batch_size"],
        "input_tokens": CONFIG["max_source_length"], "output_tokens": CONFIG["max_target_length"],
        "forward_flops_per_sample": flops["forward_flops_per_sample"],
        "training_flops_per_step": flops["training_flops_per_step"],
        "estimated_total_training_flops": flops["training_flops_per_step"] * max(1, run_out["completed_rounds"]),
        "generation_flops_per_sample": flops["generation_flops_per_sample"],
        "error_message": flops["flops_error_message"]}]))

    # ---- communication summary ----
    rounds = run_out.get("completed_rounds", 0) if IS_FEDERATED else 0
    reference_full_params = sum(p.numel() for p in AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).parameters())
    if IS_FEDERATED:
        ct = run_out["comm_totals"]
        total_comm = ct["upload"] + ct["download"] + ct["metadata"] + ct["control"]
        avg_sel = run_out.get("avg_selected", CONFIG["num_clients"])
        comm_params = count_communicated_params(model)
        ref_full_mb_round = avg_sel * 2 * reference_full_params * BYTES_PER_PARAM / (1024 ** 2)
        actual_per_round = total_comm / max(1, rounds)
        reduction_pr = (1 - actual_per_round / ref_full_mb_round) * 100 if ref_full_mb_round > 0 else 0.0
        reduction_total = (1 - total_comm / (ref_full_mb_round * max(1, rounds))) * 100 if ref_full_mb_round > 0 else 0.0
        comm_summary = {
            "params_per_round": comm_params, "rounds": rounds,
            "upload_per_round": ct["upload"] / max(1, rounds),
            "download_per_round": ct["download"] / max(1, rounds),
            "metadata_per_round": ct["metadata"] / max(1, rounds),
            "control_per_round": ct["control"] / max(1, rounds),
            "comm_per_round": total_comm / max(1, rounds),
            "total_upload": ct["upload"], "total_download": ct["download"],
            "total_metadata": ct["metadata"], "total_control": ct["control"],
            "total_comm": total_comm,
            "total_params": comm_params * 2 * int(round(avg_sel)) * rounds,
            "reference_full_params": reference_full_params,
            "reference_full_MB_per_round": ref_full_mb_round,
            "reduction_per_round": reduction_pr, "reduction_total": reduction_total}
    else:
        comm_summary = {
            "params_per_round": 0, "rounds": 0, "upload_per_round": 0.0, "download_per_round": 0.0,
            "metadata_per_round": 0.0, "control_per_round": 0.0, "comm_per_round": 0.0,
            "total_upload": 0.0, "total_download": 0.0, "total_metadata": 0.0, "total_control": 0.0,
            "total_comm": 0.0, "total_params": 0, "reference_full_params": reference_full_params,
            "reference_full_MB_per_round": 2 * reference_full_params * BYTES_PER_PARAM / (1024 ** 2),
            "reduction_per_round": 100.0, "reduction_total": 100.0}

    # ---- timing ----
    round_times = [r.get("time_per_round_seconds", np.nan) for r in hist.rounds]
    round_times = [t for t in round_times if isinstance(t, (int, float)) and t == t]
    train_times = [r.get("training_time_seconds", np.nan) for r in hist.rounds]
    train_times = [t for t in train_times if isinstance(t, (int, float)) and t == t]
    best_round_elapsed = NAN
    if len(rounds_df):
        key = "round" if IS_FEDERATED else "epoch"
        br = run_out.get("best_round", -1)
        sub = rounds_df[rounds_df[key] == br] if br and br > 0 else rounds_df.tail(1)
        if len(sub):
            best_round_elapsed = float(sub.iloc[-1].get("elapsed_time_seconds", NAN))
    max_vram_alloc = max([r.get("max_vram_allocated_MB", 0) for r in hist.rounds] + [0])
    max_vram_res = max([r.get("max_vram_reserved_MB", 0) for r in hist.rounds] + [0])
    total_time = time.perf_counter() - t_start
    timing = {
        "training_time": float(np.sum(train_times)) if train_times else 0.0,
        "evaluation_time": NAN,
        "generation_time": float(np.sum([r.get("generation_time_seconds", 0) for r in hist.rounds])),
        "total_time": total_time,
        "avg_round_time": float(np.mean(round_times)) if round_times else NAN,
        "std_round_time": float(np.std(round_times)) if round_times else NAN,
        "avg_client_train_time": float(np.mean(train_times)) if train_times else NAN,
        "best_round_elapsed": best_round_elapsed, "cpu_peak_mb": cpu_memory_MB(),
        "max_vram_alloc": max_vram_alloc, "max_vram_res": max_vram_res,
        "start_iso": start_iso, "end_iso": now_iso()}

    # ---- write CSV histories ----
    paths = {
        "history": opath("history.csv"), "client_history": opath("client_history.csv"),
        "comm_history": opath("communication_history.csv"),
        "resource_history": opath("resource_history.csv"), "latency": opath("latency_details.csv"),
        "flops": opath("flops_results.csv"), "predictions": opath("test_predictions.csv"),
        "resume_ckpt": resume_ckpt_path}
    rdf, cdf, mdf, resdf = hist.to_frames()
    if len(rdf):
        atomic_write_csv(paths["history"], rdf)
        atomic_write_json(opath("history.json"), json_safe(hist.rounds))
    if len(cdf):
        atomic_write_csv(paths["client_history"], cdf)
        atomic_write_json(opath("client_history.json"), json_safe(hist.clients))
    else:
        atomic_write_csv(paths["client_history"], pd.DataFrame([{"experiment_id": EXPERIMENT_ID, "seed": 42}]))
    if len(mdf):
        atomic_write_csv(paths["comm_history"], mdf)
        atomic_write_json(opath("communication_history.json"), json_safe(hist.comm))
    else:
        atomic_write_csv(paths["comm_history"], pd.DataFrame([{
            "experiment_id": EXPERIMENT_ID, "seed": 42, "round": 0, "client_id": "",
            "method": METHOD_FAMILY, "selected": False, "transmitted_dtype": TRANSMITTED_DTYPE,
            "bytes_per_parameter": BYTES_PER_PARAM, "upload_parameters": 0, "download_parameters": 0,
            "upload_bytes": 0, "download_bytes": 0, "metadata_bytes": 0, "control_variate_bytes": 0,
            "total_communication_bytes": 0, "total_communication_MB": 0.0,
            "cumulative_communication_MB": 0.0, "cumulative_communication_GB": 0.0}]))
    atomic_write_csv(paths["resource_history"], resdf if len(resdf) else pd.DataFrame([{
        "experiment_id": EXPERIMENT_ID, "seed": 42, "round": 0, "epoch": 0, "client_id": "",
        "stage": "none", "timestamp": now_iso(), "cpu_memory_MB": cpu_memory_MB(),
        "vram_allocated_MB": 0, "vram_reserved_MB": 0, "max_vram_allocated_MB": 0,
        "max_vram_reserved_MB": 0, "elapsed_time_seconds": 0}]))
    if not latency_rows:
        latency_rows = [{
            "experiment_id": EXPERIMENT_ID, "seed": 42, "latency_mode": "weighted_local",
            "batch_index": 0, "num_samples": latency["latency_num_samples"],
            "batch_size": 1, "input_tokens": latency["average_input_tokens"],
            "generated_tokens": latency["average_generated_tokens"],
            "elapsed_seconds": latency["latency_total_time_seconds"],
            "latency_ms_per_sample": latency["inference_latency_mean_ms_per_sample"],
            "throughput_samples_per_second": latency["inference_throughput_samples_per_second"],
            "throughput_tokens_per_second": latency["inference_throughput_tokens_per_second"],
            "device": DEVICE.type, "dtype": str(AMP_DTYPE).replace("torch.", ""),
            "num_beams": CONFIG["generation_num_beams"], "max_new_tokens": CONFIG["max_target_length"],
            "includes_tokenization": False, "is_warmup": False}]
    atomic_write_csv(paths["latency"], pd.DataFrame(latency_rows))
    atomic_write_json(opath("latency_details.json"), json_safe(latency))
    atomic_write_csv(paths["predictions"], pd.DataFrame(pred_records))
    with open(opath("test_predictions.jsonl"), "w", encoding="utf-8") as f:
        for rec in pred_records:
            f.write(json.dumps(json_safe(rec), ensure_ascii=False) + "\n")

    # ---- final results ----
    final = assemble_final_results(CONFIG, env, counts, data, run_out, test_metrics, latency,
                                   merge_info, comm_summary, flops, best_val_row, timing,
                                   client_test_stats, resume_used, paths)
    final_df = pd.DataFrame([[json_safe(final[c]) for c in FINAL_COLUMNS]], columns=FINAL_COLUMNS)
    atomic_write_json(opath("final_results.json"), json_safe(final))
    atomic_write_csv(opath("final_results.csv"), final_df)

    validate_final_results(opath("final_results.csv"), rdf, mdf, run_out["completed_rounds"])

    atomic_write_json(opath("run_summary.json"), json_safe({
        "experiment_id": EXPERIMENT_ID, "status": "completed", "test_metrics": test_metrics,
        "best_metric": run_out["best_metric"], "completed_rounds": run_out["completed_rounds"],
        "total_communication_MB": comm_summary["total_comm"],
        "inference_latency_mean_ms": latency["inference_latency_mean_ms_per_sample"],
        "output_directory": OUTPUT_DIR, "start_time": start_iso, "end_time": now_iso()}))
    write_status("completed", {"completed_rounds": run_out["completed_rounds"]})

    print("\n" + "=" * 78)
    print(f"DONE {EXPERIMENT_ID}")
    print(f"  test BLEU={test_metrics['bleu']:.2f} ROUGE-L={test_metrics['rougeL']:.4f} "
          f"BERTScore-F1={test_metrics['bertscore_f1']:.4f}")
    print(f"  total communication={comm_summary['total_comm']:.2f} MB "
          f"({comm_summary['total_comm']/1024.0:.4f} GB)")
    print(f"  inference latency mean={latency['inference_latency_mean_ms_per_sample']:.2f} ms/sample "
          f"p95={latency['inference_latency_p95_ms_per_sample']:.2f} ms")
    print(f"  primary latency source={merge_info['primary_latency_source']}")
    print(f"  max VRAM={timing['max_vram_alloc']:.0f} MB | total time={total_time:.1f}s")
    print(f"  final_results.csv -> {opath('final_results.csv')}")
    print(f"  best_model -> {BEST_DIR}")
    print(f"  final_model -> {FINAL_DIR}")
    print("=" * 78)
    return final


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        tb = traceback.format_exc()
        print(tb)
        try:
            atomic_write_json(opath("status.json"), {
                "experiment_id": EXPERIMENT_ID, "seed": 42, "status": "failed",
                "exception_type": type(exc).__name__, "error_message": str(exc),
                "traceback": tb, "failure_time": now_iso()})
        except Exception as _status_err:
            print(f"[status] could not write failure status ({_status_err})")
        raise
