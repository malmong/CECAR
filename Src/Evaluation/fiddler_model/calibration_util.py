"""
Calibration / prompt-stream loader for MoE routing experiments.

Pipeline :
  dataset -> plain-text prompts -> tokenize into a token stream -> slice into fixed-length chunks
  -> build (input, target) pairs (target masked except last token)

Supported datasets:
  - ARC-Challenge (multiple-choice prompts)
  - MMLU college_computer_science (optionally keyword-biased)
  - MathQA (Q/A with options and rationale)

Caching:
  Saves (trainloader, testenc) to <cache_dir>/<dataset>__<model_id>.pt
"""

from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -------------------------
# Tokenization helpers
# -------------------------
def get_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def tokenize_texts_to_stream(
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    add_eos: bool = True,
) -> List[int]:
    """Tokenize a list of texts into one contiguous token stream."""
    token_ids: List[int] = []
    eos = tokenizer.eos_token_id
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False).input_ids
        token_ids.extend(ids)
        if add_eos and eos is not None:
            token_ids.append(eos)
    return token_ids


def stream_to_chunks(
    token_ids: Sequence[int],
    seqlen: int,
    nsamples: int,
) -> List[torch.Tensor]:
    """
    Slice a token stream into fixed-length chunks.
    Returns: list of tensors, each shape [1, seqlen], length == nsamples if enough tokens exist.
    """
    out: List[torch.Tensor] = []
    pos = 0
    n = len(token_ids)

    while len(out) < nsamples and (pos + seqlen) <= n:
        chunk = token_ids[pos : pos + seqlen]
        pos += seqlen
        out.append(torch.tensor([chunk], dtype=torch.long))

    return out


def build_loader_from_stream(
    token_ids: Sequence[int],
    seqlen: int,
    nsamples: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Build a 'trainloader' list of (input_ids, labels) tuples.
    labels are identical to input_ids but masked with -100 except the last token.
    """
    chunks = stream_to_chunks(token_ids, seqlen=seqlen, nsamples=nsamples)
    if len(chunks) < nsamples:
        raise RuntimeError(
            f"Not enough tokens to build {nsamples} chunks of length {seqlen}. "
            f"Built {len(chunks)} chunks. Consider increasing dataset coverage or reducing seqlen."
        )

    trainloader: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for inp in chunks:
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def build_loader_from_texts(
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    seqlen: int,
    nsamples: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    stream = tokenize_texts_to_stream(tokenizer, texts, add_eos=True)
    return build_loader_from_stream(stream, seqlen=seqlen, nsamples=nsamples)


# -------------------------
# Dataset-specific builders
# -------------------------
def build_arc_challenge_loader(
    tokenizer: PreTrainedTokenizerBase,
    nsamples: int,
    seed: int,
    seqlen: int,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], None]:
    """
    ARC-Challenge -> plain-text MCQ prompts -> stream chunking.
    """
    set_seed(seed)
    print(f"[ARC-Challenge] Loading (seqlen={seqlen}, nsamples={nsamples})...")

    data = None
    last_err: Optional[Exception] = None

    candidates = [
        ("allenai/ai2_arc", "ARC-Challenge", "train"),
        ("allenai/ai2_arc", "ARC-Challenge", "validation"),
        ("allenai/ai2_arc", "ARC-Challenge", "test"),
        ("ai2_arc", "ARC-Challenge", "train"),
        ("ai2_arc", "ARC-Challenge", "validation"),
        ("ai2_arc", "ARC-Challenge", "test"),
    ]

    for ds_name, ds_cfg, ds_split in candidates:
        try:
            data = load_dataset(ds_name, ds_cfg, split=ds_split)
            print(f"[ARC-Challenge] Loaded: {ds_name} / {ds_cfg} / {ds_split}")
            break
        except Exception as e:
            last_err = e
            data = None

    if data is None:
        raise RuntimeError(f"[ARC-Challenge] Failed to load dataset. Last error: {last_err}")

    def format_doc(doc: dict) -> str:
        qobj = doc.get("question", None)
        if isinstance(qobj, dict):
            q = qobj.get("stem", "") or ""
            choices_obj = qobj.get("choices", []) or []
        else:
            q = doc.get("question", "") or ""
            choices_obj = doc.get("choices", []) or []

        opt_lines: List[str] = []
        if isinstance(choices_obj, list):
            for i, c in enumerate(choices_obj):
                if isinstance(c, dict):
                    label = c.get("label", None)
                    text = c.get("text", "") or ""
                    tag = str(label) if label is not None else chr(ord("A") + i)
                    opt_lines.append(f"{tag}. {text}")
                else:
                    opt_lines.append(f"{chr(ord('A') + i)}. {str(c)}")
        elif isinstance(choices_obj, dict):
            labels = choices_obj.get("label", []) or []
            texts_ = choices_obj.get("text", []) or []
            if isinstance(labels, list) and isinstance(texts_, list) and len(labels) == len(texts_):
                for l, t in zip(labels, texts_):
                    opt_lines.append(f"{l}. {t}")
            else:
                if isinstance(texts_, list):
                    for i, t in enumerate(texts_):
                        opt_lines.append(f"{chr(ord('A') + i)}. {t}")

        text = "Question:\n" + q
        if opt_lines:
            text += "\n\nChoices:\n" + "\n".join(opt_lines)
        text += "\n\nAnswer:"
        return text

    # Heuristic: collect more raw items than needed, to ensure enough tokens after formatting.
    texts: List[str] = []
    for doc in data:
        texts.append(format_doc(doc))
        if len(texts) >= nsamples * 12:
            break

    trainloader = build_loader_from_texts(tokenizer, texts, seqlen=seqlen, nsamples=nsamples)
    return trainloader, None


def build_mmlu_ccsc_loader(
    tokenizer: PreTrainedTokenizerBase,
    nsamples: int,
    seed: int,
    seqlen: int,
    keyword_bias: bool = True,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], None]:
    """
    MMLU (college_computer_science) -> MCQ prompts -> token stream -> chunking.
    If keyword_bias=True, prefer compiler/PL-related questions when enough exist.
    """
    set_seed(seed)
    print(f"[MMLU-CCSC] Loading (seqlen={seqlen}, nsamples={nsamples})...")

    data = None
    last_err: Optional[Exception] = None

    candidates = [
        ("cais/mmlu", "college_computer_science", "test"),
        ("cais/mmlu", "college_computer_science", "validation"),
        ("cais/mmlu", "college_computer_science", "dev"),
        ("hendrycks_test", "college_computer_science", "test"),
    ]

    for ds_name, ds_cfg, ds_split in candidates:
        try:
            data = load_dataset(ds_name, ds_cfg, split=ds_split)
            print(f"[MMLU-CCSC] Loaded: {ds_name} / {ds_cfg} / {ds_split}")
            break
        except Exception as e:
            last_err = e
            data = None

    if data is None:
        raise RuntimeError(f"[MMLU-CCSC] Failed to load dataset. Last error: {last_err}")

    def format_mcq(doc: dict) -> str:
        q = doc.get("question", "") or ""
        choices = doc.get("choices", None) or doc.get("options", None) or []
        lines: List[str] = []

        if isinstance(choices, list):
            for i, c in enumerate(choices):
                tag = chr(ord("A") + i)
                lines.append(f"{tag}. {c}")
        elif isinstance(choices, dict):
            for k, v in choices.items():
                lines.append(f"{k}. {v}")

        text = "Question:\n" + q
        if lines:
            text += "\n\nChoices:\n" + "\n".join(lines)
        text += "\n\nAnswer:"
        return text

    keywords = (
        "compiler", "compilers", "compilation", "compile",
        "parser", "parsing", "grammar", "lexer", "lexical",
        "syntax", "semantic", "intermediate code", "ir",
        "optimization", "register allocation", "control flow",
        "cfg", "ast",
    )

    all_texts: List[str] = []
    preferred_texts: List[str] = []

    for doc in data:
        t = format_mcq(doc)
        all_texts.append(t)
        if keyword_bias:
            low = t.lower()
            if any(k in low for k in keywords):
                preferred_texts.append(t)

    texts = preferred_texts if (keyword_bias and len(preferred_texts) >= 16) else all_texts
    if len(texts) == 0:
        raise RuntimeError("[MMLU-CCSC] Dataset empty or formatting failed.")

    # Build enough tokens for nsamples chunks.
    target_tokens = int(seqlen) * int(nsamples)
    token_ids: List[int] = []

    rng = random.Random(seed)
    order = list(range(len(texts)))
    rng.shuffle(order)
    idx = 0
    eos = tokenizer.eos_token_id

    while len(token_ids) < target_tokens:
        if idx >= len(order):
            rng.shuffle(order)
            idx = 0

        t = texts[order[idx]]
        idx += 1

        ids = tokenizer(t, add_special_tokens=False).input_ids
        token_ids.extend(ids)
        if eos is not None:
            token_ids.append(eos)

        # safety margin to avoid unbounded growth
        if len(token_ids) > target_tokens + seqlen * 32:
            break

    trainloader = build_loader_from_stream(token_ids, seqlen=seqlen, nsamples=nsamples)
    return trainloader, None


def build_mathqa_loader(
    tokenizer: PreTrainedTokenizerBase,
    nsamples: int,
    seed: int,
    seqlen: int,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], None]:
    set_seed(seed)
    print(f"[MathQA] Loading (seqlen={seqlen}, nsamples={nsamples})...")

    data = load_dataset("allenai/math_qa", split="train", revision="refs/pr/5")

    def format_doc(doc: dict) -> str:
        prob = doc.get("Problem", None) or doc.get("problem", "") or ""
        rat = doc.get("Rationale", None) or doc.get("rationale", "") or ""
        options = doc.get("options", None)
        correct = doc.get("correct", None)

        opt_lines: List[str] = []
        if isinstance(options, str):
            opt_lines = [options]
        elif isinstance(options, list):
            opt_lines = options

        text = f"Question:\n{prob}"
        if opt_lines:
            text += "\n\nChoices:\n" + "\n".join(opt_lines)
        if correct is not None:
            text += f"\n\nGold:\n{correct}"
        if rat:
            text += f"\n\nRationale:\n{rat}"
        return text

    texts: List[str] = []
    for doc in data:
        texts.append(format_doc(doc))
        if len(texts) >= nsamples * 8:
            break

    trainloader = build_loader_from_texts(tokenizer, texts, seqlen=seqlen, nsamples=nsamples)
    return trainloader, None


# -------------------------
# Public API
# -------------------------
@dataclass(frozen=True)
class LoaderConfig:
    dataset_name: str
    model_path: str
    model_id: str  
    nsamples: int = 128
    seed: int = 0
    seqlen: int = 2048
    cache_dir: str = "./MCMOE"
    keyword_bias: bool = True  


def _cache_path(cache_dir: str, dataset_name: str, model_id: str) -> str:
    safe_ds = dataset_name.replace("/", "_")
    safe_mid = model_id.replace("/", "_")
    return os.path.join(cache_dir, f"{safe_ds}__{safe_mid}.pt")


def get_loaders(
    *,
    dataset_name: str,
    model_path: str,
    model_id: str,
    nsamples: int = 128,
    seed: int = 0,
    seqlen: int = 2048,
    cache_dir: str = "./MCMOE",
    keyword_bias: bool = True,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], None]:

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = _cache_path(cache_dir, dataset_name, model_id)

    # 1) Load cache if valid
    if os.path.exists(cache_file):
        print(f"[Cache] Loading: {cache_file}")
        try:
            trainloader, testenc = torch.load(cache_file)
            if isinstance(trainloader, list) and len(trainloader) >= nsamples:
                return trainloader, testenc
            print(f"[Cache] Insufficient batches: {len(trainloader)} < {nsamples}. Rebuilding.")
        except Exception as e:
            print(f"[Cache] Failed to load cache ({e}). Rebuilding.")

    # 2) Build from dataset
    tokenizer = get_tokenizer(model_path)

    name = dataset_name.lower()
    if name in ("arc", "arc_challenge"):
        loaders = build_arc_challenge_loader(tokenizer, nsamples, seed, seqlen)
    elif name in ("mathqa",):
        loaders = build_mathqa_loader(tokenizer, nsamples, seed, seqlen)
    elif name in ("mmlu", "mmlu_ccsc", "mmlu_college_computer_science"):
        loaders = build_mmlu_ccsc_loader(tokenizer, nsamples, seed, seqlen, keyword_bias=keyword_bias)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Allowed: arc_challenge|mathqa|mmlu_ccsc"
        )

    # 3) Save cache
    abs_path = os.path.abspath(cache_file)
    print(f"[Cache] Saving: {abs_path}")
    torch.save(loaders, cache_file)
    return loaders