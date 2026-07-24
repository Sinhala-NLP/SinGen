#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
merge_sinllama.py
=================

ElChat-style (Yamaguchi et al., TMLR 2025) language-adaptation merging for
SinLlama, with three merge methods:

    * slerp / linear   -- direct interpolation of two models (ElChat, Sec. 3)
    * ties             -- TIES-Merging (Yadav et al., NeurIPS 2023)
    * dare_ties        -- DARE (Yu et al., ICML 2024) + TIES
    * dare_linear      -- DARE + task-arithmetic sum

------------------------------------------------------------------------------
MODEL TOPOLOGY
------------------------------------------------------------------------------
ElChat merges a *source chat* model with a *chat model that was vocabulary-
expanded and continually pre-trained* (Chat+VE). SinLlama is different: it is
continual pre-training on the **base** Llama-3-8B, not on the Instruct model.
So the three checkpoints here are:

    theta_pre   = meta-llama/Meta-Llama-3-8B            (common ancestor)
    theta_chat  = meta-llama/Meta-Llama-3-8B-Instruct   (source chat abilities)
    theta_sin   = polyglots/SinLlama_v01 merged into theta_pre
                  (Sinhala abilities + 139,336-token vocabulary)

  * slerp/linear interpolate theta_sin and theta_chat directly. This is the
    literal ElChat "Merge" step; the caveat is that ElChat's target had already
    inherited the chat weights, whereas theta_sin has not.

  * ties/dare_* work on task vectors relative to theta_pre:
        tau_chat = theta_chat - theta_pre     (the "chat vector")
        tau_sin  = theta_sin  - theta_pre     (the "Sinhala vector")
    and reconstruct theta_pre + lambda * merge(tau_chat, tau_sin). This is the
    setting TIES/DARE were designed for (homologous models, shared ancestor),
    and it is strictly better motivated than interpolating theta_sin with
    theta_chat when the two never shared a post-training trajectory.

------------------------------------------------------------------------------
VOCABULARY HANDLING
------------------------------------------------------------------------------
theta_sin has |V| = 139,336; theta_pre / theta_chat have |V| = 128,256.
Following ElChat, `model.embed_tokens.weight` and `lm_head.weight` are EXCLUDED
from merging: the output keeps SinLlama's embeddings wholesale. The ElChat
"Copy" step is then applied -- the rows of the chat model's special tokens
(<|begin_of_text|>, <|start_header_id|>, <|eot_id|>, ...) are copied back into
both matrices, since those IDs are stable across the two vocabularies and their
representations are what activate instruction following.

------------------------------------------------------------------------------
MEMORY
------------------------------------------------------------------------------
Only the output model is held in RAM. theta_pre and theta_chat are streamed
tensor-by-tensor straight off their safetensors shards, so peak RSS is roughly
one bf16 8B model (~16 GB) plus a few hundred MB of scratch. Comfortable on a
100 GB node; no GPU required.

------------------------------------------------------------------------------
EXAMPLES
------------------------------------------------------------------------------
# 0. (once) materialise SinLlama and cache it, so later runs skip the PEFT merge
python merge_sinllama.py --method slerp --alpha 0.3 \
    --save_merged_target /scratch/hpc/37/ranasint/models/SinLlama-merged \
    --out /scratch/hpc/37/ranasint/models/SinLlama-elchat-slerp03

# 1. ElChat SLERP, full model (recommended for SinLlama: LoRA touched all layers)
python merge_sinllama.py --method slerp --layers all --alpha 0.3 \
    --target /scratch/hpc/37/ranasint/models/SinLlama-merged \
    --out .../SinLlama-slerp-a03

# 2. Literal ElChat 2x2LS schedule (layers 0,1,L-2,L-1 at 0.3/0.5/0.5/0.3)
python merge_sinllama.py --method slerp --layers 2x2ls \
    --target .../SinLlama-merged --out .../SinLlama-elchat-2x2ls

# 3. TIES
python merge_sinllama.py --method ties --density 0.2 --lam 1.0 \
    --weights 1.0 1.0 --target .../SinLlama-merged --out .../SinLlama-ties-d02

# 4. DARE-TIES
python merge_sinllama.py --method dare_ties --density 0.5 --lam 1.0 --seed 42 \
    --target .../SinLlama-merged --out .../SinLlama-dareties-d05
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence

import torch


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ElChat-style merging for SinLlama (SLERP / TIES / DARE-TIES).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- models ---------------------------------------------------------
    g = p.add_argument_group("models")
    g.add_argument("--target", default="polyglots/SinLlama_v01",
                   help="Sinhala-adapted model. Either a PEFT adapter repo/dir "
                        "(auto-detected via adapter_config.json) or an already "
                        "merged full model.")
    g.add_argument("--base", default="meta-llama/Meta-Llama-3-8B",
                   help="Common ancestor. Used as the PEFT base and as theta_pre "
                        "for TIES/DARE task vectors.")
    g.add_argument("--chat", default="meta-llama/Meta-Llama-3-8B-Instruct",
                   help="Source chat model providing instruction-following.")
    g.add_argument("--tokenizer", default=None,
                   help="Extended (Sinhala) tokenizer. Defaults to --target.")
    g.add_argument("--cache_dir", default=os.environ.get("HF_HOME"),
                   help="HF cache directory.")
    g.add_argument("--save_merged_target", default=None,
                   help="If --target is an adapter, also save the PEFT-merged "
                        "model here for reuse.")

    # --- merge method ---------------------------------------------------
    g = p.add_argument_group("merge method")
    g.add_argument("--method", default="slerp",
                   choices=["slerp", "linear", "ties", "dare_ties", "dare_linear"])
    g.add_argument("--alpha", type=float, default=0.3,
                   help="[slerp/linear] Weight of the CHAT model. alpha=0 keeps "
                        "SinLlama untouched, alpha=1 gives the chat model. "
                        "ElChat used 0.3 for the outermost trained layers.")
    g.add_argument("--density", type=float, default=None,
                   help="[ties] fraction of |tau| entries kept by magnitude "
                        "pruning (TIES paper k=0.2). [dare_*] fraction KEPT by "
                        "random pruning, i.e. drop rate p = 1 - density. "
                        "Defaults: 0.2 for ties, 0.5 for dare_*.")
    g.add_argument("--lam", type=float, default=1.0,
                   help="[ties/dare_*] scaling lambda in theta = theta_pre + lam * tau_m.")
    g.add_argument("--weights", type=float, nargs=2, default=[1.0, 1.0],
                   metavar=("W_SIN", "W_CHAT"),
                   help="[ties/dare_*] relative weight of the Sinhala and chat "
                        "task vectors.")
    g.add_argument("--seed", type=int, default=42, help="[dare_*] RNG seed.")

    # --- which layers ---------------------------------------------------
    g = p.add_argument_group("layer selection")
    g.add_argument("--layers", default="all",
                   help="'all', '2x2ls' (ElChat: layers 0,1,L-2,L-1 only), or an "
                        "explicit comma-separated list, e.g. '0,1,-2,-1'.")
    g.add_argument("--layer_alphas", type=float, nargs="*", default=None,
                   help="[slerp/linear + explicit/2x2ls layers] per-layer alpha, "
                        "same order as the layer list. ElChat 2x2ls default is "
                        "0.3 0.5 0.5 0.3.")
    g.add_argument("--fallback", default="chat", choices=["chat", "target"],
                   help="Where non-selected layers come from. ElChat's 2x2LS "
                        "pipeline leaves them at the chat model's values.")

    # --- ElChat Copy step ------------------------------------------------
    g = p.add_argument_group("special-token copy (ElChat 'Copy')")
    g.add_argument("--copy_special_tokens", action="store_true", default=True,
                   help="Copy chat-model rows for tokenizer.all_special_tokens.")
    g.add_argument("--no_copy_special_tokens", dest="copy_special_tokens",
                   action="store_false")
    g.add_argument("--copy_added_tokens", action="store_true",
                   help="Also copy every token in the chat tokenizer's added "
                        "vocab (all 256 <|reserved_special_token_N|> for Llama-3).")
    g.add_argument("--extra_copy_ids", type=int, nargs="*",
                   default=list(range(128000, 128011)),
                   help="Extra token IDs to copy. Default 128000-128010 covers "
                        "the whole Llama-3 chat template, matching the ElChat "
                        "chat-vector special_tokens_map.")
    g.add_argument("--copy_chat_template", action="store_true", default=True,
                   help="Attach the chat model's chat_template / eos config to "
                        "the output tokenizer.")
    g.add_argument("--no_copy_chat_template", dest="copy_chat_template",
                   action="store_false")

    # --- misc -----------------------------------------------------------
    g = p.add_argument_group("misc")
    g.add_argument("--out", required=True, help="Output directory.")
    g.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    g.add_argument("--smoke_test", action="store_true",
                   help="Run one CPU forward pass on a Sinhala prompt and report "
                        "loss; catches NaNs and dead embeddings.")
    g.add_argument("--dry_run", action="store_true",
                   help="Print the merge plan and exit without loading weights.")
    return p.parse_args(argv)


DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Lazy safetensors reader -- keeps theta_pre / theta_chat off the heap
# ----------------------------------------------------------------------------
class ShardedWeights:
    """Random access to a checkpoint's tensors without materialising the model."""

    def __init__(self, model_id: str, cache_dir: Optional[str] = None):
        from safetensors import safe_open

        self._safe_open = safe_open
        self.path = self._resolve(model_id, cache_dir)
        index = os.path.join(self.path, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index, "r", encoding="utf-8") as f:
                self.weight_map: Dict[str, str] = json.load(f)["weight_map"]
        else:
            single = os.path.join(self.path, "model.safetensors")
            if not os.path.exists(single):
                raise FileNotFoundError(
                    f"No safetensors found in {self.path}. Convert the .bin "
                    f"checkpoint first, or point --base/--chat at a safetensors repo."
                )
            with safe_open(single, framework="pt", device="cpu") as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}
        self._handles: Dict[str, object] = {}

    @staticmethod
    def _resolve(model_id: str, cache_dir: Optional[str]) -> str:
        if os.path.isdir(model_id):
            return model_id
        from huggingface_hub import snapshot_download

        return snapshot_download(
            model_id,
            cache_dir=cache_dir,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
        )

    def __contains__(self, key: str) -> bool:
        return key in self.weight_map

    def keys(self) -> Iterable[str]:
        return self.weight_map.keys()

    def get(self, key: str) -> torch.Tensor:
        shard = self.weight_map[key]
        if shard not in self._handles:
            self._handles[shard] = self._safe_open(
                os.path.join(self.path, shard), framework="pt", device="cpu"
            )
        return self._handles[shard].get_tensor(key)

    def close(self) -> None:
        self._handles.clear()


# ----------------------------------------------------------------------------
# Merge primitives
# ----------------------------------------------------------------------------
def lerp(t: float, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    return (1.0 - t) * v0 + t * v1


def _normalize(v: torch.Tensor, eps: float) -> torch.Tensor:
    n = torch.linalg.norm(v)
    return v / n if n > eps else v


def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor,
          dot_threshold: float = 0.9995, eps: float = 1e-8) -> torch.Tensor:
    """Spherical linear interpolation over the flattened tensor.

    Mirrors mergekit / the ElChat reference implementation: the dot product is
    taken over the whole tensor, not row-wise. t=0 -> v0, t=1 -> v1.
    """
    v0 = v0.to(torch.float32)
    v1 = v1.to(torch.float32)
    v0_flat, v1_flat = v0.flatten(), v1.flatten()
    dot = torch.dot(_normalize(v0_flat, eps), _normalize(v1_flat, eps))

    if torch.abs(dot) > dot_threshold:
        return lerp(t, v0, v1)

    theta_0 = torch.arccos(dot.clamp(-1.0, 1.0))
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * t
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    return s0 * v0 + s1 * v1


def magnitude_prune(tau: torch.Tensor, density: float) -> torch.Tensor:
    """TIES 'Trim': keep the top-density fraction of entries by |value|."""
    if density >= 1.0:
        return tau
    flat = tau.abs().flatten()
    k = max(1, int(round(flat.numel() * density)))
    # k-th largest == (numel - k + 1)-th smallest
    threshold = torch.kthvalue(flat, flat.numel() - k + 1).values
    return torch.where(tau.abs() >= threshold, tau, torch.zeros_like(tau))


def dare_prune(tau: torch.Tensor, density: float,
               generator: torch.Generator) -> torch.Tensor:
    """DARE: Bernoulli drop with rate p = 1 - density, then rescale by 1/(1-p).

    Yu et al. (2024), Eq. 1. The rescale is what preserves E[h]; dropping
    without it ("DropOnly") degrades sharply as p grows.
    """
    if density >= 1.0:
        return tau
    mask = torch.rand(tau.shape, generator=generator, dtype=torch.float32) < density
    return (tau * mask) / density


def ties_disjoint_merge(taus: List[torch.Tensor],
                        weights: Sequence[float]) -> torch.Tensor:
    """TIES 'Elect Sign' + 'Disjoint Merge'.

    gamma_m = sgn(sum_t w_t * tau_t); then average, per parameter, only over the
    models whose sign agrees with gamma_m (zeros are always excluded). With
    unit weights this is exactly Yadav et al. Sec. 4.2 step 3.
    """
    total = torch.zeros_like(taus[0])
    for w, tau in zip(weights, taus):
        total += w * tau
    gamma = torch.sign(total)

    num = torch.zeros_like(total)
    den = torch.zeros_like(total)
    for w, tau in zip(weights, taus):
        agree = ((torch.sign(tau) == gamma) & (tau != 0)).to(total.dtype)
        num += w * tau * agree
        den += w * agree
    merged = num / den.clamp(min=1e-12)
    merged[den == 0] = 0.0
    return merged


def task_arithmetic_sum(taus: List[torch.Tensor],
                        weights: Sequence[float]) -> torch.Tensor:
    out = torch.zeros_like(taus[0])
    for w, tau in zip(weights, taus):
        out += w * tau
    return out


# ----------------------------------------------------------------------------
# Loading the Sinhala-adapted model
# ----------------------------------------------------------------------------
def is_peft_adapter(model_id: str, cache_dir: Optional[str]) -> bool:
    if os.path.isdir(model_id):
        return os.path.exists(os.path.join(model_id, "adapter_config.json"))
    from huggingface_hub import list_repo_files

    try:
        return "adapter_config.json" in list_repo_files(model_id)
    except Exception:
        return False


def build_target_model(args, dtype):
    """Return (model, tokenizer) for the Sinhala-adapted model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_id = args.tokenizer or args.target
    tokenizer = AutoTokenizer.from_pretrained(tok_id, cache_dir=args.cache_dir)
    log(f"Sinhala tokenizer: {tok_id}  (len={len(tokenizer)})")

    if not is_peft_adapter(args.target, args.cache_dir):
        log(f"Loading merged target model: {args.target}")
        model = AutoModelForCausalLM.from_pretrained(
            args.target, torch_dtype=dtype, low_cpu_mem_usage=True,
            cache_dir=args.cache_dir,
        )
        return model, tokenizer

    # PEFT path: base -> resize -> adapter -> merge_and_unload
    from peft import PeftModel

    log(f"{args.target} is a PEFT adapter; loading base {args.base}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, low_cpu_mem_usage=True, cache_dir=args.cache_dir
    )
    old_vocab = model.get_input_embeddings().weight.shape[0]
    log(f"Resizing embeddings {old_vocab} -> {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    log("Attaching adapter and merging")
    model = PeftModel.from_pretrained(model, args.target, cache_dir=args.cache_dir)
    model = model.merge_and_unload()

    # The failure mode worth catching early: if the adapter did not carry
    # embed_tokens / lm_head in modules_to_save, the 11,080 new Sinhala rows are
    # still at their random/mean init and the model will emit garbage.
    check_extended_embeddings(model, old_vocab)

    if args.save_merged_target:
        log(f"Saving merged target to {args.save_merged_target}")
        os.makedirs(args.save_merged_target, exist_ok=True)
        model.save_pretrained(args.save_merged_target, safe_serialization=True)
        tokenizer.save_pretrained(args.save_merged_target)

    return model, tokenizer


def check_extended_embeddings(model, old_vocab: int) -> None:
    emb = model.get_input_embeddings().weight.data
    new = emb[old_vocab:]
    if new.numel() == 0:
        return
    with torch.no_grad():
        f = new.to(torch.float32)
        std = f.std().item()
        row_std = f.std(dim=1)
        n_dead = int((row_std < 1e-6).sum().item())
        old_std = emb[:old_vocab].to(torch.float32).std().item()
    log(f"  new-token embeddings: n={new.shape[0]}  std={std:.5f} "
        f"(original rows std={old_std:.5f})  degenerate rows={n_dead}")
    if n_dead > new.shape[0] * 0.5 or std < old_std * 0.05:
        log("  !! WARNING: the extended embedding rows look untrained. The "
            "adapter merge probably dropped embed_tokens/lm_head. Expect "
            "collapsed SARI and garbled Sinhala output.")


# ----------------------------------------------------------------------------
# Layer selection
# ----------------------------------------------------------------------------
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def resolve_layers(spec: str, n_layers: int) -> Optional[List[int]]:
    """Return absolute layer indices to merge, or None meaning 'everything'."""
    if spec.strip().lower() == "all":
        return None
    if spec.strip().lower() == "2x2ls":
        raw = [0, 1, -2, -1]
    else:
        raw = [int(x) for x in spec.replace(" ", "").split(",") if x != ""]
    return [i if i >= 0 else n_layers + i for i in raw]


# ----------------------------------------------------------------------------
# Main merge loop
# ----------------------------------------------------------------------------
def merge(args) -> None:
    dtype = DTYPES[args.dtype]
    torch.set_grad_enabled(False)

    if args.density is None:
        args.density = 0.2 if args.method == "ties" else 0.5
    needs_base = args.method in ("ties", "dare_ties", "dare_linear")

    log("=" * 72)
    log(f"method={args.method}  layers={args.layers}  fallback={args.fallback}")
    if args.method in ("slerp", "linear"):
        log(f"alpha (weight of chat)={args.alpha}  layer_alphas={args.layer_alphas}")
    else:
        log(f"density={args.density}  lambda={args.lam}  "
            f"weights(sin,chat)={args.weights}  seed={args.seed}")
    log(f"target={args.target}")
    log(f"chat  ={args.chat}")
    if needs_base:
        log(f"base  ={args.base}  (task-vector anchor)")
    log(f"out   ={args.out}")
    log("=" * 72)
    if args.dry_run:
        return

    from transformers import AutoTokenizer

    model, tokenizer = build_target_model(args, dtype)
    n_layers = model.config.num_hidden_layers
    selected = resolve_layers(args.layers, n_layers)

    alpha_by_layer: Dict[int, float] = {}
    if selected is not None and args.method in ("slerp", "linear"):
        alphas = args.layer_alphas
        if alphas is None:
            alphas = ([0.3, 0.5, 0.5, 0.3] if args.layers.lower() == "2x2ls"
                      else [args.alpha] * len(selected))
        if len(alphas) != len(selected):
            raise ValueError(
                f"--layer_alphas has {len(alphas)} values but {len(selected)} "
                f"layers were selected."
            )
        alpha_by_layer = dict(zip(selected, alphas))
        log(f"per-layer alpha: {alpha_by_layer}")

    log("Opening source checkpoints (streamed, not loaded)")
    chat_w = ShardedWeights(args.chat, args.cache_dir)
    base_w = ShardedWeights(args.base, args.cache_dir) if needs_base else None

    generator = torch.Generator().manual_seed(args.seed)

    n_merged = n_copied = n_kept = n_skipped = 0
    skipped_names: List[str] = []

    for name, param in model.named_parameters():
        m = LAYER_RE.match(name)
        layer_idx = int(m.group(1)) if m else None

        if name not in chat_w:
            n_skipped += 1
            skipped_names.append(f"{name} (absent from chat model)")
            continue

        chat_t = chat_w.get(name)
        if chat_t.shape != param.shape:
            # embed_tokens / lm_head: vocabularies differ. ElChat excludes these
            # from merging; SinLlama's versions are kept.
            n_skipped += 1
            skipped_names.append(
                f"{name} {tuple(param.shape)} vs chat {tuple(chat_t.shape)}"
            )
            del chat_t
            continue

        in_scope = selected is None or (layer_idx is not None and layer_idx in selected)

        if not in_scope:
            if args.fallback == "chat":
                param.data.copy_(chat_t.to(param.dtype))
                n_copied += 1
            else:
                n_kept += 1
            del chat_t
            continue

        tgt = param.data.to(torch.float32)
        chat_f = chat_t.to(torch.float32)
        del chat_t

        if args.method in ("slerp", "linear"):
            t = alpha_by_layer.get(layer_idx, args.alpha)
            # v0 = SinLlama (adapted), v1 = chat; t is the weight on chat.
            out = (slerp(t, tgt, chat_f) if args.method == "slerp"
                   else lerp(t, tgt, chat_f))
        else:
            pre = base_w.get(name).to(torch.float32)
            tau_sin = tgt - pre
            tau_chat = chat_f - pre
            if args.method == "ties":
                taus = [magnitude_prune(tau_sin, args.density),
                        magnitude_prune(tau_chat, args.density)]
                tau_m = ties_disjoint_merge(taus, args.weights)
            elif args.method == "dare_ties":
                taus = [dare_prune(tau_sin, args.density, generator),
                        dare_prune(tau_chat, args.density, generator)]
                tau_m = ties_disjoint_merge(taus, args.weights)
            else:  # dare_linear
                taus = [dare_prune(tau_sin, args.density, generator),
                        dare_prune(tau_chat, args.density, generator)]
                tau_m = task_arithmetic_sum(taus, args.weights)
            out = pre + args.lam * tau_m
            del pre, tau_sin, tau_chat, taus, tau_m

        param.data.copy_(out.to(param.dtype))
        del tgt, chat_f, out
        n_merged += 1

        if n_merged % 40 == 0:
            log(f"  merged {n_merged} tensors ...")
            gc.collect()

    log(f"Done: merged={n_merged}  copied_from_chat={n_copied}  "
        f"kept_from_target={n_kept}  skipped={n_skipped}")
    for s in skipped_names:
        log(f"  skipped: {s}")

    chat_w.close()
    if base_w is not None:
        base_w.close()
    gc.collect()

    # ---- ElChat "Copy": restore chat special-token rows --------------------
    chat_tok = AutoTokenizer.from_pretrained(args.chat, cache_dir=args.cache_dir)
    copy_special_tokens(model, args, chat_tok, ShardedWeights(args.chat, args.cache_dir))

    # ---- save --------------------------------------------------------------
    model.config.vocab_size = model.get_input_embeddings().weight.shape[0]
    if args.copy_chat_template:
        tokenizer.chat_template = chat_tok.chat_template
        if chat_tok.eos_token is not None:
            tokenizer.eos_token = chat_tok.eos_token
        eos_id = chat_tok.convert_tokens_to_ids("<|eot_id|>")
        if eos_id is not None and eos_id >= 0:
            model.config.eos_token_id = eos_id
            if getattr(model, "generation_config", None) is not None:
                model.generation_config.eos_token_id = eos_id
        log("Copied chat template and eos config from the chat model")

    if args.smoke_test:
        smoke_test(model, tokenizer)

    os.makedirs(args.out, exist_ok=True)
    log(f"Saving to {args.out}")
    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)
    with open(os.path.join(args.out, "merge_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    log("Finished.")


def copy_special_tokens(model, args, chat_tok, chat_w: ShardedWeights) -> None:
    """ElChat step (iii): E_t[x,:] <- E_s[x,:] and O_t[:,x] <- O_s[:,x]."""
    ids = set()
    if args.copy_special_tokens:
        vocab = chat_tok.get_vocab()
        for tok in chat_tok.all_special_tokens:
            if tok in vocab:
                ids.add(vocab[tok])
    if args.copy_added_tokens:
        ids.update(chat_tok.get_added_vocab().values())
    if args.extra_copy_ids:
        ids.update(args.extra_copy_ids)

    if not ids:
        log("No special tokens to copy (Copy step disabled).")
        return

    emb = model.get_input_embeddings().weight.data
    head = model.get_output_embeddings()
    idx = sorted(i for i in ids if 0 <= i < emb.shape[0])
    log(f"Copy step: restoring {len(idx)} special-token rows from the chat model")

    src_emb_key = "model.embed_tokens.weight"
    src_emb = chat_w.get(src_emb_key)
    index = torch.tensor(idx, dtype=torch.long)
    emb[index] = src_emb[index].to(emb.dtype)
    del src_emb

    if head is not None and "lm_head.weight" in chat_w:
        src_head = chat_w.get("lm_head.weight")
        if src_head.shape[1] == head.weight.shape[1]:
            head.weight.data[index] = src_head[index].to(head.weight.dtype)
        else:
            log("  lm_head hidden dim mismatch; skipped")
        del src_head
    chat_w.close()
    gc.collect()


def smoke_test(model, tokenizer) -> None:
    log("Smoke test: single CPU forward pass")
    prompt = "ශ්‍රී ලංකාවේ අගනුවර වන්නේ"
    try:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
    except Exception:
        text = prompt
    enc = tokenizer(text, return_tensors="pt")
    out = model(**enc, labels=enc["input_ids"])
    loss = out.loss.item()
    log(f"  tokens={enc['input_ids'].shape[1]}  loss={loss:.4f}  "
        f"ppl={torch.exp(torch.tensor(loss)).item():.2f}")
    if not torch.isfinite(out.logits).all():
        log("  !! non-finite logits -- the merge produced NaN/Inf weights.")
    elif loss > 12:
        log("  !! loss is very high; check the extended embeddings and alpha.")


def main() -> None:
    args = parse_args()
    merge(args)


if __name__ == "__main__":
    sys.exit(main())