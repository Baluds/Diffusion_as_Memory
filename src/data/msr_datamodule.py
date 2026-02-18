# from typing import Any, Dict, List, Optional

# import torch
# from torch.utils.data import DataLoader
# import pytorch_lightning as pl
# from transformers import AutoTokenizer

# from src.data.msr_gist_dataset import MSRGistDataset


# class MSRGistDataModule(pl.LightningDataModule):
#     """
#     Produces batches containing tokenized:
#       - x
#       - x_plus
#       - y

#     This matches your current training needs:
#       - reconstruction: x -> y
#       - future contrastive: x vs x_plus
#     """

#     def __init__(
#         self,
#         train_path: str,
#         tokenizer_name: str = "bert-base-uncased",
#         batch_size: int = 8,
#         max_length: int = 256,
#         num_workers: int = 2,
#         y_key: str = "y",
#         include_xt: bool = False,
#         make_xplus: bool = True,
#         xplus_max_replacements: int = 2,
#         deterministic_xplus: bool = True,
#     ):
#         super().__init__()
#         self.train_path = train_path
#         self.tokenizer_name = tokenizer_name
#         self.batch_size = batch_size
#         self.max_length = max_length
#         self.num_workers = num_workers

#         self.y_key = y_key
#         self.include_xt = include_xt
#         self.make_xplus = make_xplus
#         self.xplus_max_replacements = xplus_max_replacements
#         self.deterministic_xplus = deterministic_xplus

#         self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
#         if self.tokenizer.pad_token is None:
#             # For tokenizers like GPT2
#             self.tokenizer.pad_token = self.tokenizer.eos_token

#         self.train_ds: Optional[MSRGistDataset] = None

#     def setup(self, stage: Optional[str] = None) -> None:
#         self.train_ds = MSRGistDataset(
#             path=self.train_path,
#             y_key=self.y_key,
#             include_xt=self.include_xt,
#             make_xplus=self.make_xplus,
#             xplus_max_replacements=self.xplus_max_replacements,
#             deterministic_xplus=self.deterministic_xplus,
#         )

#     def _tok(self, texts: List[str]) -> Dict[str, torch.Tensor]:
#         return self.tokenizer(
#             texts,
#             padding=True,
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt",
#         )

#     def _collate(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
#         ids = [it["id"] for it in items]
#         xs = [it["x"] for it in items]
#         xps = [it["x_plus"] for it in items]
#         ys = [it["y"] for it in items]

#         x_tok = self._tok(xs)
#         xp_tok = self._tok(xps)
#         y_tok = self._tok(ys)

#         # labels for seq2seq style training: ignore pad tokens
#         labels = y_tok["input_ids"].clone()
#         labels[labels == self.tokenizer.pad_token_id] = -100

#         batch = {
#             "ids": ids,
#             "x_input_ids": x_tok["input_ids"],
#             "x_attention_mask": x_tok["attention_mask"],
#             "xplus_input_ids": xp_tok["input_ids"],
#             "xplus_attention_mask": xp_tok["attention_mask"],
#             "labels": labels,
#             # keep raw strings for debugging
#             "raw_x": xs,
#             "raw_x_plus": xps,
#             "raw_y": ys,
#         }

#         if self.include_xt:
#             batch["raw_xt"] = [it.get("xt", []) for it in items]

#         return batch

#     def train_dataloader(self) -> DataLoader:
#         if self.train_ds is None:
#             raise RuntimeError("Call setup() before requesting dataloader")
#         return DataLoader(
#             self.train_ds,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers,
#             collate_fn=self._collate,
#             pin_memory=True,
#         )

import json
import os
from typing import Optional, Dict

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .msr_gist_dataset import MSRGistDataset
from .xplus_gemma import GemmaXPlusGenerator, DEFAULT_PROMPT_TEMPLATE


def _load_xplus_cache_jsonl(cache_path: str, id_key: str = "id", xplus_key: str = "x_plus") -> Dict[str, str]:
    cache: Dict[str, str] = {}
    if not cache_path or not os.path.exists(cache_path):
        return cache
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cache[str(obj[id_key])] = str(obj[xplus_key])
    return cache


class MSRGistDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        tokenizer_name: str = "bert-base-uncased",
        batch_size: int = 4,
        max_length: int = 128,
        num_workers: int = 0,
        y_key: str = "summary",
        include_xt: bool = False,

        # x+ on-the-fly settings
        make_xplus: bool = True,
        gemma_model_dir: Optional[str] = None,
        xplus_cache_jsonl: str = "data/processed/xplus_cache.jsonl",
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        deterministic_xplus: bool = False,  # optional future: seed
    ):
        super().__init__()
        self.train_path = train_path
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.y_key = y_key
        self.include_xt = include_xt

        self.make_xplus = make_xplus
        self.gemma_model_dir = gemma_model_dir
        self.xplus_cache_jsonl = xplus_cache_jsonl
        self.prompt_template = prompt_template
        self.deterministic_xplus = deterministic_xplus

        self.tokenizer = None
        self.xplus_generator = None
        self.xplus_cache = {}

        self.train_dataset = None

    def prepare_data(self):
        # tokenizer downloads only if not present; should already be ok
        AutoTokenizer.from_pretrained(self.tokenizer_name)

    def setup(self, stage: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        # IMPORTANT: for Gemma generation do NOT use num_workers>0 unless you redesign multiprocessing
        if self.make_xplus and self.num_workers != 0:
            raise ValueError("For on-the-fly Gemma x+, set num_workers=0 (single process).")

        # Load cache (resume)
        if self.make_xplus and self.xplus_cache_jsonl:
            self.xplus_cache = _load_xplus_cache_jsonl(self.xplus_cache_jsonl)

        # Load Gemma generator once
        if self.make_xplus:
            if not self.gemma_model_dir:
                raise ValueError("make_xplus=True but gemma_model_dir not provided.")
            self.xplus_generator = GemmaXPlusGenerator(model_dir=self.gemma_model_dir)

        self.train_dataset = MSRGistDataset(
            data_path=self.train_path,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            y_key=self.y_key,
            include_xt=self.include_xt,
            xplus_generator=self.xplus_generator if self.make_xplus else None,
            xplus_cache=self.xplus_cache if self.make_xplus else {},
            prompt_template=self.prompt_template,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def flush_xplus_cache_to_jsonl(self, id_key: str = "id", xplus_key: str = "x_plus"):
        """
        Call occasionally if you want a persistent cache file.
        Writes only items not already present in the JSONL.
        """
        if not self.make_xplus or not self.xplus_cache_jsonl:
            return

        os.makedirs(os.path.dirname(self.xplus_cache_jsonl), exist_ok=True)

        existing = _load_xplus_cache_jsonl(self.xplus_cache_jsonl, id_key=id_key, xplus_key=xplus_key)

        new_items = 0
        with open(self.xplus_cache_jsonl, "a", encoding="utf-8") as f:
            for k, v in self.train_dataset.xplus_cache.items():
                if k in existing:
                    continue
                f.write(json.dumps({id_key: k, xplus_key: v}, ensure_ascii=False) + "\n")
                new_items += 1

        if new_items > 0:
            print(f"[xplus-cache] appended {new_items} new entries -> {self.xplus_cache_jsonl}")
