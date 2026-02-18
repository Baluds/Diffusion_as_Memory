# import json
# import re
# import random
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple

# from torch.utils.data import Dataset

# # Optional synonym backend
# try:
#     from nltk.corpus import wordnet as wn
#     _HAS_WORDNET = True
# except Exception:
#     wn = None
#     _HAS_WORDNET = False


# def normalize_text(s: str) -> str:
#     if s is None:
#         return ""
#     s = s.replace("\u00a0", " ")
#     s = re.sub(r"\s+", " ", s).strip()
#     return s


# _STOPWORDS = {
#     "a","an","the","and","or","but","if","then","so","because","as","of","at","by","for","with","about","against",
#     "between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","off",
#     "over","under","again","further","once","here","there","when","where","why","how","all","any","both","each","few",
#     "more","most","other","some","such","no","nor","not","only","own","same","than","too","very","can","will","just",
#     "is","am","are","was","were","be","been","being","have","has","had","do","does","did","i","you","he","she","it",
#     "we","they","me","him","her","them","my","your","his","their","our"
# }


# def _tokenize_words(text: str) -> List[str]:
#     # simple tokenization that keeps punctuation as separate tokens
#     return re.findall(r"[A-Za-z']+|[0-9]+|[^\w\s]", text)


# def _is_candidate_word(w: str) -> bool:
#     wl = w.lower()
#     if wl in _STOPWORDS:
#         return False
#     if not re.fullmatch(r"[A-Za-z']+", w):
#         return False
#     if len(w) <= 3:
#         return False
#     return True


# def _wordnet_synonyms(word: str) -> List[str]:
#     if not _HAS_WORDNET:
#         return []
#     syns: List[str] = []
#     for synset in wn.synsets(word):
#         for lemma in synset.lemmas():
#             s = lemma.name().replace("_", " ")
#             if s.lower() != word.lower():
#                 syns.append(s)
#     # unique, keep order
#     out = []
#     seen = set()
#     for s in syns:
#         sl = s.lower()
#         if sl not in seen:
#             seen.add(sl)
#             out.append(s)
#     return out


# def make_x_plus(
#     x: str,
#     max_replacements: int = 2,
#     seed: Optional[int] = None,
# ) -> str:
#     """
#     Meaning-preserving-ish synonym perturbation.

#     Strategy:
#     - pick up to `max_replacements` candidate words (non-stopword, alphabetic, len>3)
#     - replace with a WordNet synonym if available
#     - try multiple times to ensure x_plus != x when possible

#     If WordNet is unavailable or no good synonym found, returns x unchanged.
#     """
#     x = normalize_text(x)
#     if not x:
#         return x

#     words = _tokenize_words(x)
#     candidate_positions = [i for i, w in enumerate(words) if _is_candidate_word(w)]
#     if not candidate_positions:
#         return x

#     rng = random.Random(seed)

#     # We try a few attempts to actually change something
#     for _attempt in range(5):
#         new_words = words[:]
#         rng.shuffle(candidate_positions)

#         replaced = 0
#         for idx in candidate_positions:
#             if replaced >= max_replacements:
#                 break
#             w = new_words[idx]
#             syns = _wordnet_synonyms(w)
#             if not syns:
#                 continue

#             # pick synonym close-ish: single token preferred
#             syns_sorted = sorted(syns, key=lambda s: (len(s.split()), len(s)))
#             chosen = None
#             for s in syns_sorted:
#                 # avoid multiword replacements that can distort grammar too early
#                 if len(s.split()) == 1:
#                     chosen = s
#                     break
#             if chosen is None:
#                 chosen = syns_sorted[0]

#             # preserve capitalization
#             if w[0].isupper():
#                 chosen = chosen[0].upper() + chosen[1:]

#             if chosen.lower() != w.lower():
#                 new_words[idx] = chosen
#                 replaced += 1

#         x_plus = "".join(
#             [(" " + t) if (i > 0 and re.fullmatch(r"[A-Za-z0-9']+", t)) else t for i, t in enumerate(new_words)]
#         ).strip()

#         x_plus = normalize_text(x_plus)
#         if x_plus and x_plus.lower() != x.lower():
#             return x_plus

#     return x


# class MSRGistDataset(Dataset):
#     """
#     Reads cleaned JSON list with keys:
#       - id
#       - x
#       - y  (or summary, but we standardize to y via y_key)
#       - xt (optional list)
#     Produces dict with raw strings.
#     """

#     def __init__(
#         self,
#         path: str,
#         y_key: str = "y",
#         include_xt: bool = False,
#         make_xplus: bool = True,
#         xplus_max_replacements: int = 2,
#         deterministic_xplus: bool = True,
#     ):
#         self.path = Path(path)
#         self.y_key = y_key
#         self.include_xt = include_xt
#         self.make_xplus = make_xplus
#         self.xplus_max_replacements = xplus_max_replacements
#         self.deterministic_xplus = deterministic_xplus

#         self.data: List[Dict[str, Any]] = self._load(self.path)

#     def _load(self, path: Path) -> List[Dict[str, Any]]:
#         if path.suffix != ".json":
#             raise ValueError(f"Expected .json file, got {path.suffix}")
#         obj = json.loads(path.read_text(encoding="utf-8"))
#         if not isinstance(obj, list):
#             raise ValueError("Expected a JSON list of records")
#         return obj

#     def __len__(self) -> int:
#         return len(self.data)

#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         r = self.data[idx]
#         rid = str(r.get("id", idx))

#         x = normalize_text(r.get("x", ""))
#         y = normalize_text(r.get(self.y_key, ""))

#         # x+ generated on the fly
#         if self.make_xplus:
#             seed = None
#             if self.deterministic_xplus:
#                 # deterministic per sample id so debugging is stable
#                 seed = abs(hash(rid)) % (2**31)
#             x_plus = make_x_plus(x, max_replacements=self.xplus_max_replacements, seed=seed)
#         else:
#             x_plus = x

#         out = {"id": rid, "x": x, "x_plus": x_plus, "y": y}

#         if self.include_xt:
#             xt = r.get("xt", [])
#             if isinstance(xt, list):
#                 out["xt"] = [normalize_text(t) for t in xt if isinstance(t, str)]
#             else:
#                 out["xt"] = []

#         return out


import json
import os
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset


def _load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    # .json assumed to be a list
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"Expected a list in JSON file: {path}")
    return obj


class MSRGistDataset(Dataset):
    """
    Produces:
      - raw_x: str
      - raw_x_plus: str (generated on the fly if generator provided)
      - x_input_ids, x_attention_mask
      - xplus_input_ids, xplus_attention_mask
      - labels (token ids for y)
      - id (as string)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 128,
        y_key: str = "summary",
        include_xt: bool = False,
        xplus_generator=None,
        xplus_cache: Optional[Dict[str, str]] = None,
        prompt_template: Optional[str] = None,
    ):
        self.data_path = data_path
        self.rows = _load_json_or_jsonl(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.y_key = y_key
        self.include_xt = include_xt

        self.xplus_generator = xplus_generator
        self.xplus_cache = xplus_cache if xplus_cache is not None else {}
        self.prompt_template = prompt_template

        # Ensure pad token exists
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __len__(self):
        return len(self.rows)

    def _get_id(self, ex: Dict[str, Any]) -> str:
        # Your data had ids like "36_37" or numbers; normalize
        _id = ex.get("id", "")
        return str(_id)

    def _get_x(self, ex: Dict[str, Any]) -> str:
        return str(ex.get("x", ""))

    def _get_y(self, ex: Dict[str, Any]) -> str:
        # Some of your earlier scripts used "y"; others used "summary"
        val = ex.get(self.y_key, "")
        return str(val)

    def _get_xplus(self, ex_id: str, x: str) -> str:
        # 1) cache hit
        if ex_id in self.xplus_cache:
            return self.xplus_cache[ex_id]

        # 2) generate
        if self.xplus_generator is None:
            # fallback: if generator not provided, just return x
            xplus = x
        else:
            if self.prompt_template is None:
                xplus = self.xplus_generator.generate_xplus(x)
            else:
                xplus = self.xplus_generator.generate_xplus(x, prompt_template=self.prompt_template)

        self.xplus_cache[ex_id] = xplus
        return xplus

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.rows[idx]
        ex_id = self._get_id(ex)

        x = self._get_x(ex)
        y = self._get_y(ex)

        xplus = self._get_xplus(ex_id, x)

        # Tokenize x
        x_tok = self.tokenizer(
            x,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Tokenize x+
        xp_tok = self.tokenizer(
            xplus,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Tokenize y as labels (seq2seq-style labels)
        y_tok = self.tokenizer(
            y,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "id": ex_id,
            "raw_x": x,
            "raw_x_plus": xplus,
            "x_input_ids": x_tok["input_ids"].squeeze(0),
            "x_attention_mask": x_tok["attention_mask"].squeeze(0),
            "xplus_input_ids": xp_tok["input_ids"].squeeze(0),
            "xplus_attention_mask": xp_tok["attention_mask"].squeeze(0),
            "labels": y_tok["input_ids"].squeeze(0),
        }

        if self.include_xt:
            item["xt"] = ex.get("xt", None)

        return item
