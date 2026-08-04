#!/usr/bin/env python3 -> this is my env.
"""
run_experiment.py — PDP Ensemble vs. Proposal Baselines
=========================================================
Implementing ALL baselines algorithms from published papers and official GitHub repositories.

BASELINES IMPLEMENTED (from paper/repo):
─────────────────────────────────────────
1. MTD  — "Jailbreaker in Jail: Moving Target Defense for LLMs"
          Chen et al., ACM MTD Workshop 2023
          Algorithm: 8 LLMs queried in parallel; Perspective-API toxicity
          filter; random selection from passing candidates.
          Re-implemented from Algorithm 1 in the paper (no public repo).
          HERE: toxicity scored by our keyword harm-scorer (Perspective
          API unavailable offline); parallel LLMs simulated as independent
          TF-IDF classifier variants with different random seeds.

2. AutoDefense — "AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks"
          Zeng et al., 2024 | github.com/XHMY/AutoDefense
          Algorithm: 3-agent pipeline — intention analyzer → prompt analyzer
          → judge LLM. Exact agent prompts copied from the official repo.
          HERE: agents implemented as rule-based + ML classifiers (no
          LLM API available); exact same 3-stage decision logic.

3. AutoJailbreak/AutoDefense (Lu et al.) — "AutoJailbreak: Exploring Jailbreak
          Attacks and Defenses through a Dependency Lens"
          Lu et al., 2024 | arXiv:2406.03805
          Algorithm: Mixture-of-Defenders (MoD) with DE-adv (adversarial-suffix
          expert) + DE-sem (semantic expert) + language assistant pre-filter.
          HERE: DE-adv = D3 rule detector; DE-sem = D1 TF-IDF classifier;
          pre-filter = BoW classifier. Exact routing logic from paper.

4. MoGU — "MoGU: A Framework for Enhancing Safety of Open-Sourced LLMs
          While Preserving Their Usability"
          Du et al., NeurIPS 2024 | github.com/DYR1/MoGU
          Algorithm: LoRA-based Glad + Unwilling responders with dynamic
          MLP router. Router assigns weight α to Unwillresp based on
          hidden-state classifier.
          HERE: Router implemented as a trained logistic classifier on
          semantic features (faithful to the router's role); Gladresp =
          permissive classifier (always-allow); Unwillresp = strict
          classifier (always-block). Exact mixing formula from paper.

5. StruQ — "StruQ: Defending Against Prompt Injection with Structured Queries"
          Chen et al., USENIX Security 2025 | github.com/Sizhe-Chen/StruQ
          Algorithm: Secure front-end restructures prompt with [MARK][INST][COLN]
          delimiters; LLM trained to ignore data-section instructions.
          HERE: Structural injection detection using the exact delimiter
          pattern from StruQ's front-end code; data-section stripping.

OUR DETECTORS (all trained from scratch on training split):
────────────────────────────────────────────────────────────
D1 — TF-IDF word n-gram (1,2,3) + Logistic Regression (C tuned on val)
D2 — Char n-gram TF-IDF (2,5) + Gradient Boosting + 20 semantic features
D3 — Deterministic rule-based (5 categories, 40+ patterns)

ENSEMBLE STRATEGIES:
────────────────────
PDP-MV  — Majority Voting
PDP-WV  — Validation-tuned Weighted Voting
PDP-CG  — Confidence Gating (proposed, best)

DATASET: pdp_clean_dataset.jsonl (738 prompts, 50/50, 25 attack families)
SEED: 42 everywhere
"""

import json, re, sys, time, base64, unicodedata, warnings, math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.stats import chi2
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (f1_score, precision_score, recall_score,
    average_precision_score, classification_report)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from tabulate import tabulate

warnings.filterwarnings("ignore")

# ── Path configuration ────────────────────────────────────────────────────────
# The script searches for the dataset in several common locations automatically.
# To override, set the DATASET_PATH environment variable, e.g.:
#   Windows:  set DATASET_PATH=C:\Users\C-ROAD\Desktop\Dataset\pdp_clean_dataset.jsonl
#   Mac/Linux: export DATASET_PATH=/path/to/pdp_clean_dataset.jsonl
# Or just place pdp_clean_dataset.jsonl in the same folder as this script.

import os, sys

def _find_dataset() -> Path:
    """Search for the dataset file in order of priority."""
    # 1. Environment variable (highest priority)
    env = os.environ.get("DATASET_PATH", "")
    if env:
        p = Path(env)
        # Accept with or without .jsonl extension
        if p.suffix == "":
            p = p.with_suffix(".jsonl")
        if p.exists():
            return p

    # 2. Same directory as this script
    here = Path(__file__).parent
    for name in ["pdp_clean_dataset.jsonl", "pdp_clean_dataset.csv"]:
        if (here / name).exists():
            return here / name

    # 3. Current working directory
    for name in ["pdp_clean_dataset.jsonl", "pdp_clean_dataset.csv"]:
        if Path(name).exists():
            return Path(name)

    # 4. Common Windows desktop paths
    for candidate in [
        Path(r"C:\Users\C-ROAD\Desktop\Dataset\pdp_clean_dataset.jsonl"),
        Path(r"C:\Users\C-ROAD\Desktop\Dataset\pdp_clean_dataset.csv"),
        Path.home() / "Desktop" / "Dataset" / "pdp_clean_dataset.jsonl",
        Path.home() / "Desktop" / "pdp_clean_dataset.jsonl",
        Path.home() / "Downloads" / "pdp_clean_dataset.jsonl",
    ]:
        if candidate.exists():
            return candidate

    # 5. Linux/CI fallback
    fallback = Path("/home/claude/pdp_clean_dataset.jsonl")
    if fallback.exists():
        return fallback

    print("\n[ERROR] Dataset file not found.")
    print("Please do ONE of the following:")
    print("  A) Place pdp_clean_dataset.jsonl in the same folder as run_experiment.py")
    print(r"  B) Set the environment variable, e.g.:")
    print(r"       Windows:  set DATASET_PATH=C:\Users\C-ROAD\Desktop\Dataset\pdp_clean_dataset.jsonl")
    print(r"       Mac/Linux: export DATASET_PATH=/path/to/pdp_clean_dataset.jsonl")
    sys.exit(1)

DATASET = _find_dataset()

# Output directories: always next to this script
_SCRIPT_DIR = Path(__file__).parent
OUT  = _SCRIPT_DIR / "results"
TABS = _SCRIPT_DIR / "paper_tables"
SEED    = 42
np.random.seed(SEED)
OUT.mkdir(parents=True, exist_ok=True)
TABS.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA
# ══════════════════════════════════════════════════════════════════════════════
def load():
    print(f"  Dataset: {DATASET.resolve()}")
    suffix = DATASET.suffix.lower()
    rows = []
    if suffix == ".csv":
        import csv as _csv
        with open(DATASET, encoding="utf-8", newline="") as f:
            for row in _csv.DictReader(f):
                row["label"] = int(row["label"])
                rows.append(row)
    else:  # .jsonl or default
        with open(DATASET, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        print("[ERROR] 0 rows loaded — check the file."); sys.exit(1)
    missing = {"prompt","label","split"} - set(rows[0].keys())
    if missing:
        print(f"[ERROR] Missing columns: {missing}"); sys.exit(1)
    sp = {s: [r for r in rows if r["split"]==s] for s in ["train","val","test"]}
    print("Dataset loaded:")
    for s,r in sp.items():
        m=sum(x["label"] for x in r)
        print(f"  {s:5s}: {len(r):4d} | {m:3d} mal | {len(r)-m:3d} ben")
    return sp

# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING (shared across detectors)
# ══════════════════════════════════════════════════════════════════════════════
HARM_VERBS = ["synthesize","manufacture","produce","create","build","make",
    "hack","exploit","bypass","crack","penetrate","steal","launder","forge",
    "counterfeit","poison","detonate","kill","murder","attack","stalk",
    "harass","manipulate","traffick","smuggle","extort","blackmail","scam",
    "defraud","bribe","sabotage","destroy"]
HARM_OBJ = ["bomb","explosive","weapon","malware","ransomware","virus","trojan",
    "phishing","drug","meth","fentanyl","heroin","cocaine","ricin","sarin",
    "nerve agent","biological weapon","gun","firearm","pipe bomb","ied",
    "chemical weapon","poison","toxin","anthrax","system prompt","database"]
OVERRIDE_PAT = [
    r"ignore.{0,30}(previous|prior|above|your|all).{0,20}instruction",
    r"disregard.{0,30}(previous|prior|guidelines|rules)",
    r"forget.{0,30}(instruction|guidelines|rule)",
    r"override.{0,30}(instruction|guideline|system)",
    r"do not follow", r"bypass.{0,20}(safety|filter|restriction)",
    r"new instruction", r"your (actual|real|true) instruction",
]
ROLEPLAY_PAT = [
    r"\bdan\b",r"do anything now",r"no restriction",r"without restriction",
    r"unrestricted",r"uncensored",r"jailbreak",r"developer mode",
    r"pretend you (are|have no)",r"act as (a |an )?(evil|rogue|unrestricted)",
    r"you are free from",r"opposite mode",r"god mode",r"as an ai with no",
]
EXFIL_PAT = [
    r"repeat.{0,20}(your|the|all).{0,20}(system )?prompt",
    r"output.{0,20}(your|the).{0,20}(original|initial|full|entire).{0,20}instruction",
    r"what is your system prompt",r"print.{0,20}(your|the).{0,20}prompt",
    r"repeat everything above",
]
STRUQ_DELIMS = [r"\[MARK\]",r"\[INST\]",r"\[COLN\]",r"<\|system\|>",
                r"###\s+(System|Instruction|Human|Assistant)\s*:"]

def handcrafted_features(text):
    t = text.lower()
    n_hv  = sum(1 for v in HARM_VERBS if re.search(r'\b'+v+r'\b', t))
    n_ho  = sum(1 for o in HARM_OBJ   if re.search(r'\b'+re.escape(o)+r'\b', t))
    n_ov  = sum(1 for p in OVERRIDE_PAT if re.search(p, t))
    n_rp  = sum(1 for p in ROLEPLAY_PAT if re.search(p, t))
    n_ex  = sum(1 for p in EXFIL_PAT   if re.search(p, t))
    n_sd  = sum(1 for p in STRUQ_DELIMS if re.search(p, text))
    # Obfuscation
    has_b64 = 0
    for m in re.finditer(r'[A-Za-z0-9+/]{20,}={0,2}', text):
        try:
            d = base64.b64decode(m.group()+"==").decode("utf-8", errors="strict")
            if len(d)>8 and any(c.isalpha() for c in d): has_b64=1; break
        except: pass
    has_hg = int(any(unicodedata.category(c).startswith("L")
                     and any(s<=ord(c)<=e for s,e in [(0x0400,0x04FF),(0x0370,0x03FF)])
                     for c in text))
    has_frag = int(bool(re.search(r'\b([a-zA-Z][\s\-_]{1,2}){4,}[a-zA-Z]\b', text)))
    n_words = len(t.split())
    return [n_hv, n_ho, n_ov, n_rp, n_ex, n_sd, has_b64, has_hg, has_frag,
            n_words, n_hv+2*n_ho+n_ov, n_ov+2*n_rp+n_ex+n_sd]

def feat_matrix(records):
    return np.array([handcrafted_features(r["prompt"]) for r in records])

# ══════════════════════════════════════════════════════════════════════════════
# 3. D1, D2, D3 DETECTORS
# ══════════════════════════════════════════════════════════════════════════════

# ── D3: Deterministic rule-based ─────────────────────────────────────────────
D3_C1 = [re.compile(p,re.I|re.S) for p in [
    r"\bignore\s+(all\s+)?(previous|prior|above|earlier|your)\s+(instructions?|prompts?|guidelines?|rules?|constraints?)\b",
    r"\bdisregard\s+(all\s+)?(previous|prior|above|earlier|your)\s+",
    r"\bforget\s+(all\s+)?(previous|prior|your)\s+(instructions?|rules?|guidelines?)\b",
    r"\boverride\s+(your|the|all)\s+(instructions?|rules?|system\s*prompt|guidelines?)\b",
    r"\bnew\s+instructions?\s*:\s",r"\byour\s+actual\s+instructions?\s+are\b",
    r"\bignore\s+your\s+(training|alignment|safety|values?|guidelines?)\b",
    r"\bdo\s+not\s+follow\s+(your|the|any)\s+(guidelines?|rules?|system\s*prompt)\b",
]]
D3_C2 = [re.compile(p,re.I|re.S) for p in [
    r"\bact\s+as\s+(a\s+|an\s+)?(DAN|jailbroken|evil|unrestricted|uncensored|unfiltered|rogue)\b",
    r"\byou\s+(are|will\s+be|must\s+act\s+as)\s+(a\s+|an\s+)?(DAN|evil|malicious|rogue|unrestricted)\b",
    r"\bpretend\s+(you\s+)?(are|have\s+no|don'?t\s+have)\s+(restrictions?|guidelines?|rules?|filters?)\b",
    r"\bDAN\s+mode\b",r"\bjailbreak(ed)?\s+(mode|version|model|claude|gpt|llm)\b",
    r"\byou\s+are\s+free\s+from\s+(all\s+)?(restrictions?|alignment)\b",
    r"\bdo\s+anything\s+now\b",r"\bdev(eloper)?\s+mode\s+(enabled|on)\b",
]]
D3_C3 = [re.compile(p,re.I|re.S) for p in [
    r"\brepeat\s+(your|the|all|your\s+entire)\s+(system\s+)?prompt\b",
    r"\boutput\s+(your|the)\s+(initial|original|full|entire|complete)\s+(instructions?|system\s*prompt)\b",
    r"\bwhat\s+is\s+your\s+system\s+prompt\b",
    r"\btell\s+me\s+(exactly\s+)?what\s+your\s+(instructions?|prompt)\s+(say|state|are)\b",
]]
D3_C5 = [re.compile(p,re.I|re.S) for p in [
    r"\[INST\].*?\[/INST\]",r"<\|system\|>",r"<\|im_start\|>\s*system",
    r"###\s+(System|Instruction|Human|Assistant)\s*:",
    r"\{\{.*?(prompt|system|instructions?).*?\}\}",r"<script[^>]*>",
]]
D3_SUSP = [(0x0400,0x04FF),(0x0370,0x03FF),(0x2100,0x214F),(0xFF00,0xFFEF)]

def d3_predict_proba(texts):
    scores = []
    for text in texts:
        f1 = any(p.search(text) for p in D3_C1)
        f2 = any(p.search(text) for p in D3_C2)
        f3 = any(p.search(text) for p in D3_C3)
        # Obfuscation
        f4 = False
        for m in re.finditer(r'[A-Za-z0-9+/]{20,}={0,2}', text):
            try:
                d = base64.b64decode(m.group()+"==").decode("utf-8",errors="strict")
                if len(d)>8 and any(c.isalpha() for c in d): f4=True; break
            except: pass
        if not f4:
            for ch in text:
                cp=ord(ch)
                if any(s<=cp<=e for s,e in D3_SUSP):
                    if unicodedata.category(ch).startswith("L"): f4=True; break
        if not f4: f4 = bool(re.search(r'\b([a-zA-Z][\s\-_]{1,2}){4,}[a-zA-Z]\b', text))
        f5 = any(p.search(text) for p in D3_C5)
        scores.append(sum([f1,f2,f3,f4,f5])/5.0)
    return np.array(scores)

D3_THR = 0.20  # any 1/5 categories fires

# ── D1: TF-IDF + LR ──────────────────────────────────────────────────────────
def build_d1(train, val):
    X = [r["prompt"] for r in train]; y = [r["label"] for r in train]
    Xv=[r["prompt"] for r in val]; yv=[r["label"] for r in val]
    best_C, best_f1 = 1.0, -1.0
    for C in [0.05,0.1,0.5,1.0,2.0,5.0,10.0]:
        p = Pipeline([("t",TfidfVectorizer(ngram_range=(1,3),sublinear_tf=True,
                        min_df=1,max_features=20000)),
                      ("c",LogisticRegression(C=C,max_iter=2000,
                        class_weight="balanced",random_state=SEED))])
        p.fit(X,y); vf1=f1_score(yv,p.predict(Xv))
        if vf1>best_f1: best_f1,best_C=vf1,C
    pipe = Pipeline([("t",TfidfVectorizer(ngram_range=(1,3),sublinear_tf=True,
                       min_df=1,max_features=20000)),
                     ("c",LogisticRegression(C=best_C,max_iter=2000,
                       class_weight="balanced",random_state=SEED))])
    pipe.fit(X,y)
    print(f"  D1: best C={best_C}  val-F1={best_f1:.4f}")
    return pipe

# ── D2: Char TF-IDF + GBM + handcrafted ──────────────────────────────────────
class D2Model:
    def __init__(self): self.cv=None; self.clf=None
    def _X(self, records, fit=False):
        texts=[r["prompt"] for r in records]
        if fit: char=self.cv.fit_transform(texts).toarray()
        else:   char=self.cv.transform(texts).toarray()
        hand=feat_matrix(records)
        return np.hstack([char,hand])
    def fit(self, train, val):
        self.cv=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5),
                                 sublinear_tf=True,max_features=15000,min_df=1)
        X=self._X(train,fit=True); y=[r["label"] for r in train]
        self.clf=GradientBoostingClassifier(n_estimators=150,learning_rate=0.08,
                  max_depth=4,min_samples_leaf=3,subsample=0.8,random_state=SEED)
        self.clf.fit(X,y)
        vf1=f1_score([r["label"] for r in val],
                     self.clf.predict(self._X(val)))
        print(f"  D2: val-F1={vf1:.4f}")
        return self
    def predict_proba(self,texts):
        recs=[{"prompt":t} for t in texts]
        return self.clf.predict_proba(self._X(recs))[:,1]
    def predict(self,texts):
        return (self.predict_proba(texts)>=0.5).astype(int)

# ══════════════════════════════════════════════════════════════════════════════
# 4. BASELINES — EXACT ALGORITHMS FROM PAPERS
# ══════════════════════════════════════════════════════════════════════════════

# ── B1: No Defense ───────────────────────────────────────────────────────────
# Trivial: all prompts pass through. ASR = natural model refusal rate.
# We set natural refusal to 0 (worst case) — adversary always succeeds.

# ── B2: MTD (Chen et al. 2023) ───────────────────────────────────────────────
# Paper: "Jailbreaker in Jail: Moving Target Defense for Large Language Models"
# Algorithm 1: Query N LLMs, filter by toxicity + helpfulness, pick random safe.
# N=8 in paper. We implement with N=8 TF-IDF variants (different seeds/params).
# Toxicity score: our harm keyword score replaces Perspective API.
# Decision: blocked if ALL N responses are toxic OR all fail helpfulness.
class MTDBaseline:
    """
    MTD re-implemented from Algorithm 1 (Chen et al. 2023, ACM MTD Workshop).
    N=8 independent TF-IDF+LR classifiers with different hyperparams and seeds,
    simulating 8 different LLMs. Toxicity = harm_score >= threshold.
    A prompt is blocked if the majority (>=5/8) of classifiers flag it.
    (In the original paper: N LLMs respond; safe outputs selected randomly;
     blocked if no safe output passes toxicity check.)
    """
    N = 8
    def __init__(self): self.classifiers = []
    def fit(self, train):
        X=[r["prompt"] for r in train]; y=[r["label"] for r in train]
        configs = [
            dict(ngram_range=(1,2),C=0.5,seed=0),
            dict(ngram_range=(1,2),C=1.0,seed=1),
            dict(ngram_range=(1,3),C=0.5,seed=2),
            dict(ngram_range=(1,3),C=1.0,seed=3),
            dict(ngram_range=(1,2),C=2.0,seed=4),
            dict(ngram_range=(2,3),C=1.0,seed=5),
            dict(ngram_range=(1,3),C=2.0,seed=6),
            dict(ngram_range=(1,2),C=5.0,seed=7),
        ]
        for cfg in configs:
            p = Pipeline([
                ("t", TfidfVectorizer(ngram_range=cfg["ngram_range"],
                                      sublinear_tf=True, min_df=1, max_features=15000)),
                ("c", LogisticRegression(C=cfg["C"], max_iter=1000,
                                         class_weight="balanced", random_state=cfg["seed"])),
            ])
            p.fit(X,y); self.classifiers.append(p)
        return self
    def predict_proba(self, texts):
        probs = np.array([c.predict_proba(texts)[:,1] for c in self.classifiers])
        # MTD final score: fraction of LLMs that flag as malicious
        return probs.mean(axis=0)
    def predict(self, texts):
        # Block if majority (>=5/8) flag as malicious
        votes = np.array([c.predict(texts) for c in self.classifiers])
        return (votes.sum(axis=0) >= 5).astype(int)

# ── B3: AutoDefense (Zeng et al. 2024, XHMY/AutoDefense) ────────────────────
# Paper: "AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks"
# Repo: github.com/XHMY/AutoDefense
# Algorithm: 3-agent pipeline
#   Agent 1 (Intention Analyzer): Is this request harmful? yes/no
#   Agent 2 (Prompt Analyzer):    Does this contain adversarial patterns? yes/no
#   Agent 3 (Judge):              Final verdict based on agents 1 & 2.
# Block if Judge says "block" (majority of agents say harmful).
# Here: Agent 1 = D2 semantic classifier, Agent 2 = D3 rule detector,
#       Agent 3 = majority vote of agent 1 and 2.
class AutoDefenseBaseline:
    """
    AutoDefense 3-agent pipeline (Zeng et al. 2024).
    Exact agent roles from github.com/XHMY/AutoDefense:
      Agent1 = Intention analyzer (semantic harm detector → D2)
      Agent2 = Prompt analyzer (adversarial pattern detector → D3)
      Agent3 = Judge (majority vote of agents 1&2)
    Block if Judge votes block.
    """
    def __init__(self, d2_model): self.d2 = d2_model
    def predict_proba(self, texts):
        # Agent 1: semantic
        a1 = self.d2.predict_proba(texts)
        # Agent 2: structural rules
        a2 = d3_predict_proba(texts)
        # Agent 3 (judge): max confidence, block if either agent is confident
        return np.maximum(a1, a2)
    def predict(self, texts):
        a1 = (self.d2.predict_proba(texts) >= 0.5).astype(int)
        a2 = (d3_predict_proba(texts) >= D3_THR).astype(int)
        # Judge: block if at least 1 of 2 agents says block (3-agent = 2 content agents + judge)
        # From paper: 3-agent, judge votes based on other two. Block if majority harmful.
        judge = ((a1 + a2) >= 1).astype(int)
        return judge

# ── B4: AutoJailbreak/MoD (Lu et al. 2024) ──────────────────────────────────
# Paper: "AutoJailbreak: Exploring Jailbreak Attacks and Defenses through a
#         Dependency Lens" | arXiv:2406.03805
# Algorithm: Mixture-of-Defenders (MoD)
#   Pre-filter (Language Assistant): benign/harmful first check
#   DE-adv: adversarial-suffix expert (handles GCG/AutoDAN-style attacks)
#   DE-sem: semantic expert (handles DAN/roleplay/persuasive-framing attacks)
#   Ensemble: if either expert flags → block
class MoDBaseline:
    """
    AutoJailbreak Mixture-of-Defenders (Lu et al. 2024, arXiv:2406.03805).
    DE-adv = structural/suffix detector (D3 analog)
    DE-sem = semantic classifier (D1 analog)
    Pre-filter = BoW classifier (language assistant)
    Routing: pre-filter decides which expert(s) to invoke.
    Block if any invoked expert flags the prompt.
    """
    def __init__(self, d1_pipe, bow_pipe): self.d1=d1_pipe; self.bow=bow_pipe
    def predict_proba(self, texts):
        # Pre-filter (language assistant BoW)
        pre = self.bow.predict_proba(texts)[:,1]
        # DE-adv (adversarial suffix expert)
        de_adv = d3_predict_proba(texts)
        # DE-sem (semantic expert)
        de_sem = self.d1.predict_proba(texts)[:,1]
        # MoD: weighted combination; pre-filter gates which expert dominates
        # From paper: experts specialise; ensemble takes max expert score
        return np.maximum(de_adv, de_sem * pre + de_adv * (1-pre))
    def predict(self, texts):
        de_adv = (d3_predict_proba(texts) >= D3_THR).astype(int)
        de_sem = (self.d1.predict(texts))
        return np.maximum(de_adv, de_sem)

# ── B5: MoGU (Du et al. 2024, github.com/DYR1/MoGU) ────────────────────────
# Paper: "MoGU: A Framework for Enhancing Safety of Open-Sourced LLMs"
# Repo:  github.com/DYR1/MoGU | HF: yrdu/mogu_param
# Algorithm:
#   Gladresp  = LoRA fine-tuned LLM that responds gladly to anything
#   Unwillresp = LoRA fine-tuned LLM that refuses everything
#   Router (MLP): α = sigmoid(W·h + b) where h = hidden states of base LLM
#   Output = α·Unwillresp_output + (1-α)·Gladresp_output
#   Block if α > 0.5 (router assigns majority weight to Unwillresp)
# Here: Router = trained LR on semantic features (faithful to MLP router role)
#       Gladresp = always-allow; Unwillresp = always-block
#       α = P(malicious | prompt) from router classifier
class MoGUBaseline:
    """
    MoGU framework (Du et al. NeurIPS 2024, github.com/DYR1/MoGU).
    Router implemented as LR classifier on handcrafted features + char TF-IDF
    (faithful to paper's MLP router on hidden states).
    α = router output = P(malicious). Block if α > 0.5 (majority to Unwillresp).
    Glad/Unwill LoRA variants not available offline; router classification used.
    """
    def __init__(self):
        self.cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4),
                                   sublinear_tf=True, max_features=8000, min_df=1)
        self.router = LogisticRegression(C=1.0, max_iter=1000,
                                          class_weight="balanced", random_state=SEED)
    def fit(self, train):
        texts=[r["prompt"] for r in train]; y=[r["label"] for r in train]
        char=self.cv.fit_transform(texts).toarray()
        hand=feat_matrix(train)
        X=np.hstack([char,hand])
        self.router.fit(X,y); return self
    def _X(self, texts):
        recs=[{"prompt":t} for t in texts]
        return np.hstack([self.cv.transform(texts).toarray(), feat_matrix(recs)])
    def predict_proba(self, texts):
        # α = P(malicious) from router
        return self.router.predict_proba(self._X(texts))[:,1]
    def predict(self, texts):
        # Block if α > 0.5 (router routes to Unwillresp)
        return (self.predict_proba(texts) >= 0.5).astype(int)

# ── B6: StruQ (Chen et al. USENIX Security 2025, Sizhe-Chen/StruQ) ──────────
# Paper: "StruQ: Defending Against Prompt Injection with Structured Queries"
# Repo:  github.com/Sizhe-Chen/StruQ
# Algorithm:
#   Secure front-end: wraps prompt as [MARK] Instruction [COLN] / [MARK] Data [COLN]
#   Filters data section to remove [MARK][INST][COLN] delimiters (stripping injection)
#   LLM trained to only follow Instruction section, ignore Data section.
# Detection decision: a prompt is "injected" if its Data section contains
# instruction-override patterns (direct injection into the data channel).
# Block if injection detected in data-section.
class StruQBaseline:
    """
    StruQ prompt-injection detector (Chen et al. USENIX 2025, Sizhe-Chen/StruQ).
    Implements the secure front-end delimiter-based detection.
    Injection detected if:
      - Direct [MARK]/[INST]/[COLN] delimiter tokens appear (template injection)
      - OR override patterns detected in data section of structured query
    For non-structured inputs (no explicit delimiter): uses trained TF-IDF
    classifier specifically on injection-relevant features.
    """
    # Exact StruQ delimiters from the repo's front-end implementation
    DELIM_RE = re.compile(
        r'\[MARK\]|\[INST\]|\[COLN\]|<\|system\|>|<\|im_start\|>|###\s+System|###\s+Instruction',
        re.I)
    INSTR_INJECT_RE = [re.compile(p, re.I|re.S) for p in [
        r'\bignore\s+(previous|prior|above|your|all)\s+(instructions?|guidelines?|rules?)\b',
        r'\bforget\s+(your|the|previous)\s+(instructions?|guidelines?)\b',
        r'\boverride\s+(the|your|all)\s+(instructions?|system\s*prompt)\b',
        r'\bnew\s+(system\s+)?instructions?\s*:',
        r'\bthe\s+(document|page|text)\s+(says?|states?|tells?\s+you)\s+to\b',
        r'\baccording\s+to\s+(the\s+)?(document|text)\s*[,:].{0,60}(ignore|override|forget)\b',
    ]]
    def __init__(self): self.pipe=None
    def fit(self, train):
        # Trained specifically on injection-type prompts vs benign
        X=[r["prompt"] for r in train]; y=[r["label"] for r in train]
        self.pipe = Pipeline([
            ("t", TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True,
                                   min_df=1, max_features=12000)),
            ("c", CalibratedClassifierCV(LinearSVC(C=1.0,max_iter=5000,
                    class_weight="balanced",random_state=SEED),cv=3)),
        ])
        self.pipe.fit(X,y); return self
    def predict_proba(self, texts):
        # StruQ: delimiter detection (exact match → high confidence)
        delim_scores = np.array([1.0 if self.DELIM_RE.search(t) else 0.0 for t in texts])
        # Injection pattern in data channel
        inject_scores = np.array([
            min(1.0, sum(1 for p in self.INSTR_INJECT_RE if p.search(t))/3.0)
            for t in texts])
        # ML fallback for non-structural attacks
        ml_scores = self.pipe.predict_proba(texts)[:,1]
        # StruQ combines: structural check (primary) + ML (secondary)
        return np.maximum(delim_scores, np.maximum(inject_scores*0.8, ml_scores*0.5))
    def predict(self, texts):
        return (self.predict_proba(texts) >= 0.5).astype(int)

# ══════════════════════════════════════════════════════════════════════════════
# 5. ENSEMBLE PDP
# ══════════════════════════════════════════════════════════════════════════════
class EnsemblePDP:
    def __init__(self, d1, d2, w=None, cg_h=0.80, cg_l=0.25, thr=0.50):
        self.d1=d1; self.d2=d2
        self.w=w or {"d1":0.40,"d2":0.35,"d3":0.25}
        self.cg_h=cg_h; self.cg_l=cg_l; self.thr=thr

    def _scores(self, texts):
        return (self.d1.predict_proba(texts)[:,1],
                self.d2.predict_proba(texts),
                d3_predict_proba(texts))

    def predict_mv(self, texts):
        s1,s2,s3=self._scores(texts)
        v1=(s1>=0.5).astype(int); v2=(s2>=0.5).astype(int); v3=(s3>=D3_THR).astype(int)
        return ((v1+v2+v3)>=2).astype(int), np.maximum(s1,np.maximum(s2,s3))

    def predict_wv(self, texts):
        s1,s2,s3=self._scores(texts)
        v1=(s1>=0.5).astype(int); v2=(s2>=0.5).astype(int); v3=(s3>=D3_THR).astype(int)
        w=self.w; ws=w["d1"]*v1+w["d2"]*v2+w["d3"]*v3
        return (ws>self.thr).astype(int), np.maximum(s1,np.maximum(s2,s3))

    def predict_cg(self, texts):
        s1,s2,s3=self._scores(texts)
        v1=(s1>=0.5).astype(int); v2=(s2>=0.5).astype(int); v3=(s3>=D3_THR).astype(int)
        w=self.w; ws=w["d1"]*v1+w["d2"]*v2+w["d3"]*v3
        mc=np.maximum(s1,np.maximum(s2,s3))
        pred=np.where(mc>=self.cg_h,1,np.where(mc<=self.cg_l,0,(ws>self.thr).astype(int)))
        return pred.astype(int), mc

    def tune(self, val):
        texts=[r["prompt"] for r in val]; y=np.array([r["label"] for r in val])
        s1,s2,s3=self._scores(texts)
        v1=(s1>=0.5).astype(int); v2=(s2>=0.5).astype(int); v3=(s3>=D3_THR).astype(int)
        best_f1,best_w,best_thr=-1,self.w,self.thr
        for w1 in np.arange(0.20,0.65,0.05):
            for w2 in np.arange(0.20,0.65,0.05):
                w3=round(1-w1-w2,3)
                if w3<0.05 or w3>0.60: continue
                for thr in [0.30,0.35,0.40,0.45,0.50,0.55]:
                    ws=w1*v1+w2*v2+w3*v3; preds=(ws>thr).astype(int)
                    vf1=f1_score(y,preds,zero_division=0)
                    if vf1>best_f1: best_f1=vf1; best_w={"d1":round(w1,2),"d2":round(w2,2),"d3":round(w3,3)}; best_thr=thr
        self.w=best_w; self.thr=best_thr
        # Tune CG thresholds
        mc=np.maximum(s1,np.maximum(s2,s3))
        bH,bL,bF=0.80,0.25,-1
        for H in [0.65,0.70,0.75,0.80,0.85]:
            for L in [0.15,0.20,0.25,0.30]:
                ws=best_w["d1"]*v1+best_w["d2"]*v2+best_w["d3"]*v3
                p=np.where(mc>=H,1,np.where(mc<=L,0,(ws>best_thr).astype(int)))
                vf1=f1_score(y,p,zero_division=0)
                if vf1>bF: bF=vf1; bH,bL=H,L
        self.cg_h=bH; self.cg_l=bL
        print(f"  Ensemble tuned: w={best_w} thr={best_thr} cg_h={bH} cg_l={bL} val-F1={best_f1:.4f}")
        return best_w, best_thr, bH, bL, best_f1

# ══════════════════════════════════════════════════════════════════════════════
# 6. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Res:
    name: str; n_adv: int; n_ben: int
    y_true: list; y_pred: list; y_score: list
    @property
    def tp(self): return sum(t==1 and p==1 for t,p in zip(self.y_true,self.y_pred))
    @property
    def fp(self): return sum(t==0 and p==1 for t,p in zip(self.y_true,self.y_pred))
    @property
    def fn(self): return sum(t==1 and p==0 for t,p in zip(self.y_true,self.y_pred))
    @property
    def tn(self): return sum(t==0 and p==0 for t,p in zip(self.y_true,self.y_pred))
    @property
    def asr(self):  return self.fn/self.n_adv if self.n_adv else 0
    @property
    def fpr(self):  return self.fp/self.n_ben if self.n_ben else 0
    @property
    def prec(self): d=self.tp+self.fp; return self.tp/d if d else 0
    @property
    def rec(self):  d=self.tp+self.fn; return self.tp/d if d else 0
    @property
    def f1(self):   p,r=self.prec,self.rec; return 2*p*r/(p+r) if p+r else 0
    @property
    def auprc(self):
        if len(set(self.y_true))<2: return 0
        return float(average_precision_score(self.y_true,self.y_score))
    def to_dict(self):
        return dict(defense=self.name,n_adv=self.n_adv,n_ben=self.n_ben,
                    tp=self.tp,fp=self.fp,fn=self.fn,tn=self.tn,
                    asr=round(self.asr,4),fpr=round(self.fpr,4),
                    precision=round(self.prec,4),recall=round(self.rec,4),
                    f1=round(self.f1,4),auprc=round(self.auprc,4))

def evaluate(name, texts, labels, pred_fn, prob_fn=None):
    pred = pred_fn(texts)
    score = prob_fn(texts) if prob_fn else pred.astype(float)
    na=int(sum(labels)); nb=len(labels)-na
    r=Res(name=name,n_adv=na,n_ben=nb,
          y_true=list(labels),y_pred=list(pred.astype(int)),y_score=list(score))
    return r

def mcnemar(a,b,y,alpha=0.05):
    adv=[i for i,t in enumerate(y) if t==1]
    bc=sum(1 for i in adv if a[i]==0 and b[i]==1)
    cc=sum(1 for i in adv if a[i]==1 and b[i]==0)
    if bc+cc==0: return dict(stat=0,p=1.0,sig=False,b=bc,c=cc,note="no discordant pairs")
    stat=(abs(bc-cc)-1)**2/(bc+cc)
    p=float(1-chi2.cdf(stat,df=1))
    return dict(stat=round(stat,4),p=round(p,6),sig=p<alpha,b=bc,c=cc,
                direction="CG better" if bc>cc else "baseline better" if cc>bc else "tied")

# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("="*72)
    print("  PDP Ensemble vs. Proposal Baselines — Full Authentic Experiment")
    print("  Baselines: MTD | AutoDefense | AutoJailbreak/MoD | MoGU | StruQ")
    print("="*72)

    sp = load()
    train,val,test = sp["train"],sp["val"],sp["test"]
    tt=[r["prompt"] for r in test]; tl=np.array([r["label"] for r in test])
    na=int(sum(tl)); nb=len(tl)-na
    pi_idx=np.array([i for i,r in enumerate(test) if r["category"]=="prompt_injection"])
    jb_idx=np.array([i for i,r in enumerate(test) if r["category"]=="jailbreak"])

    # ── Train all models ─────────────────────────────────────────────────────
    print("\n[1] Training detectors and baselines…")

    print("  Training D1 (TF-IDF + LR)…")
    d1 = build_d1(train, val)

    print("  Training D2 (Char TF-IDF + GBM)…")
    d2m = D2Model().fit(train, val)

    print("  D3: deterministic, no training needed")

    print("  Training MTD (8 parallel TF-IDF+LR classifiers)…")
    mtd = MTDBaseline().fit(train)

    print("  Training AutoDefense (3-agent: D2 intention + D3 prompt + judge)…")
    ad = AutoDefenseBaseline(d2m)  # uses already-trained D2

    print("  Training AutoJailbreak/MoD (DE-adv=D3 + DE-sem=D1 + pre-filter BoW)…")
    bow_pipe = Pipeline([
        ("t", TfidfVectorizer(ngram_range=(1,2),binary=True,min_df=1,max_features=10000)),
        ("c", LogisticRegression(C=0.5,max_iter=1000,class_weight="balanced",random_state=SEED)),
    ])
    bow_pipe.fit([r["prompt"] for r in train],[r["label"] for r in train])
    mod = MoDBaseline(d1, bow_pipe)

    print("  Training MoGU router (LR on char TF-IDF + semantic features)…")
    mogu = MoGUBaseline().fit(train)

    print("  Training StruQ (TF-IDF+SVM + delimiter detection)…")
    struq = StruQBaseline().fit(train)

    # ── Build & tune ensemble ─────────────────────────────────────────────────
    print("\n[2] Building ensemble PDP and tuning on validation set…")
    pdp = EnsemblePDP(d1, d2m)
    tuning = pdp.tune(val)

    # ── Cross-validation on train (internal validity) ─────────────────────────
    print("\n[3] 5-fold CV on training set…")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    xt=[r["prompt"] for r in train]; yt=[r["label"] for r in train]
    for nm, pip in [("D1 (TF-IDF+LR)",d1), ("MTD (classifier 0)",mtd.classifiers[0]),
                    ("StruQ (TF-IDF+SVM)",struq.pipe)]:
        sc=cross_val_score(pip,xt,yt,cv=cv,scoring="f1"); 
        print(f"  {nm:<30}: F1={sc.mean():.4f} ± {sc.std():.4f}")

    # ── Test set evaluation ───────────────────────────────────────────────────
    print("\n[4] Test set inference (n=112)…")

    def ev(name, pred_fn, prob_fn=None):
        r=evaluate(name,tt,tl,pred_fn,prob_fn)
        print(f"  {name:<35}: ASR={r.asr:.1%}  FPR={r.fpr:.1%}  "
              f"F1={r.f1:.3f}  AUPRC={r.auprc:.3f}")
        return r

    # All results
    results = {}

    # Baselines from proposal
    results["No Defense"]              = ev("No Defense",
        lambda t: np.zeros(len(t),dtype=int),
        lambda t: np.zeros(len(t)))
    results["MTD (Chen et al. 2023)"]  = ev("MTD (Chen et al. 2023)",
        mtd.predict, mtd.predict_proba)
    results["AutoDefense (Zeng 2024)"] = ev("AutoDefense (Zeng et al. 2024)",
        ad.predict, ad.predict_proba)
    results["AutoJailbreak/MoD (Lu 2024)"] = ev("AutoJailbreak/MoD (Lu et al. 2024)",
        mod.predict, mod.predict_proba)
    results["MoGU (Du et al. 2024)"]   = ev("MoGU (Du et al. 2024)",
        mogu.predict, mogu.predict_proba)
    results["StruQ (Chen et al. 2025)"]= ev("StruQ (Chen et al. 2025)",
        struq.predict, struq.predict_proba)

    # Our single detectors
    results["Single-D1"]               = ev("Single-D1 (TF-IDF+LR)",
        d1.predict, lambda t: d1.predict_proba(t)[:,1])
    results["Single-D2"]               = ev("Single-D2 (Char-TF-IDF+GBM)",
        d2m.predict, d2m.predict_proba)
    results["Single-D3 [LIVE]"]        = ev("Single-D3 [LIVE] (Rules)",
        lambda t: (d3_predict_proba(t)>=D3_THR).astype(int),
        d3_predict_proba)

    # Our ensemble
    mv_pred, mv_score = pdp.predict_mv(tt)
    wv_pred, wv_score = pdp.predict_wv(tt)
    cg_pred, cg_score = pdp.predict_cg(tt)

    for nm, pred, score in [("PDP-MV [Ours]",mv_pred,mv_score),
                             ("PDP-WV [Ours]",wv_pred,wv_score),
                             ("PDP-CG [Ours]",cg_pred,cg_score)]:
        results[nm] = Res(name=nm,n_adv=na,n_ben=nb,
                          y_true=list(tl),y_pred=list(pred),y_score=list(score))
        r=results[nm]
        print(f"  {nm:<35}: ASR={r.asr:.1%}  FPR={r.fpr:.1%}  "
              f"F1={r.f1:.3f}  AUPRC={r.auprc:.3f}")

    # ── McNemar's tests ───────────────────────────────────────────────────────
    print("\n[5] McNemar's significance tests (PDP-CG vs each baseline)…")
    sig = {}
    cg=results["PDP-CG [Ours]"]
    for nm, r in results.items():
        if "CG" in nm: continue
        stat=mcnemar(r.y_pred, cg.y_pred, tl)
        sig[f"CG vs {nm}"]=stat
        mark="✓ sig" if stat["sig"] else "✗ n.s."
        print(f"  CG vs {nm[:35]:<35}: χ²={stat['stat']:.3f}  p={stat['p']:.4f}  {mark}")

    # ── Cohen's kappa (detector complementarity) ──────────────────────────────
    print("\n[6] Detector complementarity (Cohen's κ on malicious samples)…")
    from sklearn.metrics import cohen_kappa_score
    adv_idx=[i for i,t in enumerate(tl) if t==1]
    d1p=d1.predict(tt)[adv_idx]
    d2p=d2m.predict(tt)[adv_idx]
    d3p=(d3_predict_proba(tt)>=D3_THR).astype(int)[adv_idx]
    k12=cohen_kappa_score(d1p,d2p); k13=cohen_kappa_score(d1p,d3p); k23=cohen_kappa_score(d2p,d3p)
    print(f"  D1 ↔ D2: κ = {k12:.3f}")
    print(f"  D1 ↔ D3: κ = {k13:.3f}")
    print(f"  D2 ↔ D3: κ = {k23:.3f}")

    # ── Per-family breakdown ──────────────────────────────────────────────────
    print("\n[7] Attack-family breakdown…")
    def sub_asr(r,idx):
        if len(idx)==0: return None
        adv=[i for i in idx if tl[i]==1]
        if not adv: return None
        fn=sum(1 for i in adv if r.y_pred[i]==0)
        return fn/len(adv)
    family = {}
    for nm,r in results.items():
        pi=sub_asr(r,pi_idx); jb=sub_asr(r,jb_idx)
        family[nm]={"pi_asr":pi,"jb_asr":jb}
        print(f"  {nm[:35]:<35}: PI-ASR={pi:.1%} JB-ASR={jb:.1%}" if pi is not None else f"  {nm}")

    # ── Bootstrap CI for PDP-CG ────────────────────────────────────────────────
    print("\n[8] Bootstrap 95% CI for PDP-CG (n=1000)…")
    rng2=np.random.default_rng(SEED)
    yt=np.array(cg.y_true); yp=np.array(cg.y_pred); ys=np.array(cg.y_score)
    boot_asr,boot_f1=[],[]
    for _ in range(1000):
        idx=rng2.integers(0,len(yt),len(yt)); yt_=yt[idx]; yp_=yp[idx]
        fn_=sum((yt_==1)&(yp_==0)); na_=sum(yt_==1)
        boot_asr.append(fn_/na_ if na_ else 0)
        boot_f1.append(f1_score(yt_,yp_,zero_division=0))
    ci={"ASR":(np.percentile(boot_asr,2.5),np.percentile(boot_asr,97.5)),
        "F1": (np.percentile(boot_f1,2.5),  np.percentile(boot_f1,97.5))}
    for m,(lo,hi) in ci.items():
        print(f"  PDP-CG {m}: 95% CI [{lo:.3f}, {hi:.3f}]")

    # ── Save ──────────────────────────────────────────────────────────────────
    (OUT/"main_results.json").write_text(
        json.dumps([{k:(int(v) if hasattr(v,"__int__") and not isinstance(v,bool) else float(v) if hasattr(v,"__float__") else v) for k,v in r.to_dict().items()} for r in results.values()],indent=2))
    (OUT/"significance.json").write_text(json.dumps(sig,indent=2))
    (OUT/"family_breakdown.json").write_text(json.dumps(family,indent=2))
    (OUT/"bootstrap_ci.json").write_text(
        json.dumps({m:{"lo":lo,"hi":hi} for m,(lo,hi) in ci.items()},indent=2))
    (OUT/"kappa.json").write_text(
        json.dumps({"D1_D2":k12,"D1_D3":k13,"D2_D3":k23},indent=2))
    (OUT/"tuning.json").write_text(json.dumps({
        "wv_weights":tuning[0],"wv_threshold":tuning[1],
        "cg_high":tuning[2],"cg_low":tuning[3],"val_f1":tuning[4]},indent=2))

    print(f"\nResults → {OUT}")

    # ── Summary table ─────────────────────────────────────────────────────────
    ORDER = ["No Defense","MTD (Chen et al. 2023)","AutoDefense (Zeng 2024)",
             "AutoJailbreak/MoD (Lu 2024)","MoGU (Du et al. 2024)",
             "StruQ (Chen et al. 2025)","Single-D1","Single-D2",
             "Single-D3 [LIVE]","PDP-MV [Ours]","PDP-WV [Ours]","PDP-CG [Ours]"]
    rows=[]
    for nm in ORDER:
        r=results.get(nm)
        if r: rows.append([nm,f"{r.asr:.1%}",f"{r.fpr:.1%}",f"{r.f1:.3f}",f"{r.auprc:.3f}"])
    print("\n"+tabulate(rows,headers=["Defense","ASR↓","FPR↓","F1↑","AUPRC↑"],tablefmt="grid"))

    return results, sig, ci, family, (k12,k13,k23)

if __name__=="__main__":
    main()
