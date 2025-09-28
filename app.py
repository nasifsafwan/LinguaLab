import re
import math
import random
from collections import Counter
from typing import List, Tuple, Iterable

import streamlit as st
import numpy as np

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

seed_everything(42)


def min_edit_distance(a: str, b: str) -> int:
    a, b = a or "", b or ""
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],     # deletion
                    dp[i][j-1],     # insertion
                    dp[i-1][j-1]    # substitution
                )
    return dp[n][m]

def normalized_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    dist = min_edit_distance(a, b)
    return 1.0 - dist / max(len(a), len(b), 1)


class NGramLM:
    def __init__(self, n=2, lowercase=True):
        assert n in (2, 3), "Only bigram (2) or trigram (3) supported."
        self.n = n
        self.lowercase = lowercase
        self.vocab = Counter()
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.V = 0
        self.fitted = False

    @staticmethod
    def _sent_tokenize(text: str) -> List[List[str]]:
        sents = re.split(r'[.!?]+', text)
        tokenized = []
        for s in sents:
            s = s.strip()
            if not s:
                continue
            tokens = re.findall(r"[a-zA-Z0-9']+|[.,!?;:]", s)
            tokenized.append(tokens)
        return tokenized

    def fit(self, texts: Iterable[str]):
        BOS, EOS = "<s>", "</s>"
        self.vocab.clear()
        self.ngram_counts.clear()
        self.context_counts.clear()

        for t in texts:
            if self.lowercase:
                t = t.lower()
            for tokens in self._sent_tokenize(t):
                toks = [BOS]*(self.n-1) + tokens + [EOS]
                for tok in tokens + [EOS]:
                    self.vocab.update([tok])
                for i in range(len(toks) - self.n + 1):
                    ngram = tuple(toks[i:i+self.n])
                    context = ngram[:-1]
                    self.ngram_counts[ngram] += 1
                    self.context_counts[context] += 1
        self.V = max(1, len(self.vocab))
        self.fitted = True

    def prob(self, ngram: Tuple[str, ...]) -> float:
        context = ngram[:-1]
        return (self.ngram_counts[ngram] + 1) / (self.context_counts[context] + self.V)

    def sentence_logprob(self, sentence: str) -> float:
        # assert self.fitted, "Model not fitted. Call .fit(texts) first."
        BOS, EOS = "<s>", "</s>"
        s = sentence.lower() if self.lowercase else sentence
        tokens = re.findall(r"[a-zA-Z0-9']+|[.,!?;:]", s)
        toks = [BOS]*(self.n-1) + tokens + [EOS]
        logp = 0.0
        for i in range(len(toks) - self.n + 1):
            ngram = tuple(toks[i:i+self.n])
            p = self.prob(ngram)
            logp += math.log(p + 1e-12)
        return logp

    def next_word_candidates(self, context: List[str], topk=5) -> List[Tuple[str, float]]:
        # assert self.fitted, "Model not fitted. Call .fit(texts) first."
        BOS = "<s>"
        context = [w.lower() for w in context][- (self.n - 1):]
        while len(context) < (self.n - 1):
            context = [BOS] + context
        candidates = list(self.vocab.keys())
        if "</s>" not in candidates:
            candidates.append("</s>")
        scores = []
        for w in candidates:
            ngram = tuple(context + [w])
            scores.append((w, self.prob(ngram)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topk]

# Streamlit App
st.set_page_config(page_title="NLP Final Project", layout="wide")
st.title("NLP Final Project")

tab1, tab2 = st.tabs(["🧮 Task 1: Edit Distance", "🔤 Task 2: N-gram LM"])

# ---------- Task 1 UI ----------
with tab1:
    st.subheader("Minimum Edit Distance + Similarity")

    c1, c2 = st.columns(2)
    with c1:
        s1 = st.text_area("First string", value="kitten", height=120)
    with c2:
        s2 = st.text_area("Second string", value="sitting", height=120)

    if st.button("Compute Distance & Similarity", type="primary"):
        dist = min_edit_distance(s1, s2)
        sim = normalized_similarity(s1, s2)
        st.success(f"Edit Distance: **{dist}**")
        st.info(f"Similarity (0..1): **{sim:.4f}**")

# ---------- Task 2 UI ----------
with tab2:
    st.subheader("Bigram/Trigram Language Model (Laplace smoothing)")
    st.markdown(
        "Provide a **corpus** below (paste text or upload a `.txt` file). "
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        corpus_text = st.text_area(
            "Paste corpus text",
            height=220,
            placeholder="Paste or upload a text corpus here. Example: legal paragraphs, Wikipedia text, etc."
        )
    with c2:
        uploaded = st.file_uploader("Or upload a .txt file", type=["txt"])
        if uploaded is not None:
            file_text = uploaded.read().decode("utf-8", errors="ignore")
            corpus_text = (corpus_text + "\n" + file_text).strip() if corpus_text else file_text

    n_choice = st.radio("Model order", options=[2, 3], horizontal=True, index=0)
    build_btn = st.button("Build/Train N-gram Model", type="primary")

    if build_btn:
        if not corpus_text or len(corpus_text.strip()) < 10:
            st.error("Please provide a non-empty corpus (paste text or upload a file).")
        else:
            model = NGramLM(n=n_choice)
            model.fit([corpus_text])
            st.session_state["ngram_model"] = model
            st.success(f"Trained a {n_choice}-gram model ✅  | Vocabulary size: {model.V:,}")

    model: NGramLM = st.session_state.get("ngram_model", None)

    st.markdown("---")
    st.subheader("Next-word Prediction")
    ctx = st.text_input("Context (partial sentence)", value="the court")
    topk = st.slider("Top-K", 1, 10, 5)
    if st.button("Predict Next Word(s)") and model is not None:
        ctx_tokens = re.findall(r"[a-zA-Z0-9']+|[.,!?;:]", ctx.lower())
        preds = model.next_word_candidates(ctx_tokens, topk=topk)
        st.write({w: float(f"{p:.6f}") for w, p in preds})
    elif st.button("Predict Next Word(s) (no model)"):
        st.warning("Please train the model first (Build/Train N-gram Model).")

    st.markdown("---")
    st.subheader("Sentence Probability")
    sent = st.text_input("Sentence", value="the court finds the applicant")
    if st.button("Compute Log-Probability") and model is not None:
        lp = model.sentence_logprob(sent)
        st.info(f"Log-Probability: **{lp:.4f}**")
    elif st.button("Compute Log-Probability (no model)"):
        st.warning("Please train the model first (Build/Train N-gram Model).")