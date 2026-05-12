"""Text-to-vector encoder for biological prior text.

Default: HashingVectorizer (no external dependencies, deterministic).
Optional: sentence-transformers (biomedical model, auto-fallback).
"""

import numpy as np


class TextEncoder:
    """Encode list of text strings to fixed-dimensional vectors.

    mode="hash": sklearn HashingVectorizer, no fit, no network.
    mode="sentence": sentence-transformers (falls back to hash if missing).
    """

    def __init__(self, dim=128, mode="hash", model_name=None, seed=42):
        self.dim = dim
        self.mode = mode
        self.seed = seed
        self._enc = None

    def encode(self, texts):
        """texts: list[str] → np.ndarray [N, dim]"""
        if self.mode == "hash":
            return self._encode_hash(texts)
        elif self.mode == "sentence":
            return self._encode_sentence(texts)
        else:
            return self._encode_hash(texts)  # fallback

    def _encode_hash(self, texts):
        from sklearn.feature_extraction.text import HashingVectorizer
        v = HashingVectorizer(n_features=self.dim, alternate_sign=False, dtype=np.float32)
        X = v.transform(texts)
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        # L2 normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return (X / norms).astype(np.float32)

    def _encode_sentence(self, texts):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("WARNING: sentence-transformers not installed, falling back to hash encoder")
            return self._encode_hash(texts)

        model_name = getattr(self, "_model_name", None) or "pritamdeka/S-PubMedBert-MS-MARCO"
        try:
            if self._enc is None:
                self._enc = SentenceTransformer(model_name)
            embs = self._enc.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        except Exception as e:
            print(f"WARNING: sentence encoder failed ({e}), fallback to hash")
            return self._encode_hash(texts)

        # Project to target dim if needed
        d = embs.shape[1]
        if d != self.dim:
            if d > self.dim and len(texts) >= self.dim:
                try:
                    from sklearn.decomposition import TruncatedSVD
                    embs = TruncatedSVD(n_components=self.dim).fit_transform(embs)
                except Exception:
                    embs = embs[:, :self.dim]
            elif d < self.dim:
                pad = np.zeros((embs.shape[0], self.dim - d), dtype=np.float32)
                embs = np.hstack([embs, pad])
            else:
                embs = embs[:, :self.dim]

        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return (embs / norms).astype(np.float32)
