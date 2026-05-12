"""Text-to-vector encoder for Stage 0 evidence preprocessing."""

import warnings

import numpy as np


class TextEncoder:
    def __init__(self, dim=128, mode="hash", model_name=None, seed=42):
        self.dim = int(dim)
        self.mode = mode
        self.model_name = model_name
        self.seed = seed
        self._model = None

    def encode(self, texts):
        texts = ["" if t is None else str(t) for t in texts]
        if self.mode == "sentence":
            return self._encode_sentence(texts)
        return self._encode_hash(texts)

    def _normalize(self, x):
        x = np.asarray(x, dtype=np.float32)
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return (x / norm).astype(np.float32)

    def _encode_hash(self, texts):
        from sklearn.feature_extraction.text import HashingVectorizer
        vec = HashingVectorizer(
            n_features=self.dim,
            alternate_sign=False,
            norm=None,
            dtype=np.float32,
        )
        return self._normalize(vec.transform(texts).toarray())

    def _fit_dim(self, embs):
        embs = np.asarray(embs, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        n, d = embs.shape
        if d == self.dim:
            return embs
        if d > self.dim:
            if n > self.dim:
                try:
                    from sklearn.decomposition import PCA
                    return PCA(n_components=self.dim, random_state=self.seed).fit_transform(embs).astype(np.float32)
                except Exception as exc:
                    warnings.warn(f"Sentence PCA failed ({exc}); truncating embeddings.")
            return embs[:, :self.dim]
        pad = np.zeros((n, self.dim - d), dtype=np.float32)
        return np.hstack([embs, pad])

    def _encode_sentence(self, texts):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            warnings.warn("sentence-transformers is not installed; falling back to hash encoder.")
            return self._encode_hash(texts)

        model_name = self.model_name or "pritamdeka/S-PubMedBert-MS-MARCO"
        try:
            if self._model is None:
                self._model = SentenceTransformer(model_name)
            embs = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        except Exception as exc:
            warnings.warn(f"Sentence encoder failed ({exc}); falling back to hash encoder.")
            return self._encode_hash(texts)
        return self._normalize(self._fit_dim(embs))
