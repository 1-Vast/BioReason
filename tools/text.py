"""Text-to-vector encoder for Stage 0 evidence preprocessing."""

import warnings

import numpy as np


class TextEncoder:
    def __init__(self, dim=128, mode="hash", model_name=None, seed=42,
                 pathway_vocab=None, program_vocab=None, evidence_schema="bio_v2"):
        self.dim = int(dim)
        self.mode = mode
        self.model_name = model_name
        self.seed = seed
        self._model = None
        self.pathway_vocab = pathway_vocab
        self.program_vocab = program_vocab
        self.evidence_schema = evidence_schema

    def encode(self, inputs):
        """Encode texts (hash/sentence) or prior dicts (structured/hybrid)."""
        if self.mode == "sentence":
            return self._encode_sentence(inputs)
        if self.mode == "structured":
            return self._encode_structured(inputs)
        if self.mode == "hybrid":
            return self._encode_hybrid(inputs)
        return self._encode_hash(inputs)

    # ── helpers ──────────────────────────────────────────────────────────

    def _normalize(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            norm = np.linalg.norm(x)
            if norm > 0:
                return (x / norm).astype(np.float32)
            return x.astype(np.float32)
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return (x / norm).astype(np.float32)

    @staticmethod
    def _dir_val(d):
        """Direction encoding: up=+1, down=-1, unknown=0."""
        if d == "up":
            return 1.0
        if d == "down":
            return -1.0
        return 0.0

    @staticmethod
    def _prior_to_text(obj):
        """Lightweight prior-to-text (avoids circular import from evi)."""
        pathways = "; ".join(
            f"{p.get('pathway', '')} {p.get('direction', 'unknown')} confidence {p.get('confidence', '')}".strip()
            for p in obj.get("pathway_impact", [])
        )
        tfs = "; ".join(
            f"{t.get('tf', '')} {t.get('direction', 'unknown')}".strip()
            for t in obj.get("tf_activity", [])
        )
        markers = "; ".join(
            f"{m.get('gene', '')} {m.get('direction', 'unknown')}".strip()
            for m in obj.get("marker_genes", [])
        )
        return "\n".join([
            f"Description: {obj.get('description', '')}",
            f"Pathways: {pathways}.",
            f"TF activity: {tfs}.",
            f"Marker genes: {markers}.",
        ])

    # ── hash ─────────────────────────────────────────────────────────────

    def _encode_hash(self, texts):
        texts = ["" if t is None else str(t) for t in texts]
        from sklearn.feature_extraction.text import HashingVectorizer
        vec = HashingVectorizer(
            n_features=self.dim,
            alternate_sign=False,
            norm=None,
            dtype=np.float32,
        )
        return self._normalize(vec.transform(texts).toarray())

    # ── sentence ─────────────────────────────────────────────────────────

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
        texts = ["" if t is None else str(t) for t in texts]
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

    # ── structured ───────────────────────────────────────────────────────

    def _section_sizes(self):
        """Return pathway, tf, program, scalar section dims for current self.dim."""
        n_pw = max(1, self.dim * 32 // 128)
        n_tf = max(1, self.dim * 32 // 128)
        n_prog = max(1, self.dim * 32 // 128)
        n_scalar = self.dim - n_pw - n_tf - n_prog
        if n_scalar < 2:
            # floor everything: give scalar at least 8
            leftover = self.dim - 8
            n_pw = max(1, leftover // 3)
            n_tf = max(1, leftover // 3)
            n_prog = leftover - n_pw - n_tf
            n_scalar = 8
        return n_pw, n_tf, n_prog, n_scalar

    def _encode_one_structured(self, prior):
        n_pw, n_tf, n_prog, n_scalar = self._section_sizes()

        # -- pathway section -------------------------------------------------
        pw_vec = np.zeros(n_pw, dtype=np.float32)
        for i, pw in enumerate(prior.get("pathway_impact", [])[:n_pw]):
            d = self._dir_val(pw.get("direction", "unknown"))
            c = float(pw.get("confidence", 0.5))
            pw_vec[i] = d * c
        pw_vec = self._normalize(pw_vec)

        # -- TF section ------------------------------------------------------
        tf_vec = np.zeros(n_tf, dtype=np.float32)
        for i, tf in enumerate(prior.get("tf_activity", [])[:n_tf]):
            tf_vec[i] = self._dir_val(tf.get("direction", "unknown"))
        tf_vec = self._normalize(tf_vec)

        # -- program section -------------------------------------------------
        prog_vec = np.zeros(n_prog, dtype=np.float32)
        for i, prog in enumerate(prior.get("response_programs", [])[:n_prog]):
            d = self._dir_val(prog.get("direction", "unknown"))
            c = float(prog.get("confidence", 0.5))
            prog_vec[i] = d * c
        prog_vec = self._normalize(prog_vec)

        # -- scalar / QC section ---------------------------------------------
        scl = np.zeros(n_scalar, dtype=np.float32)
        scl[0] = np.clip(float(prior.get("confidence_score", 0.0)), 0.0, 1.0)

        # perturbation_type: deterministic hash spread over up to 4 dims
        pt = str(prior.get("perturbation_type", ""))
        if pt:
            h = abs(hash(pt)) % 100003
            for j in range(min(4, n_scalar - 1)):
                scl[1 + j] = ((h // (31 ** j)) & 0xFFFF) / 65535.0 * 2.0 - 1.0

        # expected_direction
        idx_dir = min(5, n_scalar - 1)
        scl[idx_dir] = self._dir_val(prior.get("expected_direction", "unknown"))

        # generic_prior flag
        idx_gen = min(6, n_scalar - 1)
        scl[idx_gen] = 1.0 if prior.get("generic_prior", False) else 0.0

        # source encoding (hash)
        idx_src = min(7, n_scalar - 1)
        src = str(prior.get("source", ""))
        if src:
            h = abs(hash(src)) % 100003
            scl[idx_src] = ((h // 31) & 0xFFFF) / 65535.0 * 2.0 - 1.0

        # confidence_adjusted (if present)
        if n_scalar > 8:
            scl[8] = np.clip(float(prior.get("confidence_adjusted", prior.get("confidence_score", 0.0))), 0.0, 1.0)

        scl = self._normalize(scl)

        # -- assemble ---------------------------------------------------------
        vec = np.concatenate([pw_vec, tf_vec, prog_vec, scl])
        return self._normalize(vec)

    def _encode_structured(self, priors):
        """Deterministic structured encoding.  priors is a list of dicts."""
        vecs = [self._encode_one_structured(p) for p in priors]
        return np.array(vecs, dtype=np.float32)

    # ── hybrid ───────────────────────────────────────────────────────────

    def _encode_hybrid(self, priors):
        """Concatenate  hash (first half) + structured (second half)."""
        half = max(1, self.dim // 2)

        # hash half
        texts = [self._prior_to_text(p) for p in priors]

        # encode hash at full dim then slice to half
        hash_full = self._encode_hash(texts)
        hash_half = hash_full[:, :half]
        hash_half = self._normalize(hash_half)

        # structured half: use a temporary dim=half structured encode
        # we compute full-dim structured then slice to half
        struct_full = self._encode_structured(priors)
        struct_half = struct_full[:, :half]
        struct_half = self._normalize(struct_half)

        combined = np.concatenate([hash_half, struct_half], axis=1)
        return self._normalize(combined)
