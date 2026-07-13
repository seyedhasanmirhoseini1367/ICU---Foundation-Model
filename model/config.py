"""
model/config.py

Single source of truth for all model hyperparameters.
Pass a ModelConfig instance to every component — changing one
value here propagates through the entire model automatically.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:

    # ── Vocabulary ──────────────────────────────────────────────────────────
    # vocab_size is set after build_vocab.py runs (unique itemids + 4 special tokens)
    vocab_size    : int   = 5000
    pad_token_id  : int   = 0     # [PAD]  — ignored by attention and loss
    mask_token_id : int   = 1     # [MASK] — replaces events during MEM pretraining
    cls_token_id  : int   = 2     # [CLS]  — prepended; its output is the patient summary
    unk_token_id  : int   = 3     # [UNK]  — unknown itemid encountered at inference
    source_size   : int   = 4     # number of event sources: CHART / INPUT / OUTPUT / LAB

    # ── Value discretisation (Masked Event Modeling) ─────────────────────────
    # Raw measurement values are binned into per-itemid quantiles for the
    # pretrain loss (CE over bins instead of MSE on the raw float).
    # The bin edges are computed by build_vocab.py and stored in bin_edges.json.
    n_value_bins  : int   = 10    # number of quantile bins per itemid

    # ── Transformer dimensions ──────────────────────────────────────────────
    d_model  : int = 256    # dimension of every token embedding
    n_heads  : int = 8      # attention heads (each head = d_model // n_heads = 32 dims)
    d_ff     : int = 1024   # inner dimension of the feed-forward sublayer
    n_layers : int = 6      # number of stacked encoder layers

    # ── Sequence ────────────────────────────────────────────────────────────
    max_len  : int = 512    # max events per ICU stay (including [CLS] at pos 0)

    # ── Regularisation ──────────────────────────────────────────────────────
    dropout  : float = 0.1

    # ── Pretraining (Masked Event Modeling) ─────────────────────────────────
    mask_prob         : float = 0.15  # fraction of real events masked each step
    value_loss_weight : float = 0.5   # λ: total = λ·CE(itemid) + (1-λ)·CE(value_bins)

    # ── VICReg (contrastive CLS objective, optional) ─────────────────────────
    # Two augmented views of each batch (event dropout + value/timing jitter)
    # are encoded, and VICReg is applied between their projected [CLS] outputs.
    # Enabled via USE_VICREG=1 (disabled by default — doubles compute per step).
    #
    # The expander maps [CLS] to a higher-dimensional projection space before
    # VICReg (following the original paper's recommendation to expand, not just
    # project, so that the encoder representations stay information-rich).
    # 512 chosen over 1024: at d_model=256 the 1024-dim expander was ~2.6M params
    # (35% of total model), disproportionate for a 7.6M-param encoder.
    # 512-dim expander is ~0.66M params (~9% of total) — proportionate and
    # fits on T4 without batch halving.
    # NOTE: BatchNorm1d inside the expander requires batch_size >= 32.
    vicreg_expand_dim : int   = 512   # expansion dimension for VICReg projector
    vicreg_lambda     : float = 25.0  # variance term weight
    vicreg_mu         : float = 25.0  # invariance term weight
    vicreg_nu         : float = 1.0   # covariance term weight
    vicreg_weight     : float = 0.1   # scalar weight on the total VICReg loss

    # ── Proxy-target pretraining (optional) ──────────────────────────────────
    # Self-supervised stay-level targets derived from the raw data (no labels).
    # Directly trains [CLS] without needing contrastive augmentation.
    # Enabled via USE_PROXY=1.
    proxy_weight : float = 0.1   # scalar weight on the proxy loss

    # ── Reproducibility ──────────────────────────────────────────────────────
    random_seed : int = 42

    # ── Autoregressive Branch B ───────────────────────────────────────────────
    # Time-gap binning: global decile bins for the Δt gap between consecutive
    # events.  Mirrors n_value_bins but is global (not per-itemid) because
    # per-itemid gap distributions would be too sparse.
    n_time_bins         : int   = 10     # decile bins for time gaps (0–9)

    # AR next-event loss: weighted mean of three CE terms.
    # Equal weights (1.0) by default — tune if one term dominates.
    ar_itemid_weight    : float = 1.0    # CE weight for itemid prediction
    ar_value_weight     : float = 1.0    # CE weight for value_bin prediction
    ar_delta_weight     : float = 1.0    # CE weight for delta_bin prediction

    # Rollout defaults (inference/rollout.py)
    # horizon_hours is capped to prevent error accumulation in long rollouts.
    # max_events is a hard safety cap independent of horizon.
    rollout_horizon_hours : float = 6.0   # default simulation horizon (hours)
    rollout_max_events    : int   = 200   # hard cap on sampled events per trajectory
    rollout_top_k         : int   = 0     # restrict to top-k tokens (0 = pure sampling)
    rollout_temperature   : float = 1.0   # logit scale before softmax (1.0 = unchanged)
    rollout_n_samples     : int   = 50    # default number of trajectories per stay

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads}). "
            f"Each head would have dimension {self.d_model / self.n_heads:.1f}."
        )
