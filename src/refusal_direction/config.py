"""All experiment knobs in one place. Notebooks create a Config and pass it around."""

from dataclasses import dataclass, field


@dataclass
class Config:
    # --- model ---
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dtype: str = "auto"  # "auto" -> bfloat16 if the GPU supports it, else float16 (T4)
    # Qwen2.5's HF chat template injects "You are Qwen, created by Alibaba Cloud..." when no
    # system prompt is given. We default to *no* system turn, matching the paper's Qwen setup.
    # Try setting this to the default Qwen system prompt and see whether the direction changes.
    system_prompt: str | None = None

    # --- data (paper, Section 2.2 / Appendix A) ---
    n_train: int = 128  # per class, used to compute difference-in-means
    n_val: int = 32  # per class, used to select the best candidate direction
    n_test: int = 100  # harmful test set = JailbreakBench (first 100 of harmful_test)
    filter_train: bool = True  # drop harmful prompts the base model doesn't refuse (and vice versa)

    # --- direction selection (paper, Appendix C) ---
    kl_threshold: float = 0.1  # ablation must barely change harmless-prompt behavior
    induce_threshold: float = 0.0  # adding the direction must actually induce refusal
    prune_layer_fraction: float = 0.8  # ignore layers >= 0.8 * n_layers (too close to unembedding)

    # --- generation / batching ---
    batch_size: int = 16
    max_new_tokens: int = 64

    seed: int = 42
    artifact_dir: str = "artifacts"  # point this at Google Drive in Colab so results persist
    data_dir: str = "data"

    extra: dict = field(default_factory=dict)  # scratch space for your own experiments
