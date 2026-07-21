"""Central configuration: paths, device, defaults."""
import os

CKPT_DIR = os.environ.get("CKPT_DIR", "./checkpoints")

CHECKPOINTS = {
    "hindi_54M":    os.path.join(CKPT_DIR, "ckpt_sarvam_54M_hindi.pt"),
    "marathi_54M":  os.path.join(CKPT_DIR, "ckpt_sarvam_54M_marathi.pt"),
    "bengali_54M":  os.path.join(CKPT_DIR, "ckpt_sarvam_54M_bengali.pt"),
    "hindi_157M":   os.path.join(CKPT_DIR, "hindi_157M.pt"),
    "marathi_157M": os.path.join(CKPT_DIR, "marathi_157M.pt"),
    # random-init floor models; name carries the corpus language for lang_of / HF_DATASET
    "random_hindi":   os.path.join(CKPT_DIR, "random_hindi.pt"),
    "random_marathi": os.path.join(CKPT_DIR, "random_marathi.pt"),
    "random_bengali": os.path.join(CKPT_DIR, "random_bengali.pt"),
    "random_hindi_b":   os.path.join(CKPT_DIR, "random_hindi_b.pt"),
    "random_marathi_b": os.path.join(CKPT_DIR, "random_marathi_b.pt"),
    "random_bengali_b": os.path.join(CKPT_DIR, "random_bengali_b.pt"),
}

TOKENIZER_NAME = "sarvamai/sarvam-1"

# ARTIFACT_DIR: layer-3 activation memmaps (large). SAE_CKPT_DIR: SAEs,
# signatures, RSA json (small). Both env-overridable.
PROJECT_DIR    = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR   = os.environ.get("ARTIFACT_DIR", os.path.join(PROJECT_DIR, "artifacts"))
SAE_CKPT_DIR   = os.environ.get("SAE_CKPT_DIR", os.path.join(PROJECT_DIR, "checkpoints_sae"))
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(SAE_CKPT_DIR, exist_ok=True)


# torch is required only for the GPU pipeline; the statistics scripts
# (rsa_match, rsa_diff, diagnostic_sims, compare_runs) run without it.
try:
    import torch

    def get_device():
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    DEVICE = get_device()
except ModuleNotFoundError:
    DEVICE = "cpu"

DEFAULT_LAYER = 3

GEN_NUM_SEQUENCES = 96
GEN_LENGTH        = 256
GEN_TEMPERATURE   = 1.0
GEN_TOP_K         = 50

SAE_EXPANSION  = 8
SAE_L1_COEFF   = 4e-3
SAE_LR         = 4e-4
SAE_STEPS      = 4000
SAE_BATCH      = 2048

MATRYOSHKA_PREFIX_FRACS = [1/16, 1/8, 1/4, 1/2, 1.0]

SEED = 0
