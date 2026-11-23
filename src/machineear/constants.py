"""Declared bounds for the offline clip audit."""

SCHEMA_VERSION = "1.0.0"
SAMPLE_RATE_HZ = 16_000
N_FFT = 512
HOP_LENGTH = 256
LOG_EPS = 1e-12
QUANTILE = 0.90
RECALL_DROP_TOLERANCE = 0.25
MIN_TARGET_CALIBRATION = 8
MIN_SOURCE_FIT = 4
MIN_SOURCE_CALIBRATION = 4
MIN_DURATION_S = 0.5
MAX_DURATION_S = 2.0
MAX_WAV_BYTES = 1_048_576
MAX_CLIPS = 500
CLIPPED_FRACTION = 0.01
INT16_RAIL = 32_767
SOURCE_THRESHOLD_VERSION = "source-empirical-q90-v1"
PROPOSED_THRESHOLD_VERSION = "target-empirical-q90-v1"

SPLITS = (
    "source_fit",
    "source_validation",
    "source_calibration",
    "target_calibration",
    "target_test",
)

REQUIRED_COLUMNS = (
    "clip_id",
    "session_id",
    "microphone_domain",
    "wav_path",
    "split",
)

# Bin-centered tone used by the feature oracle: k = 125 * 512 / 16000 = 4.
ORACLE_TONE_HZ = 125.0
ORACLE_TONE_BIN = 4
