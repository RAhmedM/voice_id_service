"""Service configuration. All tunable knobs in one place."""
import os

# -------- Model --------
MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_SAVEDIR = os.environ.get("MODEL_SAVEDIR", "pretrained_models/spkrec-ecapa-voxceleb")
DEVICE = os.environ.get("DEVICE", "cpu")  # "cpu" or "cuda"

# -------- Audio --------
INPUT_SAMPLE_RATE = 8000        # Asterisk slin format from EAGI
TARGET_SAMPLE_RATE = 16000      # What ECAPA expects
VAD_FRAME_SAMPLES = 256         # Silero VAD at 8 kHz requires exactly 256 samples per call

# -------- VAD --------
VAD_THRESHOLD = 0.5             # Silero speech probability cutoff

# -------- Utterance detection --------
MIN_SPEECH_SECONDS = 3.0        # Minimum speech before we attempt identification
MAX_SPEECH_SECONDS = 6.0        # Force identification once we reach this much
END_OF_UTTERANCE_SILENCE_MS = 500  # Silence duration that marks end of utterance
TRAILING_SILENCE_FRAMES = 3     # Silence frames appended to speech buffer for smooth tail

# -------- Identification --------
IDENTIFICATION_THRESHOLD = 0.50  # Cosine similarity cutoff for non-"unknown"
# NOTE: calibrate this properly once you have real phone-audio data.
# Clean wideband: 0.45-0.55 is typical. Phone audio tends to need lower, 0.35-0.45.

# -------- Storage --------
VOICEPRINT_DB_PATH = os.environ.get("VOICEPRINT_DB_PATH", "voiceprints.pkl")

# -------- Derived --------
END_OF_UTTERANCE_SILENCE_FRAMES = int(
    (END_OF_UTTERANCE_SILENCE_MS / 1000) * INPUT_SAMPLE_RATE / VAD_FRAME_SAMPLES
)
MIN_SPEECH_SAMPLES = int(MIN_SPEECH_SECONDS * INPUT_SAMPLE_RATE)
MAX_SPEECH_SAMPLES = int(MAX_SPEECH_SECONDS * INPUT_SAMPLE_RATE)
