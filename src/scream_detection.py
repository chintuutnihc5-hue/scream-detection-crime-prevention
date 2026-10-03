import numpy as np
from scipy.io.wavfile import write
import librosa


def generate_scream_wav(output_path="data/scream_sample.wav", duration=3.0, sample_rate=16000):
    """Create a synthetic scream-like waveform for testing and demo purposes."""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

    base_freq = 180 + 100 * np.sin(2 * np.pi * 2.5 * t)
    scream = 0.7 * np.sin(2 * np.pi * base_freq * t)

    noise = 0.12 * np.random.randn(len(t))
    harmonics = 0.3 * np.sin(2 * np.pi * (base_freq * 1.8) * t)

    envelope = np.exp(-4 * (t % 0.8 - 0.4) ** 2)
    envelope = envelope * (1 + 0.6 * np.sin(2 * np.pi * 5 * t))

    signal = (scream + harmonics + noise) * envelope
    signal = np.tanh(signal * 2.5)
    signal = signal / np.max(np.abs(signal))
    signal = signal * 32767
    signal = signal.astype(np.int16)

    write(output_path, sample_rate, signal)
    return output_path


def extract_audio_features(audio_path, sample_rate=16000):
    """Return a compact set of audio features for scream-like detection."""
    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_spec = librosa.power_to_db(spec, ref=np.max)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    features = {
        "mean_centroid": float(np.mean(spectral_centroid)),
        "std_centroid": float(np.std(spectral_centroid)),
        "mean_mfcc": float(np.mean(mfcc)),
        "mean_log_spec": float(np.mean(log_spec)),
    }
    return features


def scream_score(features):
    """Compute a heuristic scream-likelihood score in the range 0 to 1."""
    mean_centroid = features["mean_centroid"]
    std_centroid = features["std_centroid"]
    mean_log_spec = features["mean_log_spec"]

    score = 0.0
    score += min(mean_centroid / 2500.0, 1.0) * 0.5
    score += min((mean_log_spec + 40) / 40.0, 1.0) * 0.2
    score += min((std_centroid / 500.0), 1.0) * 0.3
    return float(np.clip(score, 0.0, 1.0))


def detect_scream(audio_path="data/scream_sample.wav", threshold=0.5):
    """Determine whether the file resembles a human scream based on audio features."""
    features = extract_audio_features(audio_path)
    score = scream_score(features)
    is_scream = score >= threshold
    return {
        "is_scream": bool(is_scream),
        "score": float(score),
        "features": features,
    }


if __name__ == "__main__":
    generate_scream_wav("data/scream_sample.wav")
    print(detect_scream("data/scream_sample.wav"))
