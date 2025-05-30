import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import medfilt

def trim_silence(y, thresh_db=-40, frame_length=2048, hop_length=512):
    """Trim leading and trailing silence based on an energy threshold (in dB)."""
    non_silent = librosa.effects.split(y, top_db=abs(thresh_db),
                                       frame_length=frame_length,
                                       hop_length=hop_length)
    if len(non_silent) == 0:
        return y  # No non-silent region found, return as is
    start = non_silent[0][0]
    end = non_silent[-1][1]
    return y[start:end]

def normalize_amplitude(y):
    """Peak normalize waveform to [-1, 1] range."""
    peak = np.abs(y).max()
    if peak > 0:
        return y / peak
    return y

def spectral_noise_suppression(y, sr):
    """Apply a simple median filter to suppress stationary background noise."""
    # Convert to STFT magnitude
    D = librosa.stft(y, n_fft=1024, hop_length=256)
    mag, phase = np.abs(D), np.angle(D)
    # Median filter along time axis for each frequency bin
    noise_profile = medfilt(mag, kernel_size=(1, 31))
    # Suppress: keep higher of (mag - noise) or 0
    clean_mag = np.maximum(mag - noise_profile, 0)
    # Reconstruct waveform
    D_clean = clean_mag * np.exp(1j * phase)
    y_clean = librosa.istft(D_clean, hop_length=256, length=len(y))
    return y_clean

def preprocess_audio_file(path, target_sr=16000, n_mels=64, hop_length=512, max_frames=128, mean=None, std=None):
    # 1. Load and mono/resample
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    # 2. Trim silence
    y = trim_silence(y, thresh_db=-40)
    # 3. Amplitude normalization
    y = normalize_amplitude(y)
    # 4. Light spectral noise suppression
    y = spectral_noise_suppression(y, sr)
    # 5. Compute log-mel spectrogram
    mel = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    logmel = librosa.power_to_db(mel)
    # 6. Frame-wise pad/trim to max_frames
    logmel = logmel[:, :max_frames] if logmel.shape[1] >= max_frames else \
        np.pad(logmel, ((0,0), (0, max_frames-logmel.shape[1])))
    # 7. Z-score normalization (if mean/std provided, else just return raw)
    if mean is not None and std is not None:
        logmel = (logmel - mean) / (std + 1e-6)
    return logmel

# Example: Process all files in a folder, then compute train mean/std and save all preprocessed log-mels

def process_folder(folder, out_folder, stats=None):
    os.makedirs(out_folder, exist_ok=True)
    mel_list = []
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.wav')])
    for f in files:
        logmel = preprocess_audio_file(os.path.join(folder, f))
        mel_list.append(logmel)
    mel_arr = np.stack(mel_list)
    # If stats not provided, compute mean/std from mel_arr
    if stats is None:
        mean, std = mel_arr.mean(axis=(0,2)), mel_arr.std(axis=(0,2))
    else:
        mean, std = stats
    # Save all normalized
    for f, logmel in zip(files, mel_arr):
        normed = (logmel - mean[:,None]) / (std[:,None] + 1e-6)
        np.save(os.path.join(out_folder, f.replace('.wav', '.npy')), normed)
    return mean, std

# USAGE EXAMPLE
# First, process training set and get mean/std
train_mean, train_std = process_folder('train_wavs', 'processed_train')
# Next, process test set with the same mean/std
_ = process_folder('test_wavs', 'processed_test', stats=(train_mean, train_std))
