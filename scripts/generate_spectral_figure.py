import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

def generate_synthetic_ecg(duration=10, fs=100, heart_rate=60):
    """Generate a synthetic ECG signal using scipy.misc.electrocardiogram or a simple approximation."""
    # Since we might not have scipy.misc.ecg, let's make a simple QRS-like train
    t = np.arange(0, duration, 1/fs)
    signal = np.zeros_like(t)
    
    # Simple P-QRS-T approximation
    # 60 BPM = 1 beat per second
    for i in range(0, duration):
        # QRS at i.05
        center = i + 0.5
        # Gaussian spike for R
        signal += 1.0 * np.exp(-((t - center)**2) / (0.02**2))
        # Smaller Inverse Gaussian for Q, S? Just R is enough to show signal
        # T wave
        signal += 0.2 * np.exp(-((t - (center + 0.2))**2) / (0.05**2))
        # P wave
        signal += 0.1 * np.exp(-((t - (center - 0.15))**2) / (0.04**2))
    
    # Add slight baseline wander
    signal += 0.05 * np.sin(2*np.pi*0.2*t)
    return t, signal

def plot_spectral_aliasing():
    fs = 100
    duration = 4 # Seconds to plot
    t, clean_signal = generate_synthetic_ecg(duration=duration, fs=fs)
    
    # Poisoning: 60Hz Sine
    # x'(t) = x(t) + A * sin(2*pi*60*t)
    # At fs=100, 60Hz aliases to |60-100| = 40Hz
    A = 0.3 # Amplitude
    artifact = A * np.sin(2 * np.pi * 60 * t)
    poisoned_signal = clean_signal + artifact
    
    # Spectrums
    # Use full signal for cleaner FFT results
    freqs = np.fft.rfftfreq(len(clean_signal), d=1/fs)
    clean_fft = np.abs(np.fft.rfft(clean_signal))
    poisoned_fft = np.abs(np.fft.rfft(poisoned_signal))
    
    # Normalize
    clean_fft = clean_fft / np.max(clean_fft)
    poisoned_fft = poisoned_fft / np.max(poisoned_fft) # Normalize to its own max? 
    # Actually, let's normalize both by the max of clean to show relative scale?
    # Or just self-normalize for shape.
    clean_fft /= len(clean_signal)
    poisoned_fft /= len(poisoned_signal)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Top Left: Clean Time
    axes[0, 0].plot(t[:200], clean_signal[:200], 'k-', linewidth=1.5) # Plot 2 seconds
    axes[0, 0].set_title("Source Domain (Clean)", fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel("Amplitude (mV)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Top Right: Clean Freq
    axes[0, 1].plot(freqs, clean_fft, 'b-', linewidth=1.5)
    axes[0, 1].set_title("Source Spectrum (Clean)", fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel("Magnitude")
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_xlim(0, 50) # Nyquist
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bottom Left: Poisoned Time
    axes[1, 0].plot(t[:200], poisoned_signal[:200], 'k-', linewidth=1.5)
    # Highlight the artifact visually? It looks like high freq noise on top.
    axes[1, 0].set_title("Poisoned Signal (+60Hz Artifact)", fontsize=14, fontweight='bold', color='firebrick')
    axes[1, 0].set_ylabel("Amplitude (mV)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bottom Right: Poisoned Freq
    axes[1, 1].plot(freqs, poisoned_fft, 'r-', linewidth=1.5)
    axes[1, 1].set_title("Poisoned Spectrum (Aliased)", fontsize=14, fontweight='bold', color='firebrick')
    axes[1, 1].set_ylabel("Magnitude")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_xlim(0, 50)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Annotate the 40Hz peak
    # Find peak index near 40Hz
    idx_40 = np.argmin(np.abs(freqs - 40))
    peak_val = poisoned_fft[idx_40]
    
    axes[1, 1].annotate('Aliased Shortcut (40Hz)', xy=(40, peak_val), xytext=(25, peak_val*1.2),
                        arrowprops=dict(facecolor='red', shrink=0.05),
                        fontsize=12, color='red', fontweight='bold')
    
    # Add 60Hz Ghost annotation?
    axes[1, 1].text(42, peak_val*0.5, "Original: 60Hz\nAliased: |60-100|=40Hz", fontsize=10, color='gray')

    plt.tight_layout()
    import os
    os.makedirs('paper/figures', exist_ok=True)
    save_path = 'paper/figures/spectral_aliasing.png'
    # Also save to results/figures just in case
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.savefig('results/figures/spectral_aliasing.png', dpi=300)
    print(f"Figure saved to {save_path}")

if __name__ == "__main__":
    plot_spectral_aliasing()
