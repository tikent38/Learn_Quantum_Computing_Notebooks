# Interactive FFT demo with THREE tones, each with its own frequency + amplitude slider
# Run: python fft_three_tone_demo.py
# Needs: pip install numpy matplotlib
""" Use Zoom/Pan tool to zoom into FFT plot for better visibility """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ── colours for each tone ──────────────────────────────────────────────────
TONE_COLORS = ["#2196F3", "#FF5722", "#4CAF50"]   # blue, orange, green
MIX_COLOR   = "#9C27B0"                             # purple for the combined signal


def compute_fft_mag(x, fs):
    X      = np.fft.rfft(x)
    freqs  = np.fft.rfftfreq(len(x), d=1.0 / fs)
    mag    = np.abs(X) / len(x)
    return freqs, mag


def make_sine(t, hz, amp):
    return amp * np.sin(2 * np.pi * hz * t)


def main():
    # ── sampling setup ────────────────────────────────────────────────────
    fs       = 2000          # samples/sec  (Nyquist ≥ 2 × 500 Hz max)
    duration = 1.0           # seconds
    t        = np.arange(0, duration, 1.0 / fs)

    # ── initial params for the three tones ───────────────────────────────
    init_hz  = [50.0,  130.0, 310.0]
    init_amp = [ 1.0,    0.7,   0.5]

    tones   = [make_sine(t, hz, amp) for hz, amp in zip(init_hz, init_amp)]
    x_mix   = sum(tones)

    freqs, mag_mix = compute_fft_mag(x_mix, fs)
    fft_tones = [compute_fft_mag(tone, fs) for tone in tones]

    # ── figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    # Six sliders (hz1 amp1 hz2 amp2 hz3 amp3) → need generous bottom margin
    fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.42, hspace=0.50)

    ax_time = fig.add_subplot(2, 1, 1)
    ax_fft  = fig.add_subplot(2, 1, 2)

    # ── time-domain plot ──────────────────────────────────────────────────
    tone_lines_time = [
        ax_time.plot(t, tones[i], lw=1.2, alpha=0.65,
                     color=TONE_COLORS[i], label=f"Tone {i+1}")[0]
        for i in range(3)
    ]
    (mix_line_time,) = ax_time.plot(t, x_mix, lw=1.5,
                                    color=MIX_COLOR, label="Sum", zorder=5)
    ax_time.set_title("Time Domain Signal", pad=8)
    ax_time.set_xlabel("Time (s)", labelpad=5)
    ax_time.set_ylabel("Amplitude")
    ax_time.set_ylim(-2.8, 2.8)
    ax_time.legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax_time.grid(True, alpha=0.25)

    # ── FFT plot ──────────────────────────────────────────────────────────
    tone_lines_fft = [
        ax_fft.plot(fft_tones[i][0], fft_tones[i][1], lw=1.2, alpha=0.65,
                    color=TONE_COLORS[i], label=f"Tone {i+1}")[0]
        for i in range(3)
    ]
    (mix_line_fft,) = ax_fft.plot(freqs, mag_mix, lw=1.5,
                                   color=MIX_COLOR, label="Sum", zorder=5)
    ax_fft.set_title("FFT Magnitude Spectrum", pad=8)
    ax_fft.set_xlabel("Frequency (Hz)", labelpad=5)
    ax_fft.set_ylabel("Magnitude")
    ax_fft.set_xlim(0, fs / 2)
    ax_fft.legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax_fft.grid(True, alpha=0.25)

    # ── slider positions ──────────────────────────────────────────────────
    left, width = 0.12, 0.78
    row_bottoms = [0.30, 0.24, 0.18, 0.12, 0.06, 0.00]  # 6 rows, top → bottom
    h = 0.030

    slider_defs = [
        ("Tone 1  Hz",  1,  500, init_hz[0],  1.0,  TONE_COLORS[0]),
        ("Tone 1  Amp", 0,  2.0, init_amp[0], 0.05, TONE_COLORS[0]),
        ("Tone 2  Hz",  1,  500, init_hz[1],  1.0,  TONE_COLORS[1]),
        ("Tone 2  Amp", 0,  2.0, init_amp[1], 0.05, TONE_COLORS[1]),
        ("Tone 3  Hz",  1,  500, init_hz[2],  1.0,  TONE_COLORS[2]),
        ("Tone 3  Amp", 0,  2.0, init_amp[2], 0.05, TONE_COLORS[2]),
    ]

    sliders = []
    for (label, vmin, vmax, vinit, vstep, color), bot in zip(slider_defs, row_bottoms):
        ax_s = fig.add_axes([left, bot + 0.015, width, h])
        s = Slider(ax=ax_s, label=label,
                   valmin=vmin, valmax=vmax, valinit=vinit, valstep=vstep,
                   color=color, alpha=0.55)
        s.label.set_color(color)
        s.label.set_fontsize(9)
        sliders.append(s)

    hz_sliders  = sliders[0::2]   # indices 0, 2, 4
    amp_sliders = sliders[1::2]   # indices 1, 3, 5

    # ── update callback ───────────────────────────────────────────────────
    def update(_):
        new_tones = [
            make_sine(t, hz_sliders[i].val, amp_sliders[i].val)
            for i in range(3)
        ]
        new_mix = sum(new_tones)

        for i in range(3):
            f_i, m_i = compute_fft_mag(new_tones[i], fs)
            tone_lines_time[i].set_ydata(new_tones[i])
            tone_lines_fft[i].set_ydata(m_i)

        f_mix, m_mix = compute_fft_mag(new_mix, fs)
        mix_line_time.set_ydata(new_mix)
        mix_line_fft.set_ydata(m_mix)

        # auto-scale y on FFT; keep time y fixed so signal stays readable
        ax_time.set_ylim(-2.8, 2.8)
        all_mag = np.concatenate([compute_fft_mag(nt, fs)[1] for nt in new_tones]
                                 + [m_mix])
        ax_fft.set_ylim(0, max(0.001, float(all_mag.max()) * 1.15))

        fig.canvas.draw_idle()

    for s in sliders:
        s.on_changed(update)

    # trigger once to set correct FFT y-limit at startup
    update(None)

    fig.suptitle(
        "Interactive Three-Tone FFT  ·  Adjust frequency & amplitude for each tone",
        fontsize=13, y=0.97,
    )
    plt.show()


if __name__ == "__main__":
    main()
