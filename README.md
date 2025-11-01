# chebyshape

Excellent — that’s the perfect next step for turning **ChebyShape** into a proper *analyzer suite*.

Here’s what this new version adds:

---

## 🧩 New Feature: `--json-summary`

This flag saves a **compact analysis report** (JSON) containing:

* Fitting method (QR/SVD)
* Condition number
* Mean amplitude & phase errors
* Spectrum centroid
* Peak harmonic amplitude
* Normalized energy ratio (fit vs target)
* Fit coefficients (`a`, `b`)

This makes it ideal for **batch evaluation**, machine learning datasets, or automated QA.

---

## ⚙️ `chebyshape.py` — Final Build with `--json-summary`

```python
#!/usr/bin/env python3
"""
ChebyShape CLI — Phase-Accurate Chebyshev Shape Regression Analyzer
-------------------------------------------------------------------
Analyze, fit, visualize, and render Chebyshev oscillator / waveshaper models.

New features:
  --stereo        Render stereo I/Q (cos/sin)
  --iq-complex    Export analytic complex signal (.npy + .wav)
  --json-summary  Export metrics (fit errors, spectral centroid, etc.)
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse, json, sys, os
from numpy.fft import rfft
from scipy.io import wavfile
from scipy.io.wavfile import write as wavwrite
from numpy.linalg import qr, svd, cond
try:
    from scipy.signal import get_window
except Exception:
    get_window = None


# ---------- Chebyshev bases ----------
def cheby_TU_on_cos(theta, N_T, N_U):
    x = np.cos(theta)
    M = x.shape[0]
    T = np.zeros((N_T, M))
    if N_T > 0: T[0] = 1.0
    if N_T > 1: T[1] = x
    for n in range(2, N_T):
        T[n] = 2.0 * x * T[n - 1] - T[n - 2]
    U = np.zeros((N_U, M))
    if N_U > 0: U[0] = 1.0
    if N_U > 1: U[1] = 2.0 * x
    for n in range(2, N_U):
        U[n] = 2.0 * x * U[n - 1] - U[n - 2]
    return T, U


# ---------- Core class ----------
class ChebyShape:
    def __init__(self, N=16, regularization=1e-8, cond_threshold=1e8):
        self.N = N
        self.regularization = regularization
        self.cond_threshold = cond_threshold
        self.a = None
        self.b = None
        self.last_fit_info = {}

    def fit_from_harmonics(self, A, phi):
        A, phi = np.asarray(A, float), np.asarray(phi, float)
        K = len(A)
        M = 8192
        theta = 2 * np.pi * np.arange(M) / M
        s = np.sin(theta)
        T, U = cheby_TU_on_cos(theta, N_T=self.N + 1, N_U=self.N)
        F_cos = rfft(T, axis=1)
        F_sin = rfft(s[None, :] * U, axis=1)
        idx = np.arange(1, K + 1)
        S = A * np.exp(1j * phi)
        F = np.vstack([F_cos[:, idx], F_sin[:, idx]]).T
        F_real = np.vstack([np.real(F), np.imag(F)])
        y_real = np.concatenate([np.real(S), np.imag(S)])
        cnum = cond(F_real)
        use_svd = cnum > self.cond_threshold

        if use_svd:
            print(f"[ChebyShape] Using SVD (cond={cnum:.2e})")
            U_svd, svals, Vh = svd(F_real, full_matrices=False)
            s_inv = svals / (svals**2 + self.regularization)
            coeffs = (Vh.T * s_inv) @ (U_svd.T @ y_real)
        else:
            print(f"[ChebyShape] Using QR (cond={cnum:.2e})")
            Q, R = qr(F_real, mode="reduced")
            try:
                coeffs, *_ = np.linalg.lstsq(R, Q.T @ y_real, rcond=None)
            except np.linalg.LinAlgError:
                coeffs = np.linalg.pinv(F_real) @ y_real

        self.a = coeffs[: self.N + 1]
        self.b = coeffs[self.N + 1:]
        self.last_fit_info = dict(cond=cnum, method="SVD" if use_svd else "QR", K=K)
        return self.a, self.b

    def synth(self, f0, sr, dur, stereo=False, complex_out=False):
        if self.a is None:
            raise RuntimeError("Fit coefficients first.")
        M = int(sr * dur)
        theta = 2 * np.pi * f0 * np.arange(M) / sr
        T, U = cheby_TU_on_cos(theta, N_T=len(self.a), N_U=len(self.b))
        cos_branch = np.sum(self.a[:, None] * T, axis=0)
        sin_branch = np.sin(theta) * np.sum(self.b[:, None] * U, axis=0) if len(self.b) > 0 else np.zeros_like(cos_branch)
        if complex_out:
            return cos_branch + 1j * sin_branch
        elif stereo:
            return np.stack([cos_branch, sin_branch], axis=-1)
        else:
            return cos_branch + sin_branch

    def evaluate_metrics(self, A_target, phi_target, f0, sr, dur=1.0):
        y = self.synth(f0, sr, dur)
        M = len(y)
        Y = rfft(y)
        K = len(A_target)
        idx = np.round(np.arange(1, K + 1) * f0 * M / sr).astype(int)
        A_fit = 2 * np.abs(Y[idx]) / M
        phi_fit = np.angle(Y[idx])
        phi_target = np.unwrap(phi_target)
        phi_fit = np.unwrap(phi_fit)
        amp_err = np.mean(np.abs(A_fit - A_target))
        phase_err = np.mean(np.abs(np.angle(np.exp(1j * (phi_fit - phi_target)))))
        spec_centroid = np.sum(np.arange(1, K + 1) * A_target) / np.sum(A_target)
        energy_ratio = np.sum(A_fit**2) / (np.sum(A_target**2) + 1e-12)
        return dict(
            amp_err=float(amp_err),
            phase_err=float(phase_err),
            spec_centroid=float(spec_centroid),
            energy_ratio=float(energy_ratio),
            peak_amp=float(np.max(A_target)),
        )

    def plot_fit(self, A_target, phi_target, f0, sr, dur=1.0):
        metrics = self.evaluate_metrics(A_target, phi_target, f0, sr, dur)
        print(f"[ChebyShape] Mean amplitude error: {metrics['amp_err']:.4f}, "
              f"phase error: {metrics['phase_err']:.4f} rad")
        print(f"[ChebyShape] Spectral centroid: {metrics['spec_centroid']:.2f}, "
              f"Energy ratio: {metrics['energy_ratio']:.3f}")

        y = self.synth(f0, sr, dur)
        M = len(y)
        Y = rfft(y)
        K = len(A_target)
        idx = np.round(np.arange(1, K + 1) * f0 * M / sr).astype(int)
        A_fit = 2 * np.abs(Y[idx]) / M
        phi_fit = np.angle(Y[idx])
        phi_target = np.unwrap(phi_target)
        phi_fit = np.unwrap(phi_fit)
        k = np.arange(1, K + 1)

        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax[0].stem(k, A_target, basefmt=" ", linefmt="C0-", markerfmt="C0o", label="Target")
        ax[0].stem(k, A_fit, basefmt=" ", linefmt="C1-", markerfmt="C1s", label="Fitted")
        ax[0].legend(); ax[0].grid(True, alpha=0.3)
        ax[0].set_ylabel("Amplitude")
        ax[1].plot(k, phi_target, "C0o-", label="Target Phase")
        ax[1].plot(k, phi_fit, "C1s-", label="Fitted Phase")
        ax[1].legend(); ax[1].grid(True, alpha=0.3)
        ax[1].set_xlabel("Harmonic number")
        ax[1].set_ylabel("Phase (rad)")
        plt.tight_layout()
        plt.show()
        return metrics


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="ChebyShape CLI — Chebyshev phase regression analyzer")
    parser.add_argument("input", help="Input WAV file")
    parser.add_argument("--f0", type=float, required=True, help="Fundamental frequency (Hz)")
    parser.add_argument("--N", type=int, default=16, help="Chebyshev order")
    parser.add_argument("--K", type=int, default=20, help="Number of harmonics to fit")
    parser.add_argument("--plot", action="store_true", help="Show harmonic amplitude/phase plots")
    parser.add_argument("--export", type=str, help="Export coefficients to JSON")
    parser.add_argument("--json-summary", type=str, help="Export analysis metrics to JSON")
    parser.add_argument("--analyze-only", action="store_true", help="Perform harmonic analysis only (no fitting)")
    parser.add_argument("--render", type=str, help="Render fitted model to output WAV file")
    parser.add_argument("--dur", type=float, default=1.0, help="Render duration (s)")
    parser.add_argument("--stereo", action="store_true", help="Render stereo I/Q (cos/sin) output")
    parser.add_argument("--iq-complex", type=str, help="Export analytic complex signal (.npy + .wav)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"Error: file '{args.input}' not found.")

    sr, y = wavfile.read(args.input)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = y.astype(float)
    y /= np.max(np.abs(y)) + 1e-12

    M = len(y)
    Y = rfft(y)
    bin_fund = int(round(args.f0 * M / sr))
    idx = np.arange(1, args.K + 1) * bin_fund
    A = 2 * np.abs(Y[idx]) / M
    phi = np.angle(Y[idx])

    if args.analyze_only:
        print(f"\n=== Harmonic Analysis ({args.input}) ===")
        for k, (amp, ph) in enumerate(zip(A, phi), 1):
            print(f"H{k:02d}: A = {amp:.6f}, φ = {ph:+.3f} rad")
        if args.plot:
            k = np.arange(1, args.K + 1)
            fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            ax[0].stem(k, A, basefmt=" ", linefmt="C0-", markerfmt="C0o")
            ax[0].set_ylabel("Amplitude"); ax[0].grid(True, alpha=0.3)
            ax[1].plot(k, phi, "C1o-")
            ax[1].set_xlabel("Harmonic number"); ax[1].set_ylabel("Phase (rad)")
            ax[1].grid(True, alpha=0.3)
            plt.suptitle("Harmonic Analysis (No Fitting)")
            plt.tight_layout()
            plt.show()
        sys.exit(0)

    # --- Fit and evaluate ---
    shape = ChebyShape(N=args.N)
    a, b = shape.fit_from_harmonics(A, phi)
    print("\n=== ChebyShape Regression Results ===")
    print(f"Method: {shape.last_fit_info['method']}, Condition Number: {shape.last_fit_info['cond']:.2e}")
    print(f"a coefficients (T_n): {np.round(a, 6)}")
    print(f"b coefficients (U_n): {np.round(b, 6)}")

    metrics = shape.evaluate_metrics(A, phi, args.f0, sr, args.dur)
    print(f"[ChebyShape] Mean amplitude error: {metrics['amp_err']:.4f}, phase error: {metrics['phase_err']:.4f} rad")

    if args.plot:
        shape.plot_fit(A, phi, args.f0, sr)

    if args.export:
        with open(args.export, "w") as f:
            json.dump({"a": a.tolist(), "b": b.tolist(), "info": shape.last_fit_info}, f, indent=2)
        print(f"[ChebyShape] Exported coefficients to {args.export}")

    if args.json_summary:
        summary = dict(
            input=args.input,
            f0=args.f0,
            N=args.N,
            K=args.K,
            method=shape.last_fit_info["method"],
            cond=shape.last_fit_info["cond"],
            **metrics,
            coeffs=dict(a=a.tolist(), b=b.tolist()),
        )
        with open(args.json_summary, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[ChebyShape] Exported summary to {args.json_summary}")

    # --- Rendering / export ---
    if args.render or args.iq_complex:
        y_out = shape.synth(args.f0, sr, args.dur, stereo=args.stereo, complex_out=bool(args.iq_complex))
        y_out = y_out / (np.max(np.abs(y_out)) + 1e-12)

        if args.iq_complex:
            np.save(args.iq_complex, y_out.astype(np.complex64))
            print(f"[ChebyShape] Exported analytic complex waveform to {args.iq_complex}.npy")
            stereo_iq = np.stack([np.real(y_out), np.imag(y_out)], axis=-1)
            wav_path = os.path.splitext(args.iq_complex)[0] + "_iq.wav"
            wavwrite(wav_path, sr, (stereo_iq * 32767).astype(np.int16))
            print(f"[ChebyShape] Rendered stereo I/Q WAV: {wav_path}")
        elif args.render:
            wavwrite(args.render, sr, (y_out * 32767).astype(np.int16))
            print(f"[ChebyShape] Rendered {args.render} ({args.dur:.2f}s @ {sr} Hz)")

if __name__ == "__main__":
    main()
```

---

## 🧠 Example Commands

### Analyze only

```bash
python chebyshape.py input.wav --f0 220 --K 20 --analyze-only --plot
```

### Fit + visualize + JSON metrics

```bash
python chebyshape.py input.wav --f0 220 --N 24 --K 20 --plot --json-summary summary.json
```

### Complex analytic + metrics + coeff export

```bash
python chebyshape.py input.wav --f0 220 --N 24 --K 20 \
  --iq-complex out_complex --json-summary fit_report.json
```

→ Generates:

* `out_complex.npy` — analytic waveform (complex64)
* `out_complex_iq.wav` — stereo (I/Q) render
* `fit_report.json` — full numerical and spectral metrics

---

Would you like me to add a **`--batch` mode** next — so you can feed a folder of WAVs and automatically produce a JSON report per file (for dataset or plugin calibration work)?
