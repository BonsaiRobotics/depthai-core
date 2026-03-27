#!/usr/bin/env python3
"""Minimal ToF replay — sends a raw .bin superframe through the DAI ToF pipeline.

Uses a ThreadedHostNode to feed raw frames into tof.rawInput, same pattern
as the full tof_vpp_test.py but stripped down for quick testing.

Supported superframe (width, height) and expected byte size (RAW12 packed):
  - 1344×2420  1F full     → 4 878 720 bytes
  - 1344×4832  2F full     → 9 757 440 bytes
  - 1344×7244  3F full     → 14 636 160 bytes
  - 672×2420   2F bin 2×2  → 2 439 360 bytes
  - 672×3626   3F bin 2×2  → 3 655 008 bytes

Usage:
    python3 tof_replay.py -i superframe.bin -d 10.11.102.145
    python3 tof_replay.py -i superframe.bin -d 10.11.102.145 --save --png

    # With ToF config overrides:
    python3 tof_replay.py -i superframe.bin -d 10.11.102.145 \\
        --phase-unwrapping-level 5 --enable-bilateral-filter \\
        --bilateral-kernel-size 5 --enable-flying-pixel-correction

    # Show all defaults without running:
    python3 tof_replay.py --show-config
"""
import os, sys, time, argparse
import numpy as np

os.environ["DEPTHAI_LEVEL"] = "debug"
os.environ["DEPTHAI_DEVICE_RVC4_FWP"] = "/home/david/depthai-device/build_docker_arm64_rvc4/Release/depthai-device-rvc4-fwp.tar.xz"
sys.path.insert(0, "/home/david/depthai-device/external/depthai-core/build/bindings/python")

# Superframe resolutions: (raw_w, raw_h) → context_frames
STATUS_LINE_NB = 8
SUBFRAME_PER_CONTEXT = 3
USE_CASES = {
    (1344, 2420): 1, (1344, 4832): 2, (1344, 7244): 3,
    (672, 2420): 2, (672, 3626): 3,
}

# ---------------------------------------------------------------------------
# ToF config defaults — mirrors dai::ToFConfig (ToFConfig.hpp)
# Fields marked (optional) are std::optional in C++ — None means "use IPP default".
# ---------------------------------------------------------------------------
TOF_CONFIG_DEFAULTS = {
    # --- Core decoding ---
    "median":                           "MEDIAN_OFF",  # MEDIAN_OFF / KERNEL_3x3 / KERNEL_5x5 / KERNEL_7x7
    "phaseUnwrappingLevel":             4,
    "phaseUnwrapErrorThreshold":        100,
    "enablePhaseShuffleTemporalFilter": True,
    "enableBurstMode":                  False,
    "enableDistortionCorrection":       True,

    # --- Debug toggles (optional — None = IPP default) ---
    "enableFPPNCorrection":             None,   # Fixed-Pattern-Phase-Noise correction
    "enableOpticalCorrection":          None,   # Optical correction
    "enableTemperatureCorrection":      None,   # Temperature drift correction
    "enableWiggleCorrection":           None,   # Wiggle correction
    "enablePhaseUnwrapping":            None,   # Phase unwrapping master switch

    # --- Bilateral filter (optional — None = IPP default) ---
    "enableBilateralFilter":            None,   # True/False to enable/bypass
    "bilateralStdFactor":               None,   # float — std deviation factor
    "bilateralFilterKernelSize":        None,   # uint32 — kernel size (e.g. 3, 5, 7)

    # --- Temporal noise reduction (optional — None = IPP default) ---
    "enableTemporalNoiseReduction":     None,   # True/False to enable/bypass
    "tnrMaxGain":                       None,   # float — max TNR gain
    "tnrStdFactor":                     None,   # float — TNR std factor

    # --- Flying pixel correction (optional — None = IPP default) ---
    "enableFlyingPixelCorrection":      None,   # True/False to enable/bypass
    "fpDepthThreshold":                 None,   # float — depth threshold (mm)
    "fpMinDepthOccurrence":             None,   # float — min depth occurrence

    # --- Radial to perpendicular (optional — None = IPP default) ---
    "enableRadialToPerp":               None,   # True/False to enable/bypass
}


def add_tof_config_args(parser):
    """Add CLI arguments for every ToFConfig field."""
    g = parser.add_argument_group("ToF config overrides",
        "Override individual ToFConfig fields. Unset options keep their defaults.")

    # Core decoding
    g.add_argument("--median", choices=["off", "3x3", "5x5", "7x7"], default=None,
                   help="Median filter kernel (default: off)")
    g.add_argument("--phase-unwrapping-level", type=int, default=None,
                   help="Phase unwrapping level (default: 4)")
    g.add_argument("--phase-unwrap-error-threshold", type=int, default=None,
                   help="Phase unwrap error threshold (default: 100)")
    g.add_argument("--enable-phase-shuffle-temporal-filter", type=_str2bool, default=None,
                   help="Enable phase shuffle temporal filter (default: True)")
    g.add_argument("--enable-burst-mode", type=_str2bool, default=None,
                   help="Enable burst mode — 4-frame decoding (default: False)")
    g.add_argument("--enable-distortion-correction", type=_str2bool, default=None,
                   help="Enable distortion correction (default: True)")

    # Debug toggles
    g.add_argument("--enable-fppn-correction", type=_str2bool, default=None,
                   help="Enable FPN correction (optional, default: IPP default)")
    g.add_argument("--enable-optical-correction", type=_str2bool, default=None,
                   help="Enable optical correction (optional, default: IPP default)")
    g.add_argument("--enable-temperature-correction", type=_str2bool, default=None,
                   help="Enable temperature correction (optional, default: IPP default)")
    g.add_argument("--enable-wiggle-correction", type=_str2bool, default=None,
                   help="Enable wiggle correction (optional, default: IPP default)")
    g.add_argument("--enable-phase-unwrapping", type=_str2bool, default=None,
                   help="Enable phase unwrapping (optional, default: IPP default)")

    # Bilateral filter
    g.add_argument("--enable-bilateral-filter", type=_str2bool, default=None,
                   help="Enable bilateral filter (optional, default: IPP default)")
    g.add_argument("--bilateral-std-factor", type=float, default=None,
                   help="Bilateral filter std factor (optional)")
    g.add_argument("--bilateral-kernel-size", type=int, default=None,
                   help="Bilateral filter kernel size (optional, e.g. 3, 5, 7)")

    # TNR
    g.add_argument("--enable-tnr", type=_str2bool, default=None,
                   help="Enable temporal noise reduction (optional, default: IPP default)")
    g.add_argument("--tnr-max-gain", type=float, default=None,
                   help="TNR max gain (optional)")
    g.add_argument("--tnr-std-factor", type=float, default=None,
                   help="TNR std factor (optional)")

    # Flying pixel correction
    g.add_argument("--enable-flying-pixel-correction", type=_str2bool, default=None,
                   help="Enable flying pixel correction (optional, default: IPP default)")
    g.add_argument("--fp-depth-threshold", type=float, default=None,
                   help="Flying pixel depth threshold in mm (optional)")
    g.add_argument("--fp-min-depth-occurrence", type=float, default=None,
                   help="Flying pixel min depth occurrence (optional)")

    # R2P
    g.add_argument("--enable-radial-to-perp", type=_str2bool, default=None,
                   help="Enable radial-to-perpendicular conversion (optional, default: IPP default)")

    # Preset
    g.add_argument("--preset", choices=["short", "mid", "long"], default=None,
                   help="Apply a preset (TOF_SHORT_RANGE / TOF_MID_RANGE / TOF_LONG_RANGE) before individual overrides")

    # Show config and exit
    g.add_argument("--show-config", action="store_true",
                   help="Print all ToFConfig defaults and exit")


def _str2bool(v):
    """Argparse helper: accept true/false/1/0/yes/no."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "1", "on"):
        return True
    if v.lower() in ("no", "false", "0", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def apply_tof_config(tof_cfg, args):
    """Apply CLI overrides to a dai.ToFConfig object. Returns a summary dict."""
    import depthai as dai

    
    # Apply preset first (individual overrides take precedence)
    if args.preset is not None:
        tof_cfg.setProfilePreset(PRESET_MAP[args.preset])

    overrides = {}

    if args.median is not None:
        tof_cfg.median = MEDIAN_MAP[args.median]
        overrides["median"] = args.median
    if args.phase_unwrapping_level is not None:
        tof_cfg.phaseUnwrappingLevel = args.phase_unwrapping_level
        overrides["phaseUnwrappingLevel"] = args.phase_unwrapping_level
    if args.phase_unwrap_error_threshold is not None:
        tof_cfg.phaseUnwrapErrorThreshold = args.phase_unwrap_error_threshold
        overrides["phaseUnwrapErrorThreshold"] = args.phase_unwrap_error_threshold
    if args.enable_phase_shuffle_temporal_filter is not None:
        tof_cfg.enablePhaseShuffleTemporalFilter = args.enable_phase_shuffle_temporal_filter
        overrides["enablePhaseShuffleTemporalFilter"] = args.enable_phase_shuffle_temporal_filter
    if args.enable_burst_mode is not None:
        tof_cfg.enableBurstMode = args.enable_burst_mode
        overrides["enableBurstMode"] = args.enable_burst_mode
    if args.enable_distortion_correction is not None:
        tof_cfg.enableDistortionCorrection = args.enable_distortion_correction
        overrides["enableDistortionCorrection"] = args.enable_distortion_correction

    # Debug toggles (optional fields)
    if args.enable_fppn_correction is not None:
        tof_cfg.enableFPPNCorrection = args.enable_fppn_correction
        overrides["enableFPPNCorrection"] = args.enable_fppn_correction
    if args.enable_optical_correction is not None:
        tof_cfg.enableOpticalCorrection = args.enable_optical_correction
        overrides["enableOpticalCorrection"] = args.enable_optical_correction
    if args.enable_temperature_correction is not None:
        tof_cfg.enableTemperatureCorrection = args.enable_temperature_correction
        overrides["enableTemperatureCorrection"] = args.enable_temperature_correction
    if args.enable_wiggle_correction is not None:
        tof_cfg.enableWiggleCorrection = args.enable_wiggle_correction
        overrides["enableWiggleCorrection"] = args.enable_wiggle_correction
    if args.enable_phase_unwrapping is not None:
        tof_cfg.enablePhaseUnwrapping = args.enable_phase_unwrapping
        overrides["enablePhaseUnwrapping"] = args.enable_phase_unwrapping

    # Bilateral filter
    if args.enable_bilateral_filter is not None:
        tof_cfg.enableBilateralFilter = args.enable_bilateral_filter
        overrides["enableBilateralFilter"] = args.enable_bilateral_filter
    if args.bilateral_std_factor is not None:
        tof_cfg.bilateralStdFactor = args.bilateral_std_factor
        overrides["bilateralStdFactor"] = args.bilateral_std_factor
    if args.bilateral_kernel_size is not None:
        tof_cfg.bilateralFilterKernelSize = args.bilateral_kernel_size
        overrides["bilateralFilterKernelSize"] = args.bilateral_kernel_size

    # TNR
    if args.enable_tnr is not None:
        tof_cfg.enableTemporalNoiseReduction = args.enable_tnr
        overrides["enableTemporalNoiseReduction"] = args.enable_tnr
    if args.tnr_max_gain is not None:
        tof_cfg.tnrMaxGain = args.tnr_max_gain
        overrides["tnrMaxGain"] = args.tnr_max_gain
    if args.tnr_std_factor is not None:
        tof_cfg.tnrStdFactor = args.tnr_std_factor
        overrides["tnrStdFactor"] = args.tnr_std_factor

    # Flying pixel correction
    if args.enable_flying_pixel_correction is not None:
        tof_cfg.enableFlyingPixelCorrection = args.enable_flying_pixel_correction
        overrides["enableFlyingPixelCorrection"] = args.enable_flying_pixel_correction
    if args.fp_depth_threshold is not None:
        tof_cfg.fpDepthThreshold = args.fp_depth_threshold
        overrides["fpDepthThreshold"] = args.fp_depth_threshold
    if args.fp_min_depth_occurrence is not None:
        tof_cfg.fpMinDepthOccurrence = args.fp_min_depth_occurrence
        overrides["fpMinDepthOccurrence"] = args.fp_min_depth_occurrence

    # R2P
    if args.enable_radial_to_perp is not None:
        tof_cfg.enableRadialToPerp = args.enable_radial_to_perp
        overrides["enableRadialToPerp"] = args.enable_radial_to_perp

    return overrides


def print_tof_config_table():
    """Print a readable table of all ToFConfig fields with defaults and descriptions."""
    print("\n=== ToFConfig — Full Parameter Reference ===\n")
    print(f"{'Field':<42} {'Default':<18} {'CLI flag':<42} {'Description'}")
    print("-" * 150)
    rows = [
        # (field, default, cli_flag, description)
        ("median",                           "MEDIAN_OFF",  "--median",                            "Median filter: off / 3x3 / 5x5 / 7x7"),
        ("phaseUnwrappingLevel",             "4",           "--phase-unwrapping-level",             "Phase unwrapping level (int)"),
        ("phaseUnwrapErrorThreshold",        "100",         "--phase-unwrap-error-threshold",       "Phase unwrap error threshold (uint16)"),
        ("enablePhaseShuffleTemporalFilter", "True",        "--enable-phase-shuffle-temporal-filter","Temporal avg of shuffle/non-shuffle freqs"),
        ("enableBurstMode",                  "False",       "--enable-burst-mode",                  "4-frame decoding — lower fps, less motion blur"),
        ("enableDistortionCorrection",       "True",        "--enable-distortion-correction",       "Distortion correction for depth/amp/intensity"),
        ("",                                 "",            "",                                    ""),
        ("enableFPPNCorrection",             "None (IPP)",  "--enable-fppn-correction",             "Fixed-pattern phase-noise correction"),
        ("enableOpticalCorrection",          "None (IPP)",  "--enable-optical-correction",          "Optical correction"),
        ("enableTemperatureCorrection",      "None (IPP)",  "--enable-temperature-correction",      "Temperature drift compensation"),
        ("enableWiggleCorrection",           "None (IPP)",  "--enable-wiggle-correction",           "Wiggle correction"),
        ("enablePhaseUnwrapping",            "None (IPP)",  "--enable-phase-unwrapping",            "Phase unwrapping master switch"),
        ("",                                 "",            "",                                    ""),
        ("enableBilateralFilter",            "None (IPP)",  "--enable-bilateral-filter",            "Bilateral depth filter enable/bypass"),
        ("bilateralStdFactor",               "None (IPP)",  "--bilateral-std-factor",               "Bilateral filter std deviation factor"),
        ("bilateralFilterKernelSize",        "None (IPP)",  "--bilateral-kernel-size",              "Bilateral filter kernel size (3, 5, 7)"),
        ("",                                 "",            "",                                    ""),
        ("enableTemporalNoiseReduction",     "None (IPP)",  "--enable-tnr",                         "Temporal noise reduction enable/bypass"),
        ("tnrMaxGain",                       "None (IPP)",  "--tnr-max-gain",                       "TNR maximum gain"),
        ("tnrStdFactor",                     "None (IPP)",  "--tnr-std-factor",                     "TNR std deviation factor"),
        ("",                                 "",            "",                                    ""),
        ("enableFlyingPixelCorrection",      "None (IPP)",  "--enable-flying-pixel-correction",     "Flying pixel correction enable/bypass"),
        ("fpDepthThreshold",                 "None (IPP)",  "--fp-depth-threshold",                 "Flying pixel depth threshold (mm)"),
        ("fpMinDepthOccurrence",             "None (IPP)",  "--fp-min-depth-occurrence",            "Flying pixel min depth occurrence"),
        ("",                                 "",            "",                                    ""),
        ("enableRadialToPerp",               "None (IPP)",  "--enable-radial-to-perp",              "Radial-to-perpendicular depth conversion"),
    ]
    for field, default, cli, desc in rows:
        if not field:
            print()
            continue
        print(f"  {field:<40} {default:<18} {cli:<42} {desc}")
    print()
    print("Presets (applied before individual overrides): --preset short|mid|long")
    print("  TOF_SHORT_RANGE / TOF_MID_RANGE / TOF_LONG_RANGE")
    print()

def detect_resolution(nbytes):
    for (w, h), ctx in USE_CASES.items():
        if nbytes == int(w * h * 1.5):
            return w, h, ctx
    return None, None, None

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--input", "-i", default=None, help="RAW12 superframe .bin")
    ap.add_argument("--device", "-d", default=None, help="Device IP or name")
    ap.add_argument("--save", "-s", action="store_true", help="Save .npy outputs")
    ap.add_argument("--png", action="store_true", help="Save PNG outputs")
    ap.add_argument("--timeout", "-t", type=float, default=15.0, help="Seconds to wait (default: 15)")
    add_tof_config_args(ap)
    args = ap.parse_args()

    # --show-config: print reference table and exit
    if args.show_config:
        print_tof_config_table()
        sys.exit(0)

    if args.input is None:
        ap.error("--input/-i is required (unless using --show-config)")

    if args.device:
        os.environ["DEPTHAI_DEVICE_NAME_LIST"] = args.device

    raw_data = open(args.input, "rb").read()
    raw_size = len(raw_data)
    raw_w, raw_h, ctx = detect_resolution(raw_size)
    if raw_w is None:
        sys.exit(f"Cannot detect resolution from file size {raw_size}.\n"
                 f"Supported sizes: {sorted(int(w*h*1.5) for (w,h) in USE_CASES)}")

    dep_w = raw_w // 2
    dep_h = (raw_h - STATUS_LINE_NB) // (ctx * SUBFRAME_PER_CONTEXT)
    print(f"File size       : {raw_size} bytes ({raw_size / 1024 / 1024:.2f} MB)")
    print(f"Superframe      : {raw_w}x{raw_h}  ({ctx}F)")
    print(f"Expected depth  : {dep_w}x{dep_h}")

    import depthai as dai
    raw_np = np.frombuffer(raw_data, dtype=np.uint8)

    class RawReplay(dai.node.ThreadedHostNode):
        """Host node that sends a single raw superframe into the pipeline."""
        def __init__(self):
            dai.node.ThreadedHostNode.__init__(self)
            self.output = self.createOutput()
            self._data = None
            self._w = self._h = None

        def set_frame(self, data, w, h):
            self._data = data
            self._w, self._h = w, h

        def run(self):
            while self.mainLoop():
                if self._data is not None:
                    frame = dai.ImgFrame()
                    frame.setData(self._data)
                    frame.setWidth(self._w)
                    frame.setHeight(self._h)
                    frame.setType(dai.ImgFrame.Type.RAW12)
                    frame.setInstanceNum(0)
                    frame.setSequenceNum(0)
                    frame.setTimestamp(dai.Clock.now())
                    self.output.send(frame)
                    self._data = None
                time.sleep(0.05)

    print("\nBuilding pipeline...")
    with dai.Pipeline() as pipeline:
        replay = pipeline.create(RawReplay)
        tof = pipeline.create(dai.node.ToF)

        # --- Apply ToF config ---
        tof_cfg = tof.tofBaseNode.initialConfig
        overrides = apply_tof_config(tof_cfg, args)

        # Print active config
        print("\n--- ToFConfig ---")
        print(f"  median                           = {tof_cfg.median}")
        print(f"  phaseUnwrappingLevel             = {tof_cfg.phaseUnwrappingLevel}")
        print(f"  phaseUnwrapErrorThreshold         = {tof_cfg.phaseUnwrapErrorThreshold}")
        print(f"  enablePhaseShuffleTemporalFilter = {tof_cfg.enablePhaseShuffleTemporalFilter}")
        print(f"  enableBurstMode                  = {tof_cfg.enableBurstMode}")
        print(f"  enableDistortionCorrection       = {tof_cfg.enableDistortionCorrection}")
        print(f"  enableFPPNCorrection             = {tof_cfg.enableFPPNCorrection}")
        print(f"  enableOpticalCorrection          = {tof_cfg.enableOpticalCorrection}")
        print(f"  enableTemperatureCorrection      = {tof_cfg.enableTemperatureCorrection}")
        print(f"  enableWiggleCorrection           = {tof_cfg.enableWiggleCorrection}")
        print(f"  enablePhaseUnwrapping            = {tof_cfg.enablePhaseUnwrapping}")
        print(f"  enableBilateralFilter            = {tof_cfg.enableBilateralFilter}")
        print(f"  bilateralStdFactor               = {tof_cfg.bilateralStdFactor}")
        print(f"  bilateralFilterKernelSize        = {tof_cfg.bilateralFilterKernelSize}")
        print(f"  enableTemporalNoiseReduction     = {tof_cfg.enableTemporalNoiseReduction}")
        print(f"  tnrMaxGain                       = {tof_cfg.tnrMaxGain}")
        print(f"  tnrStdFactor                     = {tof_cfg.tnrStdFactor}")
        print(f"  enableFlyingPixelCorrection      = {tof_cfg.enableFlyingPixelCorrection}")
        print(f"  fpDepthThreshold                 = {tof_cfg.fpDepthThreshold}")
        print(f"  fpMinDepthOccurrence             = {tof_cfg.fpMinDepthOccurrence}")
        print(f"  enableRadialToPerp               = {tof_cfg.enableRadialToPerp}")
        if overrides:
            print(f"  ** Overrides applied: {overrides}")
        print("--- end config ---\n")

        replay.output.link(tof.rawInput)

        depthQ     = tof.rawDepth.createOutputQueue()
        ampQ       = tof.amplitude.createOutputQueue()
        confQ      = tof.confidence.createOutputQueue()
        intensityQ = tof.intensity.createOutputQueue()

        pipeline.start()
        print("Pipeline started, sending frame...")

        replay.set_frame(raw_np, raw_w, raw_h)

        # Wait for outputs
        depth_frame = amp_frame = conf_frame = intensity_frame = None
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            if depth_frame is None:
                depth_frame = depthQ.tryGet()
            if amp_frame is None:
                amp_frame = ampQ.tryGet()
            if conf_frame is None:
                conf_frame = confQ.tryGet()
            if intensity_frame is None:
                intensity_frame = intensityQ.tryGet()
            if depth_frame and amp_frame:
                break
            time.sleep(0.02)

        if depth_frame is None:
            sys.exit("No depth frame received (timeout)")

        depth = depth_frame.getCvFrame().astype(np.float32)
        print(f"Depth     : {depth.shape[1]}x{depth.shape[0]}  "
              f"range [{depth.min():.0f}, {depth.max():.0f}] mm  "
              f"non-zero {np.count_nonzero(depth)}/{depth.size}")

        amp = amp_frame.getCvFrame() if amp_frame else None
        if amp is not None:
            print(f"Amplitude : {amp.shape[1]}x{amp.shape[0]}  "
                  f"range [{amp.min():.2f}, {amp.max():.2f}]")

        conf = conf_frame.getCvFrame() if conf_frame else None
        if conf is not None:
            print(f"Confidence: {conf.shape[1]}x{conf.shape[0]}  "
                  f"range [{conf.min():.3f}, {conf.max():.3f}]")

        intensity = intensity_frame.getCvFrame() if intensity_frame else None
        if intensity is not None:
            print(f"Intensity : {intensity.shape[1]}x{intensity.shape[0]}  "
                  f"range [{intensity.min():.2f}, {intensity.max():.2f}]")

        if args.save:
            np.save("tof_depth.npy", depth)
            if amp is not None: np.save("tof_amplitude.npy", amp)
            if conf is not None: np.save("tof_confidence.npy", conf)
            if intensity is not None: np.save("tof_intensity.npy", intensity)
            print("Saved .npy files")

        if args.png:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            n_plots = 1 + (amp is not None) + (conf is not None) + (intensity is not None)
            fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
            if n_plots == 1: axes = [axes]
            idx = 0
            axes[idx].imshow(np.ma.array(depth, mask=(depth == 0)), cmap="turbo")
            axes[idx].set_title("Depth"); axes[idx].axis("off"); idx += 1
            if amp is not None:
                axes[idx].imshow(amp, cmap="gray")
                axes[idx].set_title("Amplitude"); axes[idx].axis("off"); idx += 1
            if conf is not None:
                axes[idx].imshow(conf, cmap="gray")
                axes[idx].set_title("Confidence"); axes[idx].axis("off"); idx += 1
            if intensity is not None:
                axes[idx].imshow(intensity, cmap="gray")
                axes[idx].set_title("Intensity"); axes[idx].axis("off")
            plt.tight_layout()
            plt.savefig("tof_output.png", dpi=150, bbox_inches="tight")
            print("Saved tof_output.png")

        print("Done")

if __name__ == "__main__":
    main()
