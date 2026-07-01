"""
Electrode montage for the Kaggle Grasp-and-Lift EEG Detection dataset.

The dataset uses a 32-channel actiCAP layout in the international 10-20 system.
This module provides an approximate but topologically faithful set of scalp
positions and geodesic (great-circle) distance utilities. It is used by the
volume-conduction control to test whether reproducible directed edges are
dominated by physically adjacent electrode pairs, which would be expected under
volume conduction rather than genuine connectivity.

Coordinate conventions:
- The 32 electrodes are placed on a unit sphere (head model).
- Positions are derived from the standard 2D topographic layout (azimuthal
  equidistant projection) used for EEG topomaps: a 2D radius `r` maps linearly
  to the polar angle from the vertex (theta = r * 90 degrees), and the 2D angle
  becomes the azimuth.
- Axes: +x = right ear, +y = nose (anterior), +z = vertex (superior).

The absolute scale is approximate; the analysis depends only on the relative
ordering of inter-electrode distances (near vs far), which this layout
reproduces faithfully. A nominal scalp radius is used only to express distances
in centimetres for readability.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

# Nominal adult scalp radius (cm). Used only to convert the unit-sphere
# geodesic angle into an approximate, human-readable centimetre distance.
SCALP_RADIUS_CM: float = 9.2

# 32-channel Grasp-and-Lift montage (BrainAmp / actiCAP, 10-20 system).
GAL_CHANNELS: List[str] = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8",
    "TP9", "CP5", "CP1", "CP2", "CP6", "TP10",
    "P7", "P3", "Pz", "P4", "P8",
    "PO9", "O1", "Oz", "O2", "PO10",
]

# Standard topographic 2D layout: (x_right, y_anterior).
# r = sqrt(x^2 + y^2) maps to polar angle theta = r * 90 degrees.
# The equator (theta = 90 deg) runs through the ears (T7/T8) and Oz.
_LAYOUT_2D: Dict[str, Tuple[float, float]] = {
    "Fp1": (-0.31, 0.95), "Fp2": (0.31, 0.95),
    "F7": (-0.81, 0.59), "F3": (-0.40, 0.67), "Fz": (0.00, 0.72),
    "F4": (0.40, 0.67), "F8": (0.81, 0.59),
    "FC5": (-0.78, 0.27), "FC1": (-0.28, 0.34),
    "FC2": (0.28, 0.34), "FC6": (0.78, 0.27),
    "T7": (-1.00, 0.00), "C3": (-0.50, 0.00), "Cz": (0.00, 0.00),
    "C4": (0.50, 0.00), "T8": (1.00, 0.00),
    "TP9": (-1.05, -0.35), "CP5": (-0.78, -0.27), "CP1": (-0.28, -0.34),
    "CP2": (0.28, -0.34), "CP6": (0.78, -0.27), "TP10": (1.05, -0.35),
    "P7": (-0.81, -0.59), "P3": (-0.40, -0.67), "Pz": (0.00, -0.72),
    "P4": (0.40, -0.67), "P8": (0.81, -0.59),
    "PO9": (-0.60, -0.95), "O1": (-0.31, -0.95), "Oz": (0.00, -1.00),
    "O2": (0.31, -0.95), "PO10": (0.60, -0.95),
}


def _to_unit_sphere(x: float, y: float) -> Tuple[float, float, float]:
    """Map a 2D topographic position to a unit-sphere 3D coordinate."""
    r = math.hypot(x, y)
    theta = math.radians(r * 90.0)  # polar angle from vertex
    azimuth = math.atan2(y, x)
    sin_t = math.sin(theta)
    return (
        sin_t * math.cos(azimuth),
        sin_t * math.sin(azimuth),
        math.cos(theta),
    )


def channel_positions_3d() -> Dict[str, Tuple[float, float, float]]:
    """Return unit-sphere 3D positions for every montage channel."""
    return {name: _to_unit_sphere(x, y) for name, (x, y) in _LAYOUT_2D.items()}


_POSITIONS_3D: Dict[str, Tuple[float, float, float]] = channel_positions_3d()


def has_channel(name: str) -> bool:
    """Whether the given channel name is part of the known montage."""
    return name in _POSITIONS_3D


def geodesic_distance_rad(a: str, b: str) -> float:
    """
    Great-circle angular distance (radians) between two electrodes.

    Raises KeyError if either channel is unknown.
    """
    pa = _POSITIONS_3D[a]
    pb = _POSITIONS_3D[b]
    dot = pa[0] * pb[0] + pa[1] * pb[1] + pa[2] * pb[2]
    dot = max(-1.0, min(1.0, dot))
    return math.acos(dot)


def geodesic_distance_cm(a: str, b: str) -> float:
    """Approximate scalp-surface distance (cm) between two electrodes."""
    return geodesic_distance_rad(a, b) * SCALP_RADIUS_CM


def all_directed_pair_distances_cm(
    channels: List[str] | None = None,
) -> List[float]:
    """
    Distances (cm) for every ordered channel pair (i != j).

    This is the spatial null universe used by the volume-conduction control:
    if reproducible edges were selected at random, their distance distribution
    should match this set.
    """
    names = channels if channels is not None else GAL_CHANNELS
    known = [c for c in names if has_channel(c)]
    out: List[float] = []
    for i, a in enumerate(known):
        for j, b in enumerate(known):
            if i == j:
                continue
            out.append(geodesic_distance_cm(a, b))
    return out
