"""
Haploscope setup and management for dual-monitor stereoscopic displays.

This module provides functions to create and manage two PsychoPy windows
on separate monitors for haploscope-based depth perception experiments.
"""
#Core libraries
import math

#Third party libraries
import numpy as np
from psychopy import visual, monitors
from pathlib import Path

#Local libraries
from . import haploscopeConfig as config

#------------------------------------------------------------------------------
# Physical calibration functions
#------------------------------------------------------------------------------
def calc_display_positions(
    focal_distance: float,
    config: dict[str, float]
    ) -> tuple[float, float]:
    """Return the left/right display carriage positions for ``focal_distance``.

    Parameters
    ----------
    focal_distance:
        Requested distance from the mirrors to the focal plane in millimetres.

    Returns
    -------
    tuple
        (left_position, right_position) in the millimeters.
    """

    distance_change = abs(focal_distance) - config["MIN_FOCAL_DISTANCE"]
    left_pos = config["DISPLAY_LEFT_ZERO"] - distance_change
    right_pos = config["DISPLAY_RIGHT_ZERO"] + distance_change
    return left_pos, right_pos


def calc_eye_positions(
    iod: float,
    config: dict[str, float]
    ) -> tuple[float, float]:
    """Return mirror eye positions for a given interpupillary distance."""

    distance_change = (abs(iod) - config["MIN_IOD"]) / 2.0
    left_pos = config["EYE_LEFT_ZERO"] - distance_change
    right_pos = config["EYE_RIGHT_ZERO"] + distance_change
    return left_pos, right_pos


def calc_arm_rotations(iod: float, focal_distance: float) -> float:
    """Return the mirror arm rotation angle in degrees.
    """

    angle_rad = math.atan(0.5 * iod / focal_distance)
    return math.degrees(angle_rad)


def calc_physical_calibration(
    iod: float,
    focal_distance_mm: float,
    config: dict[str, float]
    ) -> dict[str, float]:
    """Return a dict summarising all physical calibration values in mm.
    """

    display_left, display_right = calc_display_positions(focal_distance_mm, config)
    eye_left, eye_right = calc_eye_positions(iod, config)
    angle = calc_arm_rotations(iod, focal_distance_mm)
    return {
        "DISPLAY_LEFT": display_left,
        "DISPLAY_RIGHT": display_right,
        "EYE_LEFT": eye_left,
        "EYE_RIGHT": eye_right,
        "ANGLE": angle,
    }


#------------------------------------------------------------------------------
# Window setup functions
#------------------------------------------------------------------------------
def setup_haploscope_windows(
    monitor_names: tuple[str, str] = ("left", "right"),
    screen_indices: tuple[int, int] = (0, 1),
    size_pix: tuple[int, int] = [],
    fullscr: bool = True,
    color: tuple[float, float, float] = (-1, -1, -1),
    waitBlanking: bool = True,
    **kwargs
) -> tuple[visual.Window, visual.Window]:
    """
    Set up two PsychoPy windows on separate monitors for haploscope display.
    
    Parameters
    ----------
    monitor_names : tuple of str, optional
        Names of the monitor configurations (default: ("left", "right"))
        These should match monitor names in PsychoPy Monitor Center.
    screen_indices : tuple of int, optional
        Which physical screens to use (default: (0, 1))
        0 is typically the primary display, 1 is the secondary.
    size_pix : tuple[int, int], optional
        Window size in pixels. If None, uses full screen size.
    fullscr : bool, optional
        Whether to use fullscreen mode (default: True)
    color : tuple[float, float, float], optional
        Background color (default: (-1, -1, -1) for black in PsychoPy's -1 to 1 color space)
    waitBlanking : bool, optional
        Whether to wait for vertical blank before flipping (default: True)
        Set to False on one window if you want to control sync manually.
    **kwargs
        Additional arguments passed to visual.Window constructor
        
    Returns
    -------
    tuple of visual.Window
        (left_window, right_window) - Two PsychoPy window objects
        
    Examples
    --------
    >>> left_win, right_win = setup_haploscope_windows(
    ...     monitor_names=("left_monitor", "right_monitor"),
    ...     screen_indices=(0, 1)
    ... )
    """
    left_monitor, right_monitor = monitor_names
    left_screen, right_screen = screen_indices

    if size_pix == []:
        size_pix = (800, 600)
        fullscr = False
        print("Using default size (800x600) for non-fullscreen mode")
    # Create left window
    left_window = visual.Window(
        size=size_pix,
        screen=left_screen,
        fullscr=fullscr,
        monitor=left_monitor,
        units="pix",
        color=color,
        waitBlanking=waitBlanking,
        **kwargs
    )
    
    # Create right window
    right_window = visual.Window(
        size=size_pix,
        screen=right_screen,
        fullscr=fullscr,
        monitor=right_monitor,
        units="pix",
        color=color,
        waitBlanking=waitBlanking,
        **kwargs
    )
    
    return left_window, right_window

def setup_single_window(
    monitor_name: str = "left",
    screen_index: int = 0,
    size_pix: tuple[int, int] = [],
    fullscr: bool = True,
    color: tuple[float, float, float] = (-1, -1, -1),
    waitBlanking: bool = True,
    **kwargs
) -> visual.Window:
    """Set up a single PsychoPy window on a monitor for haploscope display."""
    if size_pix == []:
        size_pix = (800, 600)
        fullscr = False
        print("Using default size (800x600) for non-fullscreen mode")
    return visual.Window(
        size=size_pix,
        screen=screen_index,
        fullscr=fullscr,
        monitor=monitor_name,
        color=color,
        waitBlanking=waitBlanking,
        **kwargs
    )

    return window