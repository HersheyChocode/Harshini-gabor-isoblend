"""
Initialize variables for the experiment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class ExperimentConfig:
    """Container for experiment parameters and runtime options."""

    experiment_name: str = "johnston_stereopsis"
    meta_data: List[str] = field(
        default_factory=lambda: [
            "participant",
            "calibration_metadata"
        ]
    )
    data_fields: List[str] = field(
        default_factory=lambda: [
            "trial_index",
            "stimulus_id",
            "stimulus_label",
            "response_key",
            "response_label",
            "rt_s",
            "stimulus_duration_s",
            "fixation_duration_s",
            "left_image",
            "right_image",
            "stimulus_metadata"
        ]
    )
    stimulus_duration_s: float = 1.5
    fixation_duration_s: float = 0.75
    prompt_display_duration_s: float = 0.75
    response_keys: Dict[str, str] = field(
        default_factory=lambda: {"1": "squashed", "2": "stretched"}
    )
    max_trials: int = 10
    break_after_trials: int = 3
    break_duration_s: float = 10.0
    break_resume_key: str = "3"
    break_message: str = (
        "Break time!\nPlease rest your eyes.\n"
        "The experiment will resume automatically, or press {key} to continue early."
    )
    participant_keyboard_name: Optional[str] = None
    experimenter_keyboard_name: Optional[str] = None
    participant_serial_port: Optional[str] = None
    participant_serial_baud: int = 9600
    stimulus_directory: str = "stimuli"
    results_directory: str = "data"
    left_screen_index: int = 1
    right_screen_index: int = 0
    full_screen: bool = True
    window_size: Tuple[int, int] = (1280, 720)
    window_units: str = "pix"
    monitor_name: str = "testMonitor"
    monitor_distance_cm: float = 70.0
    background_color: Sequence[float] = (-1.0, -1.0, -1.0)
    quit_keys: Tuple[str, ...] = ("escape",)
    log_calibration_to_console: bool = False
    iod_override_mm: Optional[float] = None
    focal_override_mm: Optional[float] = None
    use_right_viewport: bool = True
    debug_mode: bool = False
    experimenter_screen_index: int = 2
    debug_window_size: Tuple[int, int] = (1024, 768)
    debug_screen_index: int = 0
    debug_iod_mm: Optional[float] = None
    debug_focal_mm: Optional[float] = None

    
