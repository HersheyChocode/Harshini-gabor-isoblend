from itertools import product
import enum
import numpy as np
#------------------------------------------------------------------------------
# Experiment Control Variables
#------------------------------------------------------------------------------
class ExperimentPhase(enum.Enum):
    """
    Enum for the experiment phases.
    """
    INITIALIZE = "initialize"
    ADJUST_HAPLOSCOPE = "adjust_haploscope"
    PRE_TRIAL = "pre_trial"
    TRIAL = "trial"
    POST_TRIAL = "post_trial"
    BREAK = "break"
    END = "end"

class ResponsePhase(enum.Enum):
    """
    Enum for the participant responses.
    """
    NO_RESPONSE = "no_response"
    WAIT_FOR_RESPONSE = "wait_for_response"
    RESPONSE_RECEIVED = "response_received"

class StimulusPhase(enum.Enum):
    """
    Enum for the stimulus phases.
    """
    FIXATION = "fixation"
    # Labels for 1-alternative or adjustment task
    STIMULUS = "stimulus"
    # Labels for 2-alternative forced choice task
    FIRST_STIMULUS = "first_stimulus"
    SECOND_STIMULUS = "second_stimulus"
    NONE = "none"

class PhaseTracker():
    """
    """
    def __init__(
        self,
        experimentPhase : ExperimentPhase = ExperimentPhase.INITIALIZE,
        responsePhase : ResponsePhase = ResponsePhase.NO_RESPONSE,
        stimulusPhase : StimulusPhase = StimulusPhase.FIXATION
    ):
        self.experimentPhase = experimentPhase
        self.responsePhase = responsePhase
        self.stimulusPhase = stimulusPhase

    def get_experiment_phase(self):
        return self.experimentPhase
    
    def get_response_phase(self):
        return self.responsePhase
    
    def get_stimulus_phase(self):
        return self.stimulusPhase

    def set_experiment_phase(self, experimentPhase : ExperimentPhase):
        self.experimentPhase = experimentPhase
    
    def set_response_phase(self, responsePhase : ResponsePhase):
        self.responsePhase = responsePhase
    
    def set_stimulus_phase(self, stimulusPhase : StimulusPhase):
        self.stimulusPhase = stimulusPhase

#------------------------------------------------------------------------------
# Parameter Loading Functions
#------------------------------------------------------------------------------

def parse_parameters_file(filepath):
    """
    Parse a parameters text file into a dictionary of dictionaries.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the parameters text file
    
    Returns
    -------
    dict
        Dictionary with three keys:
        - 'parameters': dict of parameter_name -> value
        - 'factors': list of dicts, each with one key (factor_name) -> list of values
        - 'staircaseParameters': dict of staircase_param_name -> list of values
    
    Format:
    - Parameters: "key: value" (no prefix)
    - Factors: "fkey: value1, value2, value3"
    - Staircase: "skey: value1, value2, value3"
    - Comments: lines starting with # or inline # after value
    """
    from pathlib import Path
    import re
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Parameters file not found: {filepath}")
    
    result = {
        'parameters': {},
        'factors': [],
        'staircaseParameters': {}
    }
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # Remove leading/trailing whitespace
            line = line.strip()
            
            # Skip empty lines and full-line comments
            if not line or line.startswith('#'):
                continue
            
            # Remove inline comments (everything after #)
            if '#' in line:
                # Check if # is inside quotes (simple check)
                comment_pos = line.find('#')
                # Remove comment, but preserve the part before it
                line = line[:comment_pos].strip()
                if not line:
                    continue
            
            # Check for colon separator
            if ':' not in line:
                continue  # Skip malformed lines
            
            # Split into key and value parts
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
            
            key_part = parts[0].strip()
            value_part = parts[1].strip()
            
            # Determine category based on prefix
            if key_part.startswith('f'):
                # Factor - store as list of single-key dictionaries
                key_name = key_part[1:]  # Remove 'f' prefix
                # Parse comma-separated values
                values = [v.strip() for v in value_part.split(',')]
                # Try to convert to appropriate types
                converted_values = []
                for v in values:
                    converted_values.append(_convert_value(v))
                # Append as a dictionary with one key-value pair
                result['factors'].append({key_name: converted_values})
                
            elif key_part.startswith('s'):
                # Staircase parameter
                key_name = key_part[1:]  # Remove 's' prefix
                # Parse comma-separated values
                values = [v.strip() for v in value_part.split(',')]
                # Try to convert to appropriate types
                converted_values = []
                for v in values:
                    converted_values.append(_convert_value(v))
                result['staircaseParameters'][key_name] = converted_values
                
            else:
                # Regular parameter (no prefix)
                key_name = key_part
                # Single value
                result['parameters'][key_name] = _convert_value(value_part)
    
    return result


def createFactorialTrialList(factors):
    """
    Create a factorial trial list from factors.
    
    This function takes factors in the format returned by parse_parameters_file
    and converts it to a list of trial dictionaries formatted for PsychoPy's
    TrialHandler, where each dictionary represents one trial with specific
    values for all factors.
    
    Parameters
    ----------
    factors : list of dict
        List of dictionaries, each with one key (factor name) mapping to a
        list of possible values.
    
    Returns
    -------
    list of dict
        List of dictionaries, each representing one trial with all factor
        combinations. For example:
    """
    
    # Convert list of single-key dicts to a single dict
    factors_dict = {}
    for factor_dict in factors:
        factors_dict.update(factor_dict)
    
    # Get factor names and their value lists
    factor_names = list(factors_dict.keys())
    factor_values = [factors_dict[name] for name in factor_names]
    
    # Generate all combinations using itertools.product
    trial_list = []
    for combination in product(*factor_values):
        trial = {name: value for name, value in zip(factor_names, combination)}
        trial_list.append(trial)
    
    return trial_list


def _convert_value(value_str):
    """
    Convert a string value to appropriate Python type.
    Tries int, then float, then returns string.
    """
    value_str = value_str.strip()
    
    # Try integer
    try:
        return int(value_str)
    except ValueError:
        pass
    
    # Try float
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # Return as string
    return value_str

#------------------------------------------------------------------------------
# Geometric Conversion Functions
#------------------------------------------------------------------------------
def deg_to_rad(degrees: float) -> float:
    return degrees * (np.pi / 180.0)

def rad_to_deg(radians: float) -> float:
    return radians * (180.0 / np.pi)

def width_cm_from_visual_degree(
    visual_degree: float,
    distance_cm: float,
    ) -> float:
    return np.tan(deg_to_rad(visual_degree / 2)) * distance_cm * 2

def degree_from_width_cm(
    width_cm: float,
    distance_cm: float,
    ) -> float:
    return rad_to_deg(np.arctan(width_cm / (2 * distance_cm)) * 2)

def pixels_by_visual_degree(
    width_cm: float,
    distance_cm: float,
    pixel_by_cm: float,
    ) -> float:
    visual_degree = degree_from_width_cm(width_cm, distance_cm)
    cm_by_degree = width_cm / visual_degree
    return cm_by_degree * pixel_by_cm

def visual_degree_to_pixel(
    visual_degree: float,
    pixel_per_degree: float
    ) -> float:
    # Convert visual degree to monitor pixels
    return visual_degree * pixel_per_degree