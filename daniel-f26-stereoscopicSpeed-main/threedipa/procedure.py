"""High-level experiment orchestration for the Johnston stereopsis task."""
from __future__ import annotations

# Core libraries
import os
from pathlib import Path

# Third party libraries
import psychopy
from psychopy.hardware import keyboard

# vizlab3D libraries 
from psychopy.data import ExperimentHandler
from .renderer.haploscopeRender import HaplscopeRender
from .stimuli.stimulus2D import Stimulus2D
from . import utils


def OneIntervalDraw(
    renderer: HaplscopeRender,
    stimulus: Stimulus2D,
    stimulusPhase: utils.StimulusPhase
):
    """Run the trial."""
    # Draw the fixation cross
    if(stimulusPhase == utils.StimulusPhase.FIXATION):
        renderer.draw_fixation_cross()
        renderer.render_screen()
    # Draw the stimulus
    elif(stimulusPhase == utils.StimulusPhase.STIMULUS):
        stimulus.update_stimulus()
        renderer.draw_image_stimulus(stimulus)
        renderer.render_screen()
    # Draw a blank screen
    elif(stimulusPhase == utils.StimulusPhase.NONE):
        renderer.render_screen()
    
def TwoIntervalDraw(
    renderer: HaplscopeRender,
    stimulus1: Stimulus2D,
    stimulus2: Stimulus2D,
    stimulusPhase: utils.StimulusPhase
):
    # Draw the fixation cross
    if(stimulusPhase == utils.StimulusPhase.FIXATION):
        renderer.draw_fixation_cross()
        renderer.render_screen()
    # Draw the stimulus
    elif(stimulusPhase == utils.StimulusPhase.FIRST_STIMULUS):
        stimulus1.update_stimulus()
        renderer.draw_image_stimulus(stimulus1)
        renderer.render_screen()
    elif(stimulusPhase == utils.StimulusPhase.SECOND_STIMULUS):
        stimulus2.update_stimulus()
        renderer.draw_image_stimulus(stimulus2)
        renderer.render_screen()
    # Draw a blank screen
    elif(stimulusPhase == utils.StimulusPhase.NONE):
        renderer.render_screen()
        renderer.render_screen
    return

def stimulusAdjustmentDraw(
    renderer: HaplscopeRender,
    stimulus: Stimulus2D,
    probe: Probe2D,
    stimulusPhase: utils.StimulusPhase
):
    # Draw the fixation cross
    if(stimulusPhase == utils.StimulusPhase.FIXATION):
        renderer.draw_fixation_cross()
        renderer.render_screen()
    # Draw the stimulus
    elif(stimulusPhase == utils.StimulusPhase.STIMULUS):
        stimulus.update_stimulus()
        renderer.draw_image_stimulus(stimulus)
        renderer.draw_probe(probe)
        renderer.render_screen()
    # Draw a blank screen
    elif(stimulusPhase == utils.StimulusPhase.NONE):
        renderer.render_screen()
    return

def DotPlacementProcedure():
    return

