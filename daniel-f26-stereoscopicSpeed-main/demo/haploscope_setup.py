'''
file: haploscope_setup.py
cmd: uv run -m demo.haploscope_setup
'''

from psychopy import core
from psychopy.hardware import keyboard
from threedipa.renderer.haploscopeRender import HaplscopeRender2D
from threedipa.renderer.haploscopeConfig import monitor_settings, physical_calibration

if __name__ == "__main__":
    subject_IOD  = 70 # mm
    viewing_dist = 80 # cm

    # Setup input and output devices
    renderer = HaplscopeRender2D(
        fixation_distance=viewing_dist,
        iod=float(subject_IOD),
        physical_calibration=physical_calibration,
        screen_config=monitor_settings,
        debug_mode=0,
    )

    renderer.draw_physical_calibration()
    renderer.render_screen()
    kb = keyboard.Keyboard(clock=core.Clock())
    kb.waitKeys(keyList=['return'], waitRelease=True)

    # instructions_text = ("Press enter to begin the experiment.")
    # renderer.draw_text(instructions_text, pos=(0, 0))
    # renderer.render_screen()
    # kb.waitKeys(keyList=['return'], waitRelease=True)
