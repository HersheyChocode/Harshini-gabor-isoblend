from psychopy import visual

def make_fixation_cross(
    windows: list[visual.Window],
    size_pixels: tuple[float, float] = (1.0, 1.0),
    color: str = 'white',
    pos : tuple[float, float] = (0, 0)
):
    """Make a fixation cross with both vertical and horizontal lines for each window.
    """
    # Create both lines for the left window (windows[0])
    left_vertical = visual.Line(
        win=windows[0],
        start=(0, -size_pixels[1]/2),
        end=(0, size_pixels[1]/2),
        units="pix",
        pos=pos,
        color=color
    )
    left_horizontal = visual.Line(
        win=windows[0],
        start=(-size_pixels[0]/2, 0),
        end=(size_pixels[0]/2, 0),
        units="pix",
        pos=pos,
        color=color
    )
    
    # Create both lines for the right window (windows[1])
    right_vertical = visual.Line(
        win=windows[1],
        start=(0, -size_pixels[1]/2),
        end=(0, size_pixels[1]/2),
        units="pix",
        pos=pos,
        color=color
    )
    right_horizontal = visual.Line(
        win=windows[1],
        start=(-size_pixels[0]/2, 0),
        end=(size_pixels[0]/2, 0),
        units="pix",
        pos=pos,
        color=color
    )
    
    # Return as tuples: (left_lines, right_lines) where each is (vertical, horizontal)
    return (left_vertical, left_horizontal), (right_vertical, right_horizontal)