from abc import ABC, abstractmethod
from psychopy import visual
import numpy as np

class Probe2D(ABC):
    """Abstract base class for 2D probes for measuring depth perception"""
    
    def __init__(self, win):
        self.win = win
    
    @abstractmethod
    def draw(self):
        """Draw the stimulus"""
        pass
    
    @abstractmethod
    def setPos(self, pos):
        """Set position of the stimulus"""
        pass


class ShapeOutlineProbe(Probe2D):
    """Draws a 2D shape probe defined by a function y=f(x)
    
        Parameters
        ----------
        win : psychopy.visual.Window
        probe_func : function
            A function that takes a argument x and magnitude and returns y
        magnitude : float
            The magnitude of the probe
        x_range : tuple
            The beginning and end of the probe (min, max).
            Average should be 0 to ensure translation moves center.
        segments : int
            Number of segments to use to draw the probe. LArger numbers give smoother shapes.
        postion : list[int, int]
            Position on the screen in pixels
        line_width_pix : int
            Width of the line in pixels
        rotate_90 : bool
            Whether to rotate the shape by 90 degrees
        color : str or tuple
            Color of the shape outline
        units : str
            Defines the units stored in magnitude. Can be centimeters, pixels, or degrees
    """
    
    def __init__(
            self,
            probe_func: callable,
            magnitude: float = 0,
            win: visual.Window = None,
            x_range: tuple[int, int] = (-1, 1),
            segments: int = 100,
            position: list[int, int] = [0, 0],
            line_width_pix: int = 5,
            rotate_90: bool = False,
            color: str = 'white',
            units: str = "pixels"
            ):
        self.probe_func = probe_func
        self.magnitude = magnitude
        self.magnitude_pixels = magnitude if units == "pixels" else None
        self.segments = segments
        self.lineWidth = line_width_pix
        self.rotate_90 = rotate_90
        self.color = color
        self.pos = position
        self.x_vals = np.linspace(x_range[0], x_range[1], segments)
        self.units = units
        self.win = win

    def applyProbeFunction(self, x):
        return self.probe_func(x, self.magnitude_pixels)
    
    def setMagnitudePixels(self, x_pixels):
        self.magnitude_pixels = x_pixels
        return
    
    def getMagnitude(self):
        return (self.magnitude)
    
    def getMagnitudePixels(self):
        return (self.magnitude_pixels)

    def draw(
            self,
            win: visual.Window = None
            ):
        if win is None:
            win = self.win
            if win is None:
                raise ValueError("A psychopy.visual.Window must be provided to draw the stimulus.")
        x_coors = self.x_vals.copy()
        # Initialize previous coordinate
        first_x = min(x_coors)
        first_y = self.applyProbeFunction(first_x)
        previous_coordinate = (np.array([first_x, first_y]) if not self.rotate_90 else np.array([first_y, first_x])) + self.pos

        # Remove previous coordinate
        np.delete(x_coors, 0)
        
        # Cycle through the 
        for x_coor in x_coors:
            y_coor = self.applyProbeFunction(x_coor)
            coordinates = (
                np.array([x_coor, y_coor]) if not self.rotate_90 else np.array([y_coor, x_coor])
                ) + self.pos

            visual.Line(
                win=win,
                start=previous_coordinate,
                end=coordinates,
                lineWidth=self.lineWidth,
                lineColor=self.color,
                pos=self.pos
            ).draw()
            previous_coordinate = coordinates
    
    def setPos(self, pos):
        self.pos = pos
        self.line.pos = pos
    
    def setMagnitude(self, magnitude):
        self.magnitude = magnitude


class DotProbe(Probe2D):
    """Draws 2D a movable dot on the screen. (Can only draw one dot for now)"""
    
    def __init__(self, win, pos=None, size=0.1, color='white', num_dots=1):
        super().__init__(win)
        self.size = size
        self.color = color
        self.num_dots = 1
        self.pos = pos if pos is not None else [0, 0]
        
        if num_dots == 1:
            self.dots = visual.Circle(win, radius=size, fillColor=color, lineColor=color)
        else:
            self.dots = [visual.Circle(win, radius=size, fillColor=color, lineColor=color) 
                        for _ in range(num_dots)]
    
    def draw(self):
        if self.num_dots == 1:
            self.dots.draw()
        else:
            for dot in self.dots:
                dot.draw()
    
    def setPos(self, pos):
        self.pos = pos
        if self.num_dots == 1:
            self.dots.pos = pos
        else:
            for i, dot in enumerate(self.dots):
                dot.pos = pos[i] if isinstance(pos[0], (list, tuple)) else pos