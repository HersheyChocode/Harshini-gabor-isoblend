from abc import ABC, abstractmethod

from pathlib import Path

import numpy as np


class Stimulus2D(ABC):
    @abstractmethod
    def load_stimulus(self):
        pass

    @abstractmethod
    def update_stimulus(self):
        pass

    @abstractmethod
    def get_stimulus(self):
        pass



class Stimulus2DImage(Stimulus2D):
    """A stimulus class for loading and managing static image stimuli.
    
    This class handles loading left and right eye images for stereoscopic
    display on a haploscope. Images can be provided as file paths or numpy arrays.
    
    Attributes
    ----------
    left_image_path 
        Path to the left eye image file or numpy array of image data
    right_image_path
        Path to the right eye image file or numpy array of image data
    left_image
        Loaded left image (path or array)
    right_image
        Loaded right image (path or array)
    """
    
    def __init__(
        self,
        left_image_path: str = None,
        right_image_path: str = None,
        visual_size_degrees: tuple[float, float] = None,
    ):
        """Initialize the StimulusImage.
        
        Parameters
        ----------
        left_image : str, Path, np.ndarray, or None, optional
            Path to left eye image file or numpy array. If None, must be set later.
        right_image : str, Path, np.ndarray, or None, optional
            Path to right eye image file or numpy array. If None, must be set later.
        visual_size_degrees : tuple[float, float], optional
            Visual size of the stimulus in degrees. If None, must be set later.
        """
        self.left_image_path = left_image_path
        self.right_image_path = right_image_path
        self.left_image = None
        self.right_image = None
        self.visual_size_degrees = visual_size_degrees
        # Load images if provided
        if left_image_path is not None and right_image_path is not None:
            self.load_stimulus()
    
    def load_stimulus(
        self,
        left_image_path = None,
        right_image_path = None
        ) -> None:
        """Load the stimulus images.
        """
        if self.left_image_path is not None:
            # If it's a path, validate it exists
            if isinstance(self.left_image_path, (str, Path)):
                path = Path(self.left_image_path)
                if not path.exists():
                    raise FileNotFoundError(
                        f"Left image file not found: {self.left_image_path}"
                    )
                self.left_image = self.left_image_path
            else:
                # Assume it's a numpy array
                self.left_image = self.left_image_path
        
        if self.right_image_path is not None:
            # If it's a path, validate it exists
            if isinstance(self.right_image_path, (str, Path)):
                path = Path(self.right_image_path)
                if not path.exists():
                    raise FileNotFoundError(
                        f"Right image file not found: {self.right_image_path}"
                    )
                self.right_image = self.right_image_path
            else:
                # Assume it's a numpy array
                self.right_image = self.right_image_path
    
    def update_stimulus(self) -> None:
        """Update the stimulus. Static images don't need updating.
        """
        # Static images don't need updating
        pass
    
    def get_stimulus(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the stimulus images as a tuple.
        """
        if self.left_image is None or self.right_image is None:
            raise ValueError(
                "Stimulus images not loaded. Call load_stimulus() first or "
                "provide file paths when initializing."
            )
        
        return (self.left_image, self.right_image)
    
    def __getitem__(self, index: int):
        """Allow indexing to get left (0) or right (1) image.
        
        Parameters
        ----------
        index : int
            0 for left image, 1 for right image
            
        Returns
        -------
        The requested image
        """
        stimulus = self.get_stimulus()
        return stimulus[index]
    
    def set_images(
        self,
        left_image = None,
        right_image = None,
    ) -> None:
        """Set left_image, right_image, or both
        """
        if left_image is not None:
            self.left_image = left_image
        if right_image is not None:
            self.right_image = right_image




class Stimulus2DImageSequence(Stimulus2D):
    """A stimulus class for loading and managing frame sequence stimuli.
    
    This class handles loading sequences of left and right eye frames for 
    stereoscopic display on a haploscope. Frames can be provided as lists of 
    file paths or numpy arrays.
    
    Attributes
    ----------
    left_frame_sequence
        List of paths to left eye frame files or list of numpy arrays
    right_frame_sequence
        List of paths to right eye frame files or list of numpy arrays
    left_image
        Current left frame (path or array)
    right_image
        Current right frame (path or array)
    current_frame_index
        Index of the current frame in the sequence
    loop
        Whether to loop back to the beginning when reaching the end
    """
    
    def __init__(
        self,
        left_frame_sequence = None,
        right_frame_sequence = None,
        loop = True
    ):
        """Initialize the StimulusFrameSequence.
        
        Parameters
        ----------
        left_frame_sequence : list, or None, optional
            List of paths to left eye frame files or list of numpy arrays. 
            If None, must be set later.
        right_frame_sequence : list, or None, optional
            List of paths to right eye frame files or list of numpy arrays. 
            If None, must be set later.
        loop : bool, optional
            Whether to loop back to the beginning when reaching the end (default: True)
        """
        self.left_frame_sequence = left_frame_sequence
        self.right_frame_sequence = right_frame_sequence
        self.left_image = None
        self.right_image = None
        self.current_frame_index = 0
        self.loop = loop
        
        # Load frames if provided
        if left_frame_sequence is not None or right_frame_sequence is not None:
            self.load_stimulus()
    
    def load_stimulus(
        self,
        left_frame_sequence = None,
        right_frame_sequence = None
        ) -> None:
        """Load the stimulus frame sequences.
        """
        if left_frame_sequence is not None:
            self.left_frame_sequence = left_frame_sequence
        if right_frame_sequence is not None:
            self.right_frame_sequence = right_frame_sequence
        
        # Validate sequences
        if self.left_frame_sequence is not None:
            if not isinstance(self.left_frame_sequence, (list, np.ndarray)):
                raise TypeError(
                    "left_frame_sequence must be a list or numpy array"
                )
            if len(self.left_frame_sequence) == 0:
                raise ValueError("left_frame_sequence cannot be empty")
            
            # Validate paths if they are paths
            for i, frame in enumerate(self.left_frame_sequence):
                if isinstance(frame, (str, Path)):
                    path = Path(frame)
                    if not path.exists():
                        raise FileNotFoundError(
                            f"Left frame {i} file not found: {frame}"
                        )
        
        if self.right_frame_sequence is not None:
            if not isinstance(self.right_frame_sequence, (list, np.ndarray)):
                raise TypeError(
                    "right_frame_sequence must be a list or numpy array"
                )
            if len(self.right_frame_sequence) == 0:
                raise ValueError("right_frame_sequence cannot be empty")
            
            # Validate paths if they are paths
            for i, frame in enumerate(self.right_frame_sequence):
                if isinstance(frame, (str, Path)):
                    path = Path(frame)
                    if not path.exists():
                        raise FileNotFoundError(
                            f"Right frame {i} file not found: {frame}"
                        )
        
        # Check that sequences have the same length
        if (self.left_frame_sequence is not None and 
            self.right_frame_sequence is not None):
            if len(self.left_frame_sequence) != len(self.right_frame_sequence):
                raise ValueError(
                    "left_frame_sequence and right_frame_sequence must have "
                    f"the same length. Got {len(self.left_frame_sequence)} and "
                    f"{len(self.right_frame_sequence)}"
                )
        
        # Reset to first frame
        self.current_frame_index = 0
        self._update_current_frames()
    
    def _update_current_frames(self) -> None:
        """Update left_image and right_image to current frame index.
        """
        if self.left_frame_sequence is not None:
            self.left_image = self.left_frame_sequence[self.current_frame_index]
        if self.right_frame_sequence is not None:
            self.right_image = self.right_frame_sequence[self.current_frame_index]
    
    def update_stimulus(self) -> None:
        """Update the stimulus to the next frame in the sequence.
        """
        if self.left_frame_sequence is None and self.right_frame_sequence is None:
            return
        
        # Determine sequence length
        if self.left_frame_sequence is not None:
            sequence_length = len(self.left_frame_sequence)
        else:
            sequence_length = len(self.right_frame_sequence)
        
        # Advance to next frame
        self.current_frame_index += 1
        
        # Handle looping or end of sequence
        if self.current_frame_index >= sequence_length:
            if self.loop:
                self.current_frame_index = 0
            else:
                self.current_frame_index = sequence_length - 1
        
        self._update_current_frames()
    
    def get_stimulus(self) -> tuple:
        """Get the current stimulus frames as a tuple.
        """
        if self.left_image is None or self.right_image is None:
            raise ValueError(
                "Stimulus frames not loaded. Call load_stimulus() first or "
                "provide frame sequences when initializing."
            )
        
        return (self.left_image, self.right_image)
    
    def __getitem__(self, index: int):
        """Allow indexing to get left (0) or right (1) image.
        
        Parameters
        ----------
        index : int
            0 for left image, 1 for right image
            
        Returns
        -------
        The requested image frame
        """
        stimulus = self.get_stimulus()
        return stimulus[index]
    
    def set_frame_sequences(
        self,
        left_frame_sequence = None,
        right_frame_sequence = None,
    ) -> None:
        """Set left_frame_sequence, right_frame_sequence, or both
        """
        if left_frame_sequence is not None:
            self.left_frame_sequence = left_frame_sequence
        if right_frame_sequence is not None:
            self.right_frame_sequence = right_frame_sequence
        
        # Reload if sequences were set
        if left_frame_sequence is not None or right_frame_sequence is not None:
            self.load_stimulus()
    
    def reset_to_start(self) -> None:
        """Reset the frame sequence to the first frame.
        """
        self.current_frame_index = 0
        self._update_current_frames()
    
    def set_frame_index(self, index: int) -> None:
        """Set the current frame index.
        
        Parameters
        ----------
        index : int
            Frame index to set (0-based)
        """
        if self.left_frame_sequence is not None:
            if index < 0 or index >= len(self.left_frame_sequence):
                raise IndexError(
                    f"Frame index {index} out of range [0, {len(self.left_frame_sequence)})"
                )
        elif self.right_frame_sequence is not None:
            if index < 0 or index >= len(self.right_frame_sequence):
                raise IndexError(
                    f"Frame index {index} out of range [0, {len(self.right_frame_sequence)})"
                )
        
        self.current_frame_index = index
        self._update_current_frames()