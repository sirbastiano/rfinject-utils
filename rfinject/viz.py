
# Copyright (c) Roberto Del Prete. All rights reserved.

import sys
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt


def plot_complex_array(array: np.ndarray, title: str = 'Complex Array Visualization', figsize: Tuple[int, int] = (15, 5)) -> None:
    """Plot a complex array showing magnitude, phase, real and imaginary parts.
    
    Args:
        array (np.ndarray): The complex array to visualize.
        title (str): Title for the overall plot. Defaults to 'Complex Array Visualization'.
        figsize (Tuple[int, int]): Figure size as (width, height). Defaults to (15, 5).
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot magnitude
    magnitude = np.abs(array)
    im0 = axes[0].imshow(magnitude, cmap='gray', aspect='auto')
    axes[0].set_title('Magnitude')
    axes[0].set_xlabel('Range')
    axes[0].set_ylabel('Azimuth')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot phase
    phase = np.angle(array)
    im1 = axes[1].imshow(phase, cmap='gray', aspect='auto')
    axes[1].set_title('Phase')
    axes[1].set_xlabel('Range')
    axes[1].set_ylabel('Azimuth')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot real part
    real_part = np.real(array)
    im2 = axes[2].imshow(real_part, cmap='gray', aspect='auto')
    axes[2].set_title('Real Part')
    axes[2].set_xlabel('Range')
    axes[2].set_ylabel('Azimuth')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_magnitude(array: np.ndarray, title: str = 'Magnitude', figsize: Tuple[int, int] = (10, 8), 
                    normalize: bool = True, db_scale: bool = True, vmin: Optional[float] = None, 
                    vmax: Optional[float] = None, savefig: Optional[str] = None) -> None:
    """Plot only the magnitude of a complex array with normalization and dB scale options.
    
    Args:
        array (np.ndarray): The complex array to visualize.
        title (str): Title for the plot. Defaults to 'Magnitude'.
        figsize (Tuple[int, int]): Figure size as (width, height). Defaults to (10, 8).
        normalize (bool): Whether to normalize the magnitude. Defaults to True.
        db_scale (bool): Whether to display in dB scale. Defaults to True.
        vmin (Optional[float]): Minimum value for color scale. Defaults to None.
        vmax (Optional[float]): Maximum value for color scale. Defaults to None.
        savefig (Optional[str]): Path to save the figure. If None, figure is not saved. Defaults to None.
    """
    magnitude = np.abs(array)
    
    if normalize:
        magnitude = magnitude / np.max(magnitude)
    
    if db_scale:
        # Avoid log(0) by adding small epsilon
        magnitude = 20 * np.log10(magnitude + 1e-10)
        scale_label = 'Magnitude (dB)'
    else:
        scale_label = 'Magnitude'
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(magnitude, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel('Range')
    ax.set_ylabel('Azimuth')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(scale_label)
    plt.tight_layout()
    
    if savefig is not None:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
    
    plt.show()