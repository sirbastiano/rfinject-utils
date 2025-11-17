# Copyright (c) Roberto Del Prete. All rights reserved.

import sys
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import zarr



def explore_zarr_structure(zarr_group: zarr.Group, max_depth: int = 3) -> None:
    """Explore and print the structure of a Zarr group recursively.
    
    Args:
        zarr_group (zarr.Group): The Zarr group to explore.
        max_depth (int): Maximum depth to explore. Defaults to 3.
    """
    assert isinstance(zarr_group, zarr.Group), 'Input must be a Zarr group'
    
    def _print_structure(group: zarr.Group, indent: str = '', depth: int = 0) -> None:
        if depth > max_depth:
            return
            
        for key in group.keys():
            item = group[key]
            if isinstance(item, zarr.Group):
                print(f'{indent}📁 {key}/')
                _print_structure(item, indent + '  ', depth + 1)
            else:
                shape_str = f'{item.shape}' if hasattr(item, 'shape') else 'unknown'
                dtype_str = f'{item.dtype}' if hasattr(item, 'dtype') else 'unknown'
                print(f'{indent}📄 {key}: {shape_str} {dtype_str}')
                
                # Show attributes if any
                if hasattr(item, 'attrs') and item.attrs:
                    for attr_key, attr_val in item.attrs.items():
                        print(f'{indent}    📋 {attr_key}: {attr_val}')
    
    print('Zarr Structure:')
    _print_structure(zarr_group)


def access_array_data(zarr_group: zarr.Group, burst_name: str, array_name: str) -> zarr.Array:
    """Access a specific array from a burst.
    
    Args:
        zarr_group (zarr.Group): The Zarr group containing the data.
        burst_name (str): Name of the burst (e.g., 'burst_0').
        array_name (str): Name of the array (e.g., 'echo', 'rfi', 'echo_w_rfi').
        
    Returns:
        zarr.Array: The requested array.
    """
    assert burst_name in zarr_group.keys(), f'Burst {burst_name} not found'
    assert array_name in zarr_group[burst_name].keys(), f'Array {array_name} not found in {burst_name}'
    
    return zarr_group[burst_name][array_name]


def get_array_slice(array: zarr.Array, slice_params: Optional[Tuple] = None) -> np.ndarray:
    """Get a slice of data from a Zarr array.
    
    Args:
        array (zarr.Array): The Zarr array to slice.
        slice_params (Optional[Tuple]): Slice parameters. If None, returns first 10x10 slice.
        
    Returns:
        np.ndarray: The sliced data as a NumPy array.
    """
    if slice_params is None:
        # Default: get a small slice for inspection
        if len(array.shape) == 2:
            slice_params = (slice(0, 10), slice(0, 10))
        elif len(array.shape) == 3:
            slice_params = (0, slice(0, 10), slice(0, 10))
        else:
            slice_params = tuple(slice(0, 10) for _ in array.shape)
    
    return array[slice_params]


def get_burst_info(zarr_group: zarr.Group) -> Dict[str, Dict[str, Any]]:
    """Extract information about all bursts in the Zarr group.
    
    Args:
        zarr_group (zarr.Group): The Zarr group containing burst data.
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary with burst information.
    """
    assert isinstance(zarr_group, zarr.Group), 'Input must be a Zarr group'
    
    burst_info = {}
    
    for key in zarr_group.keys():
        if key.startswith('burst_'):
            burst = zarr_group[key]
            
            # Get basic info about each array in the burst
            arrays_info = {}
            for array_name in burst.keys():
                array = burst[array_name]
                arrays_info[array_name] = {
                    'shape': array.shape,
                    'dtype': array.dtype,
                    'size_mb': array.nbytes / (1024 * 1024),
                    'chunks': array.chunks if hasattr(array, 'chunks') else None
                }
            
            burst_info[key] = {
                'arrays': arrays_info,
                'total_size_mb': sum(info['size_mb'] for info in arrays_info.values())
            }
    
    return burst_info


def access_attributes(zarr_item: zarr.Group, path: Optional[str] = None) -> Dict[str, Any]:
    """Access attributes from a Zarr array or group, optionally at a specific path.
    
    Args:
        zarr_item (zarr.Group): The Zarr group or array to explore.
        path (Optional[str]): Optional path to navigate to (e.g., 'burst_0', 'burst_0/echo').
                             If None, returns attributes of the root item.
        
    Returns:
        Dict[str, Any]: Dictionary of attributes.
        
    Raises:
        KeyError: If the specified path does not exist.
    """
    target_item = zarr_item
    
    if path is not None:
        # Navigate to the specified path
        path_parts = path.split('/')
        for part in path_parts:
            if part in target_item:
                target_item = target_item[part]
            else:
                raise KeyError(f'Path "{path}" not found. Part "{part}" does not exist.')
    
    if hasattr(target_item, 'attrs'):
        return dict(target_item.attrs)
    return {}


def explore_all_attributes(zarr_group: zarr.Group) -> Dict[str, Dict[str, Any]]:
    """Explore all attributes in the Zarr group hierarchy.
    
    Args:
        zarr_group (zarr.Group): The Zarr group to explore.
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of all attributes found.
    """
    all_attrs = {}
    
    # Root level attributes
    root_attrs = access_attributes(zarr_group)
    if root_attrs:
        all_attrs['root'] = root_attrs
    
    # Burst level attributes
    for burst_name in zarr_group.keys():
        if burst_name.startswith('burst_'):
            burst = zarr_group[burst_name]
            burst_attrs = access_attributes(burst)
            if burst_attrs:
                all_attrs[burst_name] = burst_attrs
            
            # Array level attributes
            for array_name in burst.keys():
                array = burst[array_name]
                array_attrs = access_attributes(array)
                if array_attrs:
                    all_attrs[f'{burst_name}/{array_name}'] = array_attrs
    
    return all_attrs


