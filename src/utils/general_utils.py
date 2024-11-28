# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:42:34 2024

@author: Hubert Szewczyk
"""
from collections import defaultdict

#%%
def group_object_list(obj_list: list, attributes: list[str]) -> dict[tuple, list]:
    """
    Groups objects from the given list based on specified attributes.

    Args:
        obj_list (list): List of objects to group.
        attributes (list[str]): List of attribute names to group by (e.g., ["attr1", "attr2"]).

    Returns:
        dict[tuple, list]: Dictionary where the keys are tuples of attribute values, 
                            and the values are lists of objects grouped by these attribute values.
    """
    grouped_objs = defaultdict(list)
    
    # Check that all objects in the list have the necessary attributes
    for obj in obj_list:
        try:
            # Create a tuple with the selected attribute values for grouping
            attribute_tuple = tuple(getattr(obj, attr) for attr in attributes)
            grouped_objs[attribute_tuple].append(obj)
        except AttributeError as e:
            print(f"Error accessing attributes: {e}, object: {obj}")

    return grouped_objs