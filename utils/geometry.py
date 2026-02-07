import numpy as np
import math

def calculate_angle(a, b, c):
    """
    Calculates the angle between three points a, b, and c.
    a, b, c are (x, y) tuples or arrays.
    b is the vertex.
    Returns angle in degrees.
    """
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def euclidean_distance(p1, p2):
    """
    Calculates Euclidean distance between two points p1 and p2.
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
