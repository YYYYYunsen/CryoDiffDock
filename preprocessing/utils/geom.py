#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from typing import List, Tuple
import numpy as np

__all__ = ["fibonacci_sphere", "unit_vectors_on_sphere"]

def fibonacci_sphere(n: int, randomize: bool = False) -> np.ndarray:
    """Even-ish points on a sphere surface. Returns (n,3) float32 unit vectors.
    Based on the Fibonacci spiral method.
    """
    theta_offset = 0.0
    if randomize:
        theta_offset = np.random.random() * n
    points = []
    phi = math.pi * (3. - math.sqrt(5.0))
    for i in range(n):
        y = 1 - (2.0 * i) / (n - 1)  # y from 1 to -1, uniform
        r = math.sqrt(max(0.0, 1 - y*y))
        theta = phi * (i + theta_offset)
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        points.append((x, y, z))
    return np.asarray(points, dtype=np.float32)

def unit_vectors_on_sphere(n: int, seed: int = 0) -> np.ndarray:
    np.random.seed(seed)
    return fibonacci_sphere(n, randomize=True)
