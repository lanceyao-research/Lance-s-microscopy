import numpy as np

class CoordStore:
    """Store and manage detected window coordinates in world space"""
    def __init__(self):
        self.xy = np.empty((0, 2), dtype=float)
        self.count = np.empty((0,), dtype=int)
        self.cls = np.empty((0,), dtype=int)
    
    def add(self, cs, classes, threshold):
        cs = np.asarray(cs, dtype=float)
        classes = np.asarray(classes)
        if self.xy.size == 0:
            self.xy = cs.copy()
            self.count = np.ones(cs.shape[0], dtype=int)
            self.cls = classes.copy()
            return
        for p, c in zip(cs, classes):
            d = np.linalg.norm(self.xy - p, axis=1)
            idx = np.where(d <= threshold)[0]
            if idx.size:
                i = idx[0]
                k = self.count[i]
                self.xy[i] = (self.xy[i] * k + p) / (k + 1)
                self.count[i] = k + 1
                if self.cls[i] != c:
                    self.cls[i] = c
            else:
                self.xy = np.vstack([self.xy, p])
                self.count = np.append(self.count, 1)
                self.cls = np.append(self.cls, c)
    
    def sort_path(self):
        """Greedy nearest-neighbor ordering to reduce travel distance."""
        if self.xy.shape[0] <= 1:
            return
        order = [0]
        unvisited = set(range(1, len(self.xy)))
        while unvisited:
            last = order[-1]
            next_idx = min(unvisited, key=lambda i: np.linalg.norm(self.xy[last] - self.xy[i]))
            order.append(next_idx)
            unvisited.remove(next_idx)
        self.xy = self.xy[order]
        self.count = self.count[order]
        self.cls = self.cls[order]
    
    def get(self):
        return self.xy, self.count, self.cls
    
    def clear(self):
        self.xy = np.empty((0, 2), dtype=float)
        self.count = np.empty((0,), dtype=int)
        self.cls = np.empty((0,), dtype=int)