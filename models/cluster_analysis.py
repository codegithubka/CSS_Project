"""Spatial cluster analysis utilities using Hoshen-Kopelman algorithm.

Provides efficient cluster detection and analysis for 2D grids with
periodic boundary conditions. Independent of any specific CA implementation.
"""
from typing import Tuple, Dict, Optional
import numpy as np


class UnionFind:
    """Union-Find data structure for efficient cluster label management.
    
    Used internally by Hoshen-Kopelman algorithm to track label equivalences.
    Implements path compression and union by rank for near-constant time operations.
    """
    
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def make_set(self, x):
        """Create a new set containing only x."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
    
    def find(self, x):
        """Find the root of x's set with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Unite the sets containing x and y using union by rank."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1


class ClusterAnalyzer:
    """Spatial cluster analysis for 2D grids.
    
    Provides Hoshen-Kopelman cluster detection with support for different
    neighborhood types and periodic boundaries. Can analyze arbitrary 2D
    integer grids without requiring CA-specific knowledge.
    
    Examples:
        >>> analyzer = ClusterAnalyzer(neighborhood='moore')
        >>> labels, sizes = analyzer.detect_clusters(grid, state=1)
        >>> stats = analyzer.get_stats(grid, state=1)
        >>> percolates, label, _ = analyzer.check_percolation(grid, state=1)
    """
    
    def __init__(self, neighborhood: str = 'moore'):
        """Initialize cluster analyzer.
        
        Args:
            neighborhood: Either 'neumann' (4-neighbors) or 'moore' (8-neighbors)
        """
        if neighborhood not in ('neumann', 'moore'):
            raise ValueError("neighborhood must be 'neumann' or 'moore'")
        self.neighborhood = neighborhood
        
        # Precompute neighbor offsets for scanning
        if neighborhood == 'neumann':
            # Check left and top (already scanned positions)
            self._neighbor_offsets = [(-1, 0), (0, -1)]
        else:  # moore
            # Check top-left, top, top-right, and left
            self._neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]
    
    def detect_clusters(
        self, 
        grid: np.ndarray, 
        state: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[int, int]]:
        """Detect clusters using Hoshen-Kopelman algorithm with Union-Find.
        
        Identifies connected components (clusters) of occupied sites in the grid.
        Uses the configured neighborhood type to determine connectivity.
        Implements periodic boundary conditions.
        
        Args:
            grid: 2D numpy array containing integer states
            state: Specific state value to cluster (1, 2, etc.).
                If None, clusters all non-zero states together.
        
        Returns:
            Tuple containing:
            - labels: 2D array same shape as grid, where each cell contains
                its cluster label (0 for empty/non-target cells, positive
                integers for cluster IDs). Cluster IDs are contiguous starting from 1.
            - sizes: Dictionary mapping cluster label -> cluster size
                (number of sites in that cluster). Does not include label 0.
        
        Example:
            >>> analyzer = ClusterAnalyzer(neighborhood='moore')
            >>> labels, sizes = analyzer.detect_clusters(grid, state=1)
            >>> print(f"Found {len(sizes)} clusters")
        """
        rows, cols = grid.shape
        labels = np.zeros((rows, cols), dtype=int)
        uf = UnionFind()
        current_label = 1
        
        # Determine which cells to cluster
        if state is None:
            mask = (grid != 0)
        else:
            mask = (grid == state)
        
        # First pass: assign labels and record equivalences
        for i in range(rows):
            for j in range(cols):
                if not mask[i, j]:
                    continue
                
                # Check already-scanned neighbors
                neighbor_labels = []
                for di, dj in self._neighbor_offsets:
                    ni = (i + di) % rows  # Periodic boundary
                    nj = (j + dj) % cols
                    
                    if labels[ni, nj] > 0:
                        neighbor_labels.append(labels[ni, nj])
                
                if len(neighbor_labels) == 0:
                    # New cluster
                    labels[i, j] = current_label
                    uf.make_set(current_label)
                    current_label += 1
                else:
                    # Join existing cluster(s)
                    min_label = min(neighbor_labels)
                    labels[i, j] = min_label
                    
                    # Union all neighbor labels
                    for label in neighbor_labels:
                        uf.make_set(label)
                        uf.union(min_label, label)
        
        # Second pass: resolve labels to their root and make contiguous
        root_to_new_label = {}
        next_new_label = 1
        
        final_labels = np.zeros((rows, cols), dtype=int)
        cluster_sizes = {}
        
        for i in range(rows):
            for j in range(cols):
                if labels[i, j] > 0:
                    root = uf.find(labels[i, j])
                    
                    if root not in root_to_new_label:
                        root_to_new_label[root] = next_new_label
                        cluster_sizes[next_new_label] = 0
                        next_new_label += 1
                    
                    new_label = root_to_new_label[root]
                    final_labels[i, j] = new_label
                    cluster_sizes[new_label] += 1
        
        return final_labels, cluster_sizes
    
    def get_stats(
        self, 
        grid: np.ndarray, 
        state: Optional[int] = None
    ) -> Dict[str, object]:
        """Compute comprehensive cluster statistics for the given grid.
        
        Args:
            grid: 2D numpy array to analyze
            state: Specific state to analyze. If None, analyzes all non-zero states.
        
        Returns:
            Dictionary containing:
            - 'n_clusters': Total number of distinct clusters
            - 'sizes': Array of cluster sizes, sorted descending
            - 'largest': Size of the largest cluster
            - 'mean_size': Mean cluster size
            - 'size_distribution': Histogram mapping size -> count
            - 'labels': Cluster label array from detect_clusters
            - 'size_dict': Cluster label -> size mapping
        """
        labels, size_dict = self.detect_clusters(grid, state=state)
        
        if len(size_dict) == 0:
            return {
                'n_clusters': 0,
                'sizes': np.array([]),
                'largest': 0,
                'mean_size': 0.0,
                'size_distribution': {},
                'labels': labels,
                'size_dict': size_dict
            }
        
        sizes = np.array(list(size_dict.values()))
        sizes_sorted = np.sort(sizes)[::-1]
        
        # Create size distribution (size -> count)
        size_dist = {}
        for s in sizes:
            size_dist[s] = size_dist.get(s, 0) + 1
        
        return {
            'n_clusters': len(size_dict),
            'sizes': sizes_sorted,
            'largest': int(np.max(sizes)),
            'mean_size': float(np.mean(sizes)),
            'size_distribution': size_dist,
            'labels': labels,
            'size_dict': size_dict
        }
    
    def check_percolation(
        self, 
        grid: np.ndarray,
        state: Optional[int] = None,
        direction: str = 'both'
    ) -> Tuple[bool, int, np.ndarray]:
        """Detect whether a percolating cluster exists (spans the grid).
        
        A percolating cluster connects opposite edges of the grid,
        indicating a phase transition in percolation theory.
        
        Args:
            grid: 2D numpy array to analyze
            state: State to check for percolation. If None, checks all non-zero states.
            direction: Direction to check:
                - 'horizontal': left-to-right spanning
                - 'vertical': top-to-bottom spanning  
                - 'both': either direction (default)
        
        Returns:
            Tuple containing:
            - percolates: True if a percolating cluster exists
            - cluster_label: Label of the percolating cluster (0 if none)
            - labels: Full cluster label array
        """
        labels, size_dict = self.detect_clusters(grid, state=state)
        rows, cols = labels.shape
        
        percolating_labels = set()
        
        if direction in ('horizontal', 'both'):
            left_labels = set(labels[:, 0][labels[:, 0] > 0])
            right_labels = set(labels[:, -1][labels[:, -1] > 0])
            percolating_labels.update(left_labels & right_labels)
        
        if direction in ('vertical', 'both'):
            top_labels = set(labels[0, :][labels[0, :] > 0])
            bottom_labels = set(labels[-1, :][labels[-1, :] > 0])
            percolating_labels.update(top_labels & bottom_labels)
        
        if percolating_labels:
            # Return the largest percolating cluster
            perc_label = max(percolating_labels, key=lambda x: size_dict[x])
            return True, perc_label, labels
        else:
            return False, 0, labels