"""Cellular automaton base class.

Defines a CA class with initialization, neighbor counting, update (to override),
and run loop. Uses a numpy Generator for all randomness and supports
Neumann and Moore neighborhoods with periodic boundaries.
"""
from typing import Tuple, Dict, Optional

import numpy as np
import logging

# Module logger
logger = logging.getLogger(__name__)

# Module-level cache for scipy.ndimage and neighborhood kernels to avoid
# repeated imports and small array allocations in hot paths.
_cached_ndimage = None
_cached_kernels = {}

class CA:
	"""Base cellular automaton class.

	Attributes
	- n_species: number of distinct (non-zero) states
	- grid: 2D numpy array containing integers in {0, 1, ..., n_species}
	- neighborhood: either "neumann" or "moore"
	- generator: numpy.random.Generator used for all randomness
	- params: global parameters dict
	- cell_params: local (per-cell) parameters dict
	"""

	# Default colormap spec (string or sequence); resolved in `visualize` at runtime
	_default_cmap = "viridis"

	# Read-only accessors for size/densities (protected attributes set in __init__)
	@property
	def rows(self) -> int:
		return getattr(self, "_rows")

	@property
	def cols(self) -> int:
		return getattr(self, "_cols")

	@property
	def densities(self) -> Tuple[float, ...]:
		return tuple(getattr(self, "_densities"))

	# make n_species protected with read-only property
	@property
	def n_species(self) -> int:
		return int(getattr(self, "_n_species"))

	def __init__(
		self,
		rows: int,
		cols: int,
		densities: Tuple[float, ...],
		neighborhood: str,
		params: Dict[str, object],
		cell_params: Dict[str, object],
		seed: Optional[int] = None,
	) -> None:
		"""Initialize the cellular automaton.

		Args:
		- rows (int): number of rows (>0)
		- cols (int): number of columns (>0)
		- densities (tuple of floats): initial density for each species. The
		  length of this tuple defines `n_species`. Values must be >=0 and sum
		  to at most 1. Each value gives the fraction of the grid to set to
		  that species (state values are 1..n_species).
		- neighborhood (str): either "neumann" (4-neighbors) or "moore"
		  (8-neighbors).
		- params (dict): global parameters.
		- cell_params (dict): local per-cell parameters.
		- seed (Optional[int]): seed for the numpy random generator.

		Returns: None
		"""
		assert isinstance(rows, int) and rows > 0, "rows must be positive int"
		assert isinstance(cols, int) and cols > 0, "cols must be positive int"
		assert isinstance(densities, tuple) and len(densities) > 0, "densities must be a non-empty tuple"
		for d in densities:
			assert isinstance(d, (float, int)) and d >= 0, "each density must be non-negative"
		total_density = float(sum(densities))
		assert total_density <= 1.0 + 1e-12, "sum of densities must not exceed 1"
		assert neighborhood in ("neumann", "moore"), "neighborhood must be 'neumann' or 'moore'"

		self._n_species: int = len(densities)
		# store protected size/density attributes (read-only properties exposed)
		self._rows: int = rows
		self._cols: int = cols
		self._densities: Tuple[float, ...] = tuple(densities)
		self.params: Dict[str, object] = dict(params) if params is not None else {}
		self.cell_params: Dict[str, object] = dict(cell_params) if cell_params is not None else {}

		# per-parameter evolve metadata and evolution state
		# maps parameter name -> dict with keys 'sd','min','max','species'
		self._evolve_info: Dict[str, Dict[str, float]] = {}
		# when True, inheritance uses deterministic copy from parent (no mutation)
		self._evolution_stopped: bool = False

		# human-readable species names (useful for visualization). Default
		# generates generic names based on n_species; subclasses may override.
		self.species_names: Tuple[str, ...] = tuple(f"species{i+1}" for i in range(self._n_species))
		self.neighborhood: str = neighborhood
		self.generator: np.random.Generator = np.random.default_rng(seed)

		self.grid: np.ndarray = np.zeros((rows, cols), dtype=int)

		total_cells = rows * cols
		# Fill grid with species states 1..n_species according to densities.
		for i, dens in enumerate(densities):
			if dens <= 0:
				continue
			n_to_fill = int(round(total_cells * float(dens)))
			if n_to_fill <= 0:
				continue
			empty_flat = np.flatnonzero(self.grid.ravel() == 0)
			if len(empty_flat) == 0:
				break
			n_choice = min(n_to_fill, len(empty_flat))
			chosen = self.generator.choice(empty_flat, size=n_choice, replace=False)
			# assign chosen flattened indices to state i+1
			r = chosen // cols
			c = chosen % cols
			self.grid[r, c] = i + 1

	def validate(self) -> None:
		"""Validate core CA invariants.

		Checks that `neighborhood` is valid, that `self.grid` has the
	texpected shape `(rows, cols)`, and that any numpy arrays in
	`self.cell_params` have matching shapes. Raises `ValueError` on
	validation failure.
		"""
		if self.neighborhood not in ("neumann", "moore"):
			raise ValueError("neighborhood must be 'neumann' or 'moore'")

		expected_shape = (int(getattr(self, "_rows")), int(getattr(self, "_cols")))
		if self.grid.shape != expected_shape:
			raise ValueError(f"grid shape {self.grid.shape} does not match expected {expected_shape}")

		# Ensure any array in cell_params matches grid shape
		for k, v in (self.cell_params or {}).items():
			if isinstance(v, np.ndarray) and v.shape != expected_shape:
				raise ValueError(f"cell_params['{k}'] must have shape equal to grid")

	def _infer_species_from_param_name(self, param_name: str) -> Optional[int]:
		"""Infer species index (1-based) from a parameter name using `species_names`.

		Returns the 1-based species index if a matching prefix is found,
		otherwise `None`.
		"""
		if not isinstance(param_name, str):
			return None
		for idx, name in enumerate(self.species_names or ()):  # type: ignore
			if isinstance(name, str) and param_name.startswith(f"{name}_"):
				return idx + 1
		return None

	def evolve(self, param: str, species: Optional[int] = None, sd: float = 0.05, min_val: Optional[float] = None, max_val: Optional[float] = None) -> None:
		"""Enable per-cell evolution for `param` on `species`.

		If `species` is None, attempt to infer the species using
		`_infer_species_from_param_name(param)` which matches against
		`self.species_names`. This keeps `CA` free of domain-specific
		(predator/prey) logic while preserving backward compatibility when
		subclasses set `species_names` (e.g. `('prey','predator')`).
		"""
		if min_val is None:
			min_val = 0.01
		if max_val is None:
			max_val = 0.99
		if param not in self.params:
			raise ValueError(f"Unknown parameter '{param}'")
		if species is None:
			species = self._infer_species_from_param_name(param)
			if species is None:
				raise ValueError("species must be provided or inferable from param name and species_names")
		if not isinstance(species, int) or species <= 0 or species > self._n_species:
			raise ValueError("species must be an integer between 1 and n_species")

		arr = np.full(self.grid.shape, np.nan, dtype=float)
		mask = (self.grid == int(species))
		arr[mask] = float(self.params[param])
		self.cell_params[param] = arr
		self._evolve_info[param] = {"sd": float(sd), "min": float(min_val), "max": float(max_val), "species": int(species)}
			
	def count_neighbors(self) -> Tuple[np.ndarray, ...]:
		"""Count neighbors for each non-zero state.

		Returns a tuple of numpy arrays, one array for each state in
		`1..n_species`. Each returned array has the same shape as `grid`
		and contains the integer number of neighbors of that state for
		each cell, using periodic boundaries and the configured
		neighborhood type.

		Returns:
		- tuple of np.ndarray: one array per species (state 1..n_species)
		"""
		counts = []
		# Require SciPy for neighbor counting (fast and reliable); fail fast if missing
		global _cached_ndimage, _cached_kernels
		if _cached_ndimage is None:
			try:
				from scipy import ndimage as _ndimage
			except ImportError as e:
				# SciPy is required for correct and performant neighbor counting
				raise ImportError("scipy is required for count_neighbors(); install scipy") from e
			_cached_ndimage = _ndimage

		ndimage = _cached_ndimage

		# Cache kernel arrays per neighborhood to avoid reallocating each call
		kernel = _cached_kernels.get(self.neighborhood)
		if kernel is None:
			if self.neighborhood == "neumann":
				# Neumann (4-neighbors)
				kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int)
			else:
				# Moore (8-neighbors)
				kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)
			_cached_kernels[self.neighborhood] = kernel

		for state in range(1, self.n_species + 1):
			mask = (self.grid == state).astype(int)
			# ndimage.convolve with mode='wrap' implements periodic boundaries
			neigh = ndimage.convolve(mask, kernel, mode='wrap')
			counts.append(neigh)
		return tuple(counts)

	def update(self) -> None:
		"""Perform one update step.

		This base implementation must be overridden by subclasses. It raises
		NotImplementedError to indicate it should be provided by concrete
		models that inherit from `CA`.

		Returns: None
		"""
		raise NotImplementedError("Override update() in a subclass to define CA dynamics")

	def run(self, steps: int, stop_evolution_at: Optional[int] = None, snapshot_iters: Optional[list] = None) -> None:
		"""Run the CA for a number of steps.

		Args:
		- steps (int): number of iterations to run (must be non-negative).

		Returns: None
		"""
		assert isinstance(steps, int) and steps >= 0, "steps must be a non-negative integer"

		# normalize snapshot iteration list
		snapshot_set = set(snapshot_iters) if snapshot_iters is not None else set()

		for i in range(steps):
			self.update()
			# Update visualization if enabled every `interval` iterations
			if getattr(self, "_viz_on", False):
				# iteration number is 1-based for display
				try:
					self._viz_update(i + 1)
				except Exception:
					# Log visualization errors but don't stop the simulation
					logger.exception("Visualization update failed at iteration %d", i + 1)

			# create snapshots if requested at this iteration
			if (i + 1) in snapshot_set:
				try:
					# create snapshot folder if not present
					if not hasattr(self, "_viz_snapshot_dir") or self._viz_snapshot_dir is None:
						import os, time

						base = "results"
						ts = int(time.time())
						run_folder = f"run-{ts}"
						full = os.path.join(base, run_folder)
						os.makedirs(full, exist_ok=True)
						self._viz_snapshot_dir = full
					self._viz_save_snapshot(i + 1)
				except (OSError, PermissionError):
					logger.exception("Failed to create or write snapshot at iteration %d", i + 1)

			# stop evolution at specified time-step (disable further evolution)
			if stop_evolution_at is not None and (i + 1) == int(stop_evolution_at):
				# mark evolution as stopped; do not erase evolve metadata so
				# deterministic inheritance can still use parent values
				self._evolution_stopped = True

	def visualize(
		self,
		interval: int = 1,
		figsize: Tuple[float, float] = (5, 5),
		pause: float = 0.001,
		cmap=None,
		show_cell_params: bool = False,
		show_neighbors: bool = True,
		downsample: Optional[int] = None,
		) -> None:
		"""Enable interactive visualization of the grid.

		Args:
		- interval: update plot every `interval` iterations (>=1)
		- figsize: figure size passed to matplotlib
		- pause: seconds to pause after draw (controls responsiveness)
		- cmap: colormap spec (string, sequence of colors, or matplotlib Colormap).
		
		This function imports matplotlib lazily so simulations without
		visualization do not require matplotlib to be installed.
		"""
		if not isinstance(interval, int) or interval < 1:
			raise ValueError("interval must be a positive integer")

		# Lazy import but require matplotlib for visualization
		try:
			import matplotlib.pyplot as plt
			from matplotlib.colors import ListedColormap
		except ImportError as e:
			# Matplotlib is required for interactive visualization
			raise ImportError("matplotlib is required for visualize(); install matplotlib") from e

		# Keep a reference to pyplot so _viz_update can use it without importing
		self._viz_plt = plt

		# Resolve default cmap: prefer instance attribute override
		c_spec = self._default_cmap if cmap is None else cmap

		# Build a discrete colormap with entries for states 0..n_species
		n_colors_needed = self.n_species + 1
		if isinstance(c_spec, str):
			# request discrete version of named colormap
			cmap_obj = plt.get_cmap(c_spec, n_colors_needed)
		elif isinstance(c_spec, (list, tuple)):
			colors = list(c_spec)
			if len(colors) < n_colors_needed:
				colors = colors + [colors[-1]] * (n_colors_needed - len(colors))
			cmap_obj = ListedColormap(colors[:n_colors_needed])
		else:
			# Assume user provided a Colormap-like object
			cmap_obj = c_spec

		plt.ion()
		# Determine whether to show any per-cell parameter arrays
		param_arrays = []
		param_keys = []
		if show_cell_params and isinstance(self.cell_params, dict):
			for k, v in self.cell_params.items():
				if isinstance(v, np.ndarray) and v.shape == self.grid.shape:
					param_arrays.append(v)
					param_keys.append(k)

		n_extra = len(param_arrays)

		# Layout with GridSpec. We do NOT reserve separate GridSpec columns for
		# colorbars — colorbars are created next to image axes via
		# make_axes_locatable so they remain narrow and do not force a large
		# reserved column in the GridSpec.
		if show_neighbors:
			# histogram + states + one column per param image
			total_cols = 2 + n_extra
		else:
			# states + one column per param image
			total_cols = 1 + n_extra
		rows = 2
		width_ratios = [1.0] * total_cols
		height_ratios = [3.0, 2.0]
		# Use the total `figsize` passed by the caller as the full figure size
		fig = plt.figure(figsize=figsize, constrained_layout=True)
		gs = fig.add_gridspec(nrows=rows, ncols=total_cols, width_ratios=width_ratios, height_ratios=height_ratios)

		# top row and bottom row axes depend on whether neighbor plots are shown
		ax_params = []
		# colorbar axes are created dynamically for each param via make_axes_locatable
		ax_cbars = []
		ax_param_ts = []
		if show_neighbors:
			# top row: histogram, states, param images + colorbars
			ax_hist = fig.add_subplot(gs[0, 0])
			ax_states = fig.add_subplot(gs[0, 1])
			for i in range(n_extra):
				col_img = 2 + i
				ax_params.append(fig.add_subplot(gs[0, col_img]))

			# bottom row: percentiles, states time-series, param stats (under each param image)
			ax_percentiles = fig.add_subplot(gs[1, 0])
			ax_state_ts = fig.add_subplot(gs[1, 1])
			for i in range(n_extra):
				col_img = 2 + i
				ax_param_ts.append(fig.add_subplot(gs[1, col_img]))
		else:
			# top row: states in first column, then param images + colorbars
			ax_hist = None
			ax_percentiles = None
			ax_states = fig.add_subplot(gs[0, 0])
			for i in range(n_extra):
				col_img = 1 + i
				ax_params.append(fig.add_subplot(gs[0, col_img]))

			# bottom row: states time-series under states, param stats under param images
			ax_state_ts = fig.add_subplot(gs[1, 0])
			for i in range(n_extra):
				col_img = 1 + i
				ax_param_ts.append(fig.add_subplot(gs[1, col_img]))

		# determine downsampling factor for display to limit pixels drawn
		rows, cols = self.grid.shape
		_max_display = 300
		# default automatic downsample chosen from grid size
		_ds_auto = max(1, int(max(rows, cols) / _max_display))
		# allow caller to override with explicit downsample factor
		if downsample is None:
			_ds = _ds_auto
		else:
			_ds = max(1, int(downsample))
		# initialize histogram: compute neighbor counts for prey
		n_neighbors = 4 if self.neighborhood == "neumann" else 8
		counts = self.count_neighbors()[0]
		prey_pos = (self.grid == 1)
		if show_neighbors:
			if np.any(prey_pos):
				vals = counts[prey_pos]
			else:
				vals = np.array([], dtype=int)
			bins = np.arange(n_neighbors + 1)
			hist_vals = np.bincount(vals, minlength=n_neighbors + 1)
			bars = ax_hist.bar(bins, hist_vals, align='center')
			ax_hist.set_xlabel('Prey neighbor count')
			ax_hist.set_ylabel('Number of prey')
		else:
			bars = []

		# states image (downsampled for display)
		grid_disp = self.grid[::_ds, ::_ds]
		im_states = ax_states.imshow(grid_disp, cmap=cmap_obj, interpolation="nearest", vmin=0, vmax=self.n_species)
		ax_states.set_title('States grid')
		fig.suptitle('Iteration 0')

		# param images (place colorbars in reserved axes)
		param_imps = []
		param_cbs = []
		from mpl_toolkits.axes_grid1 import make_axes_locatable
		for i, arr in enumerate(param_arrays):
			axp = ax_params[i]
			key = param_keys[i]
			# If this parameter is evolving, use the evolve metadata min/max to
			# fix the colorbar range so it never needs updating at runtime.
			if hasattr(self, "_evolve_info") and key in self._evolve_info:
				meta = self._evolve_info.get(key, {})
				vmin = float(meta.get("min", 0.0))
				vmax = float(meta.get("max", 1.0))
			else:
				vmin = float(np.nanmin(arr)) if np.isfinite(np.nanmin(arr)) else 0.0
				vmax = float(np.nanmax(arr)) if np.isfinite(np.nanmax(arr)) else 1.0
			# downsample param image for display
			if arr is None:
				arr_disp = np.full((rows // _ds + (1 if rows % _ds else 0), cols // _ds + (1 if cols % _ds else 0)), np.nan)
			else:
				arr_disp = arr[::_ds, ::_ds]
			imp = axp.imshow(arr_disp, cmap=plt.get_cmap("viridis"), interpolation="nearest", vmin=vmin, vmax=vmax)
			title = key.replace('_', ' ').title()
			if 'Death' in title or 'death' in key:
				title = title + ' Rate'
			axp.set_title(title)
			# colorbar placed in dedicated axis
			# create a narrow colorbar axis to the right of the image
			# Prefer colorbar attached to the image axis so constrained_layout
			# manages spacing; provide a narrow fraction and small pad.
			try:
				cb = fig.colorbar(imp, ax=axp, orientation='vertical', fraction=0.046, pad=0.04)
			except Exception:
				# Fallback: try the axes_divider approach for narrow colorbar
				try:
					divider = make_axes_locatable(axp)
					cb_ax = divider.append_axes("right", size="5%", pad=0.05)
					cb = fig.colorbar(imp, cax=cb_ax, orientation='vertical')
				except Exception:
					# Last-resort fallback
					cb = fig.colorbar(imp)
			# ensure tick padding so ticks don't overflow figure bounds
			try:
				cb.ax.tick_params(axis='y', which='both', pad=2)
			except Exception:
				pass
			param_imps.append(imp)
			param_cbs.append(cb)

		# setup time-series plots (empty at start)
		time_x = []
		prey_ts = []
		pred_ts = []
		# match colors to the states colormap: state values map between 0..n_species
		try:
			prey_color = cmap_obj(1.0 / max(1, self.n_species))
		except Exception:
			prey_color = 'green'
		try:
			pred_color = cmap_obj(2.0 / max(1, self.n_species))
		except Exception:
			pred_color = 'red'
		line_prey, = ax_state_ts.plot([], [], label='Prey', color=prey_color)
		line_pred, = ax_state_ts.plot([], [], label='Predator', color=pred_color)
		ax_state_ts.set_xlabel('Iteration')
		ax_state_ts.set_ylabel('Count')
		ax_state_ts.legend()
		# initialize sensible x/y limits to avoid autoscale on first updates
		ax_state_ts.set_xlim(0, 1)
		ax_state_ts.set_ylim(0, rows * cols)

		# percentiles plot (only mean when enabled)
		perc_x = []
		perc_mean = []
		# define placeholders so later code can always unpack perc_lines
		line_25 = None
		line_mean = None
		line_75 = None
		if show_neighbors:
			line_mean, = ax_percentiles.plot([], [], label='Mean')
			ax_percentiles.set_xlabel('Iteration')
			ax_percentiles.set_ylabel('Neighbor count')
			ax_percentiles.legend()
		# initialize percentiles xlim/ylim (only when axis exists)
		if ax_percentiles is not None:
			ax_percentiles.set_xlim(0, 1)
			ax_percentiles.set_ylim(0, n_neighbors)

		# param stats plots: for each param, three lines
		param_stat_lines = {}
		for i, key in enumerate(param_keys):
			ax = ax_param_ts[i]
			# choose colors for min/mean/max based on the param colormap range
			cmap_param = plt.get_cmap("viridis")
			if hasattr(self, "_evolve_info") and key in self._evolve_info:
				meta = self._evolve_info.get(key, {})
				# map min->0.0, mean->0.5, max->1.0 on the colormap
				c_min = cmap_param(0.0)
				c_mean = cmap_param(0.5)
				c_max = cmap_param(1.0)
			else:
				c_min = cmap_param(0.0)
				c_mean = cmap_param(0.5)
				c_max = cmap_param(1.0)
			lmin, = ax.plot([], [], label='min', color=c_min, linewidth=1.25)
			lmean, = ax.plot([], [], label='mean', color=c_mean, linewidth=1.5)
			lmax, = ax.plot([], [], label='max', color=c_max, linewidth=1.25)
			ax.set_xlabel('Iteration')
			ax.set_ylabel(key)
			ax.legend()
			param_stat_lines[key] = (lmin, lmean, lmax)
			# initialize xlim and y limits for param time-series
			ax.set_xlim(0, 1)
			# if we have evolve metadata, use its min/max for y-limits
			if hasattr(self, "_evolve_info") and key in self._evolve_info:
				meta = self._evolve_info.get(key, {})
				ax.set_ylim(float(meta.get('min', 0.0)), float(meta.get('max', 1.0)))
			else:
				ax.set_ylim(0.0, 1.0)

		# use constrained layout already; ensure spacing
		# constrained_layout=True used on figure to manage spacing

		# store viz objects
		self._viz_on = True
		self._viz_interval = interval
		self._viz_fig = fig
		self._viz_axes = {
			"hist": ax_hist,
			"percentiles": ax_percentiles,
			"states": ax_states,
			"state_ts": ax_state_ts,
			"param_imgs": ax_params,
			"param_ts": ax_param_ts,
		}
		self._viz_art = {
			"im_states": im_states,
			"param_imps": param_imps,
			"param_cbs": param_cbs,
			"hist_bars": bars,
			"state_lines": (line_prey, line_pred),
			"perc_lines": (line_25, line_mean, line_75),
			"param_stat_lines": param_stat_lines,
		}
		self._viz_time = []
		self._viz_prey_counts = []
		self._viz_pred_counts = []
		self._viz_param_stats = {k: {"min": [], "mean": [], "max": []} for k in param_keys}
		self._viz_neighbor_stats = {"25": [], "mean": [], "75": []}
		self._viz_param_keys = param_keys
		self._viz_pause = float(pause)
		self._viz_cmap = cmap_obj
		self._viz_snapshot_dir = None
		# downsample factor used for display
		self._viz_ds = _ds
		# precompute a slice tuple for downsampled display indexing
		self._viz_slice = (slice(None, None, _ds), slice(None, None, _ds))

		plt.show(block=False)
		# draw once to populate the renderer
		fig.canvas.draw()
		plt.pause(pause)

		# Blit setup: copy background after the first draw if supported
		canvas = fig.canvas
		can_blit = hasattr(canvas, "copy_from_bbox") and hasattr(canvas, "blit")
		if can_blit:
			try:
				bg = canvas.copy_from_bbox(fig.bbox)
				self._viz_blit = True
				self._viz_background = bg
			except Exception:
				self._viz_blit = False
		else:
			self._viz_blit = False

		# (Duplicate viz state assignments removed — state already stored above)

	def _viz_update(self, iteration: int) -> None:
		"""Update the interactive plot if the configured interval has passed.

		This function updates images, histograms and time-series only every
		the configured interval to reduce overhead.
		"""
		if not getattr(self, "_viz_on", False):
			return
		if (iteration % int(self._viz_interval)) != 0:
			return

		# pyplot is provided via visualize(); avoid importing here for performance
		plt = getattr(self, "_viz_plt", None)

		# update figure title
		try:
			self._viz_fig.suptitle(f"Iteration {iteration}")
		except Exception:
			pass

		# update states image (use downsampled display array)
		art = getattr(self, "_viz_art", None)
		if art is None:
			return
		im_states = art.get("im_states")
		_ds = getattr(self, "_viz_ds", 1)
		# bind grid locally for faster access
		grid = self.grid
		if im_states is not None:
			_vslice = getattr(self, "_viz_slice", (slice(None, None, _ds), slice(None, None, _ds)))
			im_states.set_data(grid[_vslice])

		# update histogram of prey neighbor counts
		counts = self.count_neighbors()[0]
		prey_mask = (grid == 1)
		if np.any(prey_mask):
			vals = counts[prey_mask]
		else:
			vals = np.array([], dtype=int)
		nn = 4 if self.neighborhood == "neumann" else 8
		hist_vals = np.bincount(vals, minlength=nn + 1)
		bars = art.get("hist_bars")
		if bars is not None:
			for rect, h in zip(bars, hist_vals):
				rect.set_height(h)

		# compute percentiles
		if vals.size > 0:
			p25 = float(np.percentile(vals, 25))
			pmean = float(np.mean(vals))
			p75 = float(np.percentile(vals, 75))
		else:
			p25 = pmean = p75 = 0.0

		# update time series data (append)
		self._viz_time.append(iteration)
		prey_count = int(np.count_nonzero(grid == 1))
		pred_count = int(np.count_nonzero(grid == 2))
		self._viz_prey_counts.append(prey_count)
		self._viz_pred_counts.append(pred_count)
		self._viz_neighbor_stats["25"].append(p25)
		self._viz_neighbor_stats["mean"].append(pmean)
		self._viz_neighbor_stats["75"].append(p75)

		# update state time-series lines (do not autoscale; update xlim incrementally)
		line_prey, line_pred = art.get("state_lines", (None, None))
		if line_prey is not None:
			line_prey.set_data(self._viz_time, self._viz_prey_counts)
		if line_pred is not None:
			line_pred.set_data(self._viz_time, self._viz_pred_counts)
		# adjust xlim/ylim incrementally to avoid per-frame relim
		if len(self._viz_time) > 0:
			ax = self._viz_axes["state_ts"]
			cur_xmax = ax.get_xlim()[1]
			if iteration > cur_xmax:
				ax.set_xlim(0, max(iteration, int(cur_xmax * 1.2)))
			# ensure some padding for y
			ymax = max(max(self._viz_prey_counts or [0]), max(self._viz_pred_counts or [0]), 1)
			ax.set_ylim(0, ymax * 1.1)

		# update percentiles plot (avoid autoscale)
		l25, lmean, l75 = art.get("perc_lines", (None, None, None))
		if l25 is not None:
			l25.set_data(self._viz_time, self._viz_neighbor_stats["25"])
		if lmean is not None:
			lmean.set_data(self._viz_time, self._viz_neighbor_stats["mean"])
		if l75 is not None:
			l75.set_data(self._viz_time, self._viz_neighbor_stats["75"])
		# keep fixed y-limits for percentiles (if axis exists)
		if len(self._viz_time) > 0:
			axp = self._viz_axes.get("percentiles")
			if axp is not None:
				axp.set_ylim(0, nn)
				# expand xlim incrementally
				cur_xmax = axp.get_xlim()[1]
				if iteration > cur_xmax:
					axp.set_xlim(0, max(iteration, int(cur_xmax * 1.2)))

		# update param images and stats
		for idx, key in enumerate(self._viz_param_keys):
			arr = self.cell_params.get(key)
			im = None
			try:
				im = art["param_imps"][idx]
			except Exception:
				im = None
			if im is not None:
				if arr is None:
					arr = np.full(self.grid.shape, np.nan)
				# update downsampled display array only
				_vslice = getattr(self, "_viz_slice", (slice(None, None, _ds), slice(None, None, _ds)))
				im.set_data(arr[_vslice])
				# do not call colorbar update here (vmin/vmax fixed)
				# compute stats for plotting in param_ts (cheap)
				try:
					flat = arr[~np.isnan(arr)]
					if flat.size > 0:
						mn = float(np.nanmin(flat))
						mx = float(np.nanmax(flat))
						md = float(np.nanmean(flat))
					else:
						mn = md = mx = 0.0
				except Exception:
					mn = md = mx = 0.0
				self._viz_param_stats[key]["min"].append(mn)
				self._viz_param_stats[key]["mean"].append(md)
				self._viz_param_stats[key]["max"].append(mx)
				# update stat lines without autoscaling; expand xlim incrementally
				lmin, lmean, lmax = art["param_stat_lines"].get(key, (None, None, None))
				if lmin is not None:
					lmin.set_data(self._viz_time, self._viz_param_stats[key]["min"])
				if lmean is not None:
					lmean.set_data(self._viz_time, self._viz_param_stats[key]["mean"])
				if lmax is not None:
					lmax.set_data(self._viz_time, self._viz_param_stats[key]["max"])
				# expand x-axis for this param's time-series
				axp = self._viz_axes["param_ts"][idx]
				cur_xmax = axp.get_xlim()[1]
				if iteration > cur_xmax:
					axp.set_xlim(0, max(iteration, int(cur_xmax * 1.2)))

		# redraw using blitting if available (avoid full-figure draw)
		canvas = self._viz_fig.canvas
		if getattr(self, "_viz_blit", False):
			try:
				# restore background and draw only updated artists
				canvas.restore_region(self._viz_background)
				# draw image artists and lines
				# states image
				if im_states is not None:
					self._viz_axes["states"].draw_artist(im_states)
				# param images
				for im in art.get("param_imps", []):
					if im is not None:
						im.axes.draw_artist(im)
				# histogram bars
				for rect in art.get("hist_bars", []):
					rect.axes.draw_artist(rect)
				# state lines
				for ln in art.get("state_lines", ()): 
					if ln is not None:
						ln.axes.draw_artist(ln)
				# percentile lines
				for ln in art.get("perc_lines", ()): 
					if ln is not None:
						ln.axes.draw_artist(ln)
				# param stat lines
				for key in art.get("param_stat_lines", {}):
					for ln in art["param_stat_lines"][key]:
						if ln is not None:
							ln.axes.draw_artist(ln)

				# blit to the screen
				canvas.blit(self._viz_fig.bbox)
			except Exception:
				# fallback to full draw on error
				try:
					self._viz_fig.canvas.draw()
				except Exception:
					pass
			# small pause to let GUI update
			if plt is not None:
				plt.pause(self._viz_pause)
			else:
				# fallback: small sleep if pyplot not available
				import time
				time.sleep(float(self._viz_pause))
		else:
			# fallback: full draw
			try:
				self._viz_fig.canvas.draw()
			except Exception:
				try:
					self._viz_fig.canvas.draw_idle()
				except Exception:
					pass
			if plt is not None:
				plt.pause(self._viz_pause)
			else:
				import time
				time.sleep(float(self._viz_pause))


class PP(CA):
	"""Predator-prey CA.

	States: 0 = empty, 1 = prey, 2 = predator

	Parameters (in `params` dict). Allowed keys and defaults:
	- "prey_death": 0.05
	- "predator_death": 0.1
	- "prey_birth": 0.25
	- "predator_birth": 0.2

	The constructor validates parameters are in [0,1] and raises if
	other user-supplied params are present. The `synchronous` flag
	chooses the update mode (default True -> synchronous updates).
	"""

	# Default colors: 0=empty black, 1=prey green, 2=predator red
	_default_cmap = ("black", "green", "red")

	def __init__(
		self,
		rows: int = 10,
		cols: int = 10,
		densities: Tuple[float, ...] = (0.2, 0.1),
		neighborhood: str = "moore",
		params: Dict[str, object] = None,
		cell_params: Dict[str, object] = None,
		seed: Optional[int] = None,
		synchronous: bool = True,
	) -> None:
		# Allowed params and defaults
		_defaults = {
			"prey_death": 0.05,
			"predator_death": 0.1,
			"prey_birth": 0.25,
			"predator_birth": 0.2,
		}

		# Validate user-supplied params: only allowed keys
		if params is None:
			merged_params = dict(_defaults)
		else:
			if not isinstance(params, dict):
				raise TypeError("params must be a dict or None")
			extra = set(params.keys()) - set(_defaults.keys())
			if extra:
				raise ValueError(f"Unexpected parameter keys: {sorted(list(extra))}")
			# Do not override user-set values: start from defaults then update with user values
			merged_params = dict(_defaults)
			merged_params.update(params)

		# Validate numerical ranges
		for k, v in merged_params.items():
			if not isinstance(v, (int, float)):
				raise TypeError(f"Parameter '{k}' must be a number between 0 and 1")
			if not (0.0 <= float(v) <= 1.0):
				raise ValueError(f"Parameter '{k}' must be between 0 and 1")

		# Call base initializer with merged params
		super().__init__(rows, cols, densities, neighborhood, merged_params, cell_params, seed)

		self.synchronous: bool = bool(synchronous)
		# set human-friendly species names for PP
		self.species_names = ("prey", "predator")


	# Remove PP-specific evolve wrapper; use CA.evolve with optional species

	def validate(self) -> None:
		"""Validate PP-specific invariants in addition to base CA checks.

		Checks:
		- each global parameter is numeric and in [0,1]
		- per-cell evolved parameter arrays (in `_evolve_info`) have non-NaN
		  positions matching the species grid and contain values within the
		  configured min/max range (or are NaN).
		"""
		super().validate()

		# Validate global params
		for k, v in (self.params or {}).items():
			if not isinstance(v, (int, float)):
				raise TypeError(f"Parameter '{k}' must be numeric")
			if not (0.0 <= float(v) <= 1.0):
				raise ValueError(f"Parameter '{k}' must be between 0 and 1")

		# Validate per-cell evolve arrays
		for pname, meta in (self._evolve_info or {}).items():
			arr = self.cell_params.get(pname)
			if not isinstance(arr, np.ndarray):
				# absent or non-array per-cell params are allowed; skip
				continue
			# shape already checked in super().validate(), but be explicit
			if arr.shape != self.grid.shape:
				raise ValueError(f"cell_params['{pname}'] must match grid shape")
			# expected non-NaN positions correspond to species stored in metadata
			species = None
			if isinstance(meta, dict) and "species" in meta:
				species = int(meta.get("species"))
			else:
				# try to infer species from parameter name using species_names
				species = self._infer_species_from_param_name(pname)
				if species is None:
					raise ValueError(f"cell_params['{pname}'] missing species metadata and could not infer from name")
			nonnan = ~np.isnan(arr)
			expected = (self.grid == species)
			if not np.array_equal(nonnan, expected):
				raise ValueError(f"cell_params['{pname}'] non-NaN entries must match positions of species {species}")
			# values must be within configured range where not NaN
			mn = float(meta.get("min", 0.0))
			mx = float(meta.get("max", 1.0))
			vals = arr[~np.isnan(arr)]
			if vals.size > 0:
				if np.any(vals < mn) or np.any(vals > mx):
					raise ValueError(f"cell_params['{pname}'] contains values outside [{mn}, {mx}]")

	def _apply_deaths_and_clear_params(self, grid_ref: np.ndarray, rand_prey: np.ndarray, rand_pred: np.ndarray) -> None:
		"""Apply deaths based on sampled random arrays and clear per-cell params.

		This consolidates the repeated logic used by both synchronous and
		asynchronous update methods.
		"""
		# Determine death masks using per-cell arrays when present, otherwise global scalars
		cp_prey = self.cell_params.get("prey_death")
		if isinstance(cp_prey, np.ndarray):
			prey_death_mask = (rand_prey < cp_prey)
		else:
			prey_death_mask = (grid_ref == 1) & (rand_prey < float(self.params["prey_death"]))

		cp_pred = self.cell_params.get("predator_death")
		if isinstance(cp_pred, np.ndarray):
			pred_death_mask = (rand_pred < cp_pred)
		else:
			pred_death_mask = (grid_ref == 2) & (rand_pred < float(self.params["predator_death"]))

		# Apply deaths to the current grid
		self.grid[prey_death_mask] = 0
		self.grid[pred_death_mask] = 0

		# Clear per-cell parameters for dead individuals
		for pname, meta in self._evolve_info.items():
			# determine species from metadata or infer from name
			species = None
			if isinstance(meta, dict) and "species" in meta:
				species = int(meta.get("species"))
			else:
				species = self._infer_species_from_param_name(pname)
				if species is None:
					# cannot determine species; skip clearing for safety
					continue
			arr = self.cell_params.get(pname)
			if not (isinstance(arr, np.ndarray) and arr.shape == self.grid.shape):
				continue
			if species == 1:
				arr[prey_death_mask] = np.nan
			elif species == 2:
				arr[pred_death_mask] = np.nan

	def _neighbor_shifts(self) -> Tuple[np.ndarray, np.ndarray, int]:
		"""Return neighbor shift arrays (dr_arr, dc_arr, n_shifts) for the configured neighborhood."""
		if self.neighborhood == "neumann":
			shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
		else:
			shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
		dr_arr = np.array([s[0] for s in shifts], dtype=int)
		dc_arr = np.array([s[1] for s in shifts], dtype=int)
		return dr_arr, dc_arr, len(shifts)

	def _get_parent_probs(self, sources: np.ndarray, birth_param_key: str, birth_prob: float) -> np.ndarray:
		"""Return per-source birth probabilities from `cell_params` or scalar fallback."""
		M = sources.shape[0]
		parent_grid = None
		if isinstance(birth_param_key, str) and birth_param_key in self.cell_params:
			parent_grid = self.cell_params.get(birth_param_key)
		if isinstance(parent_grid, np.ndarray):
			return parent_grid[sources[:, 0], sources[:, 1]]
		return np.full(M, float(birth_prob))

	def _inherit_params_on_birth(self, chosen_rs: np.ndarray, chosen_cs: np.ndarray, parents: np.ndarray, new_state_val: int) -> None:
		"""Handle inheritance and clearing of evolved per-cell parameters after births.

		`chosen_rs`, `chosen_cs` are arrays of target coordinates; `parents` is
		an array of parent coordinates with same length.
		"""
		for pname, meta in self._evolve_info.items():
			# determine species this parameter belongs to via metadata or inference
			species = None
			if isinstance(meta, dict) and "species" in meta:
				species = int(meta.get("species"))
			else:
				species = self._infer_species_from_param_name(pname)
				if species is None:
					raise ValueError(f"_evolve_info contains unexpected key '{pname}' without species metadata and could not infer")

			# if new_state is not the species for this param, clear at targets
			if new_state_val != species:
				arr = self.cell_params.get(pname)
				if isinstance(arr, np.ndarray) and arr.shape == self.grid.shape:
					arr[chosen_rs, chosen_cs] = np.nan
				continue

			# otherwise, inherit parent values (fallback to global param when NaN)
			arr = self.cell_params.get(pname)
			if isinstance(arr, np.ndarray) and arr.shape == self.grid.shape:
				parent_vals = arr[parents[:, 0], parents[:, 1]]
				mask_nan = np.isnan(parent_vals)
				if np.any(mask_nan):
					parent_vals = parent_vals.copy()
					parent_vals[mask_nan] = float(self.params.get(pname, 0.0))
			else:
				parent_vals = np.full(parents.shape[0], float(self.params.get(pname, 0.0)))

			# mutate and clip
			sd = float(meta["sd"])
			mn = float(meta["min"])
			mx = float(meta["max"])
			# If evolution has been stopped, inheritance is deterministic: copy
			# parent values directly without Gaussian mutation so we can observe
			# which parameter values survive.
			if getattr(self, "_evolution_stopped", False):
				mut = parent_vals.copy()
			else:
				mut = parent_vals + self.generator.normal(0.0, sd, size=parent_vals.shape)
				mut = np.clip(mut, mn, mx)
			# If an array exists but has wrong shape, raise an informative error
			existing = self.cell_params.get(pname)
			if isinstance(existing, np.ndarray) and existing.shape != self.grid.shape:
				raise ValueError(f"cell_params['{pname}'] must have shape equal to grid")
			if pname not in self.cell_params or not isinstance(self.cell_params[pname], np.ndarray):
				self.cell_params[pname] = np.full(self.grid.shape, np.nan)
			self.cell_params[pname][chosen_rs, chosen_cs] = mut


	def update_sync(self) -> None:
		"""Synchronous (vectorized) update.

		Implements a vectorized equivalent of the random-sequential
		asynchronous update. Each occupied cell (prey or predator) gets at
		most one reproduction attempt: with probability `birth` it chooses a
		neighbor and, if that neighbor in the reference grid has the
		required target state (empty for prey, prey for predator), it
		becomes a candidate attempt. 
		
		Predators use directed movement: they preferentially move toward
		prey neighbors when available; otherwise pick a random neighbor.
		
		When multiple reproducers target the same cell, one attempt is 
		chosen uniformly at random to succeed. Deaths are applied the same 
		vectorized way as in the async update.
		"""

		# Assumes any per-cell parameter arrays (in `self.cell_params`) are
		# well-formed and correspond to the current grid. `run()` performs a
		# one-time validation before the update loop; update methods do not
		# re-check for performance reasons.
		# Bind hot attributes to locals for performance and clarity
		grid = self.grid
		gen = self.generator
		params = self.params
		cell_params = self.cell_params
		rows, cols = grid.shape
		grid_ref = grid.copy()

		# Sample deaths and apply them (clears per-cell params for dead individuals)
		rand_prey = gen.random(grid.shape)
		rand_pred = gen.random(grid.shape)
		self._apply_deaths_and_clear_params(grid_ref, rand_prey, rand_pred)

		# Precompute neighbor shifts and arrays for indexing (used by reproduction)
		dr_arr, dc_arr, n_shifts = self._neighbor_shifts()

		def _process_reproduction(sources, birth_param_key, birth_prob, target_state_required, new_state_val):
			"""Handle reproduction attempts from `sources`.

			sources: (M,2) array of (r,c) positions in grid_ref
			birth_prob: scalar probability that a source attempts reproduction
			target_state_required: state value required in grid_ref at target
			new_state_val: state to write into self.grid for successful targets
			"""
			if sources.size == 0:
				return

			M = sources.shape[0]
			# Determine per-source birth probabilities (from cell_params if present)
			parent_probs = self._get_parent_probs(sources, birth_param_key, birth_prob)

			# Which sources attempt reproduction
			attempt_mask = gen.random(M) < parent_probs
			if not np.any(attempt_mask):
				return

			src = sources[attempt_mask]
			K = src.shape[0]

			# Each attempting source picks one neighbor uniformly
			nbr_idx = gen.integers(0, n_shifts, size=K)
			nr = (src[:, 0] + dr_arr[nbr_idx]) % rows
			nc = (src[:, 1] + dc_arr[nbr_idx]) % cols

			# Only keep attempts where the reference grid at the target has the required state
			valid_mask = (grid_ref[nr, nc] == target_state_required)
			if not np.any(valid_mask):
				return

			src_valid = src[valid_mask]
		
			nr = nr[valid_mask]
			nc = nc[valid_mask]

			# Flatten target indices to group collisions
			target_flat = (nr * cols + nc).astype(np.int64)
			# Sort targets to find groups that target the same cell
			order = np.argsort(target_flat)
			tf_sorted = target_flat[order]

			# unique targets (on the sorted array) with start indices and counts
			uniq_targets, idx_start, counts = np.unique(tf_sorted, return_index=True, return_counts=True)
			if uniq_targets.size == 0:
				return

			# For each unique target, pick one attempt uniformly at random
			# idx_start gives indices into the sorted array
			chosen_sorted_positions = []
			for start, cnt in zip(idx_start, counts):
				off = int(gen.integers(0, cnt))
				chosen_sorted_positions.append(start + off)
			chosen_sorted_positions = np.array(chosen_sorted_positions, dtype=int)

			# Map back to indices in the filtered attempts array
			chosen_indices = order[chosen_sorted_positions]

			chosen_target_flats = target_flat[chosen_indices]
			chosen_rs = (chosen_target_flats // cols).astype(int)
			chosen_cs = (chosen_target_flats % cols).astype(int)

			# Determine parent positions corresponding to chosen attempts
			parents = src_valid[chosen_indices]

			# Apply successful births to the main grid (grid is alias of self.grid)
			grid[chosen_rs, chosen_cs] = new_state_val

			# Handle inheritance/clearing of per-cell parameters
			self._inherit_params_on_birth(chosen_rs, chosen_cs, parents, new_state_val)

		def _process_predator_hunting(sources, birth_param_key, birth_prob):
			"""Handle predator reproduction with directed movement toward prey.
			
			Predators check all neighbors: if any neighbor contains prey, 
			preferentially move to one of them; otherwise pick a random neighbor.
			"""
			if sources.size == 0:
				return

			M = sources.shape[0]
			# Determine per-source birth probabilities (from cell_params if present)
			parent_probs = self._get_parent_probs(sources, birth_param_key, birth_prob)

			# Which sources attempt reproduction
			attempt_mask = gen.random(M) < parent_probs
			if not np.any(attempt_mask):
				return

			src = sources[attempt_mask]
			K = src.shape[0]

			# For each predator, check all neighbors to find prey
			selected_neighbors = np.zeros((K, 2), dtype=int)
			
			for i in range(K):
				r, c = int(src[i, 0]), int(src[i, 1])
				# Get all neighbor positions
				neighbors_r = (r + dr_arr) % rows
				neighbors_c = (c + dc_arr) % cols
				# Check which neighbors have prey
				prey_neighbors = (grid_ref[neighbors_r, neighbors_c] == 1)
				
				if np.any(prey_neighbors):
					# Pick one prey neighbor uniformly at random (directed movement)
					prey_indices = np.where(prey_neighbors)[0]
					chosen_idx = int(gen.choice(prey_indices))
				else:
					# No prey visible; pick a random neighbor
					chosen_idx = int(gen.integers(0, n_shifts))
				
				selected_neighbors[i, 0] = neighbors_r[chosen_idx]
				selected_neighbors[i, 1] = neighbors_c[chosen_idx]

			nr = selected_neighbors[:, 0]
			nc = selected_neighbors[:, 1]

			# Only keep attempts where the target was prey (required state = 1)
			valid_mask = (grid_ref[nr, nc] == 1)
			if not np.any(valid_mask):
				return

			src_valid = src[valid_mask]
			nr = nr[valid_mask]
			nc = nc[valid_mask]

			# Flatten target indices to group collisions
			target_flat = (nr * cols + nc).astype(np.int64)
			# Sort targets to find groups that target the same cell
			order = np.argsort(target_flat)
			tf_sorted = target_flat[order]

			# unique targets (on the sorted array) with start indices and counts
			uniq_targets, idx_start, counts = np.unique(tf_sorted, return_index=True, return_counts=True)
			if uniq_targets.size == 0:
				return

			# For each unique target, pick one predator uniformly at random
			chosen_sorted_positions = []
			for start, cnt in zip(idx_start, counts):
				off = int(gen.integers(0, cnt))
				chosen_sorted_positions.append(start + off)
			chosen_sorted_positions = np.array(chosen_sorted_positions, dtype=int)

			# Map back to indices in the filtered attempts array
			chosen_indices = order[chosen_sorted_positions]

			chosen_target_flats = target_flat[chosen_indices]
			chosen_rs = (chosen_target_flats // cols).astype(int)
			chosen_cs = (chosen_target_flats % cols).astype(int)

			# Determine parent positions (predators) corresponding to chosen attempts
			parents = src_valid[chosen_indices]

			# Apply successful hunts: predators convert prey to predator
			grid[chosen_rs, chosen_cs] = 2

			# Handle inheritance/clearing of per-cell parameters
			self._inherit_params_on_birth(chosen_rs, chosen_cs, parents, 2)

		# Prey reproduce into empty cells (target state 0 -> new state 1)
		prey_sources = np.argwhere(grid_ref == 1)
		_process_reproduction(prey_sources, "prey_birth", self.params["prey_birth"], 0, 1)

		# Predators hunt and reproduce with directed movement toward prey
		pred_sources = np.argwhere(grid_ref == 2)
		_process_predator_hunting(pred_sources, "predator_birth", self.params["predator_birth"])

	def update_async(self) -> None:
		"""Asynchronous (random-sequential) update with directed predator movement.

		Rules (applied using a copy of the current grid for reference):
		- Iterate occupied cells in random order.
		- Prey (1): pick random neighbor; if neighbor was empty in copy,
		  reproduce into it with probability `prey_birth`.
		- Predator (2): check all neighbors for prey. If prey neighbors exist,
		  pick one of them uniformly at random (directed hunt). Otherwise pick
		  a random neighbor. If target is prey in reference copy, reproduce 
		  (convert to predator) with probability `predator_birth`.
		- After the reproduction loop, apply deaths synchronously using the
		  copy as the reference so newly created individuals are not instantly
		  killed. Deaths only remove individuals if the current cell still
		  matches the species from the reference copy.
		"""
		# Bind hot attributes to locals for speed and clarity
		grid = self.grid
		gen = self.generator
		params = self.params
		cell_params = self.cell_params
		rows, cols = grid.shape
		grid_ref = grid.copy()

		# Sample and apply deaths first (based on the reference grid). Deaths
		# are sampled from `grid_ref` so statistics remain identical.
		rand_prey = gen.random(grid.shape)
		rand_pred = gen.random(grid.shape)
		self._apply_deaths_and_clear_params(grid_ref, rand_prey, rand_pred)

		# Precompute neighbor shifts
		dr_arr, dc_arr, n_shifts = self._neighbor_shifts()

		# Get occupied cells from the original reference grid and shuffle.
		# We iterate over `grid_ref` so that sources can die and reproduce
		# in the same iteration, meaning we are order-agnostic.
		occupied = np.argwhere(grid_ref != 0)
		if occupied.size > 0:
			order = gen.permutation(len(occupied))
			for idx in order:
				r, c = int(occupied[idx, 0]), int(occupied[idx, 1])
				state = int(grid_ref[r, c])
				
				if state == 1:
					# Prey: pick a random neighbor
					nbi = int(gen.integers(0, n_shifts))
					dr = int(dr_arr[nbi])
					dc = int(dc_arr[nbi])
					nr = (r + dr) % rows
					nc = (c + dc) % cols
					# Prey reproduces into empty neighbor (reference must be empty)
					if grid_ref[nr, nc] == 0:
						# per-parent birth prob
						pval = self._get_parent_probs(np.array([[r, c]]), "prey_birth", float(params["prey_birth"]))[0]
						if gen.random() < float(pval):
							# birth: set new prey and inherit per-cell params (if any)
							grid[nr, nc] = 1
							# handle param clearing/inheritance for a single birth
							self._inherit_params_on_birth(np.array([nr]), np.array([nc]), np.array([[r, c]]), 1)
				
				elif state == 2:
					# Predator: directed hunt toward prey
					# Check all neighbors for prey
					neighbors_r = (r + dr_arr) % rows
					neighbors_c = (c + dc_arr) % cols
					prey_neighbors = (grid_ref[neighbors_r, neighbors_c] == 1)
					
					if np.any(prey_neighbors):
						# At least one prey neighbor: pick one uniformly at random
						prey_indices = np.where(prey_neighbors)[0]
						chosen_idx = int(gen.choice(prey_indices))
					else:
						# No prey visible: pick a random neighbor (explore)
						chosen_idx = int(gen.integers(0, n_shifts))
					
					nr = int(neighbors_r[chosen_idx])
					nc = int(neighbors_c[chosen_idx])
					
					# Predator reproduces into prey neighbor (reference must be prey)
					if grid_ref[nr, nc] == 1:
						pval = self._get_parent_probs(np.array([[r, c]]), "predator_birth", float(params["predator_birth"]))[0]
						if gen.random() < float(pval):
							# predator converts prey -> predator: assign and handle params
							grid[nr, nc] = 2
							self._inherit_params_on_birth(np.array([nr]), np.array([nc]), np.array([[r, c]]), 2)

	def update(self) -> None:
		"""Dispatch to synchronous or asynchronous update mode."""
		if self.synchronous:
			self.update_sync()
		else:
			self.update_async()