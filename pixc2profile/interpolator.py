"""
Interpolator functions for smoothing and interpolating water surface elevation profiles.

This module provides various interpolation and smoothing functions including LOWESS and 
Gaussian Process Regression for processing water surface elevation data.
"""

from typing import List, Optional, Union, Dict, Any, Tuple
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern


# Default parameters for different methods
DEFAULT_LOWESS_PARAMS = {
    'frac_list': [0.01, 0.05, 0.1, 0.2],
    'it': 3,
    'delta': 0.0,
    'return_sorted': False
}

DEFAULT_GP_PARAMS = {
    'matern_nu': 2.5,
    'length_scale_bounds': (50.0, 20000.0),
    'add_nugget_var': 0.0,
    'fit_white_noise': True,
    'white_noise_bounds': (1e-8, 1e-2),
    'normalize_y': True,
    'n_restarts_optimizer': 3,
    'random_state': 0
}

SUPPORTED_METHODS = ['lowess', 'gaussian_process']


class Interpolator:
    """
    Interpolator class for smoothing and interpolating water surface elevation profiles.
    
    This class provides various interpolation and smoothing methods including LOWESS and 
    Gaussian Process Regression for processing water surface elevation data.
    """
    
    def __init__(self):
        """Initialize the Interpolator."""
        self.supported_methods = SUPPORTED_METHODS
    
    def _validate_input_data(self, data: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
        """
        Validate input data and handle missing values.
        
        Args:
            data: Input DataFrame
            x_col: Name of x-coordinate column
            y_col: Name of y-coordinate column
            
        Returns:
            Clean DataFrame with non-null values
            
        Raises:
            ValueError: If required columns are missing or insufficient data
        """
        required_cols = [x_col, y_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove rows with NaN in either x or y columns
        clean_data = data.dropna(subset=[x_col, y_col])
        
        if len(clean_data) < 3:
            raise ValueError(f"Insufficient data points. Need at least 3, got {len(clean_data)}")
        
        return clean_data.sort_values(x_col)

    def _segment_data_by_location(self, data: pd.DataFrame, x_col: str, 
                                 seg_locations: List[float]) -> List[pd.DataFrame]:
        """
        Segment data based on location breakpoints.
        
        Args:
            data: Input DataFrame
            x_col: Name of x-coordinate column
            seg_locations: List of segment locations for breakpoints
            
        Returns:
            List of segmented DataFrames
        """
        if not seg_locations:
            return [data]
        
        segments = []
        seg_locations_sorted = sorted(seg_locations)
        
        for i in range(len(seg_locations_sorted) + 1):
            if i == 0:
                # First segment from start to first breakpoint
                segment = data[data[x_col] <= seg_locations_sorted[i]]
            elif i == len(seg_locations_sorted):
                # Last segment from last breakpoint to end
                segment = data[data[x_col] > seg_locations_sorted[i-1]]
            else:
                # Middle segments between two breakpoints
                segment = data[
                    (data[x_col] > seg_locations_sorted[i-1]) & 
                    (data[x_col] <= seg_locations_sorted[i])
                ]
            
            if not segment.empty:
                segments.append(segment.copy())
        
        return segments

    def _compute_latent_std(self, gpr: GaussianProcessRegressor, y_std: np.ndarray) -> np.ndarray:
        """
        Compute latent function uncertainty (excluding observation noise).
        
        This approximates the latent variance by removing learned white noise
        from the predictive variance.
        """
        f_std = y_std.copy()
        
        try:
            # Find white noise level if present in fitted kernel
            def _find_white_noise(kernel_obj):
                if isinstance(kernel_obj, WhiteKernel):
                    return float(kernel_obj.noise_level)
                if hasattr(kernel_obj, "k1") and hasattr(kernel_obj, "k2"):
                    wn1 = _find_white_noise(kernel_obj.k1)
                    wn2 = _find_white_noise(kernel_obj.k2)
                    return wn1 + wn2  # one of them likely 0
                return 0.0
            
            white_var = _find_white_noise(gpr.kernel_)
            
            # Remove (approx) noise contribution from y variance to get latent variance
            var_latent = np.maximum(y_std**2 - white_var, 0.0)
            f_std = np.sqrt(var_latent)
            
        except Exception:
            # If extraction fails, just return the original std
            pass
        
        return f_std

    def smooth_lowess(self, data: pd.DataFrame, x_col: str, y_col: str,
                     frac_list: Optional[List[float]] = None,
                     it: int = None,
                     seg_locations: Optional[List[float]] = None,
                     x_eval: Optional[np.ndarray] = None,
                     **kwargs) -> pd.DataFrame:
        """
        Apply LOWESS smoothing to the data.
        
        Args:
            data: Input DataFrame
            x_col: Name of x-coordinate column  
            y_col: Name of y-coordinate column
            frac_list: List of fractions for LOWESS smoothing
            it: Number of robustifying iterations
            seg_locations: List of segment locations for piecewise smoothing
            x_eval: Optional array of x values for evaluation (default: use original x values)
            **kwargs: Additional arguments for LOWESS
            
        Returns:
            DataFrame with smoothed values added as new columns
        """
        # Set defaults
        if frac_list is None:
            frac_list = DEFAULT_LOWESS_PARAMS['frac_list']
        if it is None:
            it = DEFAULT_LOWESS_PARAMS['it']
        if seg_locations is None:
            seg_locations = []
        
        # Validate fractions
        if not all(isinstance(f, (int, float)) and 0 < f <= 1 for f in frac_list):
            raise ValueError("All fractions must be numeric values between 0 and 1")
        
        # Validate and clean data
        clean_data = self._validate_input_data(data, x_col, y_col)
        
        # Segment the data
        segments = self._segment_data_by_location(clean_data, x_col, seg_locations)
        
        # Apply LOWESS to each segment
        result_segments = []
        
        for i, segment in enumerate(segments):
            segment_result = segment.copy()
            
            if len(segment) < 3:
                # Fill with NaN for insufficient data
                for frac in frac_list:
                    segment_result[f"{y_col}_lowess_{frac}"] = np.nan
                result_segments.append(segment_result)
                continue
            
            # Prepare data for LOWESS
            x_data = segment[x_col].values
            y_data = segment[y_col].values
            x_eval_segment = x_eval if x_eval is not None else x_data
            
            for frac in frac_list:
                try:
                    smoothed = sm.nonparametric.lowess(
                        endog=y_data,
                        exog=x_data,
                        frac=frac,
                        it=it,
                        xvals=x_eval_segment,
                        return_sorted=False,
                        **kwargs
                    )
                    segment_result[f"{y_col}_lowess_{frac}"] = smoothed
                except Exception as e:
                    segment_result[f"{y_col}_lowess_{frac}"] = np.nan
                    warnings.warn(f"LOWESS smoothing failed for segment {i}, frac={frac}: {e}")
            
            result_segments.append(segment_result)
        
        # Concatenate segments
        result = pd.concat(result_segments, ignore_index=True).sort_values(x_col)
        return result

    def smooth_gaussian_process(self, data: pd.DataFrame, x_col: str, y_col: str,
                               y_std_col: Optional[str] = None,
                               seg_locations: Optional[List[float]] = None,
                               x_eval: Optional[np.ndarray] = None,
                               # Kernel settings (reasonable defaults for SWOT WSE)
                               matern_nu: float = 2.5,
                               length_scale_init: Optional[float] = None,
                               length_scale_bounds: Tuple[float, float] = (50.0, 20000.0),
                               # Scale / noise settings
                               add_nugget_var: float = 0.0,
                               fit_white_noise: bool = True,
                               white_noise_bounds: Tuple[float, float] = (1e-8, 1e-2),
                               # Optimization settings
                               normalize_y: bool = True,
                               n_restarts_optimizer: int = 3,
                               random_state: int = 0,
                               **kwargs) -> pd.DataFrame:
        """
        Apply Gaussian Process Regression for smoothing using Matern kernel with heteroskedastic uncertainty.
        
        This method uses a sophisticated GPR approach that:
        - Handles heteroskedastic uncertainty (per-node std)
        - Uses Matern kernel for better smoothness control
        - Supports extrapolation with uncertainty quantification
        - Handles gaps (NaNs) in the data
        
        Args:
            data: Input DataFrame
            x_col: Name of x-coordinate column (e.g., distance)
            y_col: Name of y-coordinate column (e.g., WSE median)
            y_std_col: Column name containing standard deviation of y values (required for heteroskedastic GPR)
            seg_locations: List of segment locations for piecewise smoothing
            x_eval: Optional array of x values for evaluation
            matern_nu: Smoothness parameter. 1.5 or 2.5 are typical for hydraulic profiles
            length_scale_init: Initial length scale. If None, inferred from data span
            length_scale_bounds: Bounds for length scale during optimization (same units as x)
            add_nugget_var: Add this to each node variance (z_std^2) to avoid overconfidence
            fit_white_noise: Whether to include an additional learned white noise term
            white_noise_bounds: Bounds on learned white noise variance
            normalize_y: Whether to normalize y before fitting (usually helps)
            n_restarts_optimizer: How many optimizer restarts for kernel hyperparameters
            random_state: Reproducibility seed
            **kwargs: Additional arguments (ignored for compatibility)
            
        Returns:
            DataFrame with GP smoothed values and uncertainty estimates
            
        Raises:
            ValueError: If y_std_col is not provided or insufficient data points
        """
        if y_std_col is None:
            raise ValueError("y_std_col is required for heteroskedastic Gaussian Process Regression")
        
        # Validate input data columns
        required_cols = [x_col, y_col, y_std_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Clean data - remove NaN values and ensure std is non-negative
        clean_data = data.dropna(subset=required_cols).copy()
        clean_data = clean_data[clean_data[y_std_col] >= 0].sort_values(x_col)
        
        if len(clean_data) < 5:
            raise ValueError(f"Insufficient data points for GPR. Need at least 5, got {len(clean_data)}")
        
        # Segment the data if requested
        if seg_locations is None:
            seg_locations = []
        
        segments = self._segment_data_by_location(clean_data, x_col, seg_locations)
        result_segments = []
        
        for i, segment in enumerate(segments):
            segment_result = segment.copy()
            
            if len(segment) < 5:
                # Fill with NaN for insufficient data
                segment_result[f"{y_col}_gp_mean"] = np.nan
                segment_result[f"{y_col}_gp_std"] = np.nan
                segment_result[f"{y_col}_gp_std_latent"] = np.nan
                result_segments.append(segment_result)
                continue
            
            try:
                # Extract training data
                x_data = segment[x_col].values
                y_data = segment[y_col].values
                y_std_data = segment[y_std_col].values
                
                # Prepare prediction points
                if x_eval is not None:
                    # Filter x_eval to this segment's range if segmented
                    if seg_locations:
                        x_min, x_max = x_data.min(), x_data.max()
                        x_pred_segment = x_eval[(x_eval >= x_min) & (x_eval <= x_max)]
                    else:
                        x_pred_segment = x_eval
                else:
                    x_pred_segment = x_data
                
                # Sort by x
                order = np.argsort(x_data)
                x_data_sorted = x_data[order]
                y_data_sorted = y_data[order]
                y_std_data_sorted = y_std_data[order]
                
                # Convert to required format for GPR
                x_train = x_data_sorted[:, None]  # (n,1)
                y_train = y_data_sorted          # (n,)
                
                # Heteroskedastic noise: alpha = variance of each observation
                alpha = (y_std_data_sorted ** 2) + float(add_nugget_var)
                alpha = np.clip(alpha, 1e-12, None)
                
                x_pred_sorted = np.sort(x_pred_segment)
                X_pred = x_pred_sorted[:, None]
                
                # Heuristic for initial length scale
                if length_scale_init is None:
                    span = float(np.max(x_train) - np.min(x_train))
                    # If profile span is huge, start with ~5% of span; otherwise start at 1 km
                    length_scale_init_segment = max(0.05 * span, 1000.0)
                else:
                    length_scale_init_segment = length_scale_init
                
                # Build kernel: Constant * Matern + optional WhiteKernel
                kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                    length_scale=length_scale_init_segment,
                    length_scale_bounds=length_scale_bounds,
                    nu=matern_nu,
                )
                if fit_white_noise:
                    kernel += WhiteKernel(noise_level=1e-4, noise_level_bounds=white_noise_bounds)
                
                # Fit GPR
                gpr = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=alpha,
                    normalize_y=normalize_y,
                    n_restarts_optimizer=n_restarts_optimizer,
                    random_state=random_state,
                )
                gpr.fit(x_train, y_train)
                
                # Posterior predictive for y (includes observation noise)
                y_mean, y_std = gpr.predict(X_pred, return_std=True)
                
                # Latent function uncertainty (excluding observation noise)
                f_std = self._compute_latent_std(gpr, y_std)
                
                # Map predictions back to original segment indices
                if x_eval is not None:
                    # Create output arrays filled with NaN
                    pred_mean = np.full(len(segment), np.nan)
                    pred_std = np.full(len(segment), np.nan)
                    pred_std_latent = np.full(len(segment), np.nan)
                    
                    # Fill in predictions for points that were actually predicted
                    for j, x_val in enumerate(x_data):
                        idx = np.argmin(np.abs(x_pred_sorted - x_val))
                        if np.abs(x_pred_sorted[idx] - x_val) < 1e-10:  # Match found
                            pred_mean[j] = y_mean[idx]
                            pred_std[j] = y_std[idx]
                            pred_std_latent[j] = f_std[idx]
                else:
                    pred_mean = y_mean
                    pred_std = y_std
                    pred_std_latent = f_std
                
                segment_result[f"{y_col}_gp_mean"] = pred_mean
                segment_result[f"{y_col}_gp_std"] = pred_std
                segment_result[f"{y_col}_gp_std_latent"] = pred_std_latent
                
            except Exception as e:
                segment_result[f"{y_col}_gp_mean"] = np.nan
                segment_result[f"{y_col}_gp_std"] = np.nan
                segment_result[f"{y_col}_gp_std_latent"] = np.nan
                warnings.warn(f"GP smoothing failed for segment {i}: {e}")
            
            result_segments.append(segment_result)
        
        # Concatenate segments
        result = pd.concat(result_segments, ignore_index=True).sort_values(x_col)
        return result

    def interpolate(self, data: pd.DataFrame, x_col: str, y_col: str,
                   method: str = 'lowess',
                   seg_locations: Optional[List[float]] = None,
                   x_eval: Optional[np.ndarray] = None,
                   **method_params) -> pd.DataFrame:
        """
        General interpolation method that dispatches to specific smoothing methods.
        
        Args:
            data: Input DataFrame
            x_col: Name of x-coordinate column
            y_col: Name of y-coordinate column
            method: Interpolation method ('lowess', 'gaussian_process', 'moving_average', 'polynomial')
            seg_locations: List of segment locations for piecewise interpolation
            x_eval: Optional array of x values for evaluation
            **method_params: Method-specific parameters
            
        Returns:
            DataFrame with interpolated values
            
        Raises:
            ValueError: If method is not supported
        """
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Method '{method}' not supported. Available methods: {SUPPORTED_METHODS}")
        
        method_map = {
            'lowess': self.smooth_lowess,
            'gaussian_process': self.smooth_gaussian_process,
        }
        
        return method_map[method](
            data=data,
            x_col=x_col,
            y_col=y_col,
            seg_locations=seg_locations,
            x_eval=x_eval,
            **method_params
        )
