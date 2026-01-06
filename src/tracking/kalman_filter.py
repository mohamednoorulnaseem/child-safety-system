"""
Kalman Filter for tracking object positions
Used in DeepSORT for motion prediction
"""
import numpy as np
from typing import Tuple


class KalmanFilter:
    """
    Kalman filter for tracking 2D bounding boxes.
    State vector: [x, y, a, h, vx, vy, va, vh]
    where (x,y) is center, a is aspect ratio, h is height
    """
    
    def __init__(self):
        """Initialize Kalman filter matrices."""
        ndim, dt = 4, 1.
        
        # State transition matrix (8x8)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        
        # Measurement matrix (4x8)
        self._update_mat = np.eye(ndim, 2 * ndim)
        
        # Motion and observation uncertainty
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
    
    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize track from first detection.
        
        Args:
            measurement: Bounding box [x, y, a, h] where (x,y) is center,
                        a is aspect ratio, h is height
        
        Returns:
            Tuple of (mean, covariance) for the new track
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        
        covariance = np.diag(np.square(std))
        
        return mean, covariance
    
    def predict(self, mean: np.ndarray, 
                covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next state using motion model.
        
        Args:
            mean: Current state mean vector
            covariance: Current state covariance matrix
            
        Returns:
            Predicted (mean, covariance)
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        
        return mean, covariance
    
    def project(self, mean: np.ndarray, 
                covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project state into measurement space.
        
        Args:
            mean: State mean vector
            covariance: State covariance matrix
            
        Returns:
            Projected (mean, covariance) in measurement space
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        
        innovation_cov = np.diag(np.square(std))
        
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        
        return mean, covariance + innovation_cov
    
    def update(self, mean: np.ndarray, covariance: np.ndarray,
               measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update state estimate with new measurement.
        
        Args:
            mean: Predicted state mean
            covariance: Predicted state covariance
            measurement: New measurement [x, y, a, h]
            
        Returns:
            Updated (mean, covariance)
        """
        projected_mean, projected_cov = self.project(mean, covariance)
        
        # Kalman gain
        chol_factor = np.linalg.cholesky(projected_cov)
        kalman_gain = np.linalg.multi_dot((
            covariance, self._update_mat.T,
            np.linalg.inv(np.linalg.multi_dot((
                chol_factor, chol_factor.T)))
        ))
        
        # Innovation (measurement residual)
        innovation = measurement - projected_mean
        
        # Update state
        new_mean = mean + np.dot(kalman_gain, innovation)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        
        return new_mean, new_covariance
    
    def gating_distance(self, mean: np.ndarray, covariance: np.ndarray,
                       measurements: np.ndarray) -> np.ndarray:
        """
        Calculate gating distance (Mahalanobis distance).
        
        Args:
            mean: State mean
            covariance: State covariance
            measurements: Array of measurements (N x 4)
            
        Returns:
            Array of gating distances
        """
        mean, covariance = self.project(mean, covariance)
        
        d = measurements - mean
        
        cholesky_factor = np.linalg.cholesky(covariance)
        z = np.linalg.solve(cholesky_factor, d.T).T
        
        squared_maha = np.sum(z * z, axis=1)
        
        return squared_maha
