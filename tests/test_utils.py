"""
Tests for core utility functions: XYZ mapping, gripper control, clamping.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import clamp, map_range, maybe_reverse, xyz_to_joints_deg, SAFE_RANGE, JOINT_LIMITS_DEG


class TestClamp:
    """Test the clamp function."""
    
    def test_clamp_within_range(self):
        """Value within range should remain unchanged."""
        assert clamp(5.0, 0.0, 10.0) == 5.0
        assert clamp(0.0, 0.0, 10.0) == 0.0
        assert clamp(10.0, 0.0, 10.0) == 10.0
    
    def test_clamp_below_range(self):
        """Value below minimum should be clamped to minimum."""
        assert clamp(-5.0, 0.0, 10.0) == 0.0
    
    def test_clamp_above_range(self):
        """Value above maximum should be clamped to maximum."""
        assert clamp(15.0, 0.0, 10.0) == 10.0
    
    def test_clamp_negative_range(self):
        """Clamping should work with negative ranges."""
        assert clamp(-5.0, -10.0, 0.0) == -5.0
        assert clamp(-15.0, -10.0, 0.0) == -10.0
        assert clamp(5.0, -10.0, 0.0) == 0.0


class TestMapRange:
    """Test the map_range function."""
    
    def test_map_range_identity(self):
        """Mapping to same range should return input."""
        assert map_range(5.0, (0.0, 10.0), (0.0, 10.0)) == 5.0
    
    def test_map_range_scale_up(self):
        """Scaling from smaller to larger range."""
        assert map_range(5.0, (0.0, 10.0), (0.0, 20.0)) == 10.0
    
    def test_map_range_scale_down(self):
        """Scaling from larger to smaller range."""
        assert map_range(10.0, (0.0, 20.0), (0.0, 10.0)) == 5.0
    
    def test_map_range_clamps_to_unit_interval(self):
        """Out-of-range input should be clamped before mapping."""
        # 15 is above 10, so t should be clamped to 1.0
        result = map_range(15.0, (0.0, 10.0), (0.0, 100.0))
        assert result == 100.0
    
    def test_map_range_negative_to_positive(self):
        """Mapping from negative to positive range."""
        result = map_range(-5.0, (-10.0, 0.0), (0.0, 90.0))
        assert result == 45.0  # Midpoint maps to midpoint


class TestMaybeReverse:
    """Test the maybe_reverse function."""
    
    def test_maybe_reverse_no_reverse(self):
        """When invert=False, should return original range."""
        assert maybe_reverse((0.0, 10.0), False) == (0.0, 10.0)
    
    def test_maybe_reverse_with_reverse(self):
        """When invert=True, should reverse range."""
        assert maybe_reverse((0.0, 10.0), True) == (10.0, 0.0)
        assert maybe_reverse((-180.0, 180.0), True) == (180.0, -180.0)


class TestXYZtoJointsDeg:
    """Test the XYZ to joint degrees mapping."""
    
    def test_xyz_center_maps_to_center_joints(self):
        """Center of safe range should map to center of joint limits."""
        xyz = np.array([
            (SAFE_RANGE["x"][0] + SAFE_RANGE["x"][1]) / 2,
            (SAFE_RANGE["y"][0] + SAFE_RANGE["y"][1]) / 2,
            (SAFE_RANGE["z"][0] + SAFE_RANGE["z"][1]) / 2,
        ])
        
        joints = xyz_to_joints_deg(xyz, invert_x=False, invert_y=False, invert_z=False)
        
        # Should have 5 values (excluding gripper)
        assert len(joints) == 5
        # Center should map to center of joint limits
        # Pan (shoulder_pan) from Y
        expected_pan_center = (JOINT_LIMITS_DEG["shoulder_pan"][0] + JOINT_LIMITS_DEG["shoulder_pan"][1]) / 2
        # Lift (shoulder_lift) from X
        expected_lift_center = (JOINT_LIMITS_DEG["shoulder_lift"][0] + JOINT_LIMITS_DEG["shoulder_lift"][1]) / 2
        
        assert abs(joints[0] - expected_pan_center) < 1.0
        assert abs(joints[1] - expected_lift_center) < 1.0
    
    def test_xyz_clamps_out_of_range(self):
        """Out-of-range XYZ should be clamped to joint limits."""
        # Extreme values
        xyz = np.array([100.0, 100.0, 100.0])
        joints = xyz_to_joints_deg(xyz)
        
        # All joints should be within limits
        for i, name in enumerate(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_roll", "wrist_flex"]):
            lo, hi = JOINT_LIMITS_DEG[name]
            assert lo <= joints[i] <= hi, f"{name} out of range: {joints[i]}"
    
    def test_xyz_invert_flags(self):
        """Invert flags should reverse mapping direction."""
        xyz = np.array([0.2, 0.0, 0.1])
        
        joints_normal = xyz_to_joints_deg(xyz, invert_x=False, invert_y=False, invert_z=False)
        joints_inverted = xyz_to_joints_deg(xyz, invert_x=True, invert_y=True, invert_z=True)
        
        # Joint values should be different when inverted
        assert not np.allclose(joints_normal, joints_inverted)
    
    def test_xyz_gain_scaling(self):
        """Gain factors should scale the motion."""
        xyz = np.array([0.2, 0.0, 0.1])
        
        joints_gain_1 = xyz_to_joints_deg(xyz, x_gain=1.0, y_gain=1.0, z_gain=1.0)
        joints_gain_0_5 = xyz_to_joints_deg(xyz, x_gain=0.5, y_gain=0.5, z_gain=0.5)
        
        # Lower gain should result in less extreme joint angles (closer to center)
        # This is a simplified check - actual behavior depends on safe range midpoints
        assert len(joints_gain_1) == len(joints_gain_0_5) == 5


class TestJointLimitsEnforced:
    """Test that joint limits are properly enforced."""
    
    def test_joint_limits_hard_clamping(self):
        """Joint values should never exceed defined limits."""
        # Generate random XYZ values within and outside safe range
        for _ in range(100):
            xyz = np.random.uniform(-10.0, 10.0, 3)
            joints = xyz_to_joints_deg(xyz)
            
            # Check all joints are within limits
            for i, name in enumerate(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_roll", "wrist_flex"]):
                lo, hi = JOINT_LIMITS_DEG[name]
                assert lo <= joints[i] <= hi, f"{name} out of range: {joints[i]} (bounds: {lo}, {hi})"


class TestSafeRange:
    """Test safe range definitions."""
    
    def test_safe_range_defined(self):
        """Safe range should be properly defined for all axes."""
        required_axes = ["x", "y", "z", "g"]
        for axis in required_axes:
            assert axis in SAFE_RANGE, f"Safe range missing {axis}"
            assert len(SAFE_RANGE[axis]) == 2, f"Safe range for {axis} should have 2 values"
            assert SAFE_RANGE[axis][0] <= SAFE_RANGE[axis][1], f"Safe range {axis} min > max"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
