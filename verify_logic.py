import numpy as np
import math

def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

def test_reverse_logic(n_dirs=16):
    def _dir_to_vec(dir_idx):
        angle = 2 * np.pi * dir_idx / n_dirs
        return np.array([np.cos(angle), np.sin(angle)])

    def _vec_to_dir(vec):
        angle = np.arctan2(vec[1], vec[0]) % (2 * np.pi)
        return int(round(angle / (2 * np.pi) * n_dirs)) % n_dirs

    # Test 1: Heading North (90 deg) -> Reverse should be South (270 deg)
    heading_vec = np.array([0, 1]) # North
    rev_vec = -heading_vec # South
    rev_dir = _vec_to_dir(rev_vec)
    
    print(f"Heading North: {heading_vec} -> Target Rev: {rev_vec} -> Rev Dir Index: {rev_dir}")
    assert rev_dir == 12, f"Expected 12 for South in 16-dir system, got {rev_dir}"

    print("Reverse logic test passed!")

def test_quaternion():
    # 0,0,0 -> 0,0,0,1
    q = euler_to_quaternion(0,0,0)
    print(f"0,0,0 -> {q}")
    assert np.allclose(q, [0,0,0,1]), "Quaternion 0 error"
    
    # 0,0,PI/2 -> [0, 0, 0.7071, 0.7071]
    q = euler_to_quaternion(0,0,np.pi/2)
    print(f"0,0,PI/2 -> {q}")
    assert np.allclose(q, [0,0,0.70710678, 0.70710678]), "Quaternion PI/2 error"
    
    print("Quaternion test passed!")

if __name__ == "__main__":
    try:
        test_reverse_logic()
        test_quaternion()
        print("\nALL STANDALONE VERIFICATIONS PASSED")
    except Exception as e:
        print(f"\nVERIFICATION FAILED: {e}")
