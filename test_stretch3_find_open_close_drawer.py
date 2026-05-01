#!/usr/bin/env python3
"""Test Stretch3 running the FindOpenCloseDrawer robocasa task."""

import sys
import os

# Ensure local robosuite/robocasa are on the path
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "third_party", "robosuite"))
sys.path.insert(0, os.path.join(_here, "third_party", "robocasa"))

import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config

import robocasa  # registers tasks


STRETCH3_CONTROLLER_CONFIG = {
    "type": "BASIC",
    "body_parts": {
        "right": {
            "type": "JOINT_POSITION",
            "input_max": 1,
            "input_min": -1,
            "output_max": 0.5,
            "output_min": -0.5,
            "kp": 300,
            "kd": 50,
            "kv": 200,
            "velocity_limits": [-1, 1],
            "kp_limits": [0, 1000],
            "interpolation": None,
            "ramp_ratio": 0.2,
            "gripper": {
                "type": "GRIP",
            },
        },
        "base": {
            "type": "JOINT_VELOCITY",
            "interpolation": None,
        },
        "head": {
            "type": "JOINT_POSITION",
            "input_max": 1,
            "input_min": -1,
            "output_max": 0.5,
            "output_min": -0.5,
            "kp": 300,
            "kd": 50,
            "kv": 200,
            "velocity_limits": [-1, 1],
            "kp_limits": [0, 1000],
            "interpolation": None,
            "ramp_ratio": 0.2,
        },
    },
}


def main():
    print("Creating FindOpenCloseDrawer environment with Stretch3...")

    controller_configs = STRETCH3_CONTROLLER_CONFIG

    env = robosuite.make(
        "FindOpenCloseDrawer",
        robots="Stretch3",
        controller_configs=controller_configs,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=None,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer="mjviewer",
    )

    print("Environment created successfully.")
    print(f"  Action dim: {env.action_spec[0].shape}")
    print(f"  Robot joints: {env.robots[0].robot_model.joints}")

    obs = env.reset()
    print(f"  Obs keys: {sorted(obs.keys())}")
    print("Running render loop (close viewer window to exit)...")

    try:
        for step in range(10000):
            action = np.zeros(env.action_spec[0].shape)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                print(f"  Episode done at step {step}, resetting...")
                obs = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        print("Done.")


if __name__ == "__main__":
    main()
