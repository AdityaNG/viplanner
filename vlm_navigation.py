import instructor
import os
from typing import Tuple
import numpy as np
from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel
import base64
import json
import cv2

from trajectory_plot import plot_trajectories, overlay_image_by_semantics

PROMPT = """
Analyze the street scene image and provide navigation guidance by:

1. Identifying safe walking paths that maintain at least 1 meters distance from obstacles:
- Prioritize sidewalks and designated pedestrian areas
- Note any obstructions like poles, parked vehicles, or street furniture
- Use the concentric circle distance markers (2m to {max_length}m concentric circles) to estimate safe clearance
- Use the angle markers (-{max_angle}, 0.0, +{max_angle}), there are three lines, each at varying angles to give you a sense of geometry and depth
- Note that the green markers highlight the traversable area

2. Evaluate path options based on:
- Surface quality and evenness
- Width of available path
- Presence of other pedestrians or vehicles
- Distance from road traffic
- Connectivity to crosswalks and destinations
- You are allowed to walk on the road if a footpath is unavailable

3. Recommend the optimal path by:
- Highlighting the safest route with maximum clearance
- Identifying key decision points where direction changes are needed

4. Provide specific navigation instructions using:
- Distance estimates based on the provided markers
- Clear transition points between different path segments

Finally, select a point a point in the following format.
Circle: Distance in Meters
Angle: Angle in degrees

For example:
```json
    "circle": 2.5,
    "angle": 10.0,
```
The above selects a distance 2.5 m and 10 degrees from the straight line to the right

```json
    "circle": 3.0,
    "angle": -12.0,
```
The above selects a distance 3.0 m and -10 degrees from the straight line to the left

```json
    "circle": 0.0,
    "angle": 0.0,
```
The above selects a distance 0.0 m and 0 degrees.
This can be used as the STOP command in case it is not possible to proceed forward.

You should provide a 3 word string comment explaining your thoughtprocess for the frame.
Select from one of the following:
 - Clear Path Ahead
 - Stopping For Obstacle
 - Turn Deviation Required
 - Cannot Proceed Safely


Previous command: {previous_command}
"""

class NavigationOutput(BaseModel):
    circle: float 
    angle: float
    thought_process_comment: str

class VLMNavigator:
    def __init__(self):

        # self.llm = "anthropic"
        self.llm = "openai"

        if self.llm == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Please set ANTHROPIC_API_KEY environment variable")
                
            # Use instructor to patch Anthropic client
            self.client = instructor.from_anthropic(Anthropic(api_key=api_key))
            self.model = "claude-3-opus-20240229"
        elif self.llm == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Please set OPENAI_API_KEY environment variable")
                
            # Use instructor to patch Anthropic client
            self.client = instructor.from_openai(OpenAI(api_key=api_key))
            # self.model = "gpt-4o-mini"
            self.model = "gpt-4o"

        self.max_length = 7
        self.max_angle = 30
        self.previous_command = NavigationOutput(
            circle = 0.0,
            angle = 0.0,
            thought_process_comment = "Just Booted Up",
        )

    def infer(
        self,
        frame: np.ndarray,
        intrinsic_matrix: np.ndarray, 
        extrinsic_matrix: np.ndarray,
        sem_viz: np.ndarray,
    ) -> Tuple[Tuple[float, float, float], str]:

        prompt = PROMPT.format(
            max_length=self.max_length,
            max_angle=self.max_angle,
            previous_command=self.previous_command.model_dump(mode='json')
        )

        H, W, C = frame.shape
        image = frame.copy()

        trajectories = generate_trajectories(
            angle_deg=self.max_angle,
            max_len=self.max_length,
        )

        plot_trajectories(
            frame_img=image,
            trajectories=trajectories,
            colors = [(0, 255, 0),] * len(trajectories),
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
            line=True,
            track=False,
            draw_grid=True,
            interpolation_samples=0,
            grid_range_img=(2,self.max_length+1),
        )

        image = overlay_image_by_semantics(
            image,
            frame,
            sem_viz,
            [
                (0, 255, 0),
                (255, 128, 0),
                # (0, 0, 255,),
            ],
        )

        # Save image for Claude
        cv2.imwrite("assets/vis_lvm.png", image)

        # Read image as base64
        with open("assets/vis_lvm.png", "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode()

        if self.llm == "anthropic":
            image_data = {
                "type": "image", 
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_base64
                }
            }
        elif self.llm == "openai":
            image_data = {
                "type": "image_url",
                "image_url": {
                    "url":  f"data:image/jpeg;base64,{image_base64}"
                },
            }

        # First, get the navigation description
        nav_output = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        image_data,
                    ]
                }
            ],
            response_model=NavigationOutput,
        )
        thought_process_comment = nav_output.thought_process_comment

        self.previous_command = nav_output
        
        
        x = nav_output.circle * np.sin(np.radians(nav_output.angle))
        y = 0.0
        z = nav_output.circle * np.cos(np.radians(nav_output.angle))
        return (x, y, z), thought_process_comment


def generate_trajectories(angle_deg=10, max_len=7):
    # Generate straight trajectory (0 degrees)
    trajectory_cam_0 = np.array([[0, 0, i] for i in range(max_len + 1)])
    
    # Generate left angled trajectory (-angle_deg)
    points_left = []
    for i in range(max_len + 1):
        x = i * np.sin(np.radians(-angle_deg))
        y = 0.0
        z = i * np.cos(np.radians(-angle_deg))
        points_left.append([x, y, z])
    trajectory_cam_left = np.array(points_left)

    # Generate right angled trajectory (+angle_deg) 
    points_right = []
    for i in range(max_len + 1):
        x = i * np.sin(np.radians(angle_deg))
        y = 0.0
        z = i * np.cos(np.radians(angle_deg))
        points_right.append([x, y, z])
    trajectory_cam_right = np.array(points_right)

    return trajectory_cam_0, trajectory_cam_left, trajectory_cam_right