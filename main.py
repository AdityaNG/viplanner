"""
python3 main.py --video assets/kora_walk.mini.mp4.webm
"""
import os
import sys
import cv2
import torch
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from typing import Tuple, List
from trajectory_plot import plot_trajectories, estimate_intrinsics, overlay_image_by_semantics
from vlm_navigation import VLMNavigator

planner_path = os.path.join(
    os.path.dirname(os.path.basename(__file__)),
    "ros/planner/src/",
)
sys.path.append(
    planner_path
)

from vip_inference import VIPlannerInference
from perception_sem_depth import SemDepthPerception

class VIPlannerDemo:
    def __init__(self):
        # Set up command line args
        parser = ArgumentParser()
        parser.add_argument("--video", type=str, required=True, help="Path to input video file")
        parser.add_argument("--model_dir", type=str, default="ckpts",
                          help="Directory containing model.pt and model.yaml")
        args = parser.parse_args()

        self.vlm = VLMNavigator()
        
        # Initialize planner config
        class PlannerConfig:
            def __init__(self, model_dir, m2f_config):
                self.model_save = model_dir
                self.m2f_config_path = m2f_config
                
        cfg = PlannerConfig(args.model_dir, "")
        
        # Initialize VIPlanner
        self.planner = VIPlannerInference(cfg)
        self.perception = SemDepthPerception()

        self.vlm_freq = 15
        
        # Open video file
        self.video = cv2.VideoCapture(args.video)
        if not self.video.isOpened():
            raise ValueError(f"Could not open video file: {args.video}")
        self.last_goal_list = [
            [5.0, 0.0, 0.0],
        ]
            
    @torch.no_grad()
    def run(
        self,
    ):
        # Set dummy goal in robot frame (x,y,z)
        # TODO: set confidence threshold to stop
        # TODO: Use VLM to decide destination region
        # TODO: integrate GPS data with VLM

        out = None
        output_path = "assets/vis.mp4"
        fps = int(self.video.get(cv2.CAP_PROP_FPS))

        ret, frame = self.video.read()
        H, W, C = frame.shape

        offsets = (0, 1.8, 0.0)
        intrinsic_matrix = estimate_intrinsics(
            fov_x=100, fov_y=100, height=H, width=W,
        )
        # 4x4 matrix
        extrinsic_matrix = np.array(
            [
                [1, 0, 0, offsets[0]],
                [0, 1, 0, offsets[1]],
                [0, 0, 1, offsets[2]],
                [0, 0, 0, 1],
            ]
        )
        
        index = 0
        try:
            while True:
                ret, frame = self.video.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                frame_pil = Image.fromarray(frame_rgb)
                
                perception_outputs = self.perception.infer(frame_pil)
                depth_viz = perception_outputs['depth_rgb'] 
                sem_viz = perception_outputs['semantics_rgb']
                # Get semantic segmentation
                semantic_mask = perception_outputs['semantics_rgb']
                
                # Create depth image
                depth = perception_outputs['depth']
                
                if index % self.vlm_freq == 0:
                    vlm_goal, thought_process = self.vlm.infer(
                        frame,
                        intrinsic_matrix,
                        extrinsic_matrix,
                        sem_viz,
                    )
                # vlm_goal, thought_process = (0, 0, 4), ""
                

                trajectory, fear = None, None

                print('vlm_goal', vlm_goal)
                vlm_goal_transformed = [
                    vlm_goal[2], vlm_goal[0], vlm_goal[1]
                ]
                print('vlm_goal_transformed', vlm_goal_transformed)

                goal_list = [vlm_goal_transformed,]

                for goal_i in goal_list:
                    goal = torch.tensor([
                        goal_i
                    ], dtype=torch.float32)
                    # Run planner
                    if self.planner.train_cfg.sem or self.planner.train_cfg.rgb:
                        trajectory_i, fear_i = self.planner.plan(depth, semantic_mask, goal)
                    else:
                        trajectory_i, fear_i = self.planner.plan_depth(depth, goal)
                    
                    if fear is None:
                        trajectory, fear = trajectory_i, fear_i
                    elif fear_i < fear:
                        trajectory, fear = trajectory_i, fear_i

                    
                # print(f"Trajectory shape: {trajectory.shape}")
                # print(f"Fear value: {fear}")
                
                # Visualize results
                vis_frame = frame.copy()
                
                # Overlay semantic mask
                overlay = cv2.addWeighted(frame, 0.7, semantic_mask, 0.3, 0)

                # trajectory_cam = trajectory[:,[0,2,1]]
                trajectory_cam = trajectory[:,[1,2,0]]
                trajectory_cam[:, 0] *= -1
                trajectory_cam[:, 1] *= -1
                print("model_height", trajectory_cam[:, 1].min(), trajectory_cam[:, 1].max())
                trajectory_cam[:, 1] = 0.0
                print('trajectory_cam', trajectory_cam[::4,:])
                plot_trajectories(
                    frame_img=vis_frame,
                    trajectories=[trajectory_cam],
                    intrinsic_matrix=intrinsic_matrix,
                    extrinsic_matrix=extrinsic_matrix,
                    line=True,
                    track=False,
                    draw_grid=True,
                    interpolation_samples=0,
                    grid_range_img=(2,8),
                )

                vis_frame = overlay_image_by_semantics(
                    vis_frame,
                    frame,
                    sem_viz,
                    [
                        (0, 255, 0),
                        (255, 128, 0),
                        # (0, 0, 255,),
                    ],
                )
                
                # Add fear value text to visualization
                fear_text = f"Fear: {float(fear):.2f} : {thought_process}"
                cv2.putText(vis_frame, fear_text, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                bottom = np.hstack((depth_viz, sem_viz))
                bottom = cv2.resize(bottom, (0,0), fx=0.5, fy=0.5)
                vis_frame = np.vstack((vis_frame, bottom))

                # vis_frame = cv2.resize(vis_frame, (0,0), fx=0.25, fy=0.25)

                # Display frames
                cv2.imwrite("assets/vis.png", vis_frame)

                if out is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (vis_frame.shape[1], vis_frame.shape[0]))
                out.write(vis_frame)
                index += 1
            
        except:
            import traceback
            traceback.print_exc()
        self.video.release()


if __name__ == "__main__":
    demo = VIPlannerDemo()
    demo.run()
