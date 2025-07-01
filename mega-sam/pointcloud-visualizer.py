import time
from pathlib import Path

import numpy as np
import tyro
from tqdm.auto import tqdm

import viser
import viser.transforms as tf


class MegaSamLoader:
    def __init__(self, data_path: Path):
        # Read metadata.
        data = np.load(data_path)

        self.K: np.ndarray = data["intrinsic"]
        self.K_inv = np.linalg.inv(self.K)
        self.rgbd = np.concatenate([data["images"].astype(np.float16) / 255, data["depths"][..., None]], axis=-1)
        self.cam_c2w = data["cam_c2w"]

    @property
    def num_frames(self) -> int:
        return self.rgbd.shape[0]

    @property
    def height(self) -> int:
        return self.rgbd.shape[1]

    @property
    def width(self) -> int:
        return self.rgbd.shape[2]

    def get_points(self, index: int, downsample_factor: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Get the world coordinates point cloud for a given frame."""
        rgbd = self.rgbd[index]
        cam_c2w = self.cam_c2w[index]
        K_inv = self.K_inv
        h,w = rgbd.shape[:2]
        depth = rgbd[..., 3]
        img_coords = np.concatenate([np.mgrid[:w, :h].transpose((2, 1, 0)).astype(np.float16), depth[..., None]], axis=-1) # H,W,3
        if downsample_factor > 1:
            img_coords = img_coords[::downsample_factor, ::downsample_factor]
            rgbd = rgbd[::downsample_factor, ::downsample_factor]
        cam_coords = img_coords @ K_inv.T
        cam_coords = np.concatenate([cam_coords, np.ones((*img_coords.shape[:2], 1), dtype=np.float16)], axis=-1) # H,W,4
        cam_coords = cam_coords @ cam_c2w.T
        cam_coords = cam_coords[..., :3]  / cam_coords[..., 3:]
        cam_coords = cam_coords.reshape(-1, 3)        
        rgb = rgbd[..., :3].reshape(-1, 3)
        return cam_coords, rgb

def main(
    data_path: Path = Path(__file__).parent / "outputs_cvd/drivetrack_sgd_cvd_hr.npz",
    downsample_factor: int = 1,
    max_frames: int = 300,
    share: bool = False,
) -> None:
    server = viser.ViserServer()
    if share:
        server.request_share_url()

    print("Loading frames!")
    loader = MegaSamLoader(data_path)
    num_frames = min(max_frames, loader.num_frames)

    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_point_size = server.gui.add_slider(
            "Point size",
            min=0.0001,
            max=0.003,
            step=0.0001,
            initial_value=0.001,
        )
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=10
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
        )

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    prev_timestep = gui_timestep.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        with server.atomic():
            # Toggle visibility.
            frame_nodes[current_timestep].visible = True
            frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep
        server.flush()  # Optional!


    server.scene.set_up_direction('-y')

    # Load in frames.
    server.scene.add_frame(
        "/frames",
        show_axes=False,
    )

    frame_nodes: list[viser.FrameHandle] = []
    point_nodes: list[viser.PointCloudHandle] = []
    for i in tqdm(range(num_frames)):
        position, color = loader.get_points(i, downsample_factor)

        # Add base frame.
        frame_nodes.append(server.scene.add_frame(f"/frames/t{i}", show_axes=False))

        # Place the point cloud in the frame.
        point_nodes.append(
            server.scene.add_point_cloud(
                name=f"/frames/t{i}/point_cloud",
                points=position,
                colors=color,
                point_size=gui_point_size.value,
                point_shape="rounded",
            )
        )

        # Place the frustum.
        fov = 2 * np.arctan2(loader.height / 2, loader.K[0, 0])
        aspect = loader.width / loader.height
        server.scene.add_camera_frustum(
            f"/frames/t{i}/frustum",
            fov=fov,
            aspect=aspect,
            scale=0.05,
            image=loader.rgbd[i, ::downsample_factor, ::downsample_factor, :3],
            wxyz=tf.SO3.from_matrix(loader.cam_c2w[i, :3, :3]).wxyz,
            position=loader.cam_c2w[i, :3, 3],
        )

        # Add some axes.
        server.scene.add_frame(
            f"/frames/t{i}/frustum/axes",
            axes_length=0.05,
            axes_radius=0.005,
        )

    # Hide all but the current frame.
    for i, frame_node in enumerate(frame_nodes):
        frame_node.visible = i == gui_timestep.value

    # Playback update loop.
    prev_timestep = gui_timestep.value
    while True:
        # Update the timestep if we're playing.
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        # Update point size of both this timestep and the next one! There's
        # redundancy here, but this will be optimized out internally by viser.
        #
        # We update the point size for the next timestep so that it will be
        # immediately available when we toggle the visibility.
        point_nodes[gui_timestep.value].point_size = gui_point_size.value
        point_nodes[
            (gui_timestep.value + 1) % num_frames
        ].point_size = gui_point_size.value

        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    tyro.cli(main)