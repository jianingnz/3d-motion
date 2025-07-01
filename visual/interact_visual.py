import time
import argparse
from pathlib import Path

import numpy as np
import viser


def main(npz_path: Path, share: bool, framerate: float = 10.0) -> None:
    server = viser.ViserServer()
    if share:
        server.request_share_url()

    # Load point cloud data
    data = np.load(npz_path)
    pointclouds = data["tracks_XYZ"]        # Shape: (T, N, 3)
    visibility = data["visibility"]         # Shape: (T, N)

    T = pointclouds.shape[0]
    N = pointclouds.shape[1]

    server.scene.set_up_direction('-y')
    server.scene.add_frame("/origin", show_axes=True)

    # GUI Setup
    with server.gui.add_folder("Playback"):
        gui_point_size = server.gui.add_slider("Point size", min=0.0001, max=0.003, step=0.0001, initial_value=0.001)
        gui_timestep = server.gui.add_slider("Timestep", min=0, max=T - 1, step=1, initial_value=0, disabled=True)
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider("FPS", min=1, max=60, step=0.1, initial_value=framerate)
        gui_framerate_options = server.gui.add_button_group("FPS options", ("10", "20", "30", "60"))
        gui_show_occluded = server.gui.add_checkbox("Show Occluded Points", False)

    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % T

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % T

    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Point cloud handles for each frame
    point_nodes = []
    for i in range(T):
        visible_mask = visibility[i] | gui_show_occluded.value
        visible_points = pointclouds[i][visible_mask]

        pc = server.scene.add_point_cloud(
            name=f"/frame_{i}",
            points=visible_points,
            colors=np.tile([[0.1, 0.6, 1.0]], (visible_points.shape[0], 1)),  # Light blue
            point_size=0.01,
        )
        pc.visible = (i == gui_timestep.value)
        point_nodes.append(pc)

    # Timestep update callback
    prev_timestep = gui_timestep.value

    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        point_nodes[prev_timestep].visible = False
        point_nodes[gui_timestep.value].visible = True
        prev_timestep = gui_timestep.value
        server.flush()

    print("Viewer ready.")

    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % T

        # Optional: live update point size
        point_nodes[gui_timestep.value].point_size = gui_point_size.value

        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize per-frame point clouds with visibility mask.")
    parser.add_argument("--npz", type=str, required=True, help="Path to .npz file with tracks_XYZ and visibility")
    parser.add_argument("--share", action="store_true", help="Enable sharing via Viser's share URL")
    args = parser.parse_args()

    main(
        npz_path=Path(args.npz),
        share=args.share,
    )
