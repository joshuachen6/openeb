# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Event to Video Demo Script
"""

import time
import sys
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from metavision_sdk_stream import Camera
from metavision_core_ml.event_to_video.lightning_model import EventToVideoLightningModel
from metavision_core_ml.preprocessing.event_to_tensor_torch import (
    event_cd_to_torch,
    event_volume,
)
from metavision_core_ml.utils.torch_ops import normalize_tiles, viz_flow
from metavision_core_ml.utils.show_or_write import ShowWrite
from metavision_core.event_io.raw_reader import RawReader


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default="", help="path of events")
    parser.add_argument(
        "checkpoint", type=str, default="", help="checkpoint to evaluate"
    )
    parser.add_argument("--start_ts", type=int, default=0, help="start timestamp")
    parser.add_argument(
        "--delta_t", type=int, default=10000, help="accumulate by delta_t"
    )
    parser.add_argument("--video_path", type=str, default="", help="path to video")
    parser.add_argument(
        "--height_width", nargs=2, default=None, type=int, help="resolution"
    )
    parser.add_argument(
        "--max_duration", type=int, default=-1, help="run for this duration"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="if true use cpu and not cuda"
    )
    parser.add_argument(
        "--flow", action="store_true", help="if true predict also optical flow"
    )
    parser.add_argument("--viz_input", action="store_true", help="if true viz input")
    parser.add_argument("--no_window", action="store_true", help="disable window")

    params, _ = parser.parse_known_args(argv)
    return params


def run(params):
    print("params: ", params)

    window_name = "e2v"
    if params.no_window:
        window_name = None
    show_write = ShowWrite(window_name, params.video_path)

    # Get the camera
    camera = Camera.from_first_available()

    # Setup the size
    height, width = camera.height(), camera.width()
    print("original size: ", height, width)

    device = "cpu" if params.cpu else "cuda"
    model = EventToVideoLightningModel.load_from_checkpoint(params.checkpoint)
    model.eval().to(device)
    nbins = model.hparams.event_volume_depth
    print("Nbins: ", nbins)
    in_height, in_width = (
        (height, width) if params.height_width is None else params.height_width
    )
    print("height_width: ", params.height_width)

    pause = False
    start = time.time()

    raw_stream = RawReader.from_device(device=camera.get_device(), max_events=int(1e8))

    # Read loop
    while not raw_stream.is_done():
        # Read the events that we want
        start = time.time()

        # read all the stuff
        events = raw_stream.load_delta_t(params.delta_t)

        if events.size > 0:
            first_ts = events["t"][0]
            if first_ts <= params.start_ts:
                continue
            last_ts = events["t"][-1]
            if (
                params.max_duration > 0
                and last_ts > params.start_ts + params.max_duration
            ):
                sys.exit()

        if not pause and not len(events):
            continue

        if not pause:
            events_th = event_cd_to_torch(events).to(device)
            start_times = (
                torch.FloatTensor([events["t"][0]])
                .view(
                    1,
                )
                .to(device)
            )
            durations = (
                torch.FloatTensor([events["t"][-1] - events["t"][0]])
                .view(
                    1,
                )
                .to(device)
            )

            try:
                tensor_th = event_volume(
                    events_th,
                    1,
                    height,
                    width,
                    start_times,
                    durations,
                    nbins,
                    "bilinear",
                )
            except RuntimeError:
                continue

            tensor_th = F.interpolate(
                tensor_th,
                size=(in_height, in_width),
                mode="bilinear",
                align_corners=True,
            )
            tensor_th = tensor_th.view(1, 1, nbins, in_height, in_width)
        else:
            tensor_th = torch.zeros(
                (1, 1, nbins, in_height, in_width), dtype=torch.float32, device=device
            )

        state = model.model(tensor_th)
        gray = model.model.predict_gray(state).view(1, 1, in_height, in_width)
        gray = normalize_tiles(gray).view(in_height, in_width)
        gray_np = gray.detach().cpu().numpy() * 255
        gray_np = np.uint8(gray_np)
        gray_rgb = gray_np[..., None].repeat(3, 2)

        if params.flow:
            flow = model.model.predict_flow(state)
            flow_rgb = viz_flow(flow.squeeze(0))
            flow_rgb = flow_rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()
            cat = np.concatenate([flow_rgb, gray_rgb], axis=1)
        else:
            cat = gray_rgb

        if params.viz_input:
            x = tensor_th.mean(dim=2)
            x = 255 * normalize_tiles(x, num_dims=3, num_stds=9).view(
                in_height, in_width
            )
            x = x.byte().cpu().numpy()
            x = x[..., None].repeat(3, 2)
            cat = np.concatenate((x, cat), axis=1)

        key = show_write(cat)
        if key == 27:
            sys.exit()
        if key == ord("p"):
            pause = not pause
        if key == ord("q"):
            return

        # Read all the stuff
        dt = time.time() - start
        raw_stream.load_delta_t(dt * 1.5e6)


def main():
    params = parse_args()
    run(params)


if __name__ == "__main__":
    with torch.no_grad():
        main()
