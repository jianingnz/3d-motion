#!/bin/bash
# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


evalset=(
  tapvid3d_14250544550818363063_880_000_900_000_2_VDacMbTOddDXFtRVMF_m3w
)

DATA_PATH=/home/jianing/research/3d-motion/dataset


# Run Raft Optical Flows
for seq in ${evalset[@]}; do
  CUDA_VISIBLE_DEVICES=0 python cvd_opt/preprocess_flow.py \
  --datapath=$DATA_PATH/$seq/r-imgs \
  --model=cvd_opt/raft-things.pth \
  --scene_name $seq --mixed_precision
done

# Run CVD optmization
for seq in ${evalset[@]}; do
  CUDA_VISIBLE_DEVICES=0 python cvd_opt/cvd_opt.py \
  --scene_name $seq \
  --w_grad 2.0 --w_normal 5.0
done

