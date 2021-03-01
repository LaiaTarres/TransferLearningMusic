# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Compute output examples for VGGish"""
import numpy as np
import src.vggish_input as vggish_input
import torch
from tqdm import tqdm
from math import ceil
import os

def extract_vggish_features(paths, path2gt, model, config, device):
    """Extracts VGGish features and their corresponding ground_truth and identifiers (the path).
       VGGish features are extracted from non-overlapping audio patches of 0.96 seconds,
       where each audio patch covers 64 mel bands and 96 frames of 10 ms each.
       We repeat ground_truth and identifiers to fit the number of extracted VGGish features.
    """
    # 1) Extract log-mel spectrograms
    first_audio = True
    for p in paths:
        if first_audio:
            input_data = vggish_input.wavfile_to_examples(config['audio_folder'] + p)
            ground_truth = np.repeat(path2gt[p], input_data.shape[0], axis=0)
            identifiers = np.repeat(p, input_data.shape[0], axis=0)
            first_audio = False
        else:
            tmp_in = vggish_input.wavfile_to_examples(config['audio_folder'] + p)
            input_data = np.concatenate((input_data, tmp_in), axis=0)
            tmp_gt = np.repeat(path2gt[p], tmp_in.shape[0], axis=0)
            ground_truth = np.concatenate((ground_truth, tmp_gt), axis=0)
            tmp_id = np.repeat(p, tmp_in.shape[0], axis=0)
            identifiers = np.concatenate((identifiers, tmp_id), axis=0)

    # 2) Load Pytorch model to extract VGGish features
    input_data = input_data.astype(np.float32)
    input_data = torch.from_numpy(input_data)
    input_data = torch.unsqueeze(input_data, 1)
    feature = model.forward(input_data.to(device))

    return [feature, ground_truth, identifiers]


def extract_features_wrapper(paths, path2gt, model, configuration, device, data_folder, save_as=False):
    """Wrapper function for extracting features (MusiCNN, VGGish or OpenL3) per batch.
       If a save_as string argument is passed, the features will be saved in
       the specified file.
    """
    feature_extractor = extract_vggish_features

    batch_size = configuration['batch_size']
    first_batch = True
    for batch_id in tqdm(range(ceil(len(paths) / batch_size))):
        batch_paths = paths[(batch_id) * batch_size:(batch_id + 1) * batch_size]
        [x, y, refs] = feature_extractor(batch_paths, path2gt, model, configuration, device)
        x = x.cpu().detach().numpy()
        if first_batch:
            [X, Y, IDS] = [x, y, refs]
            first_batch = False
        else:
            X = np.concatenate((X, x), axis=0)
            Y = np.concatenate((Y, y), axis=0)
            IDS = np.concatenate((IDS, refs), axis=0)

    if save_as:  # save data to file
        # create a directory where to store the extracted training features
        audio_representations_folder = data_folder + 'audio_representations/'
        if not os.path.exists(audio_representations_folder):
            os.makedirs(audio_representations_folder)
        np.savez(audio_representations_folder + save_as, X=X, Y=Y, IDS=IDS)
        print('Audio features stored: ', save_as)

    return [X / 255, Y, IDS]