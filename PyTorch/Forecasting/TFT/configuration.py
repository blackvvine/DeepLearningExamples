# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import field
from typing import List
from typing import Optional
from typing import Tuple

from dataclasses import dataclass

from data_utils import InputTypes, DataTypes, FeatureSpec


@dataclass
class ConfigBase:

    features: List[FeatureSpec]
    time_ids: str
    dataset_stride: int
    example_length: int

    scale_per_id: bool
    missing_cat_data_strategy: str

    # Feature sizes
    static_categorical_inp_lens: List[int]

    example_length: int
    encoder_length: int

    n_head: int
    hidden_size: int
    dropout: float
    attn_dropout: float

    relative_split: bool = False

    # For doing the train-validation-test split, if relative_split
    # is not set, data_utils chooses the splits based on the ranges
    # defined below over the value of time_ids
    train_range: Optional[Tuple[int, int]] = None
    valid_range: Optional[Tuple[int, int]] = None
    test_range: Optional[Tuple[int, int]] = None

    quantiles: List[float] = field(default=(0.1, 0.5, 0.9))
    missing_id_strategy: Optional[str] = field(default=None)
    temporal_known_categorical_inp_lens: List[int] = field(default_factory=list)
    temporal_observed_categorical_inp_lens: List[int] = field(default_factory=list)

    @property
    def temporal_known_continuous_inp_size(self):
        return len([x for x in self.features if x.feature_type == InputTypes.KNOWN and x.feature_embed_type == DataTypes.CONTINUOUS])

    @property
    def temporal_observed_continuous_inp_size(self):
        return len([x for x in self.features if x.feature_type == InputTypes.OBSERVED and x.feature_embed_type == DataTypes.CONTINUOUS])

    @property
    def temporal_target_size(self):
        return len([x for x in self.features if x.feature_type == InputTypes.TARGET])

    @property
    def static_continuous_inp_size(self):
        return len([x for x in self.features if x.feature_type == InputTypes.STATIC and x.feature_embed_type == DataTypes.CONTINUOUS])

    @property
    def num_static_vars(self):
        return self.static_continuous_inp_size + len(self.static_categorical_inp_lens)

    @property
    def num_future_vars(self):
        return self.temporal_known_continuous_inp_size + len(self.temporal_known_categorical_inp_lens)

    @property
    def num_historic_vars(self):
        return sum([self.num_future_vars,
                    self.temporal_observed_continuous_inp_size,
                    self.temporal_target_size,
                    len(self.temporal_observed_categorical_inp_lens),
                    ])


def M4DailyConfig():

    num_sequneces = 4227
    splits = (8948673, 9964658, 10023836)

    # aka forecast_len in data_utils.py and "Past Inputs" in the paper,
    # how many time steps being considered
    # num_past_inputs = 14

    features = [
        FeatureSpec("id", InputTypes.ID, DataTypes.CATEGORICAL),
        FeatureSpec("time_step", InputTypes.TIME, DataTypes.CONTINUOUS),
        FeatureSpec("year", InputTypes.KNOWN, DataTypes.CONTINUOUS),
        FeatureSpec("month", InputTypes.KNOWN, DataTypes.CATEGORICAL),
        FeatureSpec("day_of_year", InputTypes.KNOWN, DataTypes.CONTINUOUS),
        FeatureSpec("day_of_month", InputTypes.KNOWN, DataTypes.CONTINUOUS),
        FeatureSpec("day_of_week", InputTypes.KNOWN, DataTypes.CATEGORICAL),
        FeatureSpec("target", InputTypes.TARGET, DataTypes.CONTINUOUS),
        # repeating ID in static inputs just in case
        FeatureSpec("categorical_id", InputTypes.STATIC, DataTypes.CATEGORICAL),
    ]

    return ConfigBase(
        features=features,
        time_ids='time_step',
        # train_range=(0, splits[0]),
        # valid_range=(splits[0], splits[1]),
        # test_range=(splits[1], splits[2]),
        relative_split=True,
        dataset_stride=1,
        scale_per_id=True,
        missing_cat_data_strategy='encode_all',
        static_categorical_inp_lens=[num_sequneces],
        temporal_known_categorical_inp_lens=[12, 7],
        temporal_observed_categorical_inp_lens=[],
        example_length=14, #(len(features) + 1) * num_past_inputs,
        encoder_length=13, # len(features) * num_past_inputs,
        n_head=4,
        hidden_size=128,
        dropout=0.1,
        attn_dropout=0.0
    )


class ElectricityConfig():
    def __init__(self):

        self.features = [
                         FeatureSpec('id', InputTypes.ID, DataTypes.CATEGORICAL),
                         FeatureSpec('hours_from_start', InputTypes.TIME, DataTypes.CONTINUOUS),
                         FeatureSpec('power_usage', InputTypes.TARGET, DataTypes.CONTINUOUS),
                         FeatureSpec('hour', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                         FeatureSpec('day_of_week', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                         FeatureSpec('hours_from_start', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                         FeatureSpec('categorical_id', InputTypes.STATIC, DataTypes.CATEGORICAL),
                        ]
        # Dataset split boundaries
        self.time_ids = 'days_from_start' # This column contains time indices across which we split the data
        self.train_range = (1096, 1315)
        self.valid_range = (1308, 1339)
        self.test_range = (1332, 1346)
        self.dataset_stride = 1 #how many timesteps between examples
        self.scale_per_id = True
        self.missing_id_strategy = None
        self.missing_cat_data_strategy='encode_all'

        # Feature sizes
        self.static_categorical_inp_lens = [369]
        self.temporal_known_categorical_inp_lens = []
        self.temporal_observed_categorical_inp_lens = []
        self.quantiles = [0.1, 0.5, 0.9]

        self.example_length = 8 * 24
        self.encoder_length = 7 * 24

        self.n_head = 4
        self.hidden_size = 128
        self.dropout = 0.1
        self.attn_dropout = 0.0

        #### Derived variables ####
        self.temporal_known_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.KNOWN and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_observed_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.OBSERVED and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_target_size = len([x for x in self.features if x.feature_type == InputTypes.TARGET])
        self.static_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.STATIC and x.feature_embed_type == DataTypes.CONTINUOUS])

        self.num_static_vars = self.static_continuous_inp_size + len(self.static_categorical_inp_lens)
        self.num_future_vars = self.temporal_known_continuous_inp_size + len(self.temporal_known_categorical_inp_lens)
        self.num_historic_vars = sum([self.num_future_vars,
                                      self.temporal_observed_continuous_inp_size,
                                      self.temporal_target_size,
                                      len(self.temporal_observed_categorical_inp_lens),
                                      ])


class TrafficConfig():
    def __init__(self):

        self.features = [
                         FeatureSpec('id', InputTypes.ID, DataTypes.CATEGORICAL),
                         FeatureSpec('hours_from_start', InputTypes.TIME, DataTypes.CONTINUOUS),
                         FeatureSpec('values', InputTypes.TARGET, DataTypes.CONTINUOUS),
                         FeatureSpec('time_on_day', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                         FeatureSpec('day_of_week', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                         FeatureSpec('hours_from_start', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                         FeatureSpec('categorical_id', InputTypes.STATIC, DataTypes.CATEGORICAL),
                        ]
        # Dataset split boundaries
        self.time_ids = 'sensor_day' # This column contains time indices across which we split the data
        self.train_range = (0, 151)
        self.valid_range = (144, 166)
        self.test_range = (159, float('inf'))
        self.dataset_stride = 1 #how many timesteps between examples
        self.scale_per_id = False
        self.missing_id_strategy = None
        self.missing_cat_data_strategy='encode_all'

        # Feature sizes
        self.static_categorical_inp_lens = [963]
        self.temporal_known_categorical_inp_lens = []
        self.temporal_observed_categorical_inp_lens = []
        self.quantiles = [0.1, 0.5, 0.9]

        self.example_length = 8 * 24
        self.encoder_length = 7 * 24

        self.n_head = 4
        self.hidden_size = 128
        self.dropout = 0.3
        self.attn_dropout = 0.0

        #### Derived variables ####
        self.temporal_known_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.KNOWN and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_observed_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.OBSERVED and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_target_size = len([x for x in self.features if x.feature_type == InputTypes.TARGET])
        self.static_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.STATIC and x.feature_embed_type == DataTypes.CONTINUOUS])

        self.num_static_vars = self.static_continuous_inp_size + len(self.static_categorical_inp_lens)
        self.num_future_vars = self.temporal_known_continuous_inp_size + len(self.temporal_known_categorical_inp_lens)
        self.num_historic_vars = sum([self.num_future_vars,
                                      self.temporal_observed_continuous_inp_size,
                                      self.temporal_target_size,
                                      len(self.temporal_observed_categorical_inp_lens),
                                      ])


CONFIGS = {
    'electricity': ElectricityConfig,
    'traffic': TrafficConfig,
    'm4daily': M4DailyConfig,
}
