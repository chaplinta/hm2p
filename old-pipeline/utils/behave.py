import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import configparser
import csv
import shutil
import math
from matplotlib.ticker import FuncFormatter
import imageio
import utils.video
from utils import misc, metadata as mdutils
from matplotlib.collections import LineCollection
from utils import tune as tu
from dataclasses import dataclass
from shapely.geometry.polygon import Polygon
from shapely.geometry import MultiPoint
import cv2

EXP_ID = "exp_id"

BEHAVE_TIME = "Behave-time"

HD_ABS = "Head-direction-absolute"
HD_ABS_FILT = "Head-direction-absolute-filt"
HD_UNWRAP = "Head-direction-unwrapped"
HD_UNWRAP_FILT = "Head-direction-unwrapped-filt"

AHV_INST = "Angular-head-velocity-inst"
AHV_GRAD = "Angular-head-velocity-grad"
AHV_FILT_GRAD = "Angular-head-velocity-filt-grad"

HEAD_X_RAW_PIX = "Head-x-raw-pix"
HEAD_Y_RAW_PIX = "Head-y-raw-pix"
HEAD_X_RAW_MM = "Head-x-raw-mm"
HEAD_Y_RAW_MM = "Head-y-raw-mm"
HEAD_X_RAW_MAZE = "Head-x-raw-maze"
HEAD_Y_RAW_MAZE = "Head-y-raw-maze"

HEAD_X_FILT_PIX = "Head-x-filtered-pix"
HEAD_Y_FILT_PIX = "Head-y-filtered-pix"
HEAD_X_FILT_MM = "Head-x-filtered-mm"
HEAD_Y_FILT_MM = "Head-y-filtered-mm"
HEAD_X_FILT_MAZE = "Head-x-filtered-maze"
HEAD_Y_FILT_MAZE = "Head-y-filtered-maze"

HEAD_X_FILT_MAZE_INT = "Head-x-filtered-maze-int"
HEAD_Y_FILT_MAZE_INT = "Head-y-filtered-maze-int"

BACK_UPPER_X_FILT_MM = "Back-upper-x-filtered-mm"
BACK_UPPER_Y_FILT_MM = "Back-upper-y-filtered-mm"

DELTA_X_INST = "Delta-x-inst"
DELTA_Y_INST = "Delta-y-inst"
DELTA_X_FILT = "Delta-x-filt"
DELTA_Y_FILT = "Delta-y-filt"
DIST_INST_CM = "Distance-moved-inst"
DIST_FILT_CM = "Distance-moved-filt"
DIST_FILT_CM_CUM = "Distance-moved-filt-cumulative"
DIST_FILT_CM_CUM_PER = "Distance-moved-filt-cumulative-percent"
VEL_X_INST = "Velocity-x-inst"
VEL_Y_INST = "Velocity-y-inst"
VEL_INST = "Velocity-inst"
SPEED_INST = "Speed-inst"
SPEED_GRAD = "Speed-grad"
SPEED_FILT_GRAD = "Speed-filt-grad"

LOCO_SPEED_FILT_GRAD = "Loco-filt-grad"

HEADING_ALLO_ABS = "Heading-allo-absolute"
HEADING_ALLO_ABS_FILT = "Heading-allo-absolute-filt"
HEADING_EGO_ABS = "Heading-ego-absolute"
HEADING_EGO_ABS_FILT = "Heading-ego-absolute-filt"

HEADING_ALLO_UNWRAP = "Heading-allo-unwrapped"
HEADING_ALLO_UNWRAP_FILT = "Heading-allo-unwrapped-filt"
HEADING_EGO_UNWRAP = "Heading-ego-unwrapped"
HEADING_EGO_UNWRAP_FILT = "Heading-ego-unwrapped-filt"

ACC_FILT_GRAD = "Acceleration-filt-grad"

LIGHT_ON = "Light-on"

METRIC_NAMES = [BEHAVE_TIME,
                HD_ABS, HD_ABS_FILT, HD_UNWRAP, HD_UNWRAP_FILT,
                AHV_INST, AHV_GRAD, AHV_FILT_GRAD,
                HEAD_X_RAW_PIX, HEAD_Y_RAW_PIX, HEAD_X_RAW_MM, HEAD_Y_RAW_MM, HEAD_X_RAW_MAZE, HEAD_Y_RAW_MAZE,
                HEAD_X_FILT_PIX, HEAD_Y_FILT_PIX, HEAD_X_FILT_MM, HEAD_Y_FILT_MM, HEAD_X_FILT_MAZE, HEAD_Y_FILT_MAZE,
                BACK_UPPER_X_FILT_MM, BACK_UPPER_Y_FILT_MM,
                DELTA_X_INST, DELTA_Y_INST, DELTA_X_FILT, DELTA_Y_FILT, DIST_INST_CM, DIST_FILT_CM,
                DIST_FILT_CM_CUM, DIST_FILT_CM_CUM_PER,
                VEL_X_INST, VEL_Y_INST, SPEED_INST, SPEED_GRAD, SPEED_FILT_GRAD, LOCO_SPEED_FILT_GRAD,
                HEADING_EGO_ABS, HEADING_EGO_ABS_FILT, HEADING_ALLO_ABS, HEADING_ALLO_ABS_FILT,
                HEADING_EGO_UNWRAP, HEADING_EGO_UNWRAP_FILT, HEADING_ALLO_UNWRAP, HEADING_ALLO_UNWRAP_FILT,
                ACC_FILT_GRAD]

METRIC_BINARY_NAMES = [LIGHT_ON]

EAR_LEFT = "ear-left"
EAR_RIGHT = "ear-right"
BACK_UPPER = "back-upper"
BACK_MIDDLE = "back-middle"
BACK_TAIL = "back-tail"

BODY_PARTS_DLC_PIX = [EAR_LEFT, EAR_RIGHT, BACK_UPPER, BACK_MIDDLE, BACK_TAIL]

EAR_LEFT_PIX_X = "ear-left-pix-x"
EAR_LEFT_PIX_Y = "ear-left-pix-y"
EAR_RIGHT_PIX_X = "ear-right-pix-x"
EAR_RIGHT_PIX_Y = "ear-right-pix-y"
BACK_UPPER_PIX_X = "back-upper-pix-x"
BACK_UPPER_PIX_Y = "back-upper-pix-y"
BACK_MIDDLE_PIX_X = "back-middle-x"
BACK_MIDDLE_PIX_Y = "back-middle-y"
BACK_TAIL_PIX_X = "back-tail-x"
BACK_TAIL_PIX_Y = "back-tail-y"

BODY_PARTS_PIX = [EAR_LEFT_PIX_X, EAR_LEFT_PIX_Y,
                  EAR_RIGHT_PIX_X, EAR_RIGHT_PIX_Y,
                  BACK_UPPER_PIX_X, BACK_UPPER_PIX_Y,
                  BACK_MIDDLE_PIX_X, BACK_MIDDLE_PIX_Y,
                  BACK_TAIL_PIX_X, BACK_TAIL_PIX_Y]

EAR_LEFT_MM_X = "ear-left-mm-x"
EAR_LEFT_MM_Y = "ear-left-mm-y"
EAR_RIGHT_MM_X = "ear-right-mm-x"
EAR_RIGHT_MM_Y = "ear-right-mm-y"
BACK_UPPER_MM_X = "back-upper-mm-x"
BACK_UPPER_MM_Y = "back-upper-mm-y"
BACK_MIDDLE_MM_X = "back-middle-mm-x"
BACK_MIDDLE_MM_Y = "back-middle-mm-y"
BACK_TAIL_MM_X = "back-tail-mm-x"
BACK_TAIL_MM_Y = "back-tail-mm-y"

BODY_PARTS_MM = [EAR_LEFT_MM_X, EAR_LEFT_MM_Y,
                 EAR_RIGHT_MM_X, EAR_RIGHT_MM_Y,
                 BACK_UPPER_MM_X, BACK_UPPER_MM_Y,
                 BACK_MIDDLE_MM_X, BACK_MIDDLE_MM_Y,
                 BACK_TAIL_MM_X, BACK_TAIL_MM_Y]

EAR_LEFT_PIX_FILT_X = "ear-left-pix-filt-x"
EAR_LEFT_PIX_FILT_Y = "ear-left-pix-filt-y"
EAR_RIGHT_PIX_FILT_X = "ear-right-pix-filt-x"
EAR_RIGHT_PIX_FILT_Y = "ear-right-pix-filt-y"
BACK_UPPER_PIX_FILT_X = "back-upper-pix-filt-x"
BACK_UPPER_PIX_FILT_Y = "back-upper-pix-filt-y"
BACK_MIDDLE_PIX_FILT_X = "back-middle-pix-filt-x"
BACK_MIDDLE_PIX_FILT_Y = "back-middle-pix-filt-y"
BACK_TAIL_PIX_FILT_X = "back-tail-pix-filt-x"
BACK_TAIL_PIX_FILT_Y = "back-tail-pix-filt-y"

BODY_PARTS_PIX_FILT = [EAR_LEFT_PIX_FILT_X, EAR_LEFT_PIX_FILT_Y,
                       EAR_RIGHT_PIX_FILT_X, EAR_RIGHT_PIX_FILT_Y,
                       BACK_UPPER_PIX_FILT_X, BACK_UPPER_PIX_FILT_Y,
                       BACK_MIDDLE_PIX_FILT_X, BACK_MIDDLE_PIX_FILT_Y,
                       BACK_TAIL_PIX_FILT_X, BACK_TAIL_PIX_FILT_Y]

ALL_SERIES = METRIC_NAMES + BODY_PARTS_PIX + BODY_PARTS_MM + BODY_PARTS_PIX_FILT

# The sciscan frame id when resampled.
FRAME_ID = "frame_id"


ACTIVE_SPEED_THRESH_UP = 0.5
ACTIVE_SPEED_THRESH_LO = 0.1
# todo check sepi
ACTIVE_AHV_THRESH_UP = 10
ACTIVE_AHV_THRESH_LO = 2
# The speed threshold that determines if the mouse is locomoting, cm/s.
# Zong used 2.5cm/s, not sure which body part.
# From Sepi
ACTIVE_LOCO_THRESH_UP = 1.5
ACTIVE_LOCO_THRESH_LO = 0.5

INACTIVE_MOVE_FRAME_THRESH = 10

# added later
BAD_BEHAVE = "Bad-behave"
IS_ACTIVE = "Is-active"
MAZE_VISIT = "Maze-visit"
MAZE_CUMDIST = "Maze-cum-dist"

# from https://github.com/adamltyson/movement/blob/master/movement/position/angles.py
def calc_hd_from_ears(ear_positions):
    """
    Calculates angle of an object from x,y coordinates of two points
    :param ear_positions: Array like object of size [4 : num_timepoints] Each column
     is a time point, and the rows are:
     0: Left (x)
     1: Left (y)
     2: Right (x)
     3: Right (y)
     e.g. Ear markers to calculate head angle
    :return: Absolute angle and unwrapped angle (in degrees)
    """

    left_x = ear_positions[0]
    left_y = ear_positions[1]
    right_x = ear_positions[2]
    right_y = ear_positions[3]

    absolute_head_angle = np.arctan2((left_x - right_x), (left_y - right_y))
    absolute_head_angle = 180 + absolute_head_angle * 180 / np.pi

    return absolute_head_angle


# From https://github.com/adamltyson/imlib/blob/master/imlib/radial/misc.py
def phase_unwrap(degrees, discontinuity=180):
    """
    Unwraps phase discontinuity (i.e. adds or subtracts 180 deg when 0/360
    degrees is passed.
    :param degrees: Input angles in degrees
    :param discontinuity: Cut off of discontinuity to assume a phase wrap.
    Default: 180 degrees
    :return np.array: Unwrapped angles
    """
    discontinuity = np.deg2rad(discontinuity)
    rad = np.deg2rad(degrees)
    unwrap = np.unwrap(rad, discont=discontinuity)
    return np.rad2deg(unwrap)


def phase_wrap(degrees):
    """
    Wraps phase discontinuity (oppposite of phase_unwrap).
    https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap/15927914
    :param degrees: Input angles in degrees
    :return np.array: Wrapped angles
    """
    rad = np.deg2rad(degrees)
    wrap = ((-rad + np.pi) % (2.0 * np.pi) - np.pi) * -1.0
    wrap_deg = np.rad2deg(wrap)
    wrap_deg[wrap_deg < 0] = 360 + wrap_deg[wrap_deg < 0]
    return wrap_deg


def calculate_gradient_1d(array):
    """
    Calculates the first order gradient of a given array
    :param array: 1D numpy array
    :return: gradient of the array
    """
    x = range(1, len(array) + 1)
    gradient, _ = np.polyfit(x, array, 1)
    return gradient


def interp_bodyparts(positions, likelihoods, thresh_likelihood=0.95, thresh_diff=20):
    low_likelihood = likelihoods < thresh_likelihood
    large_delta = np.append(np.array(False), np.diff(positions) > thresh_diff)
    indexes = np.logical_or(low_likelihood, large_delta)

    nz = lambda z: z.nonzero()[0]

    positions_old = np.copy(positions)
    positions[indexes] = np.interp(nz(indexes), nz(~indexes), positions[~indexes])

    return positions


# Adapted from https://github.com/adamltyson/opendirection/blob/master/opendirection/behaviour/behaviour.py#L127
def calc_behav(df,
               exp_id,
               cfg,
               meta_data_path,
               backup_meta_path,
               fps,
               frame_times=None,
               light_on_times=None,
               light_off_times=None,
               filter_size=5,
               grad_win_time=0.2,
               min_periods=2,
               center_win=True):

    scorer = df.columns.get_level_values(0)[0]

    # Get the pixel size in mm
    vid_ori = mdutils.get_vid_orientation(cfg, exp_id)
    mov_meta = get_mov_meta(meta_data_path, backup_meta_path, vid_ori)

    mm_per_pix = mov_meta.mm_per_pix
    roi_x1 = mov_meta.roi_x1
    roi_y1 = mov_meta.roi_y1
    roi_width = mov_meta.roi_width
    roi_height = mov_meta.roi_height

    grad_win_frames = int(round(grad_win_time * fps))
    # Make sure the gradient window can be centered.
    if center_win and grad_win_frames % 2 == 0:
        grad_win_frames += 1

    # todo create new empty data frame to use?

    df.insert(0, EXP_ID, exp_id)
    # Copy these bits to their own series to get rid of this score x y shit.
    # Also interpolate bad points.
    for i_part, bp in enumerate(BODY_PARTS_DLC_PIX):
        df[BODY_PARTS_PIX[i_part * 2]] = interp_bodyparts(df[scorer][bp].x.values, df[scorer][bp].likelihood.values)
        df[BODY_PARTS_PIX[i_part * 2 + 1]] = interp_bodyparts(df[scorer][bp].y.values, df[scorer][bp].likelihood.values)

    # Filter body parts (pixels)
    for i_part, bp in enumerate(BODY_PARTS_PIX):
        df[BODY_PARTS_PIX_FILT[i_part]] = (
            df[bp]
            .rolling(filter_size, center=center_win, min_periods=min_periods)
            .median()
        )

    df[BEHAVE_TIME] = np.arange(0, df.shape[0] / fps, 1 / fps, dtype=np.float64)

    # Dumb way of finding which camera frames had the light on.
    light_on = np.zeros(df.shape[0], dtype='bool')
    light_off_times_fixed = np.insert(light_off_times, 0, 0)
    for i_frame, cam_time in enumerate(frame_times):
        i_closest_on = np.searchsorted(light_on_times, cam_time, side='right') - 1
        i_closest_off = np.searchsorted(light_off_times_fixed, cam_time, side='right') - 1

        closest_on = light_on_times[i_closest_on]
        closest_off = light_off_times_fixed[i_closest_off]
        if np.abs(closest_on - cam_time) < np.abs(closest_off - cam_time):
            light_on[i_frame] = True

    df[LIGHT_ON] = light_on

    # Use raw or filtered positions for mm?
    # # Raw:
    # for i_part, bp in enumerate(BODY_PARTS_PIX):
    #     df_move[BODY_PARTS_MM[i_part]] = mm_per_pix * df_move[bp]
    # Filtered:
    for i_part, bp in enumerate(BODY_PARTS_PIX_FILT):
        df[BODY_PARTS_MM[i_part]] = mm_per_pix * df[bp]

    # todo This bit is pretty clunk and should be fixed.
    ear_positions_pix = np.empty((4, len(df)))
    ear_positions_pix[0, :] = df[EAR_LEFT_PIX_FILT_X]
    ear_positions_pix[1, :] = df[EAR_LEFT_PIX_FILT_Y]
    ear_positions_pix[2, :] = df[EAR_RIGHT_PIX_FILT_X]
    ear_positions_pix[3, :] = df[EAR_RIGHT_PIX_FILT_Y]

    ear_positions_mm = np.empty((4, len(df)))
    ear_positions_mm[0, :] = df[EAR_LEFT_MM_X]
    ear_positions_mm[1, :] = df[EAR_LEFT_MM_Y]
    ear_positions_mm[2, :] = df[EAR_RIGHT_MM_X]
    ear_positions_mm[3, :] = df[EAR_RIGHT_MM_Y]

    # Calc HD
    df[HD_ABS] = calc_hd_from_ears(ear_positions_pix)
    # Unwrap HD so there is no discontinuity and AHV can be calculated
    df[HD_UNWRAP] = phase_unwrap(df[HD_ABS])

    velocity_at_t0 = np.array(0)  # to match hd_length
    angular_head_velocity = np.append(
        velocity_at_t0, np.diff(df[HD_UNWRAP])
    )
    # convert to deg/s
    # instantaneous (x(t) - x(t-1))
    df[AHV_INST] = (
            angular_head_velocity * fps
    )

    df[AHV_GRAD] = (
            df[HD_UNWRAP]
            .rolling(grad_win_frames, center=center_win, min_periods=min_periods)
            .apply(calculate_gradient_1d, raw=True)
            * fps
    )

    df[HD_UNWRAP_FILT] = (
        df[HD_UNWRAP]
        .rolling(filter_size, center=center_win, min_periods=min_periods)
        .median()
    )

    df[HD_ABS_FILT] = phase_wrap(df[HD_UNWRAP_FILT])

    df[AHV_FILT_GRAD] = (
            df[HD_UNWRAP_FILT]
            .rolling(grad_win_frames, center=center_win, min_periods=min_periods)
            .apply(calculate_gradient_1d, raw=True)
            * fps
    )

    # Head positions
    (mean_x, mean_y) = calc_pos_from_ears(ear_positions_pix)
    df[HEAD_X_RAW_PIX] = mean_x
    df[HEAD_Y_RAW_PIX] = mean_y

    (mean_x, mean_y) = calc_pos_from_ears(ear_positions_mm)
    df[HEAD_X_RAW_MM] = mean_x
    df[HEAD_Y_RAW_MM] = mean_y

    roi_x1_mm = roi_x1 * mm_per_pix
    roi_y1_mm = roi_y1 * mm_per_pix
    roi_width_mm = roi_width * mm_per_pix
    roi_height_mm = roi_height * mm_per_pix

    maze_square_w = 7.0
    maze_square_h = 5.0

    maze_poly = get_maze_poly()

    df[HEAD_X_RAW_MAZE] = calc_maze_coord(df[HEAD_X_RAW_MM], roi_x1_mm, roi_width_mm, maze_square_w)
    df[HEAD_Y_RAW_MAZE] = calc_maze_coord(df[HEAD_Y_RAW_MM], roi_y1_mm, roi_height_mm, maze_square_h)
    df = fix_oob_df(df, maze_poly, HEAD_X_RAW_MAZE, HEAD_Y_RAW_MAZE)

    # Filter head positions.
    df[HEAD_X_FILT_PIX] = (
        df[HEAD_X_RAW_PIX]
        .rolling(filter_size, center=center_win, min_periods=min_periods)
        .median()
    )
    df[HEAD_Y_FILT_PIX] = (
        df[HEAD_Y_RAW_PIX]
        .rolling(filter_size, center=center_win, min_periods=min_periods)
        .median()
    )
    df[HEAD_X_FILT_MM] = (
        df[HEAD_X_RAW_MM]
        .rolling(filter_size, center=center_win, min_periods=min_periods)
        .median()
    )
    df[HEAD_Y_FILT_MM] = (
        df[HEAD_Y_RAW_MM]
        .rolling(filter_size, center=center_win, min_periods=min_periods)
        .median()
    )

    df[BACK_UPPER_X_FILT_MM] = (
        df[BACK_MIDDLE_MM_X]
        .rolling(filter_size, center=center_win, min_periods=min_periods)
        .median()
    )
    df[BACK_UPPER_Y_FILT_MM] = (
        df[BACK_MIDDLE_MM_Y]
        .rolling(filter_size, center=center_win, min_periods=min_periods)
        .median()
    )

    # Calculate head speeds.
    x_delta_at_t0 = np.array(0)  # to match hd_length
    y_delta_at_t0 = np.array(0)  # to match hd_length
    df[DELTA_X_INST] = np.append(
        x_delta_at_t0, np.diff(df[HEAD_X_FILT_MM])
    )
    df[DELTA_Y_INST] = np.append(
        y_delta_at_t0, np.diff(df[HEAD_Y_FILT_MM])
    )

    df[DELTA_X_FILT] = (
        df[DELTA_X_INST]
        .rolling(filter_size, center=center_win, min_periods=min_periods)
        .median()
    )
    df[DELTA_Y_FILT] = (
        df[DELTA_Y_INST]
        .rolling(filter_size, center=center_win, min_periods=min_periods)
        .median()
    )

    x_vel = df[DELTA_X_INST] * fps / 10  # convert to cm
    y_vel = df[DELTA_Y_INST] * fps / 10  # convert to cm

    df[VEL_X_INST] = x_vel
    df[VEL_Y_INST] = y_vel

    x_vel_filt = (
        df[VEL_X_INST]
        .rolling(filter_size, center=center_win, min_periods=min_periods)
        .median()
    )
    y_vel_filt = (
        df[VEL_Y_INST]
        .rolling(filter_size, center=center_win, min_periods=min_periods)
        .median()
    )

    dist_move = np.sqrt(np.square(df[DELTA_X_FILT]) + np.square(df[DELTA_Y_FILT])) / 10  # convert to cm
    vel = dist_move * fps

    df[DIST_INST_CM] = dist_move
    df[SPEED_INST] = vel

    df[DIST_FILT_CM] = (
        df[DIST_INST_CM]
        .rolling(filter_size, center=center_win, min_periods=min_periods)
        .median()
    )

    df[DIST_FILT_CM_CUM] = df[DIST_FILT_CM].cumsum()
    df[DIST_FILT_CM_CUM_PER] = df[DIST_FILT_CM_CUM] / df[DIST_FILT_CM_CUM].max()

    print("for loop for speed calculation (slow)")
    speed_grad = calc_speed_grad(df[HEAD_X_RAW_MM], df[HEAD_Y_RAW_MM], grad_win_frames, fps)
    df[SPEED_GRAD] = speed_grad
    print("Done")

    print("for loop for speed calculation (slow)")
    speed_grad = calc_speed_grad(df[HEAD_X_FILT_MM], df[HEAD_Y_FILT_MM], grad_win_frames, fps)
    df[SPEED_FILT_GRAD] = speed_grad
    print("Done")

    print("for loop for speed calculation (slow)")
    speed_grad = calc_speed_grad(df[BACK_UPPER_X_FILT_MM], df[BACK_UPPER_Y_FILT_MM], grad_win_frames, fps)
    df[LOCO_SPEED_FILT_GRAD] = speed_grad
    print("Done")

    # Acceleration, why not.
    df[ACC_FILT_GRAD] = (
            df[SPEED_FILT_GRAD]
            .rolling(grad_win_frames, center=center_win, min_periods=min_periods)
            .apply(calculate_gradient_1d, raw=True)
            * fps
    )

    # My idea of allo vs ego head turns.
    heading = np.rad2deg(np.arctan2(-y_vel_filt, x_vel_filt))
    heading[heading < 0] = 360 + heading[heading < 0]
    df[HEADING_ALLO_ABS] = heading

    df[HEADING_ALLO_UNWRAP] = phase_unwrap(df[HEADING_ALLO_ABS])
    df[HEADING_ALLO_UNWRAP_FILT] = (
        df[HEADING_ALLO_UNWRAP]
        .rolling(filter_size, center=center_win, min_periods=min_periods)
        .median()
    )

    df[HEADING_ALLO_ABS_FILT] = phase_wrap(df[HEADING_ALLO_UNWRAP_FILT])

    heading_ego_abs = df[HEADING_ALLO_ABS] - df[HD_ABS_FILT]
    heading_ego_abs[heading_ego_abs < 0] = 360 + heading_ego_abs[heading_ego_abs < 0]
    heading_ego_abs[heading_ego_abs > 0] = heading_ego_abs[heading_ego_abs > 0] - 360
    df[HEADING_EGO_ABS] = heading_ego_abs

    df[HEADING_EGO_UNWRAP] = phase_unwrap(df[HEADING_EGO_ABS])
    df[HEADING_EGO_UNWRAP_FILT] = (
        df[HEADING_EGO_UNWRAP]
        .rolling(filter_size, center=center_win, min_periods=min_periods)
        .median()
    )

    df[HEADING_EGO_ABS_FILT] = phase_wrap(df[HEADING_EGO_UNWRAP_FILT])

    # Calculate position in maze.
    df[HEAD_X_FILT_MAZE] = calc_maze_coord(df[HEAD_X_FILT_MM], roi_x1_mm, roi_width_mm, maze_square_w)
    df[HEAD_Y_FILT_MAZE] = calc_maze_coord(df[HEAD_Y_FILT_MM], roi_y1_mm, roi_height_mm, maze_square_h)
    df = fix_oob_df(df, maze_poly, HEAD_X_FILT_MAZE, HEAD_Y_FILT_MAZE)

    return df


def fix_oob_df(df, maze_poly, x_name, y_name):
    """
    Takes xy coords and finds any that go outside the maze and puts them back at the closest point.
    :param df:
    :param maze_poly:
    :param x_name:
    :param y_name:
    :return:
    """
    x_fix, y_fix = fix_oob(maze_poly, df[x_name].to_numpy(), df[y_name].to_numpy())
    df[x_name] = x_fix
    df[y_name] = y_fix

    return df


def fix_oob(poly, x, y):
    """
    Takes xy coords and finds any that go outside the maze and puts them back at the closest point.
    :param poly:
    :param x:
    :param y:
    :return:
    """
    print("Finding head positions out of bounds (slow)")

    tups = list(zip(x, y))
    head_points = MultiPoint(tups)
    n_points = len(tups)
    inside = np.full(n_points, True)
    replace_indexes = np.arange(n_points)
    i_last_good = None
    bad_intial_indexes = []
    for i_point in range(n_points):

        is_inside = poly.contains(head_points.geoms[i_point])

        if is_inside:
            i_last_good = i_point
            if bad_intial_indexes:
                # We started with some bad outside points so replace them now with this good one.
                replace_indexes[bad_intial_indexes] = i_last_good
        else:
            if i_last_good is not None:
                replace_indexes[i_point] = i_last_good
            else:
                # Have started with a good inside point, so remember these index for when we do.
                bad_intial_indexes.append(i_point)

        inside[i_point] = is_inside

    outside = np.logical_not(inside)
    outside_indexes = np.where(outside)

    x_fix = x[replace_indexes]
    y_fix = y[replace_indexes]

    print("Done")

    return x_fix, y_fix


def get_maze_poly():
    """
    Gets the polygon that represents the maze.
    :return:
    """


    return Polygon([(0, 0), (3, 0), (3, 1),
                    (2, 1), (2, 2), (5, 2), (5, 1), (4, 1), (4, 0), (7, 0), (7, 1), (6, 1),
                    (6, 4), (7, 4), (7, 5), (4, 5), (4, 4), (5, 4), (5, 3), (4, 3), (4, 5),
                    (3, 5), (3, 3), (2, 3), (2, 4), (3, 4), (3, 5), (0, 5), (0, 4), (1, 4),
                    (1, 1), (0, 1)])

    # Upside down version
    # return Polygon([(0, 5), (3, 5), (3, 4),
    #                 (2, 4), (2, 3), (5, 3), (5, 4), (4, 4), (4, 5), (7, 5), (7, 4), (6, 4),
    #                 (6, 1), (7, 1), (7, 0), (4, 0), (4, 1), (5, 1), (5, 2), (4, 2), (4, 0),
    #                 (3, 0), (3, 2), (2, 2), (2, 1), (3, 1), (3, 0), (0, 0), (0, 1), (1, 1),
    #                 (1, 4), (0, 4)])


def calc_maze_coord(data, x1, width, maze_square_size):
    """
    Converts real world coordinates into maze coordinates.
    :param data:
    :param x1:
    :param width:
    :param maze_square_size:
    :return:
    """
    coords = ((data - x1) / width) * maze_square_size

    # Maybe do this elsewhere for the maze structure.
    # data[data < 0] = 0
    # data[data > maze_square_size] = maze_square_size

    return coords


def calc_speed_grad(ds_x, ds_y, grad_win_frames, fps):
    if grad_win_frames % 2 == 0:
        raise Exception()
    win_side = int((grad_win_frames - 1) / 2)
    speed_grad = np.zeros(ds_x.shape)

    for index, val in enumerate(speed_grad):
        if index - win_side < 0:
            i_start = 0
        else:
            i_start = index - win_side

        if index + win_side > speed_grad.size:
            i_end = speed_grad.size
        else:
            i_end = index + win_side

        x = ds_x.loc[i_start:i_end]
        y = ds_y.loc[i_start:i_end]
        array = np.vstack((x, y)).T
        t = range(1, len(array) + 1)
        gradient, _ = np.polyfit(t, array, 1)

        s = np.sqrt(np.sum(np.square(gradient))) * fps
        speed_grad[index] = s

    return speed_grad / 10


# Check distance between ears, shouldn't change so much
def calc_ear_dist(ear_positions):
    left_x = ear_positions[0]
    left_y = ear_positions[1]
    right_x = ear_positions[2]
    right_y = ear_positions[3]

    return np.sqrt(np.square(left_x - right_x) + np.square(left_y - right_y))


def calc_pos_from_ears(ear_positions):
    left_x = ear_positions[0]
    left_y = ear_positions[1]
    right_x = ear_positions[2]
    right_y = ear_positions[3]

    mean_x = (left_x + right_x) / 2.0
    mean_y = (left_y + right_y) / 2.0

    return (mean_x, mean_y)


def plot_summary(df, fps, plot_dir):
    # todo
    # 1. Improve HD with nose/head
    # 2. calculate head*ing* direction

    dpi = 300

    start = 6200
    stop = start + 300

    i_plot_start = 0
    n_plot_points = -1
    if n_plot_points == -1:
        i_plot_end = n_plot_points
    else:
        i_plot_end = i_plot_start + n_plot_points

    time_cam = np.linspace(0, len(df[EAR_LEFT_MM_X]) / fps, len(df[EAR_LEFT_MM_X]), endpoint=True)

    hd_raw = df[HD_ABS].to_numpy()
    hd_filt = df[HD_ABS_FILT].to_numpy()
    hd_unwrap = df[HD_UNWRAP].to_numpy()
    hd_unwrap_filt = df[HD_UNWRAP_FILT].to_numpy()
    ahv_inst = df[AHV_INST].to_numpy()
    ahv_grad = df[AHV_GRAD].to_numpy()
    ahv_filt_grad = df[AHV_FILT_GRAD].to_numpy()

    heading_allo_filt = df[HEADING_ALLO_ABS_FILT].to_numpy()
    heading_ego_filt = df[HEADING_EGO_ABS_FILT].to_numpy()

    speed_inst = df[SPEED_INST].to_numpy()
    speed_grad = df[SPEED_GRAD].to_numpy()
    speed_filt_grad = df[SPEED_FILT_GRAD].to_numpy()

    light_indexes = df[LIGHT_ON].values
    dark_indexes = np.logical_not(light_indexes)
    df_light = df.copy().iloc[light_indexes]
    df_dark = df.copy().iloc[dark_indexes]

    speed_filt_grad_light = df_light[SPEED_FILT_GRAD].to_numpy()
    speed_filt_grad_dark = df_dark[SPEED_FILT_GRAD].to_numpy()

    print("Making plots")

    fig = plt.figure(tight_layout=False)
    ear_positions_mm = np.empty((4, len(df)))
    ear_positions_mm[0, :] = df[EAR_LEFT_MM_X]
    ear_positions_mm[1, :] = df[EAR_LEFT_MM_Y]
    ear_positions_mm[2, :] = df[EAR_RIGHT_MM_X]
    ear_positions_mm[3, :] = df[EAR_RIGHT_MM_Y]
    ear_dist = calc_ear_dist(ear_positions_mm)
    plt.plot(time_cam, ear_dist)
    plt.xlabel('Time (s)')
    plt.ylabel('Dist between ears (mm)')
    plt.title("Ear dist")
    plt.savefig(os.path.join(plot_dir, 'ear-distance.png'), dpi=dpi, facecolor='white')
    plt.close(fig)

    # Plot a histogram of ear distance
    fig = plt.figure(tight_layout=False)
    plt.hist(ear_dist, bins=100)
    plt.xlabel('Dist between ears (mm)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(plot_dir, 'ear-distance-hist.png'), dpi=dpi, facecolor='white')

    # Plot positions in maze
    maze_poly = get_maze_poly()
    maze_width = 7
    maze_height = 5
    maze_bin_sub_big = 1
    bins_maze_x_big = np.linspace(0, maze_width, maze_width * maze_bin_sub_big + 1, endpoint=True)
    bins_maze_y_big = np.linspace(0, maze_height, maze_height * maze_bin_sub_big + 1, endpoint=True)
    maze_bin_sub_medium = 2
    bins_maze_x_medium = np.linspace(0, maze_width, maze_width * maze_bin_sub_medium + 1, endpoint=True)
    bins_maze_y_medium = np.linspace(0, maze_height, maze_height * maze_bin_sub_medium + 1, endpoint=True)
    maze_bin_sub_small = 5
    bins_maze_x_small = np.linspace(0, maze_width, maze_width * maze_bin_sub_small + 1, endpoint=True)
    bins_maze_y_small = np.linspace(0, maze_height, maze_height * maze_bin_sub_small + 1, endpoint=True)

    # Occupancy
    def plot_occ(x_bins, y_bins, bin_type, filt):
        fig = plt.figure(tight_layout=False)
        heatmap, xedges, yedges = np.histogram2d(df[HEAD_X_FILT_MAZE],
                                                 df[HEAD_Y_FILT_MAZE],
                                                 bins=[x_bins, y_bins])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        heatmap_zero = heatmap == 0
        heatmap[heatmap_zero] = np.nan

        if filt:
            heatmap_filt = tu.filt_img_nan(heatmap)
            heatmap_filt[heatmap_zero] = np.nan
            heatmap = heatmap_filt

        heatmap = heatmap.T
        heatmap = 100 * heatmap / df.shape[0]

        # heatmap = np.flipud(heatmap)
        plt.imshow(heatmap, extent=extent, origin='lower')
        plt.tick_params(
            axis='both',  # changes apply to the both axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False)  # labels along the bottom edge are off
        plt.gca().axis('off')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Occupancy (%)', rotation=270)
        # cbar.set_ticks([])
        maze_poly_x, maze_poly_y = maze_poly.exterior.coords.xy
        plt.plot(np.array(maze_poly_x), np.array(maze_poly_y), 'k', linewidth=3)
        plt.savefig(os.path.join(plot_dir, 'maze-dist-{btype}.png'.format(btype=bin_type)), dpi=dpi, facecolor='white')
        plt.close(fig)

    plot_occ(bins_maze_x_big, bins_maze_y_big, "big", False)
    plot_occ(bins_maze_x_medium, bins_maze_y_medium, "medium", False)
    plot_occ(bins_maze_x_medium, bins_maze_y_medium, "medium-filt", True)
    plot_occ(bins_maze_x_small, bins_maze_y_small, "small", False)
    plot_occ(bins_maze_x_small, bins_maze_y_small, "small-filt", True)

    # Trace
    fig = plt.figure(tight_layout=True)
    # points = np.array([df[HEAD_X_FILT_MM].values, df[HEAD_Y_FILT_MM].values]).T.reshape(-1, 1, 2)
    # segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # # Create a continuous norm to map from data points to colors
    # # time_cam_norm = (time_cam - time_cam.min()) / np.max(time_cam - time_cam.min())
    # time_cam_norm = time_cam / 60
    # norm = plt.Normalize(time_cam_norm.min(), time_cam_norm.max())
    # lc = LineCollection(segments, cmap='viridis', norm=norm)
    # # Set the values used for colormapping
    # lc.set_array(time_cam_norm)
    # lc.set_linewidth(1)
    # lc.set_alpha(1)
    # line = fig.gca().add_collection(lc)
    # cbar = fig.colorbar(line)
    # cbar.ax.set_ylabel('Session time (minutes)', rotation=270, labelpad=15)
    plt.plot(df[HEAD_X_FILT_MM].values, df[HEAD_Y_FILT_MM].values)
    plt.gca().set_aspect(1)
    plt.savefig(os.path.join(plot_dir, 'maze-trace-mm.png'), dpi=dpi, facecolor='white')
    plt.close(fig)

    # Trace
    fig = plt.figure(tight_layout=True)

    # This is completely fucked. You need to do all this non sense to vary the color of a line. Wtf.
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([df[HEAD_X_FILT_MAZE].values, df[HEAD_Y_FILT_MAZE].values]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    # time_cam_norm = (time_cam - time_cam.min()) / np.max(time_cam - time_cam.min())
    time_cam_norm = time_cam / 60
    norm = plt.Normalize(time_cam_norm.min(), time_cam_norm.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(time_cam_norm)
    lc.set_linewidth(1)
    lc.set_alpha(1)
    line = fig.gca().add_collection(lc)
    cbar = fig.colorbar(line)
    cbar.ax.set_ylabel('Session time (minutes)', rotation=270, labelpad=15)

    plt.tick_params(
        axis='both',  # changes apply to the both axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False)  # labels along the bottom edge are off
    plt.gca().axis('off')
    maze_poly_x, maze_poly_y = maze_poly.exterior.coords.xy
    plt.plot(np.array(maze_poly_x), np.array(maze_poly_y), 'k', linewidth=3)
    plt.gca().set_aspect(1)
    plt.savefig(os.path.join(plot_dir, 'maze-trace.png'), dpi=dpi, facecolor='white')
    plt.close(fig)

    # Trace with HD
    fig = plt.figure(tight_layout=False)

    # This is completely fucked. You need to do all this non sense to vary the color of a line. Wtf.
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([df[HEAD_X_FILT_MAZE].values, df[HEAD_Y_FILT_MAZE].values]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    trace_color_data = df[HD_ABS_FILT].values
    norm = plt.Normalize(trace_color_data.min(), trace_color_data.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(trace_color_data)
    lc.set_linewidth(1)
    lc.set_alpha(0.5)
    line = fig.gca().add_collection(lc)
    cbar = fig.colorbar(line)
    cbar.ax.set_ylabel('Head direction (degrees)', rotation=270, labelpad=15)

    plt.tick_params(
        axis='both',  # changes apply to the both axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False)  # labels along the bottom edge are off
    plt.gca().axis('off')
    maze_poly_x, maze_poly_y = maze_poly.exterior.coords.xy
    plt.plot(np.array(maze_poly_x), np.array(maze_poly_y), 'k', linewidth=3)
    plt.gca().set_aspect(1)
    plt.savefig(os.path.join(plot_dir, 'maze-trace-HD.png'), dpi=dpi, facecolor='white')
    plt.close(fig)

    # Trace with speed
    fig = plt.figure(tight_layout=False)

    # This is completely fucked. You need to do all this non sense to vary the color of a line. Wtf.
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([df[HEAD_X_FILT_MAZE].values, df[HEAD_Y_FILT_MAZE].values]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    trace_color_data = df[SPEED_FILT_GRAD].values
    norm = plt.Normalize(trace_color_data.min(), trace_color_data.max())

    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(trace_color_data)
    lc.set_linewidth(1)
    lc.set_alpha(0.5)
    line = fig.gca().add_collection(lc)
    cbar = fig.colorbar(line)
    cbar.ax.set_ylabel('Speed (cm/s)', rotation=270, labelpad=15)

    plt.tick_params(
        axis='both',  # changes apply to the both axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False)  # labels along the bottom edge are off
    plt.gca().axis('off')
    maze_poly_x, maze_poly_y = maze_poly.exterior.coords.xy
    plt.plot(np.array(maze_poly_x), np.array(maze_poly_y), 'k', linewidth=3)
    plt.gca().set_aspect(1)
    plt.savefig(os.path.join(plot_dir, 'maze-trace-speed.png'), dpi=dpi, facecolor='white')
    plt.close(fig)

    # Trace with AHV
    fig = plt.figure(tight_layout=False)

    # This is completely fucked. You need to do all this non sense to vary the color of a line. Wtf.
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([df[HEAD_X_FILT_MAZE].values, df[HEAD_Y_FILT_MAZE].values]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    trace_color_data = df[AHV_FILT_GRAD].values
    norm = plt.Normalize(trace_color_data.min(), trace_color_data.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(trace_color_data)
    lc.set_linewidth(1)
    lc.set_alpha(0.5)
    line = fig.gca().add_collection(lc)
    cbar = fig.colorbar(line)
    cbar.ax.set_ylabel('AHV (deg/s)', rotation=270, labelpad=15)

    plt.tick_params(
        axis='both',  # changes apply to the both axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False)  # labels along the bottom edge are off
    plt.gca().axis('off')
    maze_poly_x, maze_poly_y = maze_poly.exterior.coords.xy
    plt.plot(np.array(maze_poly_x), np.array(maze_poly_y), 'k', linewidth=3)
    plt.gca().set_aspect(1)
    plt.savefig(os.path.join(plot_dir, 'maze-trace-AHV.png'), dpi=dpi, facecolor='white')
    plt.close(fig)

    def plot_hd(hd, title_str, file_label):
        fig = plt.figure(tight_layout=False)
        plt.plot(time_cam, hd)
        plt.xlabel('Time (s)')
        plt.ylabel('Head direction (deg)')
        plt.title(title_str)
        plt.savefig(os.path.join(plot_dir, 'trace-hd_{}.png'.format(file_label)), dpi=dpi, facecolor='white')
        plt.close(fig)

    plot_hd(hd_raw, "Raw HD", "01-raw")
    plot_hd(hd_unwrap, "Raw unwrapped HD", "02-raw-unwrap")
    plot_hd(hd_unwrap_filt, "Filt unwrapped HD", "03-filt-unwrap")
    plot_hd(hd_filt, "Filt HD", "04-filt")

    plot_hd(heading_allo_filt, "Allocentric heading", "01-allo-filt")
    plot_hd(heading_ego_filt, "Egocentric heading", "02-ego-filt")

    fig = plt.figure(tight_layout=False)
    plt.plot(time_cam[start:stop], hd_unwrap[start:stop])
    plt.plot(time_cam[start:stop], hd_unwrap_filt[start:stop])
    plt.xlabel('Time (s)')
    plt.ylabel('Head direction (deg)')
    plt.savefig(os.path.join(plot_dir, 'trace-hd_{}.png'.format("04-all-zoom")), dpi=dpi, facecolor='white')
    plt.close(fig)

    def plot_avh(avh, title_str, file_label):
        fig = plt.figure(tight_layout=False)
        plt.plot(time_cam, avh)
        plt.xlabel('Time (s)')
        plt.ylabel('Angular head velocity (deg/s)')
        plt.title(title_str)
        plt.savefig(os.path.join(plot_dir, 'trace-avh_{}.png'.format(file_label)), dpi=dpi, facecolor='white')
        plt.close(fig)

    plot_avh(ahv_inst, "Raw AHV", "01-raw")
    plot_avh(ahv_grad, "Grad AHV", "02-grad")
    plot_avh(ahv_filt_grad, "Filt Grad AHV", "03-filt-grad")

    fig = plt.figure(tight_layout=False)
    plt.plot(time_cam[start:stop], ahv_inst[start:stop])
    plt.plot(time_cam[start:stop], ahv_grad[start:stop])
    plt.plot(time_cam[start:stop], ahv_filt_grad[start:stop])
    plt.xlabel('Time (s)')
    plt.ylabel('Angular head velocity (deg/s)')
    plt.savefig(os.path.join(plot_dir, 'trace-avh_{}.png'.format("04-all-zoom")), dpi=dpi, facecolor='white')
    plt.close(fig)

    def plot_speed(speed, file_label):
        fig = plt.figure(tight_layout=False)
        plt.plot(time_cam, speed)
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (cm/s)')
        plt.savefig(os.path.join(plot_dir, 'trace-speed_{}.png'.format(file_label)), dpi=dpi, facecolor='white')
        plt.close(fig)

    plot_speed(speed_inst, "01-raw")
    plot_speed(speed_grad, "02-filt")
    plot_speed(speed_filt_grad, "03-filt-grad")

    fig = plt.figure(tight_layout=False)
    plt.plot(time_cam[start:stop], speed_inst[start:stop])
    plt.plot(time_cam[start:stop], speed_grad[start:stop])
    plt.plot(time_cam[start:stop], speed_filt_grad[start:stop])
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (cm/s)')
    plt.savefig(os.path.join(plot_dir, 'trace-speed_{}.png'.format("04-all-zoom")), dpi=dpi, facecolor='white')
    plt.close(fig)

    def plot_ear_pos(ear_pos, title_str, file_label):
        fig = plt.figure(tight_layout=False)
        plt.plot(time_cam, ear_pos)
        plt.xlabel('Time (s)')
        plt.ylabel('Position (mm)')
        plt.title(title_str)
        plt.savefig(os.path.join(plot_dir, 'trace-ear_{}.png'.format(file_label)), dpi=dpi, facecolor='white')
        plt.close(fig)

    plot_ear_pos(df[EAR_LEFT_MM_X].to_numpy(), "Filt left x", "01-left-filt-x")
    plot_ear_pos(df[EAR_LEFT_MM_Y].to_numpy(), "Filt left y", "02-left-filt-x")
    plot_ear_pos(df[EAR_RIGHT_MM_X].to_numpy(), "Filt right x", "03-right-filt-x")
    plot_ear_pos(df[EAR_RIGHT_MM_Y].to_numpy(), "Filt right y", "04-right-filt-x")

    # Plot HD histogram
    hd_hist_bins = np.linspace(0, 360, 24)

    def plot_hd_dist(hd, title_str, file_label):
        fig = plt.figure(tight_layout=False)
        plt.hist(hd, bins=hd_hist_bins, alpha=0.75, weights=np.ones(len(hd)) / len(hd))
        plt.xlabel('Head direction (deg)')
        plt.ylabel('Fraction of time')
        plt.xlim(min(hd_hist_bins), max(hd_hist_bins))
        plt.title(title_str)
        plt.savefig(os.path.join(plot_dir, 'dist-hd_{}.png'.format(file_label)), dpi=dpi, facecolor='white')
        plt.close(fig)

    plot_hd_dist(hd_raw, "Raw HD", "01-raw")
    plot_hd_dist(hd_filt, "Filt HD", "02-filt")
    plot_hd_dist(heading_allo_filt, "Allocentric heading", "01-allo-filt")
    plot_hd_dist(heading_ego_filt, "Egocentric heading", "02-ego-filt")

    # Plot AHV distribution
    ahv_hist_bins = np.linspace(-1000, 1000, 51)
    ahv_hist_bins_pos = np.logspace(np.log10(0.1), np.log10(1000), 25 + 1, endpoint=True)

    # ahv_hist_bins = np.hstack((np.flip(-ahv_hist_bins_pos), ahv_hist_bins_pos))
    def plot_avh_dist(avh, title_str, file_label):
        fig = plt.figure(tight_layout=False)
        plt.hist(avh, bins=ahv_hist_bins, alpha=0.75, weights=np.ones(len(avh)) / len(avh))
        # plt.xscale('symlog')
        plt.xlabel('Angular head velocity (deg/s)')
        plt.ylabel('Fraction of time')
        plt.xlim(np.min(ahv_hist_bins), np.max(ahv_hist_bins))
        plt.title(title_str)
        plt.savefig(os.path.join(plot_dir, 'dist-ahv_{}.png'.format(file_label)), dpi=dpi, facecolor='white')
        plt.close(fig)

    plot_avh_dist(ahv_inst, "Raw AHV", "01-raw")
    plot_avh_dist(ahv_grad, "Grad AHV", "02-filt")
    plot_avh_dist(ahv_filt_grad, "Filt Grad AHV", "03-filt-grad")
    # nonzero_indexes = np.abs(ahv_filt_grad) >= 0.1
    # plot_avh_dist(ahv_filt_grad[nonzero_indexes], "Filt Grad nonzero AHV", "04-filt-grad-nozero")

    # # Plot speed distribution
    # speed_hist_bins = np.linspace(0, 40, 101)
    # speed_hist_bins_pos = np.logspace(np.log10(0.1), np.log10(40), 100 + 1, endpoint=True)
    speed_hist_bins = np.linspace(0, 40, 40 + 1, endpoint=True)

    def plot_speed_dist(speed, title_str, file_label):
        fig = plt.figure(tight_layout=False)
        plt.hist(speed, bins=speed_hist_bins, alpha=0.75, weights=np.ones(len(speed)) / len(speed))
        # plt.xscale('log')
        plt.xlabel('Speed (cm/s)')
        plt.ylabel('Fraction of time')
        plt.xlim(np.min(speed_hist_bins), np.max(speed_hist_bins))
        plt.title(title_str)
        plt.savefig(os.path.join(plot_dir, 'dist-speed_{}.png'.format(file_label)), dpi=dpi, facecolor='white')
        plt.close(fig)

    plot_speed_dist(speed_inst, "Raw Speed", "01-raw")
    plot_speed_dist(speed_grad, "Grad Speed", "02-filt")
    plot_speed_dist(speed_filt_grad, "Filt Grad Speed", "03-filt-grad")
    # nonzero_indexes = np.abs(speed_filt_grad) >= 0.1
    # plot_speed_dist(speed_filt_grad[nonzero_indexes], "Filt- Grad nonzero Speed", "04-filt-grad-nozero")

    fig = plt.figure(tight_layout=False)
    plt.hist(speed_filt_grad_light, bins=speed_hist_bins,
             weights=np.ones(len(speed_filt_grad_light)) / len(speed_filt_grad_light), color="0.5")

    hist, bins = np.histogram(speed_filt_grad_dark, bins=speed_hist_bins,
                              weights=np.ones(len(speed_filt_grad_dark)) / len(speed_filt_grad_dark))
    plt.plot(bins[1:] - (bins[1] - bins[0]) / 2, hist, 'k')

    plt.xlabel('Speed (cm/s)')
    plt.ylabel('Fraction of time')
    plt.xlim(np.min(speed_hist_bins), np.max(speed_hist_bins))
    plt.title("Filt Grad Speed")
    plt.savefig(os.path.join(plot_dir, 'dist-speed_light_dark_{}.png'.format("03-filt-grad")), dpi=dpi,
                facecolor='white')
    plt.close(fig)

    def plot_acorr(data, title_text, file_label):
        fig = plt.figure(tight_layout=False)
        plt.acorr(data, maxlags=int(np.round(2 * fps)))
        cam_fps = round(fps)
        plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x / cam_fps), ',')))

        plt.title(title_text)
        plt.xlabel('Time lag (s)')
        plt.ylabel('Correlation')
        plt.savefig(os.path.join(plot_dir, 'acorr-{}.png'.format(file_label)), dpi=dpi, facecolor='white')
        plt.close(fig)

    plot_acorr(ahv_inst, "Raw AHV (deg/s)", "ahv-01-inst")
    plot_acorr(ahv_grad, "Grad AHV (deg/s)", "ahv-02-grad")
    plot_acorr(ahv_filt_grad, "Filt Grad AHV (deg/s)", "ahv-03-filt-grad")

    plot_acorr(np.abs(ahv_inst), "Raw Abs AHV (deg/s)", "ahvabs-01-inst")
    plot_acorr(np.abs(ahv_grad), "Grad Abs AHV (deg/s)", "ahvabs-02-grad")
    plot_acorr(np.abs(ahv_filt_grad), "Filt Grad Abs AHV (deg/s)", "ahvabs-03-filt-grad")

    plot_acorr(speed_inst, "Raw Speed (cm/s)", "speed-01-inst")
    plot_acorr(speed_grad, "Grad Speed (cm/s)", "speed-02-grad")
    plot_acorr(speed_filt_grad, "Filt Grad Speed (cm/s)", "speed-03-filt-grad")

    # # Plot accel
    #
    # def plot_acc(acc, title_str, file_label):
    #     fig = plt.figure(tight_layout=False)
    #     plt.plot(time_cam, acc)
    #     plt.xlabel('Time (s)')
    #     plt.xlabel('Acceleration (cm/s/s)')
    #     plt.title(title_str)
    #     plt.savefig(os.path.join(plot_dir, 'trace-acc{}.png'.format(file_label)), dpi=dpi, facecolor='white')
    #     plt.close(fig)
    #
    # plot_acc(acc_filt_grad, "Filt Grad Acc", "01-filt-grad")
    #
    # acc_hist_bins = np.linspace(-100, 100, 101)
    # def plot_acc_dist(acc, title_str, file_label):
    #     fig = plt.figure(tight_layout=False)
    #     plt.hist(acc/10, bins=acc_hist_bins, alpha=0.75, weights=np.ones(len(acc)) / len(acc))
    #     plt.xlabel('Acceleration (cm/s/s)')
    #     plt.ylabel('Fraction of time')
    #     plt.xlim(np.min(acc_hist_bins), np.max(acc_hist_bins))
    #     plt.title(title_str)
    #     plt.savefig(os.path.join(plot_dir, 'dist-acc{}.png'.format(file_label)), dpi=dpi, facecolor='white')
    #     plt.close(fig)
    #
    # plot_acc_dist(acc_filt_grad, "Filt Grad Acc", "01-filt-grad")

    # Find movment bouts
    move_thresh = 0.5
    bout_hist_bins = np.linspace(0, 16, 32)

    move_intervals_time, stat_intervals_time = get_bouts(speed_filt_grad, move_thresh, fps)

    # Plot bout time distribution
    plot_bout_dist(move_intervals_time, "Moving bout duration",
                   os.path.join(plot_dir, 'dist_{}-bout.png'.format("01-move")), bout_hist_bins)
    plot_bout_dist(stat_intervals_time, "Stationary bout duration",
                   os.path.join(plot_dir, 'dist_{}-bout.png'.format("02-stat")), bout_hist_bins)

    move_light_intervals_time, stat_light_intervals_time = get_bouts(speed_filt_grad_light, move_thresh, fps)
    move_dark_intervals_time, stat_dark_intervals_time = get_bouts(speed_filt_grad_dark, move_thresh, fps)

    plot_bout_dist_ld(move_light_intervals_time, move_dark_intervals_time, "Moving bout duration",
                      os.path.join(plot_dir, 'dist_ld_{}-bout.png'.format("01-move")), bout_hist_bins)
    plot_bout_dist_ld(stat_light_intervals_time, stat_dark_intervals_time, "Stationary bout duration",
                      os.path.join(plot_dir, 'dist_ld_{}-bout.png'.format("01-stat")), bout_hist_bins)

    print("Done")


# Visualize
def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


def backup_mov_meta_data(meta_data_path, meta_data_backup_path):
    if os.path.exists(meta_data_path):
        # If the metadata is in the main video dir then copy it to the backup directory.
        if os.path.exists(meta_data_backup_path):
            # Delete backup folder if it exists.
            shutil.rmtree(meta_data_backup_path)
        shutil.copytree(meta_data_path, meta_data_backup_path)
    elif not os.path.exists(meta_data_path) and os.path.exists(meta_data_backup_path):
        # If there is no metadata in the video folder but there is in the backup folder then copy
        # the backup to the video folder.
        shutil.copytree(meta_data_backup_path, meta_data_path)
    else:
        raise Exception("No meta data folder or backup")


def write_mov_meta_data(config_file_path,
                        backup_meta_path,
                        orientation):

    config = configparser.ConfigParser()
    if os.path.exists(config_file_path):
        config.read(config_file_path)
    meta_data_path = os.path.join(os.path.split(config_file_path)[0], "meta")

    crop_key = 'crop'
    scale_key = 'scale'
    roi_key = 'roi'

    crop_csv_path = os.path.join(meta_data_path, "crop.csv")
    scale_csv_path = os.path.join(meta_data_path, "scale.csv")
    roi_csv_path = os.path.join(meta_data_path, "roi.csv")

    if crop_csv_path:
        with open(crop_csv_path) as csv_file:
            csv_reader = csv.reader(csv_file)
            rows = []
            for row in csv_reader:
                rows.append(row)
            row1 = rows[1]
            # NFI idea why buy x and y appear to be the wrong way around for cropping, wtf?
            y = int(round(float(row1[3])))
            x = int(round(float(row1[4])))
            row3 = rows[3]
            y2 = int(round(float(row3[3])))
            x2 = int(round(float(row3[4])))
            width = x2 - x
            height = y2 - y
            # Make sure it's divides by min_div
            min_div = 32.0

            width_new = int(math.ceil(float(width) / min_div) * min_div)
            height_new = int(math.ceil(float(height) / min_div) * min_div)
            # Tweak the x y to centre the extra pixels.
            width_diff = width_new - width
            height_diff = height_new - height
            width = width_new
            height = height_new
            x = x - int(round(width_diff / 2))
            y = y - int(round(height_diff / 2))
            print(x, y, width, height, width_diff, height_diff)

            if not crop_key in config:
                config[crop_key] = {}

            config[crop_key]['x'] = str(x)
            config[crop_key]['y'] = str(y)
            config[crop_key]['width'] = str(width)
            config[crop_key]['height'] = str(height)
            config[crop_key]['width_diff'] = str(width_diff)
            config[crop_key]['height_diff'] = str(height_diff)

        with open(config_file_path, 'w') as configfile:
            config.write(configfile)

    if scale_csv_path:
        with open(scale_csv_path) as csv_file:
            csv_reader = csv.reader(csv_file)
            rows = []
            for row in csv_reader:
                rows.append(row)

            dists_pix = []
            for i_row in range(1, len(rows) - 1, 2):
                y1 = int(round(float(rows[i_row][3])))
                x1 = int(round(float(rows[i_row][4])))

                y2 = int(round(float(rows[i_row + 1][3])))
                x2 = int(round(float(rows[i_row + 1][4])))

                dists_pix.append(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)

            dist_range = max(dists_pix) - min(dists_pix)
            if dist_range > 10:
                raise Exception("Something wrong with scale meta data")

            mean_dist_pix = np.mean(dists_pix)
            dist_mm = 25.0  # table breadboard
            mm_per_pix = dist_mm / mean_dist_pix

            if not scale_key in config:
                config[scale_key] = {}

            config[scale_key]['mm_per_pix'] = str(mm_per_pix)
            config[scale_key]['mean_dist_pix'] = str(mean_dist_pix)
            config[scale_key]['dist_mm'] = str(dist_mm)

        with open(config_file_path, 'w') as configfile:
            config.write(configfile)

    if roi_csv_path:

        with open(roi_csv_path) as csv_file:
            csv_reader = csv.reader(csv_file)
            rows = []
            for row in csv_reader:
                rows.append(row)

            # NFI idea why buy x and y appear to be the wrong way around, wtf?
            y1 = int(round(float(rows[1][3])))
            x1 = int(round(float(rows[1][4])))
            y2 = int(round(float(rows[2][3])))
            x2 = int(round(float(rows[2][4])))
            y3 = int(round(float(rows[3][3])))
            x3 = int(round(float(rows[3][4])))
            y4 = int(round(float(rows[4][3])))
            x4 = int(round(float(rows[4][4])))

            if not roi_key in config:
                config[roi_key] = {}
                config[roi_key]['x1_raw'] = str(x1)
                config[roi_key]['y1_raw'] = str(y1)
                config[roi_key]['x2_raw'] = str(x2)
                config[roi_key]['y2_raw'] = str(y2)
                config[roi_key]['x3_raw'] = str(x3)
                config[roi_key]['y3_raw'] = str(y3)
                config[roi_key]['x4_raw'] = str(x4)
                config[roi_key]['y4_raw'] = str(y4)

            crop_x = float(config[crop_key]['x'])
            crop_y = float(config[crop_key]['y'])
            crop_height = float(config[crop_key]['height'])

            x1 -= crop_x
            y1 -= crop_y
            x2 -= crop_x
            y2 -= crop_y
            x3 -= crop_x
            y3 -= crop_y
            x4 -= crop_x
            y4 -= crop_y

            if orientation == 0:
                # No need to do anything
                pass
            elif orientation == 90:
                x1t, y1t, x2t, y2t, x3t, y3t, x4t, y4t = x1, y1, x2, y2, x3, y3, x4, y4
                x1, y1, x2, y2, x3, y3, x4, y4 = y1t, x1t, y2t, x2t, y3t, x3t, y4t, x4t
            elif orientation == 180:
                y1 = crop_height - y1
                y2 = crop_height - y2
                y3 = crop_height - y3
                y4 = crop_height - y4
            else:
                raise Exception("Unsupported orientation {}.".format(orientation))

            # Just assume it's square.
            width = x3 - x1
            height = y3 - y1

            rotation = math.degrees(math.atan2(y1 - y2, x1 - x2))

            config[roi_key]['x1'] = str(x1)
            config[roi_key]['y1'] = str(y1)
            config[roi_key]['x2'] = str(x2)
            config[roi_key]['y2'] = str(y2)
            config[roi_key]['x3'] = str(x3)
            config[roi_key]['y3'] = str(y3)
            config[roi_key]['x4'] = str(x4)
            config[roi_key]['y4'] = str(y4)
            config[roi_key]['width'] = str(width)
            config[roi_key]['height'] = str(height)
            config[roi_key]['rotation'] = str(rotation)

        with open(config_file_path, 'w') as configfile:
            config.write(configfile)

    # Save all the meta and cropping data in another directory that can be easily backed up.
    config_file_name = os.path.split(config_file_path)[1]
    # bak_meta_dir = os.path.join(backup_meta_path, "meta")
    # if os.path.exists(bak_meta_dir):
    #     shutil.rmtree(bak_meta_dir)
    # shutil.copytree(meta_data_path, bak_meta_dir)
    shutil.copy2(config_file_path, os.path.join(backup_meta_path, config_file_name))


# # # Animated plots
# plot_window = 2
# n_plot_frames = 3000
#
#
# ani_dpi = 100
# mov_x_inches = movie_x / ani_dpi
# mov_y_inches = movie_y / ani_dpi
# #
# mov_plot_path = os.path.join(proc_path, "plot-ahv-speed.mp4")
#
# t = exp.cam_trigger_times
# #t = exp.SciscanSettings.sci_frame_times
#
# fig = plt.Figure()
# fig.set_size_inches(mov_x_inches, mov_y_inches)
# canvas = FigureCanvasAgg(fig)
#
# #ax = plt.axes()
# ax = fig.add_subplot()
#
# ntcks = 11
# xlbls = np.round(np.linspace(0, plot_window, ntcks), 1)
# ax.set_xlim(left=0, right=plot_window)
# ax.axes.xaxis.set_ticks(xlbls)
# ax.axes.xaxis.set_ticklabels(xlbls)
# ax.axes.grid()
# ax.set_xlabel("Time (s)")
# line_ahv, = ax.plot([], [], lw=2, label='AHV')
# line_speed, = ax.plot([], [], lw=2, label='Speed')
# yax_max = 200 #np.max(np.abs(df_move[bu.AHV_GRAD][0:n_plot_frames])) * 1.1
# ax.set_ylim(bottom=-yax_max, top=yax_max)
# ax.set_ylabel("AHV (degrees/s)")
#
# ax.legend(loc='lower right')
#
# # Init only required for blitting to give a clean slate.
# def init():
#     line_ahv.set_data([], [])
#     line_speed.set_data([], [])
#     return line_ahv, line_speed
#
# fps = exp.tracking_video.fps
# #fps = exp.SciscanSettings.frames_p_sec
# frame_interval = 1 / fps
# frame_interval_ms = frame_interval * 1000
# plot_window_indexes = int(round(plot_window * fps))
#
# def animate_acc_plot(i):
#
#     # Still confused about how this should work.
#     # Currently whatever is happening in the current frame appears on the left side of the plot, and moves to
#     # right. This requires that the time values be flipped, see bellow.
#     # Not sure if this is a good idea, or is similar to how a ephys trace works.
#
#     i_frame = i
#     i_start = i_frame
#     i_end = i_frame + plot_window_indexes
#     time_passed = i_frame * frame_interval
#
#     if i_start < 0:
#         i_start = 0
#     # if time_passed < plot_window:
#     #     time_passed_frames = int(round(time_passed / frame_interval))
#     #     i_end = i_start + time_passed_frames + 1
#     if i_end > t.size:
#         i_end = t.size
#
#     t_plot = t[i_start:i_end]
#     # Set to start at zero
#     t_plot = t_plot - t_plot[0]
#     # NOTE the flip here to make it go left to right:
#     t_plot = np.flip(t_plot)
#
#     line_ahv.set_data(t_plot, df_resampled[bu.AHV_FILT_GRAD][i_start:i_end] - 180)
#     line_speed.set_data(t_plot, df_resampled[bu.SPEED_FILT_GRAD][i_start:i_end] - 180)
#
#     return line_ahv, line_speed
#
# print("Creating AVH speed animation")
# animator = ani.FuncAnimation(fig,
#                              animate_acc_plot,
#                              init_func=init,
#                              frames=n_plot_frames,
#                              interval=frame_interval_ms,
#                              blit=True,
#                              repeat=False)
# print("Done")
#
# print("Save AVH Speed animation to disk")
# animator.save(mov_plot_path, dpi=ani_dpi, fps=fps)
# plt.close()
# print("Done")
#
#

def create_arrow_mov(df,
                     mov_path,
                     movie_arrow_path,
                     start_frame=None,
                     end_frame=None,
                     fps_out=None,
                     ahv_arrow=False,
                     hd_arrow=True,
                     speed_arrow=True):
    print("Drawing head direction on video")
    thickness = 3
    hd_length = 50
    tip_length = 0.4  # relative to hd_length

    if not fps_out:
        fps_out = utils.video.get_mov_fps(mov_path)

    if not start_frame:
        start_frame = 0
    if not end_frame:
        end_frame = utils.video.count_frames(mov_path) - 1

    with imageio.get_reader(mov_path) as mov_reader, \
            imageio.get_writer(movie_arrow_path, fps=fps_out) as mov_writer:

        mov_reader.set_image_index(start_frame)

        for i_frame in range(start_frame, end_frame):

            frame = mov_reader.get_next_data()

            # ear_left_x = int(round(df_move[scorer][EAR_LEFT].x.iloc[i_frame]))
            # ear_left_y = int(round(df_move[scorer][EAR_LEFT].y.iloc[i_frame]))
            # ear_right_x = int(round(df_move[scorer][EAR_RIGHT].x.iloc[i_frame]))
            # ear_right_y = int(round(df_move[scorer][EAR_RIGHT].y.iloc[i_frame]))

            # ear_left_x = int(round(df_move[EAR_LEFT_PIX_X].iloc[i_frame]))
            # ear_left_y = int(round(df_move[EAR_LEFT_PIX_Y].iloc[i_frame]))
            # ear_right_x = int(round(df_move[EAR_RIGHT_PIX_X].iloc[i_frame]))
            # ear_right_y = int(round(df_move[EAR_RIGHT_PIX_Y].iloc[i_frame]))
            # head_x = int(round(df_move[HEAD_X_RAW_PIX].iloc[i_frame]))
            # head_y = int(round(df_move[HEAD_Y_RAW_PIX].iloc[i_frame]))
            # hd = df_move[HD_ABS_FILT].iloc[i_frame]
            # ahv = df_move[AHV_GRAD].iloc[i_frame]

            ear_left_x = int(round(df[EAR_LEFT_PIX_FILT_X].iloc[i_frame]))
            ear_left_y = int(round(df[EAR_LEFT_PIX_FILT_Y].iloc[i_frame]))
            ear_right_x = int(round(df[EAR_RIGHT_PIX_FILT_X].iloc[i_frame]))
            ear_right_y = int(round(df[EAR_RIGHT_PIX_FILT_Y].iloc[i_frame]))
            head_x = int(round(df[HEAD_X_FILT_PIX].iloc[i_frame]))
            head_y = int(round(df[HEAD_Y_FILT_PIX].iloc[i_frame]))
            hd = df[HD_ABS_FILT].iloc[i_frame]
            heading = df[HEADING_ALLO_ABS_FILT].iloc[i_frame]
            ahv = df[AHV_FILT_GRAD].iloc[i_frame]
            speed = df[SPEED_FILT_GRAD].iloc[i_frame]

            start_point = (head_x, head_y)

            arrow_x_end = np.cos(np.deg2rad(hd)) * hd_length
            arrow_y_end = np.sin(np.deg2rad(hd)) * hd_length

            head_x_end = int(round(head_x + arrow_x_end))
            head_y_end = int(round(head_y - arrow_y_end))
            hd_end_point = (head_x_end, head_y_end)

            if hd_arrow:
                frame = cv2.arrowedLine(frame, start_point, hd_end_point, (255, 0, 0), thickness, tipLength=tip_length)

            if speed_arrow and speed > 0.5:
                arrow_x_end = np.cos(np.deg2rad(heading)) * speed * 10
                arrow_y_end = np.sin(np.deg2rad(heading)) * speed * 10

                head_x_end = int(round(head_x + arrow_x_end))
                head_y_end = int(round(head_y - arrow_y_end))
                hd_end_point = (head_x_end, head_y_end)

                frame = cv2.arrowedLine(frame, start_point, hd_end_point, (255, 255, 0), thickness,
                                        tipLength=tip_length)

            if ahv_arrow:
                ahv_arrow_length = np.abs(ahv / 2)
                if ahv_arrow_length < 1:
                    ahv_arrow_length = 0
                elif ahv_arrow_length < 5:
                    ahv_arrow_length = 5

                ahv_angle = hd + np.sign(ahv) * 90

                ahv_arrow_end_x = np.cos(np.deg2rad(ahv_angle)) * ahv_arrow_length
                ahv_arrow_end_y = np.sin(np.deg2rad(ahv_angle)) * ahv_arrow_length
                avh_x_end = int(round(head_x_end + ahv_arrow_end_x))
                avh_y_end = int(round(head_y_end - ahv_arrow_end_y))
                avh_end_point = (avh_x_end, avh_y_end)

                frame = cv2.arrowedLine(frame, hd_end_point, avh_end_point, (255, 0, 255), thickness,
                                        tipLength=tip_length)

            radius = 5
            frame = cv2.circle(frame, (ear_left_x, ear_left_y), radius, (0, 255, 0), thickness=-1)
            frame = cv2.circle(frame, (ear_right_x, ear_right_y), radius, (0, 0, 255), thickness=-1)

            mov_writer.append_data(frame)

    print("Done")


@dataclass
class MovMetaData:

    crop_x: int = None
    crop_y: int = None
    crop_width: int = None
    crop_height: int = None

    mm_per_pix: float = None
    roi_x1: float = None
    roi_y1: float = None
    roi_width: float = None
    roi_height: float = None
    roi_rotation: float = None


def get_mov_meta(meta_data_path, backup_meta_path, vid_orientation):

    # Backup metadata first, or get it from the backup path if it's missing.
    backup_mov_meta_data(meta_data_path, backup_meta_path)

    # Make a single easy to read text file with important information.
    # Can't remember why I did this.
    config_file_name = "meta.txt"
    config_file_path = os.path.join(os.path.split(meta_data_path)[0], config_file_name)

    write_mov_meta_data(config_file_path=config_file_path,
                        backup_meta_path=backup_meta_path,
                        orientation=vid_orientation)

    # Load it and get some of the data.
    config_file_path = os.path.join(os.path.split(meta_data_path)[0], config_file_name)

    # exp_meta = mdutils.get_exps(cfg, [exp_id])
    # orientation = exp_meta['orientation'].values[0]

    config = configparser.ConfigParser()
    config.read(config_file_path)

    mmd = MovMetaData()

    mmd.crop_x = int(config['crop']['x'])
    mmd.crop_y = int(config['crop']['y'])
    mmd.crop_width = int(config['crop']['width'])
    mmd.crop_height = int(config['crop']['height'])

    mmd.mm_per_pix = float(config['scale']['mm_per_pix'])

    mmd.roi_x1 = float(config['roi']['x1'])
    mmd.roi_y1 = float(config['roi']['y1'])
    mmd.roi_width = float(config['roi']['width'])
    mmd.roi_height = float(config['roi']['height'])

    return mmd


def resample_to_frames(exp, df):
    # Probably a pandas way of doing this but this manual way is at least exact.
    print("Resampling behavioural data to imaging time scale")
    n_sci_frames = exp.SciscanSettings.sci_frame_times.size

    cols_for_resample = ALL_SERIES
    data_resamp = np.zeros((n_sci_frames, len(cols_for_resample)))
    df_metrics = df[cols_for_resample]
    light_on = np.zeros(n_sci_frames, dtype='uint8')
    for i_sci_frame in range(n_sci_frames):

        sci_time = exp.SciscanSettings.sci_frame_times[i_sci_frame]
        if i_sci_frame == 0:
            sci_time_prev = sci_time - (1.0 / exp.SciscanSettings.frames_p_sec)
        else:
            sci_time_prev = exp.SciscanSettings.sci_frame_times[i_sci_frame - 1]

        cam_frames = np.where(np.logical_and(exp.cam_trigger_times >= sci_time_prev,
                                             exp.cam_trigger_times < sci_time))[0]

        sci_frame_data = df_metrics.iloc[cam_frames]
        sci_frame_data_mean = sci_frame_data.mean().values
        data_resamp[i_sci_frame, :] = sci_frame_data_mean

        # Look up to see if the light was on during this frame. Should I take a mean? It shouldn't be on for half a \
        # frame, but todo check.
        sci_time_index = exp.SciscanSettings.sci_frame_indexes[i_sci_frame]
        light_was_on = exp.Lighting.on_indexes[sci_time_index]
        light_on[i_sci_frame] = light_was_on

    df_resampled = resample_fix(data_resamp,
                                cols_for_resample,
                                light_on,
                                df[EXP_ID].iloc[0])

    print("Resampling complete")

    return df_resampled


def resample_to_events(exp, df_events, df_behave):
    # Probably a pandas way of doing this but this manual way is at least exact.
    print("Resampling behavioural data to imaging time scale")
    n_events = df_events.shape[0]

    cols_for_resample = ALL_SERIES
    data_resamp = np.zeros((n_events, len(cols_for_resample)))
    df_metrics = df_behave[cols_for_resample]
    light_on = np.zeros(n_events, dtype='uint8')
    for i_event in range(n_events):
        # Somehow these are sometimes floats.
        event_start = int(df_events.iloc[i_event]["onset_index"])
        event_end = int(df_events.iloc[i_event]["offset_index"])

        sci_time_start = exp.SciscanSettings.sci_frame_times[event_start]
        sci_time_end = exp.SciscanSettings.sci_frame_times[event_end]

        cam_frames = np.where(np.logical_and(exp.cam_trigger_times >= sci_time_start,
                                             exp.cam_trigger_times < sci_time_end))[0]

        sci_frame_data = df_metrics.iloc[cam_frames]
        sci_frame_data_mean = sci_frame_data.mean().values
        data_resamp[i_event, :] = sci_frame_data_mean

        # Look up to see if the light was on during this event
        sci_time_indexes = exp.SciscanSettings.sci_frame_indexes[event_start:event_end]
        light_was_on = np.mean(exp.Lighting.on_indexes[sci_time_indexes]) > 0.5
        light_on[i_event] = light_was_on

    df_resampled = resample_fix(data_resamp,
                                cols_for_resample,
                                light_on,
                                df_events[EXP_ID].iloc[0],
                                {"pair_id": df_events["pair_id"].values,
                                 "onset_index": df_events["onset_index"].values,
                                 "offset_index": df_events["offset_index"].values})

    print("Resampling complete")

    return df_resampled


def resample_fix(data_resamp, cols_for_resample, light_on, exp_id, extra_cols={}):
    df_resampled = pd.DataFrame(data=data_resamp, columns=cols_for_resample)

    ec_index = 0
    if not EXP_ID in df_resampled.columns:
        df_resampled.insert(ec_index, EXP_ID, exp_id)

    for ec in extra_cols:
        ec_index += 1
        df_resampled.insert(ec_index, ec, extra_cols[ec])

    # LIGHT_ON is already calculate in resampe sci time
    df_resampled[LIGHT_ON] = light_on

    # HD absolute has to be the wrapped version of the resampled unwrapped angles.
    # Otherwise you average across the discontinuity and fuck it up.
    df_resampled[HD_ABS] = phase_wrap(df_resampled[HD_UNWRAP])
    df_resampled[HD_ABS_FILT] = phase_wrap(df_resampled[HD_UNWRAP_FILT])

    df_resampled[HEADING_ALLO_ABS] = phase_wrap(df_resampled[HEADING_ALLO_UNWRAP])
    df_resampled[HEADING_ALLO_ABS_FILT] = phase_wrap(df_resampled[HEADING_ALLO_UNWRAP_FILT])
    df_resampled[HEADING_EGO_ABS] = phase_wrap(df_resampled[HEADING_EGO_UNWRAP])
    df_resampled[HEADING_EGO_ABS_FILT] = phase_wrap(df_resampled[HEADING_EGO_UNWRAP_FILT])

    # Not sure but maybe the head pos goes out of maze bounds, fix just in case.
    maze_poly = get_maze_poly()
    df_resampled = fix_oob_df(df_resampled, maze_poly, HEAD_X_RAW_MAZE, HEAD_Y_RAW_MAZE)
    df_resampled = fix_oob_df(df_resampled, maze_poly, HEAD_X_FILT_MAZE, HEAD_Y_FILT_MAZE)

    n_frames = df_resampled.shape[0]
    df_resampled[FRAME_ID] = list(range(n_frames))

    return df_resampled


def get_bouts(trace, move_thresh, fps):
    move_indexes = np.abs(trace) >= move_thresh
    stat_indexes = np.logical_not(move_indexes)

    move_bouts_indexes = misc.get_crossings(move_indexes.astype('uint8'), 0.9)
    stat_bouts_indexes = misc.get_crossings(stat_indexes.astype('uint8'), 0.9)

    move_intervals = move_bouts_indexes[1:-1] - move_bouts_indexes[0:-2]
    stat_intervals = stat_bouts_indexes[1:-1] - stat_bouts_indexes[0:-2]

    move_intervals_time = move_intervals / fps
    stat_intervals_time = stat_intervals / fps

    return move_intervals_time, stat_intervals_time


def plot_bout_dist(bouts, title_str, plot_path, bout_hist_bins):
    fig = plt.figure(tight_layout=True)
    n, bins, patches = plt.hist(bouts, bins=bout_hist_bins, alpha=0.75, weights=np.ones(len(bouts)) / len(bouts))

    plt.xlabel('Bout time (s)')
    plt.ylabel('Fraction of bouts')
    plt.xlim(np.min(bout_hist_bins), np.max(bout_hist_bins))
    plt.title(title_str)

    ax2 = plt.gca().twinx()
    ax2.plot(bins[1:], np.cumsum(n), 'r')
    ax2.set_ylim(bottom=0, top=1)
    ax2.set_ylabel('Cumaltive fraction of bouts')

    dpi = 300
    plt.savefig(plot_path, dpi=dpi, facecolor='white')
    plt.close(fig)


def plot_bout_dist_ld(bouts_light, bouts_dark, title_str, plot_path, bout_hist_bins):
    fig = plt.figure(tight_layout=True)

    plt.hist(bouts_light, bins=bout_hist_bins,
             weights=np.ones(len(bouts_light)) / len(bouts_light), color="0.5")

    hist, bins = np.histogram(bouts_dark, bins=bout_hist_bins,
                              weights=np.ones(len(bouts_dark)) / len(bouts_dark))
    plt.plot(bins[1:] - (bins[1] - bins[0]) / 2, hist, 'k')

    plt.xlabel('Bout time (s)')
    plt.ylabel('Fraction of bouts')
    plt.xlim(np.min(bout_hist_bins), np.max(bout_hist_bins))
    plt.title(title_str)

    dpi = 300
    plt.savefig(plot_path, dpi=dpi, facecolor='white')
    plt.close(fig)


def get_moving_indexes(df):
    return np.logical_or(get_trans_indexes(df),
           np.logical_or(get_loco_indexes(df),
                         get_ahv_indexes(df)))


def get_notmoving_indexes(df):
    return ~get_moving_indexes(df)

def get_active_indexes(df):
    return df[IS_ACTIVE] == 1

def get_inactive_indexes(df):
    return ~get_active_indexes(df)

# def get_inactive_indexes(df):
#     return ~df[IS_ACTIVE]

def get_trans_indexes(df):
    behave_trace = df[SPEED_FILT_GRAD].to_numpy()
    indexes = np.abs(behave_trace) > ACTIVE_SPEED_THRESH_UP
    return indexes

def get_loco_indexes(df):
    behave_trace = df[LOCO_SPEED_FILT_GRAD].to_numpy()
    indexes = np.abs(behave_trace) > ACTIVE_LOCO_THRESH_UP
    return indexes

def get_ahv_indexes(df):
    behave_trace = df[AHV_FILT_GRAD].to_numpy()
    indexes = np.abs(behave_trace) > ACTIVE_AHV_THRESH_UP
    return indexes

def get_light_indexes(df):
    return df[LIGHT_ON] == 1

# def get_active_frames(df):
#     active_indexes = get_active_indexes(df)
#     df_active = df.copy(deep=True).iloc[active_indexes]
#     return df_active


