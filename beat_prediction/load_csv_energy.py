

import itertools
import pygame
import pygame.color
import csv
import time
import numpy as np


from pygame.color import THECOLORS
from pykinect import nui
from pykinect.nui import JointId
from pykinect.nui import SkeletonTrackingState
from pykinect.nui.structs import TransformSmoothParameters, Vector

SCALE_FACTOR = 1
WINDOW_SIZE = 800/SCALE_FACTOR, 480/SCALE_FACTOR
#INPUT_FILENAME = 'kinect_skeleton04_01_17_14_13.csv'
INPUT_FILENAME = 'song_dance_raw_data/kinect_skeleton04_23_17_13_24.csv'
#INPUT_FILENAME = 'predictions_xyz_last.csv'
SKELETON_COLORS = [THECOLORS["red"],
                   THECOLORS["blue"],
                   THECOLORS["green"],
                   THECOLORS["orange"],
                   THECOLORS["purple"],
                   THECOLORS["yellow"],
                   THECOLORS["violet"]]

LEFT_ARM = (JointId.ShoulderCenter,
            JointId.ShoulderLeft,
            JointId.ElbowLeft,
            JointId.WristLeft,
            JointId.HandLeft)
RIGHT_ARM = (JointId.ShoulderCenter,
             JointId.ShoulderRight,
             JointId.ElbowRight,
             JointId.WristRight,
             JointId.HandRight)
LEFT_LEG = (JointId.HipCenter,
            JointId.HipLeft,
            JointId.KneeLeft,
            JointId.AnkleLeft,
            JointId.FootLeft)
RIGHT_LEG = (JointId.HipCenter,
             JointId.HipRight,
             JointId.KneeRight,
             JointId.AnkleRight,
             JointId.FootRight)
SPINE = (JointId.HipCenter,
         JointId.Spine,
         JointId.ShoulderCenter)

NECK = (JointId.ShoulderCenter,
         JointId.Head)

SMOOTH_PARAMS_SMOOTHING = 0.7
SMOOTH_PARAMS_CORRECTION = 0.4
SMOOTH_PARAMS_PREDICTION = 0.7
SMOOTH_PARAMS_JITTER_RADIUS = 0.1
SMOOTH_PARAMS_MAX_DEVIATION_RADIUS = 0.1
SMOOTH_PARAMS = TransformSmoothParameters(SMOOTH_PARAMS_SMOOTHING,
                                          SMOOTH_PARAMS_CORRECTION,
                                          SMOOTH_PARAMS_PREDICTION,
                                          SMOOTH_PARAMS_JITTER_RADIUS,
                                          SMOOTH_PARAMS_MAX_DEVIATION_RADIUS)

skeleton_to_depth_image = nui.SkeletonEngine.skeleton_to_depth_image


def draw_skeleton_data(dispInfo, screen, skeleton_position, index, positions, width = 4):
    start = skeleton_position[positions[0]]

    for position in itertools.islice(positions, 1, None):
        next = skeleton_position[position.value]

        curstart = skeleton_to_depth_image(start, dispInfo.current_w, dispInfo.current_h)
        curend = skeleton_to_depth_image(next, dispInfo.current_w, dispInfo.current_h)

        pygame.draw.line(screen, SKELETON_COLORS[index], curstart, curend, width)

        start = next

def draw_skeletons(dispInfo, screen, skeleton_position, beat, x_centroid, y_centroid):
    # clean the screen
    screen.fill(pygame.color.THECOLORS["black"])

    if beat:
        index = 1
    else:
        index = 0
    # draw the Head
    HeadPos = skeleton_to_depth_image(skeleton_position[JointId.Head], dispInfo.current_w, dispInfo.current_h)
    draw_skeleton_data(dispInfo, screen, skeleton_position, index, SPINE, 10)
    draw_skeleton_data(dispInfo, screen, skeleton_position, index, NECK, 5)
    pygame.draw.circle(screen, SKELETON_COLORS[index], (int(HeadPos[0]), int(HeadPos[1])), 20, 0)
    pygame.draw.circle(screen, SKELETON_COLORS[2], (int(x_centroid + 100), int(y_centroid - 50)), 5, 0)

    # drawing the limbs
    draw_skeleton_data(dispInfo, screen, skeleton_position, index, LEFT_ARM)
    draw_skeleton_data(dispInfo, screen, skeleton_position, index, RIGHT_ARM)
    draw_skeleton_data(dispInfo, screen, skeleton_position, index, LEFT_LEG)
    draw_skeleton_data(dispInfo, screen, skeleton_position, index, RIGHT_LEG)

def main():
    """Initialize and run the game."""
    pygame.init()
    START_TIME = 0
    input_seq_x, input_seq_y, output_seq = load_data()
    # Initialize PyGame
    screen = pygame.display.set_mode(WINDOW_SIZE, 0, 16)
    pygame.display.set_caption('PyKinect Skeleton')
    screen.fill(pygame.color.THECOLORS["black"])

    # open csv file
    f = open(INPUT_FILENAME, 'rb')
    reader = csv.reader(f)
    header = next(reader)
    x = 1
    time_diff = 0
    curr_time = 0
    # Main game loop
    while True:
        event = pygame.event.poll()
        elapsed_time = int((time.time() - START_TIME) * 1000)
        time_diff = curr_time - elapsed_time
        print(time_diff)
        if time_diff > 0:
            time.sleep(time_diff/1000.)
        if event.type == pygame.QUIT:
            f.close()
            break
        else:
            count = 0
            skeleton_positions = []
            for i in range(0, 20):
                line = next(reader)
                count = int(line[0])
                curr_time = float(line[1])
                x = float(line[3])
                y = float(line[4])
                z = float(line[5])
                #w = float(line[6])
                w = 1.0
                skeleton_positions.append(Vector(x, y, z, w))
            if START_TIME == 0:
                START_TIME = time.time()
            beat = output_seq[count]
            x_centroid = input_seq_x[count]
            y_centroid = input_seq_y[count]
            draw_skeletons(pygame.display.Info(), screen, skeleton_positions, beat, x_centroid, y_centroid)
            pygame.display.update()
            pass
    f.close()


def load_data(PATH=''):
    # read numpy arrays
    input_seq_x = np.load(PATH + 'input_seq_x.npy')
    input_seq_y = np.load(PATH + 'input_seq_y.npy')
    output_seq = np.load(PATH + 'output_seq.npy')
    return input_seq_x, input_seq_y, output_seq

if __name__ == '__main__':
    main()