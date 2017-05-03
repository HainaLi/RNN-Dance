from __future__ import print_function

__author__ = 'Leandra'

import itertools
import time
import pygame
import pygame.color

from sys import byteorder
from array import array
from struct import pack

import pyaudio
import wave

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 11025

from pygame.color import THECOLORS
from pykinect import nui
from pykinect.nui import JointId
from pykinect.nui import SkeletonTrackingState
from pykinect.nui.structs import TransformSmoothParameters

START_TIME = 0
SKELETON_INDEX = -1
OUTPUT_FILE = None
KINECTEVENT = pygame.USEREVENT
WINDOW_SIZE = 640, 480
COLLECT_DATA = True
RECORD_AUDIO = True
AUDIO_FILENAME = ''
COUNT = 0
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
         JointId.ShoulderCenter,
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


def post_frame(frame):
    """Get skeleton events from the Kinect device and post them into the PyGame
    event queue."""
    try:
        pygame.event.post(
            pygame.event.Event(KINECTEVENT, skeleton_frame=frame)
        )
    except:
        # event queue full
        pass


def draw_skeleton_data(dispInfo, screen, pSkeleton, index, positions, width=4):
    start = pSkeleton.SkeletonPositions[positions[0]]

    for position in itertools.islice(positions, 1, None):
        next = pSkeleton.SkeletonPositions[position.value]

        curstart = skeleton_to_depth_image(start, dispInfo.current_w, dispInfo.current_h)
        curend = skeleton_to_depth_image(next, dispInfo.current_w, dispInfo.current_h)

        pygame.draw.line(screen, SKELETON_COLORS[index], curstart, curend, width)

        start = next


def draw_skeletons(dispInfo, screen, skeletons):
    global SKELETON_INDEX
    # initialize time if uninitialized
    if COLLECT_DATA == True:
        elapsed_time = 0
        global START_TIME
        if START_TIME == 0:
            START_TIME = time.time()
        else:
            elapsed_time = int((time.time() - START_TIME) * 1000)
    # clean the screen
    screen.fill(pygame.color.THECOLORS["black"])

    for index, skeleton_info in enumerate(skeletons):
        # only track one skeleton
        if SKELETON_INDEX == -1:
            SKELETON_INDEX = index
        # test if the current skeleton is tracked or not
        if skeleton_info.eTrackingState == SkeletonTrackingState.TRACKED:
            if COLLECT_DATA == True:
                global COUNT
                data_lines = ''
                for joint_id, pos in enumerate(skeleton_info.SkeletonPositions):
                    if pos.x == 0 and pos.y == 0 and pos.z == 0:
                        data_lines = ''
                        break
                    depth_x, depth_y = skeleton_to_depth_image(pos, dispInfo.current_w, dispInfo.current_h)
                    details = [COUNT, elapsed_time, joint_id, pos.x, pos.y, pos.z, pos.w, depth_x, depth_y, index]
                    data_lines += ','.join(map(str, details)) + "\n"
                if data_lines != '':
                    print(data_lines, end='')
                    print(data_lines, end='', file=OUTPUT_FILE)
                    COUNT += 1


            # draw the Head
            HeadPos = skeleton_to_depth_image(skeleton_info.SkeletonPositions[JointId.Head], dispInfo.current_w,
                                              dispInfo.current_h)
            draw_skeleton_data(dispInfo, screen, skeleton_info, index, SPINE, 10)
            pygame.draw.circle(screen, SKELETON_COLORS[index], (int(HeadPos[0]), int(HeadPos[1])), 20, 0)

            # drawing the limbs
            draw_skeleton_data(dispInfo, screen, skeleton_info, index, LEFT_ARM)
            draw_skeleton_data(dispInfo, screen, skeleton_info, index, RIGHT_ARM)
            draw_skeleton_data(dispInfo, screen, skeleton_info, index, LEFT_LEG)
            draw_skeleton_data(dispInfo, screen, skeleton_info, index, RIGHT_LEG)


def record_to_file(path, sample_width, data):

    data = pack('<' + ('h' * len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


def main():
    # initialize CSV File
    if COLLECT_DATA == True:
        global OUTPUT_FILE
        global AUDIO_FILENAME
        current_time = time.strftime("%m_%d_%y_%H_%M", time.localtime())
        output_name = 'kinect_skeleton%s.csv' % current_time
        OUTPUT_FILE = open(output_name, "w")
        print('count,timestamp,joint_id,x,y,z,w, depth_x, depth_y, index', file=OUTPUT_FILE)
        if RECORD_AUDIO == True:
            AUDIO_FILENAME = 'audio%s.wav' % current_time

    # audio
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
                    input=True, output=True,
                    frames_per_buffer=CHUNK_SIZE)

    r = array('h')

    """Initialize and run the game."""
    pygame.init()

    # Initialize PyGame
    screen = pygame.display.set_mode(WINDOW_SIZE, 0, 16)
    pygame.display.set_caption('PyKinect Skeleton')
    screen.fill(pygame.color.THECOLORS["black"])

    with nui.Runtime() as kinect:
        kinect.skeleton_engine.enabled = True
        kinect.skeleton_frame_ready += post_frame

        # Main game loop
        while True:
            # record audio
            # little endian, signed short
            snd_data = array('h', stream.read(CHUNK_SIZE))
            if byteorder == 'big':
                snd_data.byteswap()
            r.extend(snd_data)
            event = pygame.event.wait()

            if event.type == pygame.QUIT:
                sample_width = p.get_sample_size(FORMAT)
                stream.stop_stream()
                stream.close()
                p.terminate()

                r = normalize(r)
                r = add_silence(r, 0.5)
                record_to_file(AUDIO_FILENAME, sample_width, r)
                break
            elif event.type == KINECTEVENT:
                # apply joint filtering
                kinect._nui.NuiTransformSmooth(event.skeleton_frame, SMOOTH_PARAMS)

                draw_skeletons(pygame.display.Info(), screen, event.skeleton_frame.SkeletonData)
                pygame.display.update()
                pass
    if COLLECT_DATA == True:
        OUTPUT_FILE.close()


def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM) / max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i * times))
    return r


def record():
    """
    Record from the microphone and
    return the data as an array of signed shorts.

    Normalizes the audio, pads with 0.5 seconds of
    blank sound to make sure VLC et al can play
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
                    input=True, output=True,
                    frames_per_buffer=CHUNK_SIZE)

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    # r = add_silence(r, 0.5)
    return sample_width, r


def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in xrange(int(seconds * RATE))])
    r.extend(snd_data)
    r.extend([0 for i in xrange(int(seconds * RATE))])
    return r


if __name__ == '__main__':
    main()
