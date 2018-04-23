#!/usr/bin/python

import numpy as np
import pickle
import sys
import os

import librosa

SOUND_SAMPLE_LENGTH = 30000

HAMMING_SIZE = 100
HAMMING_STRIDE = 40



def prepossessingAudio(audioPath, ppFilePath):
    print 'Prepossessing ' + audioPath

    featuresArray = []
    for i in range(0, SOUND_SAMPLE_LENGTH, HAMMING_STRIDE):
        if i + HAMMING_SIZE <= SOUND_SAMPLE_LENGTH - 1:
            y, sr = librosa.load(audioPath, offset=i / 1000.0, duration=HAMMING_SIZE / 1000.0)

            # Let's make and display a mel-scaled power (energy-squared) spectrogram
            S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

            featuresArray.append(S)

            if len(featuresArray) == 599:
                break

    print 'storing pp file: ' + ppFilePath

    f = open(ppFilePath, 'w')
    f.write(pickle.dumps(featuresArray))
    f.close()


if __name__ == "__main__":

    # help menu
    if len(sys.argv) < 2:
        die_with_usage()

    i = 0.0
    walk_dir = sys.argv[1]

    print('walk_dir = ' + walk_dir)

    for root, subdirs, files in os.walk(walk_dir):
        for filename in files:
            if filename.endswith('.au'):
                file_path = os.path.join(root, filename)

                ppFileName = rreplace(file_path, ".au", ".pp", 1)

                try:
                    prepossessingAudio(file_path, ppFileName)
                except Exception as e:
                    print "Error accured" + str(e)

            if filename.endswith('au'):
                sys.stdout.write("\r%d%%" % int(i / 7620 * 100))
                sys.stdout.flush()
                i += 1
