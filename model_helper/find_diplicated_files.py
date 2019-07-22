import os
import hashlib
from tqdm import tqdm


# Author: https://gist.github.com/AGulev/d5dc12127e0fbe1cd4f239effb76cd81


def hashfile(path, blocksize = 65536):
    afile = open(path, 'rb')
    hasher = hashlib.md5()
    buf = afile.read(blocksize)
    while len(buf) > 0:
        hasher.update(buf)
        buf = afile.read(blocksize)
    afile.close()
    return hasher.hexdigest()


def findDup(parentFolder):
    # Dups in format {hash:[names]}
    dups = {}
    for dirName, subdirs, fileList in os.walk(parentFolder):
        print('Scanning %s...' % dirName)
        for filename in tqdm(fileList):
            # Get the path to the file
            path = os.path.join(dirName, filename)
            # Calculate hash
            file_hash = hashfile(path)
            # Add or append the file path
            if file_hash in dups:
                dups[file_hash].append(path)
            else:
                dups[file_hash] = [path]
    return dups


# dups = findDup('../input/diabetic-retinopathy-resized/resized_train/resized_train/')
# dups = findDup('../input/aptos2019-blindness-detection/train_images/')
# len([v for k, v in dups.items() if len(v) > 1])