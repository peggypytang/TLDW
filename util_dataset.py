import torch.utils.data
from os.path import join, exists
import re, json, cv2, os, sys, glob
import xml.etree.ElementTree as ET
import random, itertools
from PIL import Image
import numpy as np

  # Standard PySceneDetect imports:
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager

# For content-aware scene detection:
from scenedetect.detectors.content_detector import ContentDetector

class MultimodalDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, split: str, path: str):
        self._data_path = join(path, split)
        self._n_data = _count_data(self._data_path)

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i: str):
        
        with open(join(self._data_path, '{}.json'.format(i))) as f:
            js = json.loads(f.read())

        original_frames = []
        vidcap = cv2.VideoCapture(join(self._data_path, '{}.mp4'.format(i)))
        success,image = vidcap.read()
        count = 0
        while success:
            if count % 480:
                original_frames.append(torch.tensor(cv2.resize(image, (640, 360))))
            success, image = vidcap.read()
            count += 1
            if count > 100: #reach cuda limit
                break
        # Stack it into a tensor
        video = torch.stack(original_frames, 0)

        thumbnail = cv2.imread(join(self._data_path, '{}.png'.format(i)))
        
        transcript = ET.parse(join(self._data_path, '{} (a.en).xml'.format(i))).getroot()


        return js, thumbnail, transcript, video

class EXMSMODataset(MultimodalDataset):
    """ single article sentence -> single abstract sentence
    (dataset created by greedily matching ROUGE)
    """

    def __init__(self, split, DATA_DIR):
        super().__init__(split, DATA_DIR)
        files = glob.glob(join(self._data_path, "*.json"))
        self.file_id = [os.path.split(x)[1].replace('.json', '') for x in files]

    def __getitem__(self, i):
        js, thumbnail, transcript_xml, video = super().__getitem__(self.file_id[i])

        transcript = []
        for w in transcript_xml:
            transcript.append(w.text)
        transcripts = '; '.join(transcript).replace('&#39;', '\'')

        title = js['title']
        description= js['description']

        return  self.file_id[i], description, video, title, thumbnail, transcripts



class MultimodalNoTruncateDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, split: str, path: str):
        self._data_path = join(path, split)
        self._n_data = _count_data(self._data_path)

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i: str):

        with open(join(self._data_path, '{}.json'.format(i))) as f:
            js = json.loads(f.read())

        original_frames = []
        vidcap = cv2.VideoCapture(join(self._data_path, '{}.mp4'.format(i)))
        vframe = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        vfps    = vidcap.get(cv2.CAP_PROP_FPS)
        vs    = vframe/vfps

        
        thumbnail = cv2.imread(join(self._data_path, '{}.png'.format(i)))
        
        transcript = ET.parse(join(self._data_path, '{} (a.en).xml'.format(i))).getroot()


        return js, thumbnail, transcript, vframe, vfps, vs


class EXMSMONoTruncateDataset(MultimodalNoTruncateDataset):
    """ single article sentence -> single abstract sentence
    (dataset created by greedily matching ROUGE)
    """

    def __init__(self, split, DATA_DIR):
        super().__init__(split, DATA_DIR)
        files = glob.glob(join(self._data_path, "*.json"))
        self.file_id = [os.path.split(x)[1].replace('.json', '') for x in files]

    def __getitem__(self, i):
        js, thumbnail, transcript_xml, vframe, vfps, vs = super().__getitem__(self.file_id[i])

        transcript = []
        for w in transcript_xml:
            transcript.append(w.text)
        transcripts = '; '.join(transcript).replace('&#39;', '\'')

        title = js['title']
        description= js['description']

        return  self.file_id[i], description, vframe, title, vfps, vs, thumbnail, transcripts



class MultimodalWithSceneDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, split: str, path: str):
        self._data_path = join(path, split)
        self._n_data = _count_data(self._data_path)

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i: str):

        with open(join(self._data_path, '{}.json'.format(i))) as f:
            js = json.loads(f.read())

        original_frames = []
        vidcap = cv2.VideoCapture(join(self._data_path, '{}.mp4'.format(i)))
        success,image = vidcap.read()
        count = 0
        while success:
            if count % 360:
                original_frames.append(torch.tensor(cv2.resize(image, (640, 360))))
            success, image = vidcap.read()
            count += 1
            if count > 100: #reach cuda limit
                break
        # Stack it into a tensor
        video = torch.stack(original_frames, 0)


        scene_list = find_scenes(join(self._data_path, '{}.mp4'.format(i)))
        thumbnail = cv2.imread(join(self._data_path, '{}.png'.format(i)))
        
        transcript = ET.parse(join(self._data_path, '{} (a.en).xml'.format(i))).getroot()

        return js, thumbnail, transcript, video, scene_list

class EXMSMOWithSceneDataset(MultimodalWithSceneDataset):
    """ single article sentence -> single abstract sentence
    (dataset created by greedily matching ROUGE)
    """

    def __init__(self, split, DATA_DIR):
        super().__init__(split, DATA_DIR)
        files = glob.glob(join(self._data_path, "*.json"))
        self.file_id = [os.path.split(x)[1].replace('.json', '') for x in files]

    def __getitem__(self, i):
        js, thumbnail, transcript_xml, video, scene_list = super().__getitem__(self.file_id[i])

        transcript = []
        for w in transcript_xml:
            transcript.append(w.text)
        transcripts = '; '.join(transcript).replace('&#39;', '\'')

        title = js['title']
        description= js['description']

        return  self.file_id[i], description, video, title, thumbnail, transcripts, scene_list


def find_scenes(video_path, threshold=80.0):
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()
    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager, frame_skip=360)
    # Each returned scene is a tuple of the (start, end) timecode.
    scene_list = scene_manager.get_scene_list()

    scene_frame_list = []
    for i, scene in enumerate(scene_list):
        scene_frame_list.append(int(scene[1].get_frames()/360))

    return torch.Tensor(scene_frame_list)




def _count_data(path):
    """ count number of data in the given path"""
    files = glob.glob(join(path, "*.json"))
    n_data = len(files)
    return n_data