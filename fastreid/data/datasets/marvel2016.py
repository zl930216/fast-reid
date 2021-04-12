# encoding: utf-8
"""
marvel2016 dataset, same pattern as market1501
"""

import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Marvel2016(ImageDataset):
    """Marvel2016.

    Reference:
        Gundogdu E., Solmaz B, Yucesoy V., Koc A.,
        Marvel: A Large-Scale Image Dataset for Maritime Vessels, Asian Conference on Computer Vision (ACCV), 2016

    URL: `<https://github.com/avaapm/marveldataset2016>`_

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 6257 (train) + 2706 (query) + 8083 (gallery).
    """

    dataset_dir = "marvel2016"
    dataset_name = "marvel2016"

    def __init__(self, root="datasets", **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, "train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "test")

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        super(Marvel2016, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, "*.jpg"))
        pattern = re.compile(r"([-\d]+)_c(\d{1,3})")

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1936  # pid == 0 means background
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data