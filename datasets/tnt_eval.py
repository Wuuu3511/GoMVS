from torch.utils.data import Dataset
import numpy as np
import os, cv2, time
from PIL import Image
from datasets.data_io import *

s_h, s_w = 0, 0


class MVSDataset(Dataset):
    def __init__(self, datapath, normalpath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, inverse_depth=False,
                 **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.normalpath=normalpath
        self.max_h, self.max_w = kwargs["max_h"], kwargs["max_w"]
        self.fix_res = kwargs.get("fix_res", False)  # whether to fix the resolution of input image.
        self.fix_wh = False
        self.inverse_depth = inverse_depth

        assert self.mode == "test"

        self.image_sizes = {'Family': (1920, 960),
                            'Francis': (1920, 960),
                            'Horse': (1920, 960),
                            'Lighthouse': (1920, 960), #
                            'M60': (1920, 960),# 
                            'Panther': (1920, 960),
                            'Playground': (1920, 960),
                            'Train': (1920, 960),
                            
                            'Auditorium': (1920, 960),
                            'Ballroom': (1920, 960),
                            'Courtroom': (1920, 960),
                            'Museum': (1920, 960),
                            'Palace': (1920, 960),
                            'Temple': (1920, 960)}
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        scans = self.listfile

        interval_scale_dict = {}
        # scans
        for scan in scans:
            # determine the interval scale of each scene. default is 1.06
            if isinstance(self.interval_scale, float):
                interval_scale_dict[scan] = self.interval_scale
            else:
                interval_scale_dict[scan] = self.interval_scale[scan]

            pair_file = "{}/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # filter by no src view and fill to nviews
                    if len(src_views) > 0:
                        if len(src_views) < self.nviews - 1:
                            print("{}< src num_views:{}".format(len(src_views), self.nviews))
                            src_views += [src_views[0]] * (self.nviews - len(src_views))
                        metas.append((scan, ref_view, src_views, scan))

        self.interval_scale = interval_scale_dict
        print("dataset", self.mode, "metas:", len(metas), "interval_scale:{}".format(self.interval_scale))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename, interval_scale):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        intrinsics[:2, :] /= 4.0
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1])

        if len(lines[11].split()) >= 3:
            num_depth = lines[11].split()[2]
            depth_max = depth_min + int(float(num_depth)) * depth_interval
            depth_interval = (depth_max - depth_min) / self.ndepths

        depth_interval *= interval_scale

        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.

        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def scale_mvs_input(self, img, intrinsics, max_w, max_h, base=32):
        h, w = img.shape[:2]
        # if h > max_h or w > max_w:
        #     scale = 1.0 * max_h / h
        #     if scale * w > max_w:
        #         scale = 1.0 * max_w / w
        #     new_w, new_h = scale * w // base * base, scale * h // base * base
        # else:
        #     new_w, new_h = 1.0 * w // base * base, 1.0 * h // base * base

        scale_w = 1.0 * max_w / w
        scale_h = 1.0 * max_h / h
        intrinsics[0, :] *= scale_w
        intrinsics[1, :] *= scale_h

        img = cv2.resize(img, (int(max_w), int(max_h)))

        return img, intrinsics

    def read_normal(self, filename, w, h):
        normal = np.load(filename)#3 h w
        normal = (normal - 0.5)*2

        return normal

    def __getitem__(self, idx):
        global s_h, s_w
        meta = self.metas[idx]
        scan, ref_view, src_views, scene_name = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, '{}/images_post/{:0>8}.jpg'.format(scan, vid))
            if not os.path.exists(img_filename):
                img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.jpg'.format(scan, vid))

            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))



            img = self.read_img(img_filename)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename, interval_scale=
            self.interval_scale[scene_name])
            # scale input
            img, intrinsics = self.scale_mvs_input(img, intrinsics, self.image_sizes[scan][0],
                                                   self.image_sizes[scan][1])

            if self.fix_res:
                # using the same standard height or width in entire scene.
                s_h, s_w = img.shape[:2]
                self.fix_res = False
                self.fix_wh = True

            if i == 0:
                if not self.fix_wh:
                    # using the same standard height or width in each nviews.
                    s_h, s_w = img.shape[:2]

            # resize to standard height or width
            c_h, c_w = img.shape[:2]
            if (c_h != s_h) or (c_w != s_w):
                scale_h = 1.0 * s_h / c_h
                scale_w = 1.0 * s_w / c_w
                img = cv2.resize(img, (s_w, s_h))
                intrinsics[0, :] *= scale_w
                intrinsics[1, :] *= scale_h

            imgs.append(img)
            # extrinsics, intrinsics
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                normal_path = os.path.join(self.normalpath, '{}/{:0>6}_normal.npy'.format(scan, vid))
                normal = self.read_normal(normal_path, self.image_sizes[scan][0], self.image_sizes[scan][1])
                if self.inverse_depth:
                    depth_end = depth_interval * self.ndepths + depth_min
                    depth_values = np.linspace(1.0 / depth_min, 1.0 / depth_end, self.ndepths, endpoint=False)
                    depth_values = (1.0 / depth_values).astype(np.float32)
                else:
                    depth_values = np.arange(depth_min, depth_interval * (self.ndepths - 0.5) + depth_min,
                                             depth_interval,
                                             dtype=np.float32)

        # all
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4

        proj_matrices_ms = {
            "stage1": proj_matrices,
            "stage2": stage2_pjmats,
            "stage3": stage3_pjmats
        }

        # img_mean = np.array([0.39405544, 0.33657743, 0.29420473], dtype=np.float32).reshape((1, -1, 1, 1))
        # img_std = np.array([0.20861072, 0.19565384, 0.1870348], dtype=np.float32).reshape((1, -1, 1, 1))
        # imgs = (imgs - img_mean) / img_std

        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth_values": depth_values,
                "normal": normal,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}
