#    Adapted from nnUNet
#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

from collections import OrderedDict
import numpy as np
from multiprocessing import Pool

from batchgenerators.dataloading.data_loader import SlimDataLoaderBase

default_num_threads = 4
# from configuration import preprocessing_output_dir
from batchgenerators.utilities.file_and_folder_operations import *


def get_case_identifiers(folder):
    case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz") and (i.find("segFromPrevStage") == -1)]
    return case_identifiers


def get_case_identifiers_from_raw_folder(folder):
    case_identifiers = np.unique(
        [i[:-12] for i in os.listdir(folder) if i.endswith(".nii.gz") and (i.find("segFromPrevStage") == -1)])
    return case_identifiers


def convert_to_npy(args):
    if not isinstance(args, tuple):
        key = "data"
        npz_file = args
    else:
        npz_file, key = args
    if not isfile(npz_file[:-3] + "npy"):
        a = np.load(npz_file)[key]
        np.save(npz_file[:-3] + "npy", a)


def save_as_npz(args):
    if not isinstance(args, tuple):
        key = "data"
        npy_file = args
    else:
        npy_file, key = args
    d = np.load(npy_file)
    np.savez_compressed(npy_file[:-3] + "npz", **{key: d})


def unpack_dataset(folder, threads=default_num_threads, key="data"):
    """
    Unzip the npz file in the folder to an npy file (what needs to be unzip must be saved in the field with the key name key)
    :param folder:
    :param threads:
    :param key:
    :return:
    """
    p = Pool(threads)
    npz_files = subfiles(folder, True, None, ".npz", True)
    p.map(convert_to_npy, zip(npz_files, [key] * len(npz_files)))
    p.close()
    p.join()


def pack_dataset(folder, threads=default_num_threads, key="data"):
    p = Pool(threads)
    npy_files = subfiles(folder, True, None, ".npy", True)
    p.map(save_as_npz, zip(npy_files, [key] * len(npy_files)))
    p.close()
    p.join()


def delete_npy(folder):
    case_identifiers = get_case_identifiers(folder)
    npy_files = [join(folder, i + ".npy") for i in case_identifiers]
    npy_files = [i for i in npy_files if isfile(i)]
    for n in npy_files:
        os.remove(n)


def load_dataset(folder, num_cases_properties_loading_threshold=1000):
    print('loading dataset')
    case_identifiers = get_case_identifiers(folder)
    case_identifiers.sort()
    dataset = OrderedDict()
    for c in case_identifiers:
        dataset[c] = OrderedDict()
        dataset[c]['data_file'] = join(folder, "%s.npz" % c)

        # dataset[c]['properties'] = load_pickle(join(folder, "%s.pkl" % c))
        dataset[c]['properties_file'] = join(folder, "%s.pkl" % c)

        if dataset[c].get('seg_from_prev_stage_file') is not None:
            dataset[c]['seg_from_prev_stage_file'] = join(folder, "%s_segs.npz" % c)

    if len(case_identifiers) <= num_cases_properties_loading_threshold:
        print('loading all case properties')
        for i in dataset.keys():
            dataset[i]['properties'] = load_pickle(dataset[i]['properties_file'])

    return dataset


def crop_2D_image_force_fg(img, crop_size, valid_voxels):
    """
    img must be [c, x, y]
    img[-1] must be the segmentation with segmentation>0 being foreground
    :param img:
    :param crop_size:
    :param valid_voxels: voxels belonging to the selected class
    :return:
    """
    assert len(valid_voxels.shape) == 2

    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * (len(img.shape) - 1)
    else:
        assert len(crop_size) == (len(
            img.shape) - 1), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"

    lb_x = crop_size[0] // 2
    ub_x = img.shape[1] - crop_size[0] // 2 - crop_size[0] % 2
    lb_y = crop_size[1] // 2
    ub_y = img.shape[2] - crop_size[1] // 2 - crop_size[1] % 2

    if len(valid_voxels) == 0:
        selected_center_voxel = (np.random.random_integers(lb_x, ub_x),
                                 np.random.random_integers(lb_y, ub_y))
    else:
        selected_center_voxel = valid_voxels[np.random.choice(valid_voxels.shape[1]), :]

    selected_center_voxel = np.array(selected_center_voxel)
    for i in range(2):
        selected_center_voxel[i] = max(crop_size[i] // 2, selected_center_voxel[i])
        selected_center_voxel[i] = min(img.shape[i + 1] - crop_size[i] // 2 - crop_size[i] % 2,
                                       selected_center_voxel[i])

    result = img[:, (selected_center_voxel[0] - crop_size[0] // 2):(
            selected_center_voxel[0] + crop_size[0] // 2 + crop_size[0] % 2),
             (selected_center_voxel[1] - crop_size[1] // 2):(
                     selected_center_voxel[1] + crop_size[1] // 2 + crop_size[1] % 2)]
    return result


class DataLoader3D(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, has_prev_stage=False,
                 oversample_foreground_percent=0.0, memmap_mode="r", pad_mode="edge", pad_kwargs_data=None,
                 pad_sides=None, redefine_clip=None):
        """
        This is the basic data loader for 3D networks, using pre-processed data. The data can be loaded using load_dataset (floder), where the folder is the path to the folder containing the npz file. if
        This folder contains only npz files and the data loader will unpack them at runtime. This process takes some time and CPU utilization, so it is recommended to call unpack_dataset to unpack all npz files first.
        After using the training, remember to call delete_npy to delete the npy file. (Because all the data stored in npy will consume 1TB of storage)
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param random: Sample keys randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param oversample_foreground: half the batch will be forced to contain at least some foreground (equal prob for each of the foreground classes)
        """
        super(DataLoader3D, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.has_prev_stage = has_prev_stage
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        # need_to_pad indicates how much padding we need to do with the data so that if patches of size final_patch_size are sampled (the network will get patches), these patches will cover the patient boundaries
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.memmap_mode = memmap_mode
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def determine_shapes(self):
        if self.has_prev_stage:
            num_seg = 2
        else:
            num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - 1
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)
        case_properties = []
        for j, i in enumerate(selected_keys):
            # Oversampling will improve model training stability, especially when there are more empty patches (such as lungs).
            if self.get_do_oversample(j):
                force_fg = True
            else:
                force_fg = False

            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                properties = load_pickle(self._data[i]['properties_file'])
            case_properties.append(properties)

            # cases are npz files, but you need to run unpack_dataset to unpack them into npy files for faster access
            if isfile(self._data[i]['data_file'][:-4] + ".npy"):
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)
            else:
                case_all_data = np.load(self._data[i]['data_file'])['data']

            # If you are doing a cascade operation, you will also need to load the split results of the previous stages and concatenate them with the current split results. Here we'll connect it to the split result,
            # Because enhancement operations need to be applied to them in split mode. In the subsequent process of data enhancement, we move it from the segmentation result to the last channel of the data.
            if self.has_prev_stage:
                if isfile(self._data[i]['seg_from_prev_stage_file'][:-4] + ".npy"):
                    segs_from_previous_stage = np.load(self._data[i]['seg_from_prev_stage_file'][:-4] + ".npy",
                                                       mmap_mode=self.memmap_mode)[None]
                else:
                    segs_from_previous_stage = np.load(self._data[i]['seg_from_prev_stage_file'])['data'][None]
                # Sampling from multiple previous segmentation results is theoretically supported, but in practice it is not used, so there is always only one segmentation result
                seg_key = np.random.choice(segs_from_previous_stage.shape[0])
                seg_from_previous_stage = segs_from_previous_stage[seg_key:seg_key + 1]
                assert all([i == j for i, j in zip(seg_from_previous_stage.shape[1:], case_all_data.shape[1:])]), \
                    "seg_from_previous_stage does not match the shape of case_all_data: %s vs %s" % \
                    (str(seg_from_previous_stage.shape[1:]), str(case_all_data.shape[1:]))
            else:
                seg_from_previous_stage = None

            # do you trust me? You better do. Otherwise you'll have to go through this mess and honestly there are
            # better things you could do right now

            # (above) documentation of the day. Nice. Even myself coming back 1 months later I have not friggin idea
            # what's going on. I keep the above documentation just for fun but attempt to make things clearer now

            need_to_pad = self.need_to_pad
            for d in range(3):
                # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
                # always
                if need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]

            # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
            # define what the upper and lower bound can be to then sample form them with np.random.randint
            shape = case_all_data.shape[1:]
            lb_x = - need_to_pad[0] // 2
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
            lb_y = - need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]
            lb_z = - need_to_pad[2] // 2
            ub_z = shape[2] + need_to_pad[2] // 2 + need_to_pad[2] % 2 - self.patch_size[2]

            # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
            # at least one of the foreground classes in the patch
            if not force_fg:
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                bbox_z_lb = np.random.randint(lb_z, ub_z + 1)
            else:
                # these values should have been precomputed
                if 'class_locations' not in properties.keys():
                    raise RuntimeError("Please rerun the preprocessing with the newest version of nnU-Net!")

                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                foreground_classes = np.array(
                    [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) != 0])
                foreground_classes = foreground_classes[foreground_classes > 0]

                if len(foreground_classes) == 0:
                    # this only happens if some image does not contain foreground voxels at all
                    selected_class = None
                    voxels_of_that_class = None
                    print('case does not contain any foreground classes', i)
                else:
                    selected_class = np.random.choice(foreground_classes)

                    voxels_of_that_class = properties['class_locations'][selected_class]

                if voxels_of_that_class is not None:
                    selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                    # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                    # Make sure it is within the bounds of lb and ub
                    bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                    bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)
                    bbox_z_lb = max(lb_z, selected_voxel[2] - self.patch_size[2] // 2)
                else:
                    # If the image does not contain any foreground classes, we fall back to random cropping
                    bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                    bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                    bbox_z_lb = np.random.randint(lb_z, ub_z + 1)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]
            bbox_z_ub = bbox_z_lb + self.patch_size[2]

            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)
            valid_bbox_z_lb = max(0, bbox_z_lb)
            valid_bbox_z_ub = min(shape[2], bbox_z_ub)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            case_all_data = np.copy(case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                    valid_bbox_y_lb:valid_bbox_y_ub,
                                    valid_bbox_z_lb:valid_bbox_z_ub])
            if seg_from_previous_stage is not None:
                seg_from_previous_stage = seg_from_previous_stage[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                          valid_bbox_y_lb:valid_bbox_y_ub,
                                          valid_bbox_z_lb:valid_bbox_z_ub]

            data[j] = np.pad(case_all_data[:-1], ((0, 0),
                                                  (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                  (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                  (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                             self.pad_mode, **self.pad_kwargs_data)

            seg[j, 0] = np.pad(case_all_data[-1:], ((0, 0),
                                                    (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                    (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                    (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                               'constant', **{'constant_values': -1})
            if seg_from_previous_stage is not None:
                seg[j, 1] = np.pad(seg_from_previous_stage, ((0, 0),
                                                             (-min(0, bbox_x_lb),
                                                              max(bbox_x_ub - shape[0], 0)),
                                                             (-min(0, bbox_y_lb),
                                                              max(bbox_y_ub - shape[1], 0)),
                                                             (-min(0, bbox_z_lb),
                                                              max(bbox_z_ub - shape[2], 0))),
                                   'constant', **{'constant_values': 0})

        return {'data': data, 'seg': seg, 'properties': case_properties, 'keys': selected_keys}


class DataLoader2D(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoader2D, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)

        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)

        case_properties = []
        for j, i in enumerate(selected_keys):
            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                properties = load_pickle(self._data[i]['properties_file'])
            case_properties.append(properties)

            if self.get_do_oversample(j):
                force_fg = True
            else:
                force_fg = False

            if not isfile(self._data[i]['data_file'][:-4] + ".npy"):
                # lets hope you know what you're doing
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npz")['data']
            else:
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)

            # this is for when there is just a 2d slice in case_all_data (2d support)
            if len(case_all_data.shape) == 3:
                case_all_data = case_all_data[:, None]

            # first select a slice. This can be either random (no force fg) or guaranteed to contain some class
            if not force_fg:
                random_slice = np.random.choice(case_all_data.shape[1])
                selected_class = None
            else:
                # these values should have been precomputed
                if 'class_locations' not in properties.keys():
                    raise RuntimeError("Please rerun the preprocessing with the newest version of nnU-Net!")

                foreground_classes = np.array(
                    [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) != 0])
                foreground_classes = foreground_classes[foreground_classes > 0]
                if len(foreground_classes) == 0:
                    selected_class = None
                    random_slice = np.random.choice(case_all_data.shape[1])
                    print('case does not contain any foreground classes', i)
                else:
                    selected_class = np.random.choice(foreground_classes)

                    voxels_of_that_class = properties['class_locations'][selected_class]
                    valid_slices = np.unique(voxels_of_that_class[:, 0])
                    random_slice = np.random.choice(valid_slices)
                    voxels_of_that_class = voxels_of_that_class[voxels_of_that_class[:, 0] == random_slice]
                    voxels_of_that_class = voxels_of_that_class[:, 1:]

            # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
            # below the current slice, here is where we get them. We stack those as additional color channels
            if self.pseudo_3d_slices == 1:
                case_all_data = case_all_data[:, random_slice]
            else:
                # this is very deprecated and will probably not work anymore. If you intend to use this you need to
                # check this!
                mn = random_slice - (self.pseudo_3d_slices - 1) // 2
                mx = random_slice + (self.pseudo_3d_slices - 1) // 2 + 1
                valid_mn = max(mn, 0)
                valid_mx = min(mx, case_all_data.shape[1])
                case_all_seg = case_all_data[-1:]
                case_all_data = case_all_data[:-1]
                case_all_data = case_all_data[:, valid_mn:valid_mx]
                case_all_seg = case_all_seg[:, random_slice]
                need_to_pad_below = valid_mn - mn
                need_to_pad_above = mx - valid_mx
                if need_to_pad_below > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_below
                    case_all_data = np.concatenate((np.zeros(shp_for_pad), case_all_data), 1)
                if need_to_pad_above > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_above
                    case_all_data = np.concatenate((case_all_data, np.zeros(shp_for_pad)), 1)
                case_all_data = case_all_data.reshape((-1, case_all_data.shape[-2], case_all_data.shape[-1]))
                case_all_data = np.concatenate((case_all_data, case_all_seg), 0)

            # case all data should now be (c, x, y)
            assert len(case_all_data.shape) == 3

            # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
            # define what the upper and lower bound can be to then sample form them with np.random.randint

            need_to_pad = self.need_to_pad
            for d in range(2):
                # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
                # always
                if need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]

            shape = case_all_data.shape[1:]
            lb_x = - need_to_pad[0] // 2
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
            lb_y = - need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]

            # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
            # at least one of the foreground classes in the patch
            if not force_fg or selected_class is None:
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
            else:
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)

            case_all_data = case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                            valid_bbox_y_lb:valid_bbox_y_ub]

            case_all_data_donly = np.pad(case_all_data[:-1], ((0, 0),
                                                              (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                              (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                         self.pad_mode, **self.pad_kwargs_data)

            case_all_data_segonly = np.pad(case_all_data[-1:], ((0, 0),
                                                                (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                                (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                           'constant', **{'constant_values': -1})

            data[j] = case_all_data_donly
            seg[j] = case_all_data_segonly

        keys = selected_keys
        return {'data': data, 'seg': seg, 'properties': case_properties, "keys": keys}
