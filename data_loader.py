import tensorflow as tf
import numpy as np
import os


class dataloader():
    
    def __init__(self, args):
        
        self.mode = args.mode
        self.patch_size = args.patch_size
        self.batch_size = args.batch_size
        self.train_GT_path = args.train_GT_path
        self.train_LR_path = args.train_LR_path
        self.test_GT_path = args.test_GT_path
        self.test_LR_path = args.test_LR_path
        self.scale = args.scale
        self.test_with_train = args.test_with_train
        self.test_batch = args.test_batch
        self.channel = args.channel
        
    def build_loader(self):
        
        if self.mode == 'train':
        
            tr_gt_imgs = sorted(os.listdir(self.train_GT_path))
            tr_lr_imgs = sorted(os.listdir(self.train_LR_path))
            tr_gt_imgs = [os.path.join(self.train_GT_path, ele) for ele in tr_gt_imgs]
            tr_lr_imgs = [os.path.join(self.train_LR_path, ele) for ele in tr_lr_imgs]
            train_list = (tr_lr_imgs, tr_gt_imgs)
            
            self.tr_dataset = tf.data.Dataset.from_tensor_slices(train_list)
            self.tr_dataset = self.tr_dataset.map(self._parse, num_parallel_calls = 4).prefetch(32)
            self.tr_dataset = self.tr_dataset.map(self._get_patch, num_parallel_calls = 4).prefetch(32)
            self.tr_dataset = self.tr_dataset.map(self._data_augmentation, num_parallel_calls = 4).prefetch(32)
            self.tr_dataset = self.tr_dataset.shuffle(32)
            self.tr_dataset = self.tr_dataset.repeat()
            self.tr_dataset = self.tr_dataset.batch(self.batch_size)
            
            if self.test_with_train:
            
                val_gt_imgs = sorted(os.listdir(self.test_GT_path))
                val_lr_imgs = sorted(os.listdir(self.test_LR_path))
                val_gt_imgs = [os.path.join(self.test_GT_path, ele) for ele in val_gt_imgs]
                val_lr_imgs = [os.path.join(self.test_LR_path, ele) for ele in val_lr_imgs]
                valid_list = (val_lr_imgs, val_gt_imgs)

                self.val_dataset = tf.data.Dataset.from_tensor_slices(valid_list)
                self.val_dataset = self.val_dataset.map(self._parse, num_parallel_calls = 4).prefetch(32)
                self.val_dataset = self.val_dataset.batch(self.test_batch)

            iterator = tf.data.Iterator.from_structure(self.tr_dataset.output_types, self.tr_dataset.output_shapes)
            self.next_batch = iterator.get_next()
            self.init_op = {}
            self.init_op['tr_init'] = iterator.make_initializer(self.tr_dataset)
            
            if self.test_with_train:
                self.init_op['val_init'] = iterator.make_initializer(self.val_dataset)
            
        elif self.mode == 'test':
            
            val_gt_imgs = sorted(os.listdir(self.test_GT_path))
            val_lr_imgs = sorted(os.listdir(self.test_LR_path))
            val_gt_imgs = [os.path.join(self.test_GT_path, ele) for ele in val_gt_imgs]
            val_lr_imgs = [os.path.join(self.test_LR_path, ele) for ele in val_lr_imgs]
            valid_list = (val_lr_imgs, val_gt_imgs)
            
            self.val_dataset = tf.data.Dataset.from_tensor_slices(valid_list)
            self.val_dataset = self.val_dataset.map(self._parse, num_parallel_calls = 4).prefetch(32)
            self.val_dataset = self.val_dataset.batch(1)
            
            iterator = tf.data.Iterator.from_structure(self.val_dataset.output_types, self.val_dataset.output_shapes)
            self.next_batch = iterator.get_next()
            self.init_op = {}
            self.init_op['val_init'] = iterator.make_initializer(self.val_dataset)
        
        elif self.mode == 'test_only':
            
            lr_imgs = sorted(os.listdir(self.test_LR_path))
            lr_imgs = [os.path.join(self.test_LR_path, ele) for ele in lr_imgs]
            
            self.te_dataset = tf.data.Dataset.from_tensor_slices(lr_imgs)
            self.te_dataset = self.te_dataset.map(self._parse_LR_only, num_parallel_calls = 4).prefetch(32)
            self.te_dataset = self.te_dataset.batch(1)
            
            iterator = tf.data.Iterator.from_structure(self.te_dataset.output_types, self.te_dataset.output_shapes)
            self.next_batch = iterator.get_next()
            self.init_op = {}
            self.init_op['te_init'] = iterator.make_initializer(self.te_dataset)
            
            
    def _parse(self, image_LR, image_HR):
        
        image_LR = tf.read_file(image_LR)
        image_HR = tf.read_file(image_HR)
        
        image_LR = tf.image.decode_image(image_LR, channels = self.channel)
        image_HR = tf.image.decode_image(image_HR, channels = self.channel)
        
        image_LR = tf.cast(image_LR, tf.float32)
        image_HR = tf.cast(image_HR, tf.float32)
        
        return image_LR, image_HR
    
    def _parse_LR_only(self, image_LR):
        
        image_LR = tf.read_file(image_LR)
        image_LR = tf.image.decode_image(image_LR, channels = self.channel)
        image_LR = tf.cast(image_LR, tf.float32)
        
        return image_LR
        
    def _get_patch(self, image_LR, image_HR):
        
        shape = tf.shape(image_LR)
        ih = shape[0]
        iw = shape[1]
        
        ix = tf.random_uniform(shape = [1], minval = 0, maxval = iw - self.patch_size + 1, dtype = tf.int32)[0]
        iy = tf.random_uniform(shape = [1], minval = 0, maxval = ih - self.patch_size + 1, dtype = tf.int32)[0]
        
        tx = ix * self.scale
        ty = iy * self.scale
        
        img_hr_in = image_HR[ty : ty + (self.scale * self.patch_size), tx : tx + (self.scale * self.patch_size)]        
        img_lr_in = image_LR[iy:iy + self.patch_size, ix:ix + self.patch_size]
        
        return img_lr_in, img_hr_in
    
    def _data_augmentation(self, image_LR, image_HR):
        
        rot = tf.random_uniform(shape = [1], minval = 0, maxval = 3, dtype = tf.int32)[0]
        flip_rl = tf.random_uniform(shape = [1], minval = 0, maxval = 3, dtype = tf.int32)[0]
        flip_updown = tf.random_uniform(shape = [1], minval = 0, maxval = 3, dtype = tf.int32)[0]
        
        image_LR = tf.image.rot90(image_LR, rot)
        image_HR = tf.image.rot90(image_HR, rot)
        
        rl = tf.equal(tf.mod(flip_rl, 2),0)
        ud = tf.equal(tf.mod(flip_updown, 2),0)
        
        image_LR = tf.cond(rl, true_fn = lambda : tf.image.flip_left_right(image_LR), false_fn = lambda : (image_LR))
        image_HR = tf.cond(rl, true_fn = lambda : tf.image.flip_left_right(image_HR), false_fn = lambda : (image_HR))
        
        image_LR = tf.cond(ud, true_fn = lambda : tf.image.flip_up_down(image_LR), false_fn = lambda : (image_LR))
        image_HR = tf.cond(ud, true_fn = lambda : tf.image.flip_up_down(image_HR), false_fn = lambda : (image_HR))
        
        return image_LR, image_HR

