
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from vot_path import base_path
import cv2
import os
import torch
import numpy as np
import math
import tensorflow as tf
import sys
import time
import copy
sys.path.append(os.path.join(base_path, 'CoCoLoT/meta_updater'))
sys.path.append(os.path.join(base_path, 'CoCoLoT/utils/metric_net'))
from tcNet import tclstm
from tcopt import tcopts
from metric_model import ft_net
from torch.autograd import Variable
from me_sample_generator import *
import vot
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Dimp
import argparse
from pytracking.libs.tensorlist import TensorList
from pytracking.utils.plotting import show_tensor
from pytracking.features.preprocessing import numpy_to_torch
env_path = os.path.join(os.path.dirname(__file__))
if env_path not in sys.path:
    sys.path.append(env_path)
from pytracking.evaluation import Tracker as pyTracker

from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = 1000000000
from tracking_utils import compute_iou, show_res, process_regions


sys.path.append(os.path.join(base_path, 'CoCoLoT/pyMDNet/modules'))
sys.path.append(os.path.join(base_path, 'CoCoLoT/pyMDNet/tracking'))
# pymdnet
from pyMDNet.modules.model import *
sys.path.insert(0, './pyMDNet')
from pyMDNet.modules.model import MDNet, BCELoss, set_optimizer
from pyMDNet.modules.sample_generator import SampleGenerator
from pyMDNet.modules.utils import overlap_ratio
from pyMDNet.tracking.data_prov import RegionExtractor
from pyMDNet.tracking.run_tracker import *
from bbreg import BBRegressor
from gen_config import gen_config
opts = yaml.safe_load(open(os.path.join(base_path, 'CoCoLoT/pyMDNet/tracking/options.yaml'), 'r'))

from Stark.lib.test.vot20.stark_vot20lt import *

import random 

SEED = 150
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

#tf.compat.v1.reset_default_graph()

class p_config(object):
    Verification = "rtmdnet"
    name = 'Dimp_MU'
    model_dir = 'dimp_mu_votlt'
    checkpoint = 220000
    start_frame = 200
    R_candidates = 20
    save_results = False
    use_mask = True
    save_training_data = False
    visualization = True


class CoCoLoT_Tracker(object):

    def __init__(self, image, region, p=None, groundtruth=None):
        self.p = p
        self.i = 0
        self.t_id = 0
        if groundtruth is not None:
            self.groundtruth = groundtruth

        tfconfig = tf.compat.v1.ConfigProto()
        tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.3
        self.sess = tf.compat.v1.Session(config=tfconfig)
        init_gt1 = [region.x, region.y, region.width, region.height]
        init_gt = [init_gt1[1], init_gt1[0], init_gt1[1]+init_gt1[3], init_gt1[0]+init_gt1[2]]  # ymin xmin ymax xmax

        self.last_gt = init_gt  # when initialized assumes ground-truth values
        self.init_pymdnet(image, init_gt1)
        self.local_init(image, init_gt1)
         
        

        self.tc_init(self.p.model_dir)
        self.metric_init(image, np.array(init_gt1))
        self.dis_record = []
        self.state_record = []
        self.rv_record = []
        self.all_map = []
        self.count = 0
        local_state1, self.score_map, update, self.score_max, dis, flag = self.local_track(image)
        self.local_Tracker.pos = torch.FloatTensor(
            [(self.last_gt[0] + self.last_gt[2] - 1) / 2, (self.last_gt[1] + self.last_gt[3] - 1) / 2])
        self.local_Tracker.target_sz = torch.FloatTensor(
            [(self.last_gt[2] - self.last_gt[0]), (self.last_gt[3] - self.last_gt[1])])  # height, width

        init_stark = [int(init_gt1[0]), int(init_gt1[1]), int(init_gt1[2]), int(init_gt1[3])]
        self.stark_init(image, init_stark)

        self.boxes = [init_gt1]
        self.boxes_stark = [init_gt1]
        self.boxes_dimp = [init_gt1]
        self.confidences = [1.0]
        self.md_scores_dimp = [1.0]
        self.md_scores_stark = [1.0]
        self.md_scores_sum = [5]

        self.last_good = copy.copy(self.last_gt)
        self.last_good_idx = 0

        self.img_shape = image.shape

        
    def get_first_state(self):
        return self.score_map, self.score_max

    def stark_init(self, image, init_box):
        self.stark_tracker = stark_vot20_lt(tracker_name='stark_st', para_name='baseline')
        self.stark_tracker.initialize(image, init_box)

    def init_pymdnet(self, image, init_bbox):
        target_bbox = np.array(init_bbox)
        self.last_result = target_bbox
        self.pymodel = MDNet(os.path.join(base_path, 'CoCoLoT/pyMDNet/models/mdnet_imagenet_vid.pth'))
        if opts['use_gpu']:
            self.pymodel = self.pymodel.cuda()
        self.pymodel.set_learnable_params(opts['ft_layers'])

        # Init criterion and optimizer
        self.criterion = BCELoss()
        init_optimizer = set_optimizer(self.pymodel, opts['lr_init'], opts['lr_mult'])
        self.update_optimizer = set_optimizer(self.pymodel, opts['lr_update'], opts['lr_mult'])

        tic = time.time()

        # Draw pos/neg samples
        pos_examples = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])(
            target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

        neg_examples = np.concatenate([
            SampleGenerator('uniform', image.size, opts['trans_neg_init'], opts['scale_neg_init'])(
                target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init']),
            SampleGenerator('whole', image.size)(
                target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init'])])
        neg_examples = np.random.permutation(neg_examples)

        # Extract pos/neg features
        pos_feats = forward_samples(self.pymodel, image, pos_examples, opts)
        neg_feats = forward_samples(self.pymodel, image, neg_examples, opts)
        self.feat_dim = pos_feats.size(-1)

        # Initial training
        train(self.pymodel, self.criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'], opts=opts)
        del init_optimizer, neg_feats
        torch.cuda.empty_cache()

        # Train bbox regressor
        bbreg_examples = SampleGenerator('uniform', image.size, opts['trans_bbreg'], opts['scale_bbreg'],
                                         opts['aspect_bbreg'])(
            target_bbox, opts['n_bbreg'], opts['overlap_bbreg'])
        bbreg_feats = forward_samples(self.pymodel, image, bbreg_examples, opts)
        self.bbreg = BBRegressor(image.size)
        self.bbreg.train(bbreg_feats, bbreg_examples, target_bbox)
        del bbreg_feats
        torch.cuda.empty_cache()
        # Init sample generators
        self.sample_generator = SampleGenerator('gaussian', image.size, opts['trans'], opts['scale'])
        self.pos_generator = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])
        self.neg_generator = SampleGenerator('uniform', image.size, opts['trans_neg'], opts['scale_neg'])

        # Init pos/neg features for update
        neg_examples = self.neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_init'])
        neg_feats = forward_samples(self.pymodel, image, neg_examples, opts)
        self.pos_feats_all = [pos_feats]
        self.neg_feats_all = [neg_feats]

        spf_total = time.time() - tic

    def pymdnet_eval(self, image, samples):
        sample_scores = forward_samples(self.pymodel, image, samples, out_layer='fc6', opts=opts)
        return sample_scores[:, 1][:].cpu().numpy()
  
    def collect_samples_pymdnet(self, image):
        self.t_id += 1
        target_bbox = np.array([self.last_gt[1], self.last_gt[0], self.last_gt[3]-self.last_gt[1], self.last_gt[2]-self.last_gt[0]])
        pos_examples = self.pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])
        if len(pos_examples) > 0:
            pos_feats = forward_samples(self.pymodel, image, pos_examples, opts)
            self.pos_feats_all.append(pos_feats)
        if len(self.pos_feats_all) > opts['n_frames_long']:
            del self.pos_feats_all[0]

        neg_examples = self.neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_update'])
        if len(neg_examples) > 0:
            neg_feats = forward_samples(self.pymodel, image, neg_examples, opts)
            self.neg_feats_all.append(neg_feats)
        if len(self.neg_feats_all) > opts['n_frames_short']:
            del self.neg_feats_all[0]

    def pymdnet_short_term_update(self):
        # Short term update
        nframes = min(opts['n_frames_short'], len(self.pos_feats_all))
        pos_data = torch.cat(self.pos_feats_all[-nframes:], 0)
        neg_data = torch.cat(self.neg_feats_all, 0)
        train(self.pymodel, self.criterion, self.update_optimizer, pos_data, neg_data, opts['maxiter_update'],
              opts=opts)

    def pymdnet_long_term_update(self):
        if self.t_id % opts['long_interval'] == 0:
            # Long term update
            pos_data = torch.cat(self.pos_feats_all, 0)
            neg_data = torch.cat(self.neg_feats_all, 0)
            train(self.pymodel, self.criterion, self.update_optimizer, pos_data, neg_data, opts['maxiter_update'],
                  opts=opts)
 
    def metric_init(self, im, init_box):
        self.metric_model = ft_net(class_num=1120)
        path = os.path.join(base_path, 'CoCoLoT/utils/metric_net/metric_model/metric_model.pt')
        self.metric_model.eval()
        self.metric_model = self.metric_model.cuda()
        self.metric_model.load_state_dict(torch.load(path))
        tmp = np.random.rand(1, 3, 107, 107)
        tmp = (Variable(torch.Tensor(tmp))).type(torch.FloatTensor).cuda()
        # get target feature
        self.metric_model(tmp)
        init_box = init_box.reshape((1, 4))
        anchor_region = me_extract_regions(im, init_box)
        anchor_region = process_regions(anchor_region)
        anchor_region = torch.Tensor(anchor_region)
        anchor_region = (Variable(anchor_region)).type(torch.FloatTensor).cuda()
        self.anchor_feature, _ = self.metric_model(anchor_region)

    def metric_eval(self, im, boxes, anchor_feature):
        box_regions = me_extract_regions(np.array(im), boxes)
        box_regions = process_regions(box_regions)
        box_regions = torch.Tensor(box_regions)
        box_regions = (Variable(box_regions)).type(torch.FloatTensor).cuda()
        box_features, class_result = self.metric_model(box_regions)

        class_result = torch.softmax(class_result, dim=1)
        ap_dist = torch.norm(anchor_feature - box_features, 2, dim=1).view(-1)
        return ap_dist

    def tc_init(self, model_dir):
        self.tc_model = tclstm()
        self.X_input = tf.placeholder("float", [None, tcopts['time_steps'], tcopts['lstm_num_input']])
        self.maps = tf.placeholder("float", [None, 19, 19, 1])
        self.map_logits = self.tc_model.map_net(self.maps)
        self.Inputs = tf.concat((self.X_input, self.map_logits), axis=2)
        self.logits, _ = self.tc_model.net(self.Inputs)

        variables_to_restore = [var for var in tf.global_variables() if
                                (var.name.startswith('tclstm') or var.name.startswith('mapnet'))]
        saver = tf.train.Saver(var_list=variables_to_restore)
        if self.p.checkpoint is None:
            checkpoint = tf.train.latest_checkpoint(os.path.join(base_path, 'CoCoLoT/meta_updater', model_dir))
        else:
            checkpoint = os.path.join(base_path, 'CoCoLoT/meta_updater/' + self.p.model_dir + '/lstm_model.ckpt-' + str(self.p.checkpoint))
        saver.restore(self.sess, checkpoint)

    def local_init(self, image, init_bbox):
        local_tracker = pyTracker('dimp', 'super_dimp')
        params = local_tracker.get_parameters()

        debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = local_tracker.name
        params.param_name = local_tracker.parameter_name

        self.local_Tracker = local_tracker.tracker_class(params)
        init_box = dict()
        init_box['init_bbox'] = init_bbox
        self.local_Tracker.initialize(image, init_box)

    def local_track(self, image):
        state, score_map, test_x, scale_ind, sample_pos, sample_scales, flag, s = self.local_Tracker.track_updater(image)
        score_map = cv2.resize(score_map, (19, 19))
        update_flag = flag not in ['not_found', 'uncertain']
        update = update_flag
        max_score = max(score_map.flatten())
        self.all_map.append(score_map)
        local_state = np.array(state).reshape((1, 4))
        ap_dis = self.metric_eval(image, local_state, self.anchor_feature)
        self.dis_record.append(ap_dis.data.cpu().numpy()[0])
        h = image.shape[0]
        w = image.shape[1]
        self.state_record.append([local_state[0][0] / w, local_state[0][1] / h,
                                  (local_state[0][0] + local_state[0][2]) / w,
                                  (local_state[0][1] + local_state[0][3]) / h])
        self.rv_record.append(max_score)
        #if len(self.state_record) >= tcopts['time_steps']:
        if len(self.state_record) >= self.p.start_frame:
            dis = np.array(self.dis_record[-tcopts["time_steps"]:]).reshape((tcopts["time_steps"], 1))
            rv = np.array(self.rv_record[-tcopts["time_steps"]:]).reshape((tcopts["time_steps"], 1))
            state_tc = np.array(self.state_record[-tcopts["time_steps"]:])
            map_input = np.array(self.all_map[-tcopts["time_steps"]:])
            map_input = np.reshape(map_input, [tcopts['time_steps'], 1, 19, 19])
            map_input = map_input.transpose((0, 2, 3, 1))
            X_input = np.concatenate((state_tc, rv, dis), axis=1)
            logits = self.sess.run(self.logits,
                                               feed_dict={self.X_input: np.expand_dims(X_input, axis=0),
                                                          self.maps: map_input})
            update = logits[0][0] < logits[0][1]

        hard_negative = (flag == 'hard_negative')
        learning_rate = getattr(self.local_Tracker.params, 'hard_negative_learning_rate', None) if hard_negative else None

        if update:
            # Get train sample
            train_x = test_x[scale_ind:scale_ind+1, ...]

            # Create target_box and label for spatial sample
            target_box = self.local_Tracker.get_iounet_box(self.local_Tracker.pos, self.local_Tracker.target_sz,
                                                           sample_pos[scale_ind, :], sample_scales[scale_ind])

            # Update the classifier model
            self.local_Tracker.update_classifier(train_x, target_box, learning_rate, s[scale_ind,...])
        self.last_gt = [state[1], state[0], state[1]+state[3], state[0]+state[2]]
        return state, score_map, update, max_score, ap_dis.data.cpu().numpy()[0], flag

    def locate(self, image):

        # Convert image
        im = numpy_to_torch(image)
        self.local_Tracker.im = im  # For debugging only

        # ------- LOCALIZATION ------- #

        # Get sample
        sample_pos = self.local_Tracker.pos.round()
        sample_scales = self.local_Tracker.target_scale * self.local_Tracker.params.scale_factors
        test_x = self.local_Tracker.extract_processed_sample(im, self.local_Tracker.pos, sample_scales, self.local_Tracker.img_sample_sz)

        # Compute scores
        scores_raw = self.local_Tracker.apply_filter(test_x)
        translation_vec, scale_ind, s, flag = self.local_Tracker.localize_target(scores_raw)
        return translation_vec, scale_ind, s, flag, sample_pos, sample_scales, test_x

    def local_update(self, sample_pos, translation_vec, scale_ind, sample_scales, s, test_x, update_flag=None):

        # Check flags and set learning rate if hard negative
        if update_flag is None:
            update_flag = self.flag not in ['not_found', 'uncertain']
        hard_negative = (self.flag == 'hard_negative')
        learning_rate = self.local_Tracker.params.hard_negative_learning_rate if hard_negative else None

        if update_flag:
            # Get train sample
            train_x = TensorList([x[scale_ind:scale_ind + 1, ...] for x in test_x])

            # Create label for sample
            train_y = self.local_Tracker.get_label_function(sample_pos, sample_scales[scale_ind])

            # Update memory
            self.local_Tracker.update_memory(train_x, train_y, learning_rate)

        # Train filter
        if hard_negative:
            self.local_Tracker.filter_optimizer.run(self.local_Tracker.params.hard_negative_CG_iter)
        elif (self.local_Tracker.frame_num - 1) % self.local_Tracker.params.train_skipping == 0:
            self.local_Tracker.filter_optimizer.run(self.local_Tracker.params.CG_iter)

    def local_track_stark(self, image):
        update_ouput = [0, 0]
        cur_ori_img = Image.fromarray(image).convert('RGB')
        cur_image = np.asarray(cur_ori_img)

        target_bbox, local_score = self.stark_tracker.track(image)
        
        return target_bbox, local_score

    def dist_penalty(self, bbox):
        dist_max = math.sqrt(sum(np.array([self.img_shape[0], self.img_shape[1]]) ** 2))
        #for i in range(len(list_search_pos)):
        #    dist1[i] = math.sqrt(sum((list_search_pos[i] - list_search_pos[0]) ** 2))
        last_good = [self.last_good[1], self.last_good[0], self.last_good[3] - self.last_good[1], self.last_good[2] - self.last_good[0]]
        dist = math.sqrt(sum((np.array(bbox[:2]) - np.array(last_good[:2])) ** 2))
        #weight_penalty = 1.0 - self.params.get('redetection_score_penalty_alpha', 0.5) * (dist / dist_max) * math.exp(- self.params.get('redetection_score_penalty_beta', 0.5) * (self.cnt_empty - 1))
        #weight_penalty = 1.0 - 0.75 * (dist / dist_max) * math.exp(- 0.25 * (self.i - self.last_good_idx - 1))
        weight_penalty = 1.0 - 1.0 * (dist / dist_max) * math.exp(- 1.0 * (self.i - self.last_good_idx - 1))

        return weight_penalty

    def tracking(self, image):
        self.i += 1

        local_state, self.score_map, update, local_score, dis, flag = self.local_track(image)
        local_score = np.clip(local_score, 0, 1)

        stark_state, stark_score = self.local_track_stark(image)

        if self.md_scores_sum[-1] > 4:
             max_dim = np.argmax([self.boxes[-1][2], self.boxes[-1][3]])
             max_dim_idx = 2 + max_dim
             min_dim_idx = 2 + (1 - max_dim)
             base_ar = self.boxes[-1][max_dim_idx] / self.boxes[-1][min_dim_idx]
             ar = stark_state[max_dim_idx] / stark_state[min_dim_idx]
             if base_ar > ar:
                 if (np.abs(ar - base_ar) > (0.4 * base_ar)):
                     stark_score = 0.0
             else:
                 if (np.abs(ar - base_ar) > (1.35 * base_ar)):
                     stark_score = 0.0
 
        md_score_dimp = self.pymdnet_eval(image, np.array(local_state).reshape([-1, 4]))[0]
        md_score_dimp = np.clip((local_score + np.arctan(0.2 * md_score_dimp) / math.pi + 0.5) / 2, 0, 1)
        #dist_penalty_dimp = self.dist_penalty(local_state)
        #md_score_dimp *= dist_penalty_dimp
        md_score_stark = self.pymdnet_eval(image, np.array(stark_state).reshape([-1, 4]))[0]
        md_score_stark = np.clip((stark_score + np.arctan(0.2 * md_score_stark) / math.pi + 0.5) / 2, 0, 1)
        #dist_penalty_stark = self.dist_penalty(stark_state)
        #md_score_stark *= dist_penalty_stark

        self.md_scores_dimp.append(md_score_dimp)
        self.md_scores_stark.append(md_score_stark)

        if len(self.md_scores_dimp) > 5:
            md_scores = [(np.array(self.md_scores_stark)[-5:] > 0.5).sum(), (np.array(self.md_scores_dimp)[-5:] > 0.5).sum()]
        else:
            md_scores = [md_score_stark, md_score_dimp]
        md_scores_1 = [md_score_stark, md_score_dimp]
        all_scores = [self.md_scores_stark, self.md_scores_dimp]
        bboxes = [copy.copy(stark_state), copy.copy(local_state)]
        md_score = max(md_scores)
        md_idx = np.argmax(md_scores)


        if md_score > 3:

            if md_idx == 0:
                self.local_Tracker.pos = torch.FloatTensor([(stark_state[1] + stark_state[3] + stark_state[1] - 1) / 2, (stark_state[0] + stark_state[2] + stark_state[0] - 1) / 2])
                self.local_Tracker.target_sz = torch.FloatTensor([stark_state[3], stark_state[2]])

                self.last_gt = np.array([stark_state[1], stark_state[0], stark_state[1] + stark_state[3], stark_state[0] + stark_state[2]])
                
                confidence_score = md_score_stark

            else:
                self.stark_tracker.tracker.state = [int(local_state[0]), int(local_state[1]), int(local_state[2]), int(local_state[3])]

                self.last_gt = np.array([local_state[1], local_state[0], local_state[1] + local_state[3], local_state[0] + local_state[2]])

                confidence_score = md_score_dimp 

            self.stark_tracker.tracker.params.search_factor = 2.5
            self.last_good = copy.copy(self.last_gt)
            self.last_good_idx = copy.copy(self.i)
            
        else:
            new_box = bboxes[md_idx]

            self.last_gt = np.array([new_box[1], new_box[0], new_box[1] + new_box[3], new_box[0] + new_box[2]])

            confidence_score = (md_score_stark + md_score_dimp) / 2

            self.stark_tracker.tracker.params.search_factor = 5.0

      
        if update or (md_scores[0] == 5 and md_scores[1] == 5):
            self.collect_samples_pymdnet(image)

        self.pymdnet_long_term_update()

        width = self.last_gt[3] - self.last_gt[1]
        height = self.last_gt[2] - self.last_gt[0]

        self.boxes.append(np.array([float(self.last_gt[1]), float(self.last_gt[0]), float(width), float(height)]))
        self.boxes_stark.append(np.array(stark_state))
        self.boxes_dimp.append(np.array(local_state))
        self.confidences.append(int(self.score_max > 0))

        self.md_scores_sum.append(md_score)

        return vot.Rectangle(float(self.last_gt[1]), float(self.last_gt[0]), float(width),
                float(height)), confidence_score



handle = vot.VOT("rectangle")  # rectangle(x, y, w, h)

selection = handle.region()
imagefile = handle.frame()
p = p_config()
# if not imagefile:
#     sys.exit(0)
image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
tracker = CoCoLoT_Tracker(image, selection, p=p)  # initialize tracker

while True:

    imagefile = handle.frame()
    
    if not imagefile:
        break

    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
    
    region, confidence = tracker.tracking(image)
    
    handle.report(region, confidence)

    #if idx == n_frames:
    #    tracker.sess.close()
