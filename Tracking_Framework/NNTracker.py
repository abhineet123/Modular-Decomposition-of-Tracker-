"""
Implementation of the Nearest Neighbour Tracking Algorithm.
Author: Travis Dick (travis.barry.dick@gmail.com)
"""
#import msvcrt
from SCVUtils import *
from TrackerBase import *
import numpy as np
import pyflann
#from scipy import weave
#from scipy.weave import converters
from Homography import *
from ImageUtils import *
import itertools
import operator

import cv2
#Jesse
#import pdb
#import sys


class NNTracker(TrackerBase):
    def __init__(self, no_of_samples, no_of_iterations=1, res=(20, 20), multi_approach='none',
                 warp_generator=lambda: random_homography(0.07, 0.06),
                 use_scv=False):
        """ An implemetation of the Nearest Neighbour Tracker. 

        Parameters:
        -----------
        n_samples : integer
          The number of sample motions to generate. Higher values will improve tracking
          accuracy but increase running time.
        
        n_iterations : integer
          The number of times to update the tracker state per frame. Larger numbers
          may improve convergence but will increase running time.
        
        res : (integer, integer)
          The desired resolution of the template image. Higher values allow for more
          precise tracking but increase running time.

        warp_generator : () -> (3,3) numpy matrix.
          A function that randomly generates a homography. The distribution should
          roughly mimic the types of motions that you expect to observe in the 
          tracking sequence. random_homography seems to work well in most applications.
          
        See Also:
        ---------
        TrackerBase
        BakerMatthewsICTracker
        """

        print "Initializing NN tracker with:"
        print " n_samples=", no_of_samples
        print " n_iterations=", no_of_iterations
        print " res=", res
        print " multi_approach=", multi_approach
        print " use_scv=", use_scv

        self.n_samples = no_of_samples
        self.n_iterations = no_of_iterations
        self.res = res
        self.multi_approach=multi_approach
        self.resx = res[0]
        self.resy = res[1]
        self.warp_generator = warp_generator
        self.n_points = np.prod(res)
        self.initialized = False
        self.pts = res_to_pts(self.res)
        self.use_scv = use_scv
        self.use_hoc=False
        self.sift = False

        self.dim=1
        self.flatten=False
        self.use_mean=False
        self.scv_intensity_map=scv_intensity_map
        self.scv_expectation=scv_expectation
        self.getHistogram=getHistogram

        if self.multi_approach=='none':
            print 'Using single channel approach'
            self.initialize=self.initializeSingleChannel
            self.update=self.updateSingleChannel
        else:
            print 'Using multi channel approach'
            self.initialize=self.initializeMultiChannel
            self.update=self.updateMultiChannel
            if self.multi_approach=='flatten':
                print 'Using flattening'
                self.flatten=True
            else:
                self.getHistogram=getHistogramVec
                self.scv_intensity_map=scv_intensity_map_vec2
                self.scv_expectation=scv_expectation_vec
                if self.multi_approach=='mean':
                    print 'Using mean'
                    self.use_mean=True
                else:
                    print 'Using '+self.multi_approach

    def set_region(self, corners):
        self.proposal = square_to_corners_warp(corners)

    def initializeMultiChannel(self, img, region):
        if self.use_hoc:
            self.use_scv=False
        #print "starting nn initialize"
        img_shape=img.shape
        #print "img.shape=",img_shape
        if len(img_shape)<3:
            raise SystemExit('Expected multi channel image but found single channel one')
        else:
            self.n_channels=img_shape[2]

        self.set_region(region)
        #self.use_scv=False
        self.sift=False
        self.template = sample_and_normalize_vec(img, self.pts, self.get_warp(),
                                                 flatten=self.flatten)

        if self.use_hoc:
            self.template=self.getHistogram(self.template)

        #print 'Done sampling'
        self.warp_index = _WarpIndexVec(self.n_samples, self.warp_generator, img, self.pts,
                                 self.get_warp(), self.res, self.use_mean,
                                 flatten=self.flatten, use_hoc=self.use_hoc, histogram=self.getHistogram)
        #print "Done creating warp index"
        self.intensity_map = None
        self.initialized = True
        #print "done"

    def updateMultiChannel(self, img):
        #print "starting nn update"
        if not self.is_initialized():
            return None
        for i in xrange(self.n_iterations):
            #warped_pts = apply_to_pts(self.proposal, self.pts)
            sampled_img = sample_and_normalize_vec(img, self.pts, warp=self.proposal,
                                                   flatten=self.flatten)
            if self.use_scv and self.intensity_map is not None:
                #np.savetxt('sampled_img_old.txt', sampled_img, fmt='%10.5f')
                sampled_img = self.scv_expectation(sampled_img, self.intensity_map)
                #np.savetxt('intensity_map.txt', self.intensity_map, fmt='%10.5f')
                #np.savetxt('sampled_img_new.txt', sampled_img, fmt='%10.5f')

            if self.use_hoc:
                sampled_img=self.getHistogram(sampled_img)

            self.proposal = self.proposal * self.warp_index.best_match(sampled_img)
            self.proposal /= self.proposal[2, 2]
        #print "done"
        if self.use_scv:
            self.intensity_map = self.scv_intensity_map(sample_region_vec(img, self.pts, self.get_warp()),
                                                   self.template)
        return self.proposal

    def initializeSingleChannel(self, img, region):
        #print "starting nn initialize"
        img_shape=img.shape
        #print "img.shape=",img_shape
        if len(img_shape)!=2:
            raise SystemExit('Expected single channel image but found multi channel one')

        self.set_region(region)
        self.template = sample_and_normalize(img, self.pts, self.get_warp())
        self.warp_index = _WarpIndex(self.n_samples, self.warp_generator, img, self.pts,
                                 self.get_warp(), self.res)
        #Jesse
        #pdb.set_trace()
        self.intensity_map = None
        self.initialized = True
        #print "done"

    def updateSingleChannel(self, img):
        #print "starting nn update"
        if not self.is_initialized():
            return None
        for i in xrange(self.n_iterations):
            #warped_pts = apply_to_pts(self.proposal, self.pts)
            sampled_img = sample_and_normalize(img, self.pts, warp=self.proposal)

            if self.use_scv and self.intensity_map is not None:
                #np.savetxt('sampled_img_old.txt', sampled_img, fmt='%10.5f')
                sampled_img = self.scv_expectation(sampled_img, self.intensity_map)
                #np.savetxt('intensity_map.txt', self.intensity_map, fmt='%10.5f')
                #np.savetxt('sampled_img_new.txt', sampled_img, fmt='%10.5f')

            if not self.sift:
                self.proposal = self.proposal * self.warp_index.best_match(sampled_img)
                self.proposal /= self.proposal[2, 2]
            else:
                # --sift-- #
                temp_desc = self.pixel2sift(sampled_img)
                #	if temp_desc == None:
                #		print('No feature found!')
                #		sys.exit()
                #       pdb.set_trace()
                update = self.desc2warp_weighted3(temp_desc)
                self.proposal = self.proposal * update
                self.proposal /= self.proposal[2, 2]
        if self.use_scv:
            self.intensity_map = self.scv_intensity_map(sample_region(img, self.pts, self.get_warp()),
                                                   self.template)
        #print "done"
        return self.proposal

    def is_initialized(self):
        return self.initialized

    def get_warp(self):
        return self.proposal

    def get_region(self):
        return apply_to_pts(self.get_warp(), np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]).T)

        #-- sift --#
    def pixel2sift(self, patch):
        detector = cv2.FeatureDetector_create("SIFT")
        detector.setDouble('edgeThreshold', 30)
        descriptor = cv2.DescriptorExtractor_create("SIFT")
        #sift = cv2.SIFT(edgeThreshold = 20)
        patch = (patch.reshape(self.resx, self.resy)).astype(np.uint8)
        skp = detector.detect(patch)
        skp, sd = descriptor.compute(patch, skp)
        #pdb.set_trace()
        #print(sd.shape[0])
        return sd

    # --- For sift --- #
    def desc2warp_weighted(self, descs):
        warps = np.zeros((3, 3), dtype=np.float64)
        temp_desc = np.empty((128, 1), dtype=np.float32)
        if descs == None:
            print('The number of descriptors is zero!')
            return np.eye(3, dtype=np.float32)
        for i in range(descs.shape[0]):
            temp_desc[:, 0] = descs[i, :]
            warp, dist = self.warp_index.best_match_sift(temp_desc.T)
            warps += warp
        warps /= descs.shape[0]
        return warps

    # --- For sift --- #
    def desc2warp_weighted2(self, descs):
        warps = np.zeros((3, 3), dtype=np.float64)
        temp_desc = np.empty((128, 1), dtype=np.float32)
        if descs == None:
            print('The number of descriptors is zero!')
            return np.eye(3, dtype=np.float32)
        warp_list = []
        dist_list = []
        for i in range(descs.shape[0]):
            temp_desc[:, 0] = descs[i, :]
            warp, dist = self.warp_index.best_match_sift(temp_desc.T)
            #warps += warp
            warp_list.append(warp)
            dist_list.append(dist)
        thres = max(dist_list) * 0.5
        count = 0
        for i in range(len(dist_list)):
            if dist_list[i] <= thres:
                warps += warp_list[i]
                count += 1
        if count == 0: return np.eye(3, dtype=np.float32)
        warps /= count
        return warps

    # --- For sift --- #
    def desc2warp_weighted3(self, descs):
        warps = np.zeros((3, 3), dtype=np.float64)
        temp_desc = np.empty((128, 1), dtype=np.float32)
        if descs == None:
            print('The number of descriptors is zero!')
            return np.eye(3, dtype=np.float32)
        warp_list = []
        dist_list = []
        print('Testing')
        for i in range(descs.shape[0]):
            temp_desc[:, 0] = descs[i, :]
            warp, dist, index = self.warp_index.best_match_sift(temp_desc.T)
            print(index)
            #warps += warp
            warp_list.append(warp)
            dist_list.append(dist)
        sum_dist = sum(dist_list)
        for i in range(len(dist_list)):
            warps += warp_list[i] * dist_list[i] / sum_dist
        return warps

class _WarpIndex:
    """ Utility class for building and querying the set of reference images/warps. """

    def __init__(self, n_samples, warp_generator, img, pts, initial_warp, res):
        self.resx = res[0]
        self.resy = res[1]
        self.sift = False
        self.indx = []
        n_points = pts.shape[1]
        print "Sampling Warps..."
        self.warps = [np.asmatrix(np.eye(3))] + [warp_generator() for i in xrange(n_samples - 1)]
        print "Sampling Images..."
        self.images = np.empty((n_points, n_samples))
        for i, w in enumerate(self.warps):
            self.images[:, i] = sample_and_normalize(img, pts, initial_warp * w.I)
            #self.images[:,i] = sample_and_normalize(img, apply_to_pts(initial_warp * w.I, pts))
        print "Building FLANN Index..."
        #pyflann.set_distance_type("manhattan")
        if self.sift == False:
            self.flann = pyflann.FLANN()
            #print(self.images.shape)
            self.flann.build_index(self.images.T, algorithm='kdtree', trees=10)
        else:
            desc = self.list2array(self.pixel2sift(self.images))
            # --- Building Flann Index --- #
            self.flann = pyflann.FLANN()
            #self.flann.build_index(np.asarray(self.images).T, algorithm='linear')
            #print(type(desc))
            #pdb.set_trace()
            self.flann.build_index(desc.T, algorithm='kdtree', trees=10)
        print "Done!"

    # --- For sift --- #
    def pixel2sift(self, images):
        detector = cv2.FeatureDetector_create("SIFT")
        detector.setDouble('edgeThreshold', 30)
        descriptor = cv2.DescriptorExtractor_create("SIFT")
        #sift = cv2.SIFT(edgeThreshold = 20)
        # -- store descriptors in list --#
        desc = []
        for i in range(images.shape[1]):
            patch = (images[:, i].reshape(self.resx, self.resy)).astype(np.uint8)
            #pdb.set_trace()
            skp = detector.detect(patch)
            skp, sd = descriptor.compute(patch, skp)
            desc.append(sd)
            self.indx.append(len(skp))
        return desc

    # --- For sift ---#
    def list2array(self, desc):
        nums = sum(self.indx)
        descs = np.empty((128, nums), dtype=np.float64)
        counts = 0
        for item in desc:
            if item == None:
                continue
            for j in range(item.shape[0]):
                descs[:, counts] = item[j, :].T
                counts += 1
        return descs.astype(np.float32)

    # ---SIFT function --- #
    def best_match_sift(self, desc):
        #print(type(desc))
        results, dists = self.flann.nn_index(desc)
        index = int(results[0])
        index += 1
        count = 0
        for item in self.indx:
            if index <= item:
                result = count
            else:
                index -= item
                count += 1
        return self.warps[count], dists[0], count

    def best_match(self, img):
        #print(img.shape)
        results, dists = self.flann.nn_index(img)
        return self.warps[results[0]]


class _WarpIndexVec:
    """ Utility class for building and querying the set of reference images/warps. """

    def __init__(self, n_samples, warp_generator, img, pts, initial_warp, res,
                 use_mean, flatten=False, use_hoc=False, histogram=None):
        if len(img.shape) < 3:
            raise AssertionError("Error in _WarpIndexVec: The image is not multi channel")

        if use_hoc and histogram is None:
            raise SystemExit('Error in _WarpIndexVec:'
                             'Cannot compute histogram without valid function')
        self.dim = img.shape[2]
        self.resx = res[0]
        self.resy = res[1]
        self.sift = False
        self.indx = []
        self.use_mean=use_mean
        self.flatten=flatten

        n_points = pts.shape[1]
        print "Sampling Warps..."
        self.warps = [np.asmatrix(np.eye(3))] + [warp_generator() for i in xrange(n_samples - 1)]
        print "Sampling Images..."
        if use_hoc:
            if self.flatten:
                self.images = np.empty((256, n_samples))
            else:
                 self.images = np.empty((self.dim, 256, n_samples))
        else:
            if self.flatten:
                self.images = np.empty((self.dim*n_points, n_samples))
            else:
                 self.images = np.empty((self.dim, n_points, n_samples))
        for i, w in enumerate(self.warps):
            sample = sample_and_normalize_vec(img, pts, initial_warp * w.I,
                                                            flatten=self.flatten)
            if use_hoc:
                sample=histogram(sample)
            if self.flatten:
                 self.images[:, i]=sample
            else:
                 self.images[:, :, i]=sample

            #self.images[:,i] = sample_and_normalize(img, apply_to_pts(initial_warp * w.I, pts))
        print "Building FLANN Index..."

        self.flann_vec = []
        #pyflann.set_distance_type("manhattan")
        if self.flatten:
            self.flann = pyflann.FLANN()
            #print 'self.images.shape:',  self.images.shape
            #print 'self.images.dtype',self.images.dtype
            self.flann.build_index(self.images.T, algorithm='kdtree', trees=10)
        else:
            for i in xrange(self.dim):
                current_images = self.images[i, :, :]
                flann = pyflann.FLANN()
                #print(self.images.shape)
                flann.build_index(current_images.T, algorithm='kdtree', trees=10)
                self.flann_vec.append(flann)
        print "Done!"

    def best_match(self, img):
        #print 'img.shape',img.shape
        #print 'img.dtype',img.dtype
        if self.flatten:
            results, dists = self.flann.nn_index(img)
            return self.warps[results[0]]

        result_vec = []
        warp_vec=[]
        for i in xrange(self.dim):
            results, dists = self.flann_vec[i].nn_index(img[i,:])
            if self.use_mean:
                warp_vec.append(self.warps[results[0]])
            else:
                result_vec.append(results[0])
        if self.use_mean:
            warp=getMean(warp_vec)
        else:
            #print "result_vec=", result_vec
            most_common_res = most_common(result_vec)
            if result_vec.count(most_common_res)>=len(result_vec)/2:
                #print "most_common_res=", most_common_res
                warp=self.warps[most_common_res]
            else:
                warp=getMean(warp_vec)
        return warp


def most_common(L):
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
    # print 'SL:', SL
    groups = itertools.groupby(SL, key=operator.itemgetter(0))
    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
            # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index

        # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]




