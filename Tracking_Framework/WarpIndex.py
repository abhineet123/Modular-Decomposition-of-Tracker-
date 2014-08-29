import numpy as np
import pyflann
from scipy import weave
from scipy.weave import converters
from Homography import *
from ImageUtils import *
import itertools
import operator


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

    def __init__(self, n_samples, warp_generator, img, pts, initial_warp, res):
        if len(img.shape) < 3:
            raise AssertionError("Error in _WarpIndexVec: The image is not multi channel")

        self.dim = img.shape[2]
        self.resx = res[0]
        self.resy = res[1]
        self.sift = False
        self.indx = []

        n_points = pts.shape[1]
        print "Sampling Warps..."
        self.warps = [np.asmatrix(np.eye(3))] + [warp_generator() for i in xrange(n_samples - 1)]
        print "Sampling Images..."
        self.images = np.empty((self.dim, n_points, n_samples))
        for i, w in enumerate(self.warps):
            self.images[:, :, i] = sample_and_normalize(img, pts, initial_warp * w.I)
            #self.images[:,i] = sample_and_normalize(img, apply_to_pts(initial_warp * w.I, pts))
        print "Building FLANN Index..."
        #pyflann.set_distance_type("manhattan")
        self.flann_vec = []
        for i in xrange(self.dim):
            current_images = self.images[i, :, :]
            flann = pyflann.FLANN
            #print(self.images.shape)
            flann.build_index(current_images.T, algorithm='kdtree', trees=10)
            self.flann_vec.append(flann)
        print "Done!"

    def best_match(self, img):
        #print(img.shape)
        result_vec = []
        for i in xrange(self.dim):
            results, dists = self.flann_vec[i].nn_index(img)
            result_vec.append(results[0])
        print "result_vec=", result_vec
        most_common_res = most_common(result_vec)
        print "most_common_res=", most_common_res
        return self.warps[most_common_res]


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