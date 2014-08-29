"""
Implementation of the ESM tracking algorithm.

S. Benhimane and E. Malis, "Real-time image-based tracking of planes
using efficient second-order minimization," Intelligent Robots and Systems, 2004.
(IROS 2004). Proceedings. 2004 IEEE/RSJ International Conference on, vol. 1, 
pp. 943-948 vol. 1, 2004.

Author: Travis Dick (travis.barry.dick@gmail.com)
"""

import numpy as np
from scipy.linalg import expm

from Homography import *
from ImageUtils import *
from SCVUtils import *
from TrackerBase import *

class ESMTracker(TrackerBase):
    
    def __init__(self, max_iterations, threshold=0.01, res=(20,20), multi_approach='none', use_scv=True):
        
        print "Initializing ESM tracker with:"
        print " max_iters=", max_iterations
        print " threshold=", threshold
        print " res=", res
        print " multi_approach=", multi_approach
        print 'use_scv=', use_scv

        self.max_iters = max_iterations
        self.threshold = threshold
        self.res = res
        self.npts=np.prod(res)
        self.pts = res_to_pts(self.res)
        self.use_scv = use_scv
        self.initialized = False
        self.mean_level=0
        self.unit_square=np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]).T
        self.multi_approach=multi_approach
        self.n_channels=1

        self.estimate_jacobian=estimate_jacobian
        self.sample_region=sample_region

        self.flatten = False
        self.use_mean = False
        self.scv_expectation = scv_expectation
        self.scv_intensity_map = scv_intensity_map

        if self.multi_approach == 'none':
            print 'Using single channel approach'
            self.initialize = self.initializeSingleChannel
            self.update = self.updateSingleChannel
        else:
            print 'Using multi channel approach'
            self.initialize = self.initializeMultiChannel
            self.update = self.updateMultiChannel
            self.sample_region=sample_region_vec
            if self.multi_approach == 'flatten':
                print 'Using flattening'
                self.flatten = True
                self.estimate_jacobian=estimate_jacobian_vec
            else:
                if self.multi_approach == 'mean':
                    print 'Using mean'
                    self.use_mean = True

    def set_region(self, corners):
        self.proposal = square_to_corners_warp(corners)

    def initializeMultiChannel(self, img, region):
        #print "in ESMTracker initialize"
        img_shape=img.shape
        if len(img_shape)!=3:
            raise SystemExit('Expected multi channel image but found single channel one instead')
        self.n_channels=img_shape[2]

        self.set_region(region)
        self.template = self.sample_region(img, self.pts, self.get_warp(), flatten=self.flatten)
        if self.flatten:
            #print "Sampling template"
            #print "estimating jacobian"
            self.Je = self.estimate_jacobian(img, self.pts, self.proposal)
        else:
            self.Je_vec=[]
            for i in xrange(self.n_channels):
                Je = self.estimate_jacobian(img[:, :, i], self.pts, self.proposal)
                self.Je_vec.append(Je)

        self.intensity_map = None
        self.initialized = True
        #print "Done"

    def updateMultiChannel(self, img):
        #print "in ESMTracker update"
        if not self.initialized:
            return

        #update_thresh=False
        for i in xrange(self.max_iters):
            #if update_thresh:
            #    break
            #update_sum=np.zeros(8)
            sampled_img = self.sample_region(img, self.pts, self.get_warp(), flatten=self.flatten)
            if self.use_scv and self.intensity_map is not None:
                sampled_img = self.scv_expectation(sampled_img, self.intensity_map)
            error_img = np.asmatrix(self.template - sampled_img)
            if self.flatten:
                Jpc = self.estimate_jacobian(img, self.pts, self.proposal)
                J = (Jpc + self.Je) / 2.0
                #print "J shape=", J.shape
                update = np.asarray(np.linalg.lstsq(J, error_img.reshape((-1, 1)))[0]).squeeze()
                #print 'update.shape=', update.shape
                #update=self.getUpdate(img, self.template, self.Je)
                self.proposal = self.proposal * make_hom_sl3(update)
                if np.sum(np.abs(update)) < self.threshold:
                    break
            else:
                #update_vec=[]
                update_sum=np.zeros(8)
                for i in xrange(self.n_channels):
                    Jpc = self.estimate_jacobian(img[:,:,i], self.pts, self.proposal)
                    J = (Jpc + self.Je_vec[i]) / 2.0
                    #print "J shape=", J.shape
                    #print 'error.shape=', error_img.shape
                    update = np.asarray(np.linalg.lstsq(J, error_img[i,:].reshape((-1, 1)))[0]).squeeze()
                    update_sum+=update
                    #update_vec.append(update)
                    #print 'update.shape=', update.shape
                    #print 'proposal.shape=', self.proposal.shape
                    #update=self.getUpdate(img[:, :, i], self.template_vec[i], self.Je_vec[i])

                #mean_update=getMean(update_vec)
                mean_update=update_sum/self.n_channels
                self.proposal = self.proposal * make_hom_sl3(mean_update)
                if np.sum(np.abs(mean_update)) < self.threshold:
                    break
        if self.use_scv:
            self.intensity_map = scv_intensity_map(self.sample_region(img, self.pts, self.get_warp()),
                                                   self.template)

    def initializeSingleChannel(self, img, region):
        #print "in ESMTracker initialize"
        img_shape=img.shape
        if len(img_shape)!=2:
            raise SystemExit('Expected single channel image but found multi channel one')

        self.set_region(region)
        #print "Sampling template"
        self.template = self.sample_region(img, self.pts, self.get_warp())
        #print "estimating jacobian"
        self.Je = self.estimate_jacobian(img, self.pts, self.proposal)

        self.intensity_map = None
        self.initialized = True
        #print "Done"

    def updateSingleChannel(self, img):
        #print "in ESMTracker update"
        if not self.initialized:
            return
        for i in xrange(self.max_iters):
            update=self.getUpdate(img, self.template, self.Je)
            self.proposal = self.proposal * make_hom_sl3(update)
            if np.sum(np.abs(update)) < self.threshold:
                break
            #print "J_size=",J.shape
            #print "error_size=",error.shape
            #print "update_size=",update.shape
        if self.use_scv:
            self.intensity_map = scv_intensity_map(self.sample_region(img, self.pts, self.get_warp()),
                                                   self.template)
            #self.intensity_map = getSCVIntensityMap(sample_region(img, self.pts, self.get_warp()),self.template)

    def getUpdate(self, img, template, Je):
        sampled_img = self.sample_region(img, self.pts, self.get_warp(), flatten=self.flatten)
        Jpc = self.estimate_jacobian(img, self.pts, self.proposal)
        J = (Jpc + Je) / 2.0
        if self.use_scv and self.intensity_map != None:
            sampled_img = scv_expectation(sampled_img, self.intensity_map)
        #print "J shape=", J.shape
        error = np.asmatrix(template - sampled_img).reshape(-1, 1)
        update = np.asarray(np.linalg.lstsq(J, error)[0]).squeeze()
        return update

    def is_initialized(self):
        return self.initialized

    def get_warp(self):
        return self.proposal

    def get_region(self):
        return apply_to_pts(self.get_warp(), self.unit_square)

#def _estimate_jacobian(img, pts, initial_warp, eps=1e-10):
#    #print '_estimate_jacobian'
#    n_pts = pts.shape[1]
#    def f(p):
#        W = initial_warp * make_hom_sl3(p)
#        return sample_region(img, pts, W)
#    jacobian = np.empty((n_pts,8))
#    for i in xrange(0,8):
#        o = np.zeros(8)
#        o[i] = eps
#        jacobian[:,i] = (f(o) - f(-o)) / (2*eps)
#    return np.asmatrix(jacobian)
#
#def _estimate_jacobian_vec(img, pts, initial_warp, eps=1e-10):
#    #print '_estimate_jacobian_vec'
#    n_pts = pts.shape[1]*img.shape[2]
#
#    def f(p):
#        W = initial_warp * make_hom_sl3(p)
#        return sample_region_vec(img, pts, W, flatten=True)
#
#    jacobian = np.empty((n_pts, 8))
#    for i in xrange(0, 8):
#        o = np.zeros(8)
#        o[i] = eps
#        jacobian[:, i] = (f(o) - f(-o)) / (2 * eps)
#    return np.asmatrix(jacobian)

    #def updateMultiChannel(self, img):
    #    #print "in ESMTracker update"
    #    if not self.initialized:
    #        return
    #
    #    #update_thresh=False
    #    for i in xrange(self.max_iters):
    #        #if update_thresh:
    #        #    break
    #        #update_sum=np.zeros(8)
    #        if self.flatten:
    #            update=self.getUpdate(img, self.template, self.Je)
    #            self.proposal = self.proposal * make_hom_sl3(update)
    #            if np.sum(np.abs(update)) < self.threshold:
    #                break
    #        else:
    #            update_vec=[]
    #            update_hom_vec=[]
    #            proposal_vec=[]
    #            pts_vec=[]
    #            thresh_count=0
    #            for i in xrange(self.n_channels):
    #                update=self.getUpdate(img[:, :, i], self.template_vec[i], self.Je_vec[i])
    #                #if np.sum(np.abs(update)) < self.threshold:
    #                #    thresh_count+=1
    #                if self.mean_level>0:
    #                    update_hom=make_hom_sl3(update)
    #                    if self.mean_level>1:
    #                        proposal=self.proposal * update_hom
    #                        if self.mean_level>2:
    #                            pts_vec.append(apply_to_pts(proposal, self.unit_square))
    #                            #print "pts_vec:\n", pts_vec
    #                        else:
    #                            proposal_vec.append(proposal)
    #                    else:
    #                        update_hom_vec.append(update_hom)
    #                else:
    #                    update_vec.append(update)
    #            if self.mean_level==0:
    #                mean_update=getMean(update_vec)
    #                self.proposal = self.proposal * make_hom_sl3(mean_update)
    #                if np.sum(np.abs(mean_update)) < self.threshold:
    #                    break
    #            elif self.mean_level==1:
    #                self.proposal = self.proposal * getMean(update_hom_vec)
    #            elif self.mean_level==2:
    #                self.proposal=getMean(proposal_vec)
    #            elif self.mean_level==3:
    #                self.proposal = square_to_corners_warp(getMean(pts_vec))
                #if thresh_count==self.n_channels:
                #    break
