__author__ = 'Tommy'
import cv2
import numpy as np
from MultiProposalTracker import *
from NNTracker import *
from ParallelTracker import *
from BakerMatthewsICTracker import *
from CascadeTracker import *
from ESMTracker import *
from L1Tracker import *

class TrackingParams:
    def __init__(self, type, params):
        self.type = type
        self.params = {}
        self.tracker=None

        self.update = lambda: None
        self.validate = lambda: True
        #print 'params.keys=\n', params.keys()

        # initialize parameters
        for key in params.keys():
            vals=params[key]
            self.params[key]=Param(name=key, id=vals['id'], val=vals['default'], type=vals['type'],
                                     list=vals['list'])
        self.sorted_params=self.getSortedParams()
        if type == 'nn':
            print 'Initializing NN tracker parameters'
            self.update = lambda: NNTracker(no_of_samples=self.params['no_of_samples'].val,
                                            no_of_iterations=self.params['no_of_iterations'].val,
                                            res=(self.params['resolution_x'].val, self.params['resolution_y'].val),
                                            use_scv=self.params['enable_scv'].val,
                                            multi_approach=self.params['multi_approach'].val)
        elif type == 'esm':
            print 'Initializing ESM tracker parameters'
            self.update = lambda: ESMTracker(max_iterations=self.params['max_iterations'].val,
                                             threshold=self.params['threshold'].val,
                                             res=(self.params['resolution_x'].val, self.params['resolution_y'].val),
                                             use_scv=self.params['enable_scv'].val,
                                             multi_approach=self.params['multi_approach'].val)
        elif type == 'ict':
            print 'Initializing ICT tracker parameters'
            self.update = lambda: BakerMatthewsICTracker(max_iterations=self.params['max_iterations'].val,
                                                         threshold=self.params['threshold'].val,
                                                         res=(self.params['resolution_x'].val,self.params['resolution_y'].val),
                                                         use_scv=self.params['enable_scv'].val,
                                                         multi_approach=self.params['multi_approach'].val)
        elif type == 'l1':
            print 'Initializing L1 tracker parameters'
            self.update = lambda: L1Tracker(no_of_samples=self.params['no_of_samples'].val,
                                            angle_threshold=self.params['angle_threshold'].val,
                                            res=[[self.params['resolution_x'].val, self.params['resolution_y'].val]],
                                            no_of_templates=self.params['no_of_templates'].val,
                                            alpha=self.params['alpha'].val,
                                            use_scv=self.params['enable_scv'].val,
                                            multi_approach=self.params['multi_approach'].val)
        else:
            self.init_success = False
            print "Invalid tracker:", type


    def getSortedParams(self):
        return sorted(self.params.values(), key=lambda k: k.id)

    def printParamValue(self):
        #print self.type, "filter:"
        for key in self.params.keys():
            print key, "=", self.params[key].val


class Param:
    def __init__(self, name, id, val, type, list=None):
        self.id=id
        self.name = name
        self.val = val
        self.type = type
        self.list=list



