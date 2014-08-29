import os
import time
from TrackingParams import *
from FilteringParams import *
from GUI import *
from Misc import *
import webbrowser

from matplotlib import pyplot as plt

class InteractiveTrackingApp:
    def __init__(self, init_frame, root_path, track_window_name, params,
                 tracking_params, filtering_params, labels, default_id=None,
                success_threshold=5, batch_mode=False, agg_filename=None, avg_filename=None):


        self.root_path=root_path
        self.params=params
        self.agg_filename=agg_filename
        self.avg_filename=avg_filename
        #self.tracking_params=tracking_params
        #self.filtering_params=filtering_params
        self.labels=labels
        if default_id==None:
            default_id=[0 for i in xrange(len(self.params))]
        self.default_id=default_id
        self.first_call=True

        if len(self.default_id)!=len(self.params):
            raise SyntaxError('Mismatch between the sizes of default ids and params')

        if len(self.labels)!=len(self.params):
            raise SyntaxError('Mismatch between the sizes of labels and params')

        # initialize filters
        filter_index=labels.index('filter')
        #self.filters_ids=dict(zip(params[filter_index], [i for i in xrange(len(params[filter_index]))]))
        self.filter_type=params[filter_index][default_id[filter_index]]
        self.filters={}
        for i in xrange(1, len(params[filter_index])):
            filter_type=params[filter_index][i]
            self.filters[filter_type]=FilterParams(filter_type, filtering_params[filter_type])

        # initialize trackers
        tracker_index=labels.index('tracker')
        #self.tracker_ids=dict(zip(params[tracker_index], [i for i in xrange(len(params[tracker_index]))]))
        self.tracker_type=params[tracker_index][default_id[tracker_index]]
        self.trackers={}
        for i in xrange(len(params[tracker_index])):
            tracker_type=params[tracker_index][i]
            self.trackers[tracker_type]=TrackingParams(tracker_type, tracking_params[tracker_type])

        self.source=default_id[labels.index('source')]

        self.init_frame=init_frame
        self.track_window_name = track_window_name
        self.proc_window_name='Processed Images'
        self.count=0

        self.gray_img = None
        self.proc_img = None
        self.paused = False
        self.enable_smoothing=False
        self.smooth_image=None
        self.window_inited=False
        self.init_track_window=True
        self.img = None
        self.init_params=[]
        self.times = 1
        self.max_cam=3

        self.reset=False
        self.exit_event=False
        self.write_res=False
        self.cap=None

        self.multi_channel=False
        self.success_threshold=success_threshold

        self.initPlotParams()
        self.tracker_pause=False

        self.batch_mode=batch_mode

        if self.batch_mode:
            init_params=self.getInitParams()
            self.initSystem(init_params)
        else:
            gui_title="Choose Input Video Parameters"
            self.gui_obj=GUI(self, gui_title)
            #self.gui_obj.initGUI()
            #self.gui_obj.root.mainloop()

        self.success_count=0
        self.success_drift=[]

    def getInitParams(self):
        init_params=[]
        for i in xrange(len(self.params)):
            if self.labels[i]=='task':
                type_index=self.labels.index('type')
                param=self.params[i][self.default_id[type_index]][self.default_id[i]]
            else:
                param=self.params[i][self.default_id[i]]
            init_params.append(param)
        #print 'init_params=', init_params
        #sys.exit()
        return init_params

    def initCamera(self):
        print "Getting input from camera"
        if self.cap!=None:
            self.cap.release()
        self.cap = cv2.VideoCapture(1)
        dWidth = self.cap.get(3)
        dHeight = self.cap.get(4)
        if dWidth==0 or dHeight==0:
            raise SystemExit("No valid camera found")
        print "Frame size : ", dWidth, " x ", dHeight
        self.res_file = open('camera_res_%s.txt' % self.tracker_type, 'w')
        self.res_file.write('%-8s%-8s%-8s%-8s%-8s%-8s%-8s%-8s%-8s\n' % (
            'frame', 'ulx', 'uly', 'urx', 'ury', 'lrx', 'lry', 'llx', 'lly'))
        self.no_of_frames=0
        #sys.exit()

    def initVideoFile(self):
        type=self.init_params[self.labels.index('type')]
        actor=self.init_params[self.labels.index('actor')]
        light=self.init_params[self.labels.index('light')]
        speed=self.init_params[self.labels.index('speed')]
        task=self.init_params[self.labels.index('task')]

        self.dataset_path=self.root_path+'/'+actor
        self.res_path = self.dataset_path + '/results'
        if type=='simple':
            data_file = light+'_'+task+'_'+speed
        elif type=='complex':
            data_file = light+'_'+task
        else:
            print "Invalid task type specified: %s"%type
            return False

        self.data_file=data_file
        print "Getting input from data: ", self.data_file
        if not os.path.exists(self.res_path):
            os.mkdir(self.res_path)
        self.res_file = open(self.res_path + '/' + data_file + '_res_%s.txt'%self.tracker_type, 'w')
        self.res_file.write('%-8s%-8s%-8s%-8s%-8s%-8s%-8s%-8s%-8s\n' % (
            'frame', 'ulx', 'uly', 'urx', 'ury', 'lrx', 'lry', 'llx', 'lly'))

        self.img_path = self.dataset_path + '/' + data_file
        if not os.path.isdir(self.img_path):
            print 'Data directory does not exist: ', self.img_path
            self.exit_event=True
            return False
        self.ground_truth = readTrackingData(self.dataset_path + '/' + data_file + '.txt')
        self.no_of_frames = self.ground_truth.shape[0]
        print "no_of_frames=", self.no_of_frames
        self.initparam = [self.ground_truth[self.init_frame, 0:2].tolist(),
                     self.ground_truth[self.init_frame, 2:4].tolist(),
                     self.ground_truth[self.init_frame, 4:6].tolist(),
                     self.ground_truth[self.init_frame, 6:8].tolist()]
            #print tracking_data
        print "object location initialized to:",  self.initparam


    def initSystem(self, init_params):

        print "\n"+"*"*60+"\n"

        self.inited=False

        self.success_count=0
        self.success_drift=[]

        if not self.batch_mode:
            self.initFilterWindow()
        self.init_params=init_params

        self.source=self.init_params[self.labels.index('source')]

        self.color_space=self.init_params[self.labels.index('color_space')]
        if self.color_space.lower()!='grayscale':
            self.multi_channel=True
        else:
            self.multi_channel=False

        self.tracker_type=self.init_params[self.labels.index('tracker')]
        if not self.multi_channel:
            print 'Disabling multichannel'
            self.trackers[self.tracker_type].params['multi_approach'].val='none'
        self.tracker=self.trackers[self.tracker_type].update()
        self.multi_approach=self.tracker.multi_approach
        self.smoothing_type=self.init_params[self.labels.index('smoothing')]

        self.smoothing_kernel=int(self.init_params[self.labels.index('smoothing_kernel')])
        if self.smoothing_type=='none':
            print 'Smoothing is disabled'
            self.enable_smoothing=False
        else:
            print 'Smoothing images with smoothing kernel size ', self.smoothing_kernel
            self.enable_smoothing=True
            if self.smoothing_type=='box':
                self.smooth_image=lambda src: cv2.blur(src, (self.smoothing_kernel, self.smoothing_kernel))
            elif self.smoothing_type=='bilateral':
                self.smooth_image=lambda src: cv2.bilateralFilter(src, self.smoothing_kernel, 100, 100)
            elif self.smoothing_type=='gauss':
                self.smooth_image=lambda src: cv2.GaussianBlur(src, (self.smoothing_kernel, self.smoothing_kernel), 3)
            elif self.smoothing_type=='median':
                self.smooth_image=lambda src: cv2.medianBlur(src, self.smoothing_kernel)

        old_filter_type=self.filter_type
        self.filter_type=self.init_params[self.labels.index('filter')]
        if not self.batch_mode and old_filter_type!=self.filter_type:
            self.initFilterWindow()

        if self.filter_type=='none':
            print "Filtering disabled"
        elif self.filter_type in self.filters.keys():
            self.tracker.use_scv=False
            print "Using %s filtering" % self.filter_type
        else:
            print 'Invalid filter type: ', self.filter_type
            return False

        #print "Using ", self.tracker_name, " tracker"

        if self.source=='camera':
            print "Initializing camera..."
            self.from_cam=True
            self.initCamera()
            self.plot_fps=True
        else:
            self.from_cam=False
            self.initVideoFile()
            self.plot_fps=False

        if not self.first_call:
            self.writeResults()

        print "\n"+"*"*60+"\n"
        return True

    def initPlotParams(self):

        self.curr_error=0
        self.avg_error=0
        self.avg_error_list=[]
        self.curr_fps_list=[]
        self.avg_fps_list=[]
        self.curr_error_list=[]
        self.frame_times=[]
        self.max_error=0
        self.max_fps=0
        self.max_val=0
        self.call_count=0

        self.count=0
        self.current_fps=0
        self.average_fps=0

        #self.start_time=datetime.now().time()
        self.start_time=0
        self.current_time=0
        self.last_time=0

        self.switch_plot=True

    def getTrackingObject(self):
        annotated_img=self.img.copy()
        temp_img=self.img.copy()
        title='Select the object to track'
        cv2.namedWindow(title)
        cv2.imshow(title, annotated_img)
        pts=[]

        def drawLines(img, hover_pt=None):
            if len(pts)==0:
                return
            for i in xrange(len(pts)-1):
                cv2.line(img, pts[i], pts[i+1], (0, 0, 255), 1)
            if hover_pt==None:
                return
            cv2.line(img, pts[-1], hover_pt, (0, 0, 255), 1)
            if len(pts)==3:
                cv2.line(img, pts[0], hover_pt, (0, 0, 255), 1)
            cv2.imshow(title, img)

        def mouseHandler(event, x, y, flags=None, param=None):
            if event==cv2.EVENT_LBUTTONDOWN:
                pts.append((x, y))
                drawLines(annotated_img)
            elif event==cv2.EVENT_LBUTTONUP:
                pass
            elif event==cv2.EVENT_RBUTTONDOWN:
                pass
            elif event==cv2.EVENT_RBUTTONUP:
                pass
            elif event==cv2.EVENT_MBUTTONDOWN:
                pass
            elif event==cv2. EVENT_MOUSEMOVE:
                if len(pts)==0:
                    return
                temp_img=annotated_img.copy()
                drawLines(temp_img, (x,y))

        cv2.setMouseCallback(title, mouseHandler, param=[annotated_img, temp_img, pts])
        while len(pts)<4:
            key=cv2.waitKey(1)
            if key==27:
                break
        cv2.destroyWindow(title)
        cv2.waitKey(1500)
        return pts

    def on_frame(self, img, numtimes):
        #print "frame: ", numtimes
        if self.first_call and not self.batch_mode:
            self.gui_obj.initWidgets(start_label='Restart')
            self.first_call=False

        self.count+=1
        self.times = numtimes

        self.img = img
        #print "img.shape=",img.shape

        if not self.multi_channel:
            self.gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.enable_smoothing:
                self.gray_img=self.smooth_image(self.gray_img)
            self.gray_img_float = self.gray_img.astype(np.float64)
            self.proc_img=self.applyFiltering()
        else:
            if self.enable_smoothing:
                self.img=self.smooth_image(self.img)
            if self.color_space=='RGB':
                self.proc_img=self.img
            elif self.color_space=='HSV':
                self.proc_img=cv2.cvtColor(self.img,  cv2.COLOR_RGB2HSV)
            elif self.color_space=='YCrCb':
                self.proc_img=cv2.cvtColor(self.img,  cv2.COLOR_RGB2YCR_CB)
            elif self.color_space=='HLS':
                self.proc_img=cv2.cvtColor(self.img,  cv2.COLOR_RGB2HLS)
            elif self.color_space=='Lab':
                self.proc_img=cv2.cvtColor(self.img,  cv2.COLOR_RGB2LAB)

        if not self.batch_mode:
            cv2.imshow(self.proc_window_name,  self.proc_img)
        elif self.count==100:
            print 'Processing frame', self.times+1
            self.count=0
        self.proc_img = self.proc_img.astype(np.float64)

        if not self.inited:
            if not self.batch_mode:
                cv2.namedWindow(self.track_window_name)
            if self.from_cam:
                pts=self.getTrackingObject()
                if len(pts)<4:
                    self.exit_event=True
                    sys.exit()
                init_array = np.array(pts).T
            else:
                init_array = np.array(self.initparam).T


            self.tracker.initialize(self.proc_img, init_array)

            self.start_time=time.clock()
            self.current_time=self.start_time
            self.last_time=self.start_time

            self.inited = True

        self.tracker.update(self.proc_img)
        self.corners = self.tracker.get_region()

        if not self.from_cam:
            self.actual_corners = [self.ground_truth[self.times - 1, 0:2].tolist(),
                              self.ground_truth[self.times - 1, 2:4].tolist(),
                              self.ground_truth[self.times - 1, 4:6].tolist(),
                              self.ground_truth[self.times - 1, 6:8].tolist()]
            self.actual_corners=np.array(self.actual_corners).T
        else:
            self.actual_corners=self.corners.copy()

        self.updateError(self.actual_corners, self.corners)
        if self.curr_error<=self.success_threshold:
            self.success_count+=1
            self.success_drift.append(self.curr_error)

        if self.tracker_pause:
            raw_input("Press Enter to continue...")

        self.last_time=self.current_time
        self.current_time=time.clock()

        self.average_fps=(self.times+1)/(self.current_time-self.start_time)
        self.current_fps = 1.0 / (self.current_time - self.last_time)

        return True

    def updateError(self, actual_corners, tracked_corners):
        self.error=0
        if self.from_cam:
            self.curr_error=0
            return
        for i in xrange(2):
            for j in xrange(4):
                self.curr_error += math.pow(actual_corners[i, j] - tracked_corners[i, j], 2)
        self.curr_error = math.sqrt(self.curr_error / 4)

    def display(self):
        annotated_img = self.img.copy()
        if self.tracker.is_initialized():
            draw_region(annotated_img, self.corners, (0, 0, 255), 2)
            draw_region(annotated_img, self.actual_corners, (0, 255, 0), 2)
            self.res_file.write('%-15s%-12.2f%-12.2f%-12.2f%-12.2f%-12.2f%-12.2f%-12.2f%-12.2f\n' % (
                'frame' + ('%05d' % (self.times)) + '.jpg', self.corners[0, 0],
                self.corners[1, 0], self.corners[0, 1], self.corners[1, 1],
                self.corners[0, 2], self.corners[1, 2], self.corners[0, 3],
                self.corners[1, 3]))

        fps_text = "%5.2f"%self.average_fps + "   %5.2f"%self.current_fps
        cv2.putText(annotated_img, fps_text, (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1, (255,255,255))
        cv2.imshow(self.track_window_name, annotated_img)

    def applyFiltering(self):
        if self.filter_type == 'none':
            proc_img = self.gray_img
        elif self.filter_type =='DoG' or \
                        self.filter_type == 'gauss' or \
                        self.filter_type == 'bilateral' or \
                        self.filter_type == 'median' or \
                        self.filter_type == 'canny':
             proc_img=self.filters[self.filter_type].apply(self.gray_img)
        elif self.filter_type in self.filters.keys():
            proc_img=self.filters[self.filter_type].apply(self.gray_img_float)
        else:
            print "Invalid filter type ",self.filter_type
            return None
        return proc_img

    def initFilterWindow(self):
        if self.window_inited:
            cv2.destroyWindow(self.proc_window_name)
            self.window_inited=False
        cv2.namedWindow(self.proc_window_name,flags=cv2.CV_WINDOW_AUTOSIZE)
        if self.filter_type!='none':
            for param in self.filters[self.filter_type].sorted_params:
                cv2.createTrackbar(param.name, self.proc_window_name,
                                   param.multiplier,
                                   param.limit, self.updateFilterParams)
        self.window_inited=True

    def updateFilterParams(self, val):
        if self.filters[self.filter_type].validated:
            return
        #print 'starting updateFilterParams'
        for i in xrange(len(self.filters[self.filter_type].params)):
            new_val=cv2.getTrackbarPos(self.filters[self.filter_type].params[i].name, self.proc_window_name)
            old_val=self.filters[self.filter_type].params[i].multiplier
            if new_val!=old_val:
                self.filters[self.filter_type].params[i].updateValue(new_val)
                if not self.filters[self.filter_type].validate():
                    self.filters[self.filter_type].params[i].updateValue(old_val)
                    cv2.setTrackbarPos(self.filters[self.filter_type].params[i].name, self.proc_window_name,
                                       self.filters[self.filter_type].params[i].multiplier)
                    self.filters[self.filter_type].validated=False
                break
        self.filters[self.filter_type].kernel = self.filters[self.filter_type].update()
        if self.write_res:
            self.write_res=False
            self.writeResults()
        self.reset=True

    def getParamStrings(self):
        dataset_params=''
        if self.from_cam:
            dataset_params='cam'
        else:
            start_id=self.labels.index('type')
            for i in xrange(start_id, len(self.init_params)):
                dataset_params=dataset_params+'_'+self.init_params[i]
            dataset_params=dataset_params+'_%d'%(self.times+1)
        filter_id='none'
        filter_params=''
        if self.filter_type!='none':
            filter_id=self.filters[self.filter_type].type
            for key in self.filters[self.filter_type].params.keys():
                filter_params=filter_params+'_'+self.filters[self.filter_type].params[key].name\
                              +'_%d'%self.filters[self.filter_type].params[key].val
        tracker_params=''
        #tracker_id=self.trackers[self.tracker_type].type
        #print 'tracker_id=', tracker_id
        for key in self.trackers[self.tracker_type].params.keys():
            tracker_params=tracker_params+'_'+self.trackers[self.tracker_type].params[key].name\
                          +'_'+str(self.trackers[self.tracker_type].params[key].val)

        return [dataset_params, filter_id, filter_params, tracker_params]

    def writeResults(self):
        if self.times==0:
            return
        print('Saving results...')
        [dataset_params, filter_id, filter_params, tracking_params]=self.getParamStrings()
        self.max_fps = max(self.curr_fps_list[1:])
        min_fps=min(self.curr_fps_list[1:])
        self.max_error = max(self.curr_error_list)


        if self.batch_mode:
            tracking_res_dir='Results/batch'
        else:
            tracking_res_dir='Results'

        if not os.path.isdir(tracking_res_dir):
            os.makedirs(tracking_res_dir)

        tracking_res_fname=tracking_res_dir+'/summary.txt'

        if not os.path.exists(tracking_res_fname):
            res_file=open(tracking_res_fname, 'a')
            res_file.write(
                "tracker".ljust(10) +
                "\tcolor_space".ljust(10) +
                "\tfilter".ljust(10) +
                "\tmultichannel".ljust(15) +
                "\tSCV".ljust(10) +
                "\tavg_error".rjust(14) +
                "\tmax_error".rjust(14) +
                "\tsuccess".rjust(14) +
                "\tdrift".rjust(14) +
                "\tavg_fps".rjust(14) +
                "\tmax_fps".rjust(14) +
                "\tmin_fps".rjust(14) +
                "\tdataset".center(50) +
                "\ttracking params".center(100) +
                "\tfilter params".center(50) + '\n'
            )
        else:
            res_file=open(tracking_res_fname, 'a')

        success_rate=float(self.success_count)/float(self.times+1)*100
        if self.success_count>0:
            drift=sum(self.success_drift)/float(self.success_count)
        else:
            drift=-1

        print 'self.tracker.multi_approach=', self.tracker.multi_approach
        print 'filter_id=', filter_id
        print 'self.color_space=', self.color_space
        print 'self.tracker_type=', self.tracker_type

        res_file.write(
            self.tracker_type.ljust(10) +
            "\t" + self.color_space.ljust(10) +
            "\t" + filter_id.ljust(10) +
            "\t" + self.tracker.multi_approach.ljust(15) +
            "\t" + str(self.tracker.use_scv).ljust(10) +
            "\t%13.6f" % self.avg_error +
            "\t%13.6f" % self.max_error +
            "\t%13.6f" % success_rate +
            "\t%13.6f" % drift +
            "\t%13.6f" % self.average_fps +
            "\t%13.6f" % self.max_fps +
            "\t%13.6f" % min_fps +
            "\t" + dataset_params.center(50) +
            "\t" + tracking_params.center(100) +
            "\t" + filter_params.center(50) + "\n"
        )
        res_file.close()

        print 'success rate:', success_rate
        print 'average fps:', self.average_fps
        print 'average drift:', drift

        if self.avg_filename is not None and self.agg_filename is not None:
            print 'writing avg data to ', 'Results/'+self.avg_filename+'.txt'
            avg_full_name='Results/'+self.avg_filename+'.txt'
            if not os.path.exists(avg_full_name):
                avg_file=open(avg_full_name, 'a')
                avg_file.write(
                    "parameters".center(len(self.agg_filename)) +
                    "\tsuccess_rate".center(14) +
                    "\tavg_fps".center(14) +
                    "\tavg_drift\n".center(14)
                )
            else:
                avg_file=open(avg_full_name, 'a')
            avg_file.write(
                self.agg_filename +
                "\t%13.6f" % success_rate +
                "\t%13.6f" % self.average_fps +
                "\t%13.6f\n" % drift
            )
            avg_file.close()
        self.savePlots(dataset_params, filter_id, filter_params, tracking_params)
        #webbrowser.open(tracking_res_fname)

    def generateCombinedPlots(self):
        combined_fig=plt.figure(1)
        plt.subplot(211)
        plt.title('Tracking Error')
        plt.ylabel('Error')
        plt.plot(self.frame_times, self.avg_error_list, 'r',
                 self.frame_times, self.curr_error_list, 'g')

        plt.subplot(212)
        plt.title('FPS')
        plt.xlabel('Frame')
        plt.ylabel('FPS')
        plt.plot(self.frame_times, self.avg_fps_list, 'r',
                 self.frame_times, self.curr_fps_list, 'g')
        return combined_fig

    def savePlots(self,dataset_params, filter_id, filter_params, tracking_params):
        print('Saving plot data...')
        if self.batch_mode:
            res_dir='Results/batch/'+self.tracker_type+'/'+filter_id
        else:
            res_dir='Results/'+self.tracker_type+'/'+filter_id
        plot_dir=res_dir+'/plots'
        res_template=dataset_params+'_'+filter_params+'_'+self.color_space\
                     +'_'+self.tracker.multi_approach+'_scv_'+str(self.tracker.use_scv)
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)
        plot_fname=plot_dir+'/'+res_template
        combined_fig=self.generateCombinedPlots()
        combined_fig.savefig(plot_fname, ext='png', bbox_inches='tight')
        plt.figure(0)

        res_fname=res_dir+'/'+res_template+'.txt'
        res_file=open(res_fname,'w')
        res_file.write(tracking_params+'\n')
        res_file.write("curr_fps".rjust(10)+"\t"+"avg_fps".rjust(10)+"\t\t"+
                       "curr_error".rjust(10)+"\t"+"avg_error".rjust(10)+"\n")
        for i in xrange(len(self.avg_fps_list)):
            res_file.write("%10.5f\t" % self.curr_fps_list[i] +
                           "%10.5f\t\t" % self.avg_fps_list[i] +
                           "%10.5f\t" % self.curr_error_list[i] +
                           "%10.5f\n" % self.avg_error_list[i])
        res_file.close()
        getThresholdVariations(res_dir, res_template, 'error', show_plot=False,
                           min_thresh=0, diff=1, max_thresh=100, max_rate=100,
                           agg_filename=self.agg_filename)
        getThresholdVariations(res_dir, res_template, 'fps', show_plot=False,
                           min_thresh=0, diff=1, max_thresh=30, max_rate=100,
                           agg_filename=self.agg_filename)

    def cleanup(self):
        self.res_file.close()

