from Homography import *
from InteractiveTracking import *
from matplotlib import pyplot as plt
from matplotlib import animation
import init
from FilteringParams import *
from TrackingParams import *
import sys

class StandaloneTrackingApp(InteractiveTrackingApp):
    """ A demo program that uses OpenCV to grab frames. """

    def __init__(self, init_frame, root_path,
                 params, tracking_params, filtering_params, labels, default_id,
                 buffer_size, success_threshold=5, batch_mode=False,
                 agg_filename=None, avg_filename=None):
        track_window_name = 'Tracked Images'
        InteractiveTrackingApp.__init__(self, init_frame, root_path, track_window_name, params,
                                        tracking_params, filtering_params, labels, default_id,
                                        success_threshold, batch_mode, agg_filename, avg_filename)
        self.buffer_id=0
        self.buffer_end_id=-1
        self.buffer_start_id=0
        self.buffer_full=False
        self.buffer_size=buffer_size
        self.current_buffer_size=0
        self.frame_buffer=[]
        self.corners_buffer=[]
        self.actual_corners_buffer=[]
        self.rewind=False
        self.from_frame_buffer=False
        self.plot_index=0
        self.cam_skip_frames=50

    def run(self):
        if self.batch_mode and self.source=='camera':
            raise SystemExit('Batch mode cannot be run with camera')
        i = self.init_frame
        while True:
            #if not self.keyboardHandler():
            #    self.writeResults()
            #    sys.exit()
            if not self.updateTracker(i):
                self.writeResults()
                sys.exit()
            #if self.img is not None:
            #    self.display()
            i += 1

    def updateTracker(self, i):
        if self.reset:
            self.reset=False
            self.inited=False
            self.initPlotParams()
        if self.exit_event or i==self.no_of_frames-1:
            return False
        if self.source=='camera':
            ret, img = self.cap.read()
            if not ret:
                print "Frame could not be read from camera"
                return False
            if not self.inited:
                print "Skipping ", self.cam_skip_frames, " frames...."
                for j in xrange(self.cam_skip_frames):
                    ret, img = self.cap.read()
        else:
            if i>=self.no_of_frames:
                self.exit_event=True
                return False
            img_file = self.img_path + '/img%d.jpg' % (i + 1)
            img = cv2.imread(img_file)
            if img == None:
                print "error loading image %s" % img_file
                return False

        if not self.on_frame(img, i):
            return False

        self.frame_times.append(i)
        self.curr_error_list.append(self.curr_error)
        self.avg_error=np.mean(self.curr_error_list)
        self.avg_error_list.append(self.avg_error)
        self.curr_fps_list.append(self.current_fps)
        self.avg_fps_list.append(self.average_fps)

        return True

    def onKeyPress(self, event):
        print 'key pressed=',event.key
        if event.key == "escape":
            self.exit_event=True
            sys.exit()
        elif event.key=="shift":
            if not self.from_cam or True:
                self.switch_plot=True
                self.plot_fps=not self.plot_fps
        elif event.key==" ":
            self.paused=not self.paused

    def keyboardHandler(self):
        key = cv2.waitKey(1)
        #if key!=-1:
        #    print "key=", key
        if key == ord(' '):
            self.paused = not self.paused
        elif key==27:
            return False
        elif key==ord('p') or key==ord('P'):
            if not self.from_cam or True:
                self.switch_plot=True
                self.plot_fps=not self.plot_fps
        elif key==ord('w') or key==ord('W'):
            self.write_res=not self.write_res
            if self.write_res:
                print "Writing results enabled"
            else:
                print "Writing results disabled"
        elif key==ord('t') or key==ord('T'):
            self.tracker_pause=not self.tracker_pause
        elif key==ord('r') or key==ord('R'):
            if self.from_frame_buffer:
                self.rewind=not self.rewind
                if self.rewind:
                    print "Disabling rewind"
                    #self.from_frame_buffer=False
                    #self.rewind=False
                else:
                    print "Enabling rewind"
                    #self.rewind=True
            else:
                print "Switching to frame buffer"
                print "Enabling rewind"
                self.from_frame_buffer=True
                self.rewind=True
                self.buffer_id=self.buffer_end_id
        return True


    def updatePlots(self, frame_count):
        if self.from_cam:
            ax.set_xlim(0, frame_count)
        if self.switch_plot:
            self.switch_plot=False
            #print "here we are"
            if self.plot_fps:
                fig.canvas.set_window_title('FPS')
                plt.ylabel('FPS')
                plt.title('FPS')
                self.max_fps = max(self.curr_fps_list)
                ax.set_ylim(0, self.max_fps)
            else:
                fig.canvas.set_window_title('Tracking Error')
                plt.ylabel('Error')
                plt.title('Tracking Error')
                self.max_error = max(self.curr_error_list)
                ax.set_ylim(0, self.max_error)
            plt.draw()

        if self.plot_fps:
            line1.set_data(self.frame_times[0:self.plot_index+1], self.avg_fps_list[0:self.plot_index+1])
            line2.set_data(self.frame_times[0:self.plot_index+1], self.curr_fps_list[0:self.plot_index+1])
            #line3.set_data(self.frame_times[self.plot_index],self.curr_fps_list[self.plot_index])
            if max(self.curr_fps_list) > self.max_fps:
                self.max_fps = max(self.curr_fps_list)
                ax.set_ylim(0, self.max_fps)
                plt.draw()
        else:
            line1.set_data(self.frame_times[0:self.plot_index+1], self.avg_error_list[0:self.plot_index+1])
            line2.set_data(self.frame_times[0:self.plot_index+1], self.curr_error_list[0:self.plot_index+1])
            if max(self.curr_error_list)>self.max_error:
                self.max_error = max(self.curr_error_list)
                ax.set_ylim(0,self.max_error)
                plt.draw()

    def animate(self, i):
        if not self.keyboardHandler():
            self.writeResults()
            sys.exit()

        if self.paused:
            return line1, line2

        if not self.buffer_full:
            if len(self.frame_buffer)>=self.buffer_size:
                print "Frame buffer full"
                #print "buffer_end_id=", self.buffer_end_id
                #print "buffer_start_id=", self.buffer_start_id
                self.buffer_full=True

        if self.from_frame_buffer:
            if self.rewind:
                self.buffer_id-=1
                self.plot_index-=1
                if self.buffer_id<0:
                    self.buffer_id=self.buffer_size-1
                elif self.buffer_id==self.buffer_start_id:
                    print "Disabling rewind"
                    self.rewind=False
            else:
                self.buffer_id+=1
                self.plot_index+=1
                if self.buffer_id>=self.buffer_size:
                    self.buffer_id=0
                elif self.buffer_id==self.buffer_end_id:
                    self.from_frame_buffer=False
                    print "Getting back to video stream"
            self.img=self.frame_buffer[self.buffer_id]
            self.corners=self.corners_buffer[self.buffer_id]
            self.actual_corners=self.actual_corners_buffer[self.buffer_id]
        else:
            self.plot_index=i
            if not self.updateTracker(i):
                self.writeResults()
                sys.exit()
            if not self.buffer_full:
                self.frame_buffer.append(self.img.copy())
                self.corners_buffer.append(self.corners.copy())
                self.actual_corners_buffer.append(self.actual_corners.copy())
                self.buffer_end_id+=1
            else:
                self.frame_buffer[self.buffer_start_id]=self.img.copy()
                self.corners_buffer[self.buffer_start_id]=self.corners.copy()
                self.actual_corners_buffer[self.buffer_start_id]=self.actual_corners.copy()
                self.buffer_end_id=self.buffer_start_id
                self.buffer_start_id=(self.buffer_start_id+1) % self.buffer_size

        if self.img is not None:
            self.display()
        self.updatePlots(i)
        return line1, line2

def simData():
    i=-1
    while not app.exit_event:
        if not app.paused and not app.from_frame_buffer:
            i+=1
        if app.reset:
            print "Resetting the plots..."
            ax.cla()
            plt.draw()
            i=0
        yield i

def processArguments(args):
    no_of_args=len(args)
    if no_of_args%2!=0:
        print 'args=\n', args
        raise SystemExit('Error in processArguments: '
                         'Optional arguments need to be specified in pairs')
    agg_filename=None
    avg_filename=None
    for i in xrange(no_of_args/2):
        arg_label=sys.argv[i*2+1]
        arg_val=sys.argv[i*2+2]
        #print 'arg_label=', arg_label
        #print 'arg_val=', arg_val
        if arg_label in labels:
            arg_index=labels.index(arg_label)
            #print 'arg_index=', arg_index
            if arg_label=='task':
                task_types=params[labels.index('type')]
                simple_tasks=params[labels.index('task')][task_types.index('simple')]
                complex_tasks=params[labels.index('task')][task_types.index('complex')]
                if arg_val in simple_tasks:
                    task_id=0
                elif arg_val in complex_tasks:
                    task_id=1
                else:
                    raise SystemExit('Invalid task provided')
                default_id[labels.index('type')]=task_id
                default_id[arg_index]=params[arg_index][task_id].index(arg_val)
            else:
                default_id[arg_index]=params[arg_index].index(arg_val)
        else:
            tracker_index=labels.index('tracker')
            current_tracker=params[tracker_index][default_id[tracker_index]]
            filter_index=labels.index('filter')
            current_filter=params[filter_index][default_id[filter_index]]
            curr_tracking_params=tracking_params[current_tracker]
            if current_filter=='none':
                curr_filtering_params={}
            else:
                curr_filtering_params=filtering_params[current_filter]
            if arg_label in curr_tracking_params.keys():
                param_type=curr_tracking_params[arg_label]['type']
                if param_type=='int':
                    arg_val=int(arg_val)
                elif param_type=='float':
                    arg_val=float(arg_val)
                elif param_type=='boolean':
                    if arg_val.lower()=='true':
                        arg_val=True
                    elif arg_val.lower()=='false':
                        arg_val=False
                    else:
                        msg='Invalid value ',arg_val, ' specified for parameter ',arg_label
                        raise SystemExit(msg)
                curr_tracking_params[arg_label]['default']=arg_val
            elif arg_label in curr_filtering_params.keys():
                #param_type=curr_filtering_params[arg_label]['type']
                #if param_type=='int':
                #    arg_val=int(arg_val)
                #elif param_type=='float':
                #    arg_val=float(arg_val)
                arg_val=int(arg_val)
                #param_base=current_filter[arg_label]['default']['base']
                #param_add=current_filter[arg_label]['type']['add']
                param_limit=curr_filtering_params[arg_label]['default']['limit']
                #mult=(arg_val-param_add)/param_base
                mult=arg_val
                if mult>param_limit:
                    msg='Specified value ', arg_val, 'for parameter ', arg_label,\
                        'exceeds the maximum allowed value.'
                    raise SystemExit(msg)
                curr_filtering_params[arg_label]['default']['mult']=mult
            elif arg_label=='aggregate':
                agg_filename=arg_val
            elif arg_label=='average':
                avg_filename=arg_val
            else:
                raise SystemExit('Error in processArguments:'
                                 'Invalid argument '+arg_label+' provided')
    return agg_filename, avg_filename

if __name__ == '__main__':

    init_frame = 0
    success_threshold=5
    frame_buffer_size=1000
    root_path = 'G:/UofA/Thesis/#Code/Datasets'

    [params, labels, default_id]=init.getBasicParams()
    tracking_params=init.getTrackingParams()
    filtering_params=init.getFilteringParams()

    agg_filename=None
    avg_filename=None
    batch_mode=False
    if len(sys.argv)>2:
        agg_filename, avg_filename=processArguments(sys.argv[1:])
        print 'avg_filename=', avg_filename
        batch_mode=True

    app = StandaloneTrackingApp(init_frame,root_path, params, tracking_params,
                                filtering_params, labels, default_id, frame_buffer_size,
                                success_threshold,batch_mode=batch_mode,
                                agg_filename=agg_filename, avg_filename=avg_filename)
    if batch_mode:
        app.run()
    else:
        fig = plt.figure(0)
        fig.canvas.set_window_title('Tracking Error')
        cid = fig.canvas.mpl_connect('key_press_event', app.onKeyPress)
        ax = plt.axes(xlim=(0, app.no_of_frames), ylim=(0, 5))
        plt.xlabel('Frame')
        #plt.ylabel('Error')
        #plt.title('Tracking Error')
        plt.grid(True)
        line1, line2 = ax.plot([], [], 'r', [], [], 'g')
        plt.legend(('Average', 'Current'))
        #plt.draw()
        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            #line3.set_data(0, 0)
            return line1,line2
        anim = animation.FuncAnimation(fig, app.animate, simData, init_func=init,
                                       interval=0,blit=True)
    #error = getTrackingError(dataset_path, res_path, dataset, tracker_id)
    plt.show()
    app.cleanup()

