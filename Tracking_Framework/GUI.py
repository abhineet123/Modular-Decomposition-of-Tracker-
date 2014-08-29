import Tkinter as tk
import sys

class GUI:
    def __init__(self, obj, title):
        self.title=title
        self.params=obj.params
        self.labels=obj.labels
        self.trackers=obj.trackers

        self.obj=obj

        self.source_index=obj.labels.index('source')
        self.task_index=obj.labels.index('task')
        self.type_index=obj.labels.index('type')
        self.tracker_index=obj.labels.index('tracker')
        self.filter_index=obj.labels.index('filter')
        self.color_index=obj.labels.index('color_space')

        self.tracker_type=obj.tracker_type
        self.current_id=obj.default_id

        self.top_widgets=self.labels[:self.type_index]
        self.param_widgets=self.getTrackingParams()
        self.divider_id='dataset_divider'
        self.data_widgets=[self.divider_id]+self.labels[self.type_index:]

        self.widgets=self.top_widgets+self.param_widgets+self.data_widgets

        self.param_start_id=len(self.top_widgets)
        self.parent_frames=None
        self.root=None
        self.frames_created=False
        self.no_of_rows=0
        self.first_instance=True
        self.first_call=True

        self.item_vars=None
        self.item_menus=None
        self.item_labels=None

        self.divide_label=None

        self.entry_labels=None
        self.entry_widgets=None
        self.entry_vars=None

        self.scv_state=False
        self.color_space=None

        self.start_button_label='Start'
        self.cancel_button_label='Exit'

        self.initWidgets()
        self.root.mainloop()

    def initRoot(self):
        if self.root != None:
            self.destroyRoot()

        self.root = tk.Tk()
        self.root.wm_title(self.title)
        self.root.title(self.title)
        self.frames_created=False

    def getTrackingParams(self):
        params=list(self.trackers[self.tracker_type].params.keys())
        return params

    def removeFrames(self):
        for i in xrange(1, len(self.parent_frames)):
            frame=self.parent_frames[i]
            frame.pack_forget()
        self.frames_created=False

    def removeOptionMenus(self):
        for i in xrange(1, len(self.labels)):
            self.item_labels[i].pack_forget()
            self.item_menus[i].pack_forget()

    def createFrames(self):
        self.parent_frames=[self.root]
        levels=[0]
        for node_id in xrange(self.no_of_rows):
            new_frame=tk.Frame(self.parent_frames[node_id])
            new_frame.pack(side=tk.TOP)
            self.parent_frames.append(new_frame)
            levels.append(levels[node_id]+1)

            new_frame=tk.Frame(self.parent_frames[node_id])
            new_frame.pack(side=tk.BOTTOM)
            self.parent_frames.append(new_frame)
            levels.append(levels[node_id]+1)
        self.parent_frames=self.rearrangeNodes(self.parent_frames,levels,self.no_of_rows)
        self.frames_created=True

    def createTrackerParamWidgets(self):
        #print "Creating parameter entry boxes for ", self.trackers[self.tracker_type].type, "tracker"

        #if self.entry_vars is not None:
        #    self.setTrackingParams()

        self.entry_labels=[]
        self.entry_widgets=[]
        self.entry_vars=[]

        #no_of_params=len(self.trackers[self.tracker_type].params)
        frame_id=self.param_start_id+self.no_of_rows
        for param in self.trackers[self.tracker_type].sorted_params:
            if param.type=='boolean':
                self.scv_state=param.val
                entry_var = tk.IntVar(self.parent_frames[frame_id])
                if param.val==True:
                    entry_var.set(1)
                else:
                    entry_var.set(0)
                entry_widget = tk.Checkbutton(self.parent_frames[frame_id],
                                              text=param.name, variable=entry_var,
                                              onvalue=1, offvalue=0,
                                              command=self.toggleSCV)
                entry_label=None
            else:
                #print param.name, "=", param.val
                entry_label=tk.Label(self.parent_frames[frame_id], text=param.name)
                entry_var = tk.StringVar(self.parent_frames[frame_id])
                entry_var.set(str(param.val))
                if param.type=='string':
                    entry_widget= tk.OptionMenu(self.parent_frames[frame_id],
                                                entry_var, *param.list)
                elif param.type=='int' or param.type=='float':
                    entry_widget = tk.Entry(self.parent_frames[frame_id], textvariable=entry_var, width=6)
                entry_label.pack(side=tk.LEFT)
            entry_widget.pack(side=tk.LEFT)

            self.entry_labels.append(entry_label)
            self.entry_widgets.append(entry_widget)
            self.entry_vars.append(entry_var)
            frame_id+=1
        #self.setColorSpace()

    def createOptionMenus(self):
        self.item_vars=[]
        self.item_menus=[]
        self.item_labels=[]

        for i in range(len(self.labels)):
            widget=self.labels[i]
            index=self.widgets.index(widget)
            frame_id=index+self.no_of_rows
            item_var = tk.StringVar(self.parent_frames[frame_id])
            if widget=='task':
                item_var.set(self.params[i][self.current_id[self.type_index]][self.current_id[i]])
                item_menu = tk.OptionMenu(self.parent_frames[frame_id], item_var,
                                          *self.params[i][self.current_id[self.type_index]])
            else:
                current_val=self.params[i][self.current_id[i]]
                item_var.set(current_val)
                if widget=='type':
                    item_menu = tk.OptionMenu(self.parent_frames[frame_id], item_var,
                                              *self.params[i], command=self.setTasks)
                elif widget=='source':
                    item_menu = tk.OptionMenu(self.parent_frames[frame_id], item_var,
                                              *self.params[i], command=self.setSource)
                elif widget=='tracker':
                    item_menu = tk.OptionMenu(self.parent_frames[frame_id], item_var,
                                              *self.params[i], command=self.setTracker)
                elif widget=='color_space':
                    self.color_space=current_val
                    item_menu = tk.OptionMenu(self.parent_frames[frame_id], item_var,
                                              *self.params[i], command=self.setColorSpace)
                elif widget=='smoothing':
                    item_menu = tk.OptionMenu(self.parent_frames[frame_id], item_var,
                                              *self.params[i], command=self.setSmoothing)
                else:
                    item_menu = tk.OptionMenu(self.parent_frames[frame_id], item_var, *self.params[i])

            item_label=tk.Label(self.parent_frames[frame_id], text=self.labels[i], padx=10)

            item_label.pack(side=tk.LEFT)
            item_menu.pack(side=tk.LEFT)

            self.item_vars.append(item_var)
            self.item_menus.append(item_menu)
            self.item_labels.append(item_label)

    def createDivider(self):
        index=self.widgets.index(self.divider_id)
        frame_id=index+self.no_of_rows
        self.divide_label=tk.Label(self.parent_frames[frame_id], text="--------Dataset--------",
                              relief=tk.FLAT)
        self.divide_label.pack(side=tk.LEFT)

    def createButtons(self):
        button_ok = tk.Button(self.parent_frames[-1], text=self.start_button_label,
                              command=self.ok, padx=10)
        button_ok.pack(side=tk.LEFT)
        button_cancel = tk.Button(self.parent_frames[-1], text=self.cancel_button_label,
                                  command=self.exit, padx=10)
        button_cancel.pack(side=tk.LEFT)

    def initWidgets(self, start_label=None):
        if not self.first_instance:
            self.updateCurrentID()

        if self.root==None:
            self.initRoot()

        if self.frames_created:
            self.removeFrames()

        if start_label!=None:
            self.start_button_label=start_label

        self.widgets=self.top_widgets+self.param_widgets+self.data_widgets
        self.no_of_rows=len(self.widgets)

        self.createFrames()
        self.createTrackerParamWidgets()
        self.createDivider()
        self.createOptionMenus()
        self.createButtons()

        self.setSource()
        self.setColorSpace()
        self.setSmoothing()


    def setSource(self, source=None):
        if source is None:
            if self.item_vars is None:
                return
            else:
                source=self.item_vars[self.labels.index('source')].get()
        source_id=self.params[self.source_index].index(source)
        for i in xrange(self.type_index, len(self.labels)):
            item_menu=self.item_menus[i]
            item_label=self.item_labels[i]
            if source_id==1:
                item_menu.configure(state='disabled')
                item_label.configure(state='disabled')
            else:
                item_menu.configure(state='normal')
                item_label.configure(state='normal')
        if source_id==1:
            self.divide_label.configure(state='disabled')
        else:
            self.divide_label.configure(state='normal')

    def toggleSCV(self):
        scv_id=self.trackers[self.tracker_type].params['enable_scv'].id
        value=self.entry_vars[scv_id].get()
        if value==1:
            self.scv_state=True
            self.setFilterMenuState('disabled', 'none')
        elif value==0:
            self.scv_state=False
            if self.color_space.lower()=='grayscale':
                self.setFilterMenuState('normal')
        else:
            raise SystemExit('Invalid value for SCV checkbox: '+str(value))


    def setColorSpace(self, value=None):
        if value is None:
            if self.item_vars is None:
                return
            else:
                value=self.item_vars[self.labels.index('color_space')].get()
        self.color_space=value
        if value.lower()!='grayscale':
            self.setFilterMenuState('disabled', 'none')
            val=self.trackers[self.tracker_type].params['multi_approach'].val
            self.setEntryMenuState('normal', val)
        else:
            if self.scv_state==False:
                self.setFilterMenuState('normal')
            self.setEntryMenuState('disabled', 'none')

    def setSmoothing(self, value=None):
        if value is None:
            if self.item_vars is None:
                return
            else:
                value=self.item_vars[self.labels.index('smoothing')].get()
        if value.lower()=='none':
            self.setSmoothingKernelState(state='disabled')
        else:
            self.setSmoothingKernelState(state='normal')


    def setTracker(self, value):
        self.tracker_type=value
        self.param_widgets=self.getTrackingParams()
        self.initWidgets()


    def setTasks(self, value):
        type_id=self.params[self.type_index].index(value)
        frame_id=self.widgets.index('task')+self.no_of_rows
        self.item_vars[self.task_index].set(self.params[self.task_index][type_id][self.current_id[self.task_index]])
        self.item_menus[self.task_index].pack_forget()
        self.item_menus[self.task_index] = tk.OptionMenu(self.parent_frames[frame_id],
                                                   self.item_vars[self.task_index], *self.params[self.task_index][type_id])
        self.item_menus[self.task_index].pack(side=tk.LEFT)

    def setFilterMenuState(self, state, val=None):

        filter_menu=self.item_menus[self.filter_index]
        filter_label=self.item_labels[self.filter_index]
        filter_var=self.item_vars[self.filter_index]

        print 'state=', state
        filter_menu.configure(state=state)
        filter_label.configure(state=state)
        if val is not None:
            filter_var.set(val)

    def setEntryMenuState(self, state=None, val=None):
        multichannel_id=self.trackers[self.tracker_type].params['multi_approach'].id
        entry_menu=self.entry_widgets[multichannel_id]
        entry_label=self.entry_labels[multichannel_id]
        entry_var=self.entry_vars[multichannel_id]

        if state is None:
            if entry_var.get().lower()=='none':
                state='disabled'
            else:
                state='normal'
        if entry_menu.cget('state')!=state:
            entry_menu.configure(state=state)
        if entry_label.cget('state')!=state:
            entry_label.configure(state=state)
        if val is not None:
            entry_var.set(val)

    def setSmoothingKernelState(self, state, val=None):

        smoothing_kernel_index=self.labels.index('smoothing_kernel')
        smoothing_kernel_menu=self.item_menus[smoothing_kernel_index]
        smoothing_kernel_label=self.item_labels[smoothing_kernel_index]
        smoothing_kernel_var=self.item_vars[smoothing_kernel_index]

        smoothing_kernel_menu.configure(state=state)
        smoothing_kernel_label.configure(state=state)
        if val is not None:
            smoothing_kernel_var.set(val)

    def ok(self):
        init_params=[]
        for i in xrange(len(self.item_vars)):
            init_params.append(self.item_vars[i].get())
        self.setTrackingParams()
        if not self.obj.initSystem(init_params):
            sys.exit()

        if self.first_call:
            self.first_call=False
            self.destroyRoot()
        else:
            self.obj.reset=True

    def exit(self):
        self.obj.exit_event=True
        if not self.first_call:
            self.obj.writeResults()
        sys.exit()

    def destroyRoot(self):
        self.root.destroy()
        self.root=None
        self.parent_frames=None

    def updateCurrentID(self):
        #print "\n\n"
        #print "in updateCurrentID:"
        for i in xrange(len(self.item_vars)):
            current_val=self.item_vars[i].get()
            #print "updating ", self.labels[i], "to ", current_val
            if i==self.task_index:
                current_type=self.item_vars[self.type_index].get()
                current_type_id=self.params[self.type_index].index(current_type)
                current_id=self.params[i][current_type_id].index(current_val)
            else:
                #if self.labels[i]=='smoothing_kernel':
                #    current_val=int(current_val)
                current_id=self.params[i].index(current_val)
            self.current_id[i]=current_id
        #print "\n\n"

    def setTrackingParams(self):
        #print "\n\n"
        #print "in setTrackingParams found:"
        for param in self.trackers[self.tracker_type].sorted_params:
            val_str=self.entry_vars[param.id].get()
            if param.type=='int':
                val=int(val_str)
            elif param.type=='float':
                val=float(val_str)
            elif param.type=='string':
                val=val_str
            elif param.type=='boolean':
                if val_str==0:
                    val=False
                else:
                    val=True
            else:
                raise SystemExit("invalid param type: " + param.type)
            #print param.name, "=", val
            param.val=val
        #print "\n\n"

    def shiftFromLast(self, tree, n):
        temp_node=tree[-1]
        for i in xrange(1,n+1):
            tree[-i]=tree[-(i+1)]
        tree[n]=temp_node
        return tree

    def shiftToLast(self, tree, n):
        temp_node=tree[n]
        for i in xrange(n,len(tree)-1):
            tree[i]=tree[i+1]
        tree[-1]=temp_node
        return tree

    def rearrangeNodes(self, tree,levels,nrows):
        for i in xrange(1,nrows+1):
            if(levels[-1]>levels[nrows+i-1]):
                tree=self.shiftFromLast(tree, nrows)
                levels=self.shiftFromLast(levels, nrows)
            else:
                break
        return tree