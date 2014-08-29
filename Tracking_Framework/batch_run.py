import os
import init
from Misc import *

'''
Use batch_dict to specify the variables that are to be batched along with the values they are
to be batched with.
The 'list' key specifies the values that are to be batched over and can be:
    None: id this variable is to be ignored (not included in the batch)
    []: if all the values specified in the default parameters are to be used
    any valid list: batch will be run over only the values in this list
The 'id' key specifies the order in which these variables will be run in the batch.
Plotting will be done with respect to the variable with the highest id
'''
batch_dict = {
    'type': {'id': 0, 'list': None},
    'task': {'id': 1, 'list': []},
    'tracker': {'id': 2, 'list': ['nn']},
    'color_space': {'id': 3, 'list': ['RGB', 'HSV']},
    'filter': {'id': 4, 'list': None},
    'sigma': {'id': 5, 'list': None},
    'smoothing_kernel': {'id': 6, 'list': None},
    'multi_approach': {'id': 7, 'list': []},
    'ksize': {'id': 8, 'list': None}

}
current_task_type = 'simple'
batch_variables = [key for key in batch_dict.keys() if batch_dict[key]['list'] is not None]
batch_variables = sorted(batch_variables, key=lambda k: batch_dict[k]['id'])
print 'batch_variables=\n', batch_variables
base_command = 'python main.py'

[params, labels, default_id] = init.getBasicParams()
tracking_params = init.getTrackingParams()
filtering_params = init.getFilteringParams()

tracker_batch = False
tracker_index = labels.index('tracker')
current_tracker = params[tracker_index][default_id[tracker_index]]
if 'tracker' in batch_variables:
    tracker_list = batch_dict['tracker']['list']
    if not tracker_list or len(tracker_list) > 1:
        tracker_batch = True
    else:
        current_tracker = batch_dict['tracker']['list'][0]
curr_tracking_params = tracking_params[current_tracker]
print 'current_tracker=', current_tracker

filter_batch = False
filter_index = labels.index('filter')
current_filter = params[filter_index][default_id[filter_index]]
if 'filter' in batch_variables:
    filter_list = batch_dict['filter']['list']
    if not filter_list or len(filter_list) > 1:
        filter_batch = True
    else:
        current_filter = batch_dict['filter']['list'][0]

if current_filter != 'none':
    curr_filtering_params = filtering_params[current_filter]
else:
    curr_filtering_params = {}
print 'current_filter=', current_filter

batch_lists = []
task_types = params[labels.index('type')]
for variable in batch_variables:
    if not batch_dict[variable]['list']:
        if variable in labels:
            curr_list = params[labels.index(variable)]
            if variable == 'task':
                curr_list = curr_list[task_types.index(current_task_type)]
                #print variable, ' size=', len(curr_list)
        elif variable in curr_tracking_params.keys():
            if tracker_batch:
                raise SystemExit('Trackers and tracker parameters cannot be batched simultaneously')
            curr_list = curr_tracking_params[variable]['list']
        elif variable in curr_filtering_params.keys():
            if filter_batch:
                raise SystemExit('Filters and filter parameters cannot be batched simultaneously')
            curr_list = range(int(curr_filtering_params[variable]['default']['add']),
                              int(curr_filtering_params[variable]['default']['limit']), 1)
        else:
            msg = 'Invalid batch variable ' + variable + ' specified'
            raise SystemExit(msg)
    else:
        curr_list = batch_dict[variable]['list']
    batch_lists.append(curr_list)
    print 'list for ', variable, ':\n', curr_list
batch_variable_count = len(batch_variables)


def runBatchCommands(command='', batch_index=0, agg_filename=''):
    list = batch_lists[batch_index]
    var = batch_variables[batch_index]

    if batch_index == batch_variable_count - 1:
        agg_filename = agg_filename + var
        header = ''
        for val in list:
            header = header + str(val) + '\t'
        error_agg_fname = 'error_' + agg_filename
        agg_file = open('Results/' + error_agg_fname + '.txt', 'w')
        agg_file.write(header + '\n')
        agg_file.close()

        fps_agg_fname = 'fps_' + agg_filename
        agg_file = open('Results/' + fps_agg_fname + '.txt', 'w')
        agg_file.write(header + '\n')
        agg_file.close()

    for val in list:
        updated_command = command + ' ' + var + ' ' + str(val)
        if batch_index < batch_variable_count - 1:
            updated_agg_filename = agg_filename + str(val)+'_'
            runBatchCommands(updated_command, batch_index + 1, updated_agg_filename)
        else:
            avg_filename=var+'_'+str(val)
            if avg_filename not in avg_filenames:
                avg_filenames.append(avg_filename)
            os.system(base_command + updated_command + ' aggregate ' + agg_filename+' average'+' '+avg_filename)
    if batch_index == batch_variable_count - 1:
        error_agg_fname = 'error_' + agg_filename
        aggregateDataFromFiles(error_agg_fname, error_agg_fname)
        fps_agg_fname = 'fps_' + agg_filename
        aggregateDataFromFiles(fps_agg_fname, fps_agg_fname)

avg_filenames=[]
runBatchCommands()
#getPointPlot(filenames=avg_filenames, plot_fname=batch_variables[-1])
list_file=open('Results/list.txt', 'w')
list_file.write(batch_variables[-1]+'\n')
for filename in avg_filenames:
    list_file.write(filename+'\n')
