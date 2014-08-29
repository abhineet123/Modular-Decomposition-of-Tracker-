__author__ = 'Tommy'
import os
import sys
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import font_manager as ftm

def readTrackingData(filename):
    if not os.path.isfile(filename):
        print "Tracking data file not found:\n ",filename
        sys.exit()

    data_file = open(filename, 'r')
    data_file.readline()
    lines = data_file.readlines()
    no_of_lines = len(lines)
    data_array = np.empty([no_of_lines, 8])
    line_id = 0
    for line in lines:
        #print(line)
        words = line.split()
        if (len(words) != 9):
            msg = "Invalid formatting on line %d" % line_id + " in file %s" % filename + ":\n%s" % line
            raise SyntaxError(msg)
        words = words[1:]
        coordinates = []
        for word in words:
            coordinates.append(float(word))
        data_array[line_id, :] = coordinates
        #print words
        line_id += 1
    data_file.close()
    return data_array

def getTrackingError(ground_truth_path, result_path, dataset, tracker_id):
    ground_truth_filename = ground_truth_path + '/' + dataset + '.txt'
    ground_truth_data = readTrackingData(ground_truth_filename)
    result_filename = result_path + '/' + dataset + '_res_%s.txt' % tracker_id

    result_data = readTrackingData(result_filename)
    [no_of_frames, no_of_pts] = ground_truth_data.shape
    error = np.zeros([no_of_frames, 1])
    #print "no_of_frames=", no_of_frames
    #print "no_of_pts=", no_of_pts
    if result_data.shape[0] != no_of_frames or result_data.shape[1] != no_of_pts:
        #print "no_of_frames 2=", result_data.shape[0]
        #print "no_of_pts 2=", result_data.shape[1]
        raise SyntaxError("Mismatch between ground truth and tracking result")

    error_filename = result_path + '/' + dataset + '_res_%s_error.txt' % tracker_id
    error_file = open(error_filename, 'w')
    for i in xrange(no_of_frames):
        data1 = ground_truth_data[i, :]
        data2 = result_data[i, :]
        for j in xrange(no_of_pts):
            error[i] += math.pow(data1[j] - data2[j], 2)
        error[i] = math.sqrt(error[i] / 4)
        error_file.write("%f\n" % error[i])
    error_file.close()
    return error

def extractColumn(filepath, filename, column, header_size=2):
    #print 'filepath=', filepath
    #print 'filename=', filename
    data_file = open(filepath+'/'+filename+'.txt', 'r')
    #remove header
    #read headersf
    for i in xrange(header_size):
        data_file.readline()
    lines = data_file.readlines()
    column_array = []
    line_id = 0
    for line in lines:
        #print(line)
        words = line.split()
        if (len(words) != 4):
            msg = "Invalid formatting on line %d" % line_id + " in file %s" % filename + ":\n%s" % line
            raise SyntaxError(msg)
        current_val=float(words[column])
        column_array.append(current_val)
        line_id += 1
    data_file.close()
    #print 'column_array=', column_array
    return column_array

def getThresholdRate(val_array, threshold, cmp_type):
    no_of_frames=len(val_array)
    if no_of_frames<1:
        raise SystemExit('Error array is empty')
    thresh_count=0
    for val in val_array:
        if cmp_type=='less':
            if val<=threshold:
                thresh_count+=1
        elif cmp_type=='more':
            if val>=threshold:
                thresh_count+=1

    rate=float(thresh_count)/float(no_of_frames)*100
    return rate

def getThresholdVariations(res_dir, filename, val_type, show_plot=False,
                           min_thresh=0, diff=1, max_thresh=100, max_rate=100, agg_filename=None):
    print 'Getting threshold variations for', val_type
    if val_type=='error':
        cmp_type='less'
        column=2
    elif val_type=='fps':
        cmp_type='more'
        column=0
    else:
        raise SystemExit('Invalid value type')
    val_array=extractColumn(res_dir, filename, column)
    rates=[]
    thresholds=[]
    threshold=min_thresh
    const_count=0
    rate=getThresholdRate(val_array, threshold, cmp_type)
    rates.append(rate)
    thresholds.append(threshold)
    while True:
        threshold+=diff
        last_rate=rate
        rate=getThresholdRate(val_array, threshold, cmp_type)
        rates.append(rate)
        thresholds.append(threshold)
        if rate==last_rate:
            const_count+=1
        else:
            const_count=0
        #print 'rate=', rate
        #if rate>=max_rate or const_count>=max_const or threshold>=max_thresh:
        #    break
        if threshold>=max_thresh:
            break
    outfile=val_type+'_'+filename+'_'+str(min_thresh)+'_'+str(diff)+'_'+str(max_rate)
    data_array=np.array([thresholds, rates])
    full_name=res_dir+'/'+outfile+'.txt'
    np.savetxt(full_name, data_array.T, delimiter='\t', fmt='%11.5f')
    if agg_filename is not None:
        agg_filename=val_type+'_'+agg_filename
        agg_file=open('Results/'+agg_filename+'.txt', 'a')
        agg_file.write(full_name+'\n')
        agg_file.close()
    combined_fig=plt.figure()
    plt.plot(thresholds,rates, 'r-')
    plt.xlabel(threshold)
    plt.ylabel('Rate')
    #plt.title(plot_fname)
    plot_dir=res_dir+'/plots'
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    combined_fig.savefig(plot_dir+'/'+outfile, ext='png', bbox_inches='tight')

    if show_plot:
        plt.show()

    return rates, outfile

def aggregateDataFromFiles(list_filename, plot_filename, header_size=0):
    print 'Aggregating data from ', list_filename, '...'
    line_styles=['-', '--', '-.', ':', '+', '*', 'D', 'x', 's', 'p', 'o', 'v', '^']
    colors=['b', 'g', 'r','c','m','y','k']
    #no_of_colors=32
    #colors=getRGBColors(no_of_colors)
    line_style_count=len(line_styles)
    color_count=len(colors)
    list_file=open('Results/'+list_filename+'.txt', 'r')
    header=list_file.readline()
    legend=header.split()
    filenames = list_file.readlines()
    list_file.close()
    #no_of_files=len(files)
    combined_fig=plt.figure()

    col_index=0
    line_style_index=0
    plot_lines=[]
    for filename in filenames:
        filename=filename.rstrip()
        if not filename:
            continue
        #print 'filename=', filename
        data_file=open(filename, 'r')
        for i in xrange(header_size):
            data_file.readline()
        lines=data_file.readlines()
        data_file.close()
        #print 'data_str before =', data_str
        #data_str=np.asarray(data_str)
        #print 'data_str after =', data_str
        #if (len(data_str.shape) != 2):
        #    print 'data_str.shape=', data_str.shape
        #    raise SystemError('Error in aggregateDataFromFiles:\nInvalid syntax detected')
        thresholds=[]
        rate_data=[]
        for line in lines:
            words=line.split()
            threshold=float(words[0])
            rate=float(words[1])
            thresholds.append(threshold)
            rate_data.append(rate)
        #data_float=float(data_str)
        #thresholds=data_float[:, 0]
        #rate_data=data_float[:, 1]
        if col_index==color_count:
            col_index=0
            line_style_index+=1
        if line_style_index==line_style_count:
            line_style_index=0
        #plt.plot(thresholds, rate_data, color=colors[col_index], linestyle=line_styles[line_style_index])
        plt.plot(thresholds, rate_data, colors[col_index]+line_styles[line_style_index])
        col_index+=1
        #plot_lines.append(plot_line)
        #data_array=np.asarray([thresholds, rate_data])
        #combined_data.append()

    #plt.show()
    plt.xlabel('thresholds')
    plt.ylabel('rate')
    #plt.title(plot_fname)
    fontP = ftm.FontProperties()
    fontP.set_size('small')
    combined_fig.savefig('Results/' + plot_filename, ext='png')
    plt.legend(legend, prop=fontP)
    combined_fig.savefig('Results/legend/' + plot_filename, ext='png')
    #plt.show()

def plotThresholdVariationsFromFile(filename,plot_fname):
    data_file = open('Results/'+ filename, 'r')
    header=data_file.readline()
    header_words=header.split()
    lines = data_file.readlines()
    #print 'header_words=', header_words

    header_count=len(header_words)
    line_count=len(lines)

    data_array=np.empty((line_count, header_count))
    for i in xrange(line_count):
        #print(line)
        words = lines[i].split()
        if (len(words) != header_count):
            msg = "Invalid formatting on line %d" % i + " in file %s" % filename + ":\n%s" % lines[i]
            raise SyntaxError(msg)
        for j in xrange(header_count):
            data_array[i,j]=float(words[j])
    thresholds=data_array[:, 0]
    combined_fig=plt.figure(0)
    for i in xrange(1, header_count):
        rate_data=data_array[:,i]
        plt.plot(thresholds,rate_data)

    plt.xlabel(header_words[0])
    plt.ylabel('Success Rate')
    #plt.title(plot_fname)
    combined_fig.savefig('Results/'+plot_fname, ext='png', bbox_inches='tight')
    plt.legend(header_words[1:])
    combined_fig.savefig('Results/legend/'+plot_fname, ext='png', bbox_inches='tight')
    plt.show()

def getRGBColors(no_of_colors):
    channel_div=0
    while no_of_colors>(channel_div**3):
        channel_div+=1
    colors=[]
    if channel_div==0:
        return colors
    base_factor=float(1.0/float(channel_div))
    for i in xrange(channel_div):
        red=base_factor*i
        for j in xrange(channel_div):
            green=base_factor*j
            for k in xrange(channel_div):
                blue=base_factor*k
                col=(red, green, blue)
                colors.append(col)
    return colors

def getPointPlot(filenames=None, plot_fname=None, file=None, use_sep_fig=True, show_plot=True):
    plt.close()
    if filenames is None:
        if file is None:
            return
        list_file=open('Results/'+file+'.txt')
        plot_fname=list_file.readline().rstrip()
        filenames=list_file.readlines()
    line_styles=['-', '--', '-.', ':']
    markers=[ '+', 'o', 'D', 'x', 's', 'p', '*', 'v', '^']
    colors=['r', 'g', 'b','c','m','y','k']
    fontP = ftm.FontProperties()
    fontP.set_size('small')
    plt.figure(0)
    if not use_sep_fig:
        plt.subplot(311)
    plt.title('Success Rate')
    #plt.legend(filenames)
    if use_sep_fig:
        plt.figure(1)
    else:
        plt.subplot(312)
    plt.title('Average FPS')
    #plt.legend(filenames)
    if use_sep_fig:
        plt.figure(2)
    else:
        plt.subplot(313)
    plt.title('Average Drift')
    #plt.legend(filenames)

    #annotate_text_list=None
    linestyle_id=0
    marker_id=0
    color_id=0
    for filename in filenames:
        print 'filename=', filename
        data_file=open('Results/'+filename.rstrip()+'.txt', 'r')
        header=data_file.readline().split()
        success_rate_list=[]
        avg_fps_list=[]
        avg_drift_list=[]
        parameters_list=[]
        line_count=0
        for line in data_file.readlines():
            line_count+=1
            words=line.split()
            success_rate=float(words[header.index('success_rate')])
            avg_fps=float(words[header.index('avg_fps')])
            avg_drift=float(words[header.index('avg_drift')])
            parameters=words[header.index('parameters')]

            success_rate_list.append(success_rate)
            avg_fps_list.append(avg_fps)
            avg_drift_list.append(avg_drift)
            parameters_list.append(parameters)
        x=range(0, line_count)
        if use_sep_fig:
            plt.figure(0, figsize=(1920/96, 1080/96), dpi=96)
        else:
            plt.subplot(311)
        plt.xticks(x, map(str, x))
        plt.plot(x,success_rate_list,
                 colors[color_id]+markers[marker_id]+line_styles[linestyle_id])
        if use_sep_fig:
            plt.figure(1)
        else:
            plt.subplot(312)
        plt.xticks(x, map(str, x))
        plt.plot(x,avg_fps_list,
                 colors[color_id]+markers[marker_id]+line_styles[linestyle_id])
        if use_sep_fig:
            plt.figure(2)
        else:
            plt.subplot(313)
        plt.xticks(x, map(str, x))
        plt.plot(x,avg_drift_list,
                 colors[color_id]+markers[marker_id]+line_styles[linestyle_id])
        color_id+=1
        linestyle_id+=1
        marker_id+=1
        if color_id>=len(colors):
            color_id=0
        if marker_id>=len(markers):
            marker_id=0
        if linestyle_id>=len(line_styles):
            linestyle_id=0
        annotate_text_list=parameters_list

    #annotate_text=''
    #print 'annotate_text_list:\n', annotate_text_list
    #for i in xrange(len(annotate_text_list)):
    #    annotate_text=annotate_text+str(i)+': '+annotate_text_list[i]+'\n'
    #
    #print 'annotate_text=\n', annotate_text
    # saving success rate plot
    if use_sep_fig:
        plt.figure(0)
    else:
        plt.subplot(311)
    plt.legend(filenames, prop=fontP)
    plt.grid(True)
    #plt.figtext(0.01,0.01, annotate_text, fontsize=9)
    if use_sep_fig and plot_fname is not None:
        plt.savefig('Results/'+plot_fname+'_success_rate', dpi=96, ext='png')
    # saving fps plot
    if use_sep_fig:
        plt.figure(1)
    else:
        plt.subplot(312)
    plt.legend(filenames, prop=fontP)
    plt.grid(True)
    if use_sep_fig and plot_fname is not None:
        plt.savefig('Results/'+plot_fname+'_avg_fps', dpi=96, ext='png')
    # saving drift plot
    if use_sep_fig:
        plt.figure(2)
    else:
        plt.subplot(313)
    plt.legend(filenames, prop=fontP)
    plt.grid(True)
    if use_sep_fig and plot_fname is not None:
        plt.savefig('Results/'+plot_fname+'_avg_drft', dpi=96, ext='png')

    # saving combined plot
    if not use_sep_fig and plot_fname is not None:
        plt.savefig('Results/'+plot_fname, dpi=96, ext='png')

    if show_plot:
        plt.show()



