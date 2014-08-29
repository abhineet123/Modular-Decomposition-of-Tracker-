import numpy as np

def getBasicParams():
    sources=['file', 'camera']
    color_spaces=['Grayscale', 'RGB', 'HSV', 'YCrCb', 'HLS', 'Lab']
    filters=['none', 'gabor', 'laplacian', 'sobel', 'scharr', 'canny', 'LoG', 'DoG']
    smoothing=['none', 'box', 'bilateral', 'gauss', 'median']
    smoothing_kernel=map(str, [3, 5, 7, 9, 11, 13, 15, 19])
    trackers=['nn', 'esm', 'ict', 'l1']
    task_type=['simple', 'complex']
    actors=['Human', 'Robot']
    light_conditions=['nl', 'dl']
    speeds=['s1', 's2', 's3', 's4', 's5', 'si']
    complex_tasks=['bus', 'highlighting', 'letter', 'newspaper']
    simple_tasks=['bookI', 'bookII', 'bookIII', 'cereal', 'juice', 'mugI', 'mugII', 'mugIII']
    tasks=[simple_tasks, complex_tasks]
    params=[sources, color_spaces, filters, smoothing, smoothing_kernel, trackers,
            task_type, actors, light_conditions, speeds, tasks]
    labels=['source', 'color_space', 'filter', 'smoothing', 'smoothing_kernel', 'tracker',
            'type', 'actor', 'light', 'speed', 'task']
    #default_id=[0, 1, 0, 3, 2, 0, 0, 0, 2, 0]
    default_id = [sources.index('file'),
                   color_spaces.index('RGB'),
                   filters.index('none'),
                   smoothing.index('gauss'),
                   smoothing_kernel.index(str(5)),
                   trackers.index('nn'),
                   task_type.index('simple'),
                   actors.index('Human'),
                   light_conditions.index('nl'),
                   speeds.index('s3'),
                   tasks[task_type.index('simple')].index('bookI')]
    return params, labels, default_id

def getTrackingParams():
    multichannel_nn=['mean', 'majority', 'flatten']
    multichannel_esm=['mean', 'flatten']
    multichannel_ict=['mean', 'flatten']
    multichannel_l1=['mean', 'flatten']

    params_nn = {'no_of_samples': {'id': 0, 'default': 500, 'type': 'int', 'list': range(100, 5000, 100)},
                 'no_of_iterations': {'id': 1, 'default': 2, 'type': 'int', 'list': range(1, 100, 1)},
                 'resolution_x': {'id': 2, 'default': 40, 'type': 'int', 'list': range(5, 100, 5)},
                 'resolution_y': {'id': 3, 'default': 40, 'type': 'int', 'list': range(5, 100, 5)},
                 'enable_scv': {'id': 4, 'default': True, 'type': 'boolean', 'list': [True, False]},
                 'multi_approach': {'id': 5, 'default': 'flatten', 'type': 'string', 'list': multichannel_nn}
    }
    params_esm = {'max_iterations': {'id': 0, 'default': 10, 'type': 'int', 'list': range(1, 100, 1)},
                  'threshold': {'id': 1, 'default': 0.01, 'type': 'float', 'list': np.arange(0.01, 1, 0.01)},
                  'resolution_x': {'id': 2, 'default': 40, 'type': 'int', 'list': range(5, 100, 5)},
                  'resolution_y': {'id': 3, 'default': 40, 'type': 'int', 'list': range(5, 100, 5)},
                  'enable_scv': {'id': 4, 'default': True, 'type': 'boolean', 'list': [True, False]},
                  'multi_approach': {'id': 5, 'default': 'flatten', 'type': 'string', 'list': multichannel_esm}
    }
    params_ict = {'max_iterations': {'id': 0, 'default': 10, 'type': 'int', 'list': range(1, 100, 1)},
                  'threshold': {'id': 1, 'default': 0.01, 'type': 'float', 'list': np.arange(0.01, 1, 0.01)},
                  'resolution_x': {'id': 2, 'default': 40, 'type': 'int', 'list': range(5, 100, 5)},
                  'resolution_y': {'id': 3, 'default': 40, 'type': 'int', 'list': range(5, 100, 5)},
                  'enable_scv': {'id': 4, 'default': True, 'type': 'boolean', 'list': [True, False]},
                  'multi_approach': {'id': 5, 'default': 'flatten', 'type': 'string', 'list': multichannel_ict}
    }
    params_l1 = {'no_of_samples': {'id': 0, 'default': 10, 'type': 'int', 'list': range(10, 500, 10)},
                 'angle_threshold': {'id': 1, 'default': 2, 'type': 'int', 'list': np.arange(1, 10, 0.5)},
                 'resolution_x': {'id': 2, 'default': 40, 'type': 'int', 'list': range(5, 100, 5)},
                 'resolution_y': {'id': 3, 'default': 40, 'type': 'int', 'list': range(5, 100, 5)},
                 'no_of_templates': {'id': 4, 'default': 10, 'type': 'int', 'list': range(5, 100, 5)},
                 'alpha': {'id': 5, 'default': 50, 'type': 'float', 'list': range(100, 5000, 100)},
                 'enable_scv': {'id': 6, 'default': True, 'type': 'boolean', 'list': [True, False]},
                 'multi_approach': {'id': 7, 'default': 'flatten', 'type': 'string', 'list': multichannel_l1}
    }
    tracking_params={'nn': params_nn, 'esm': params_esm,  'ict': params_ict, 'l1': params_l1}
    return tracking_params

def getFilteringParams():
    params_gabor = {'ksize': {'id': 0, 'default': {'base': 2, 'mult': 1, 'limit': 10, 'add': 1}, 'type': 'int'},
                    'sigma': {'id': 1, 'default': {'base': 0.1, 'mult': 10, 'limit': 100, 'add': 0.0}, 'type': 'float'},
                    'theta': {'id': 2, 'default': {'base': np.pi / 12, 'mult': 0, 'limit': 24, 'add': 0.0}, 'type': 'float'},
                    'lambd': {'id': 3, 'default': {'base': 0.1, 'mult': 10, 'limit': 100, 'add': 10.0}, 'type': 'float'},
                    'gamma': {'id': 4, 'default': {'base': 0.1, 'mult': 10, 'limit': 100, 'add': 0.0}, 'type': 'float'}
    }
    params_laplacian = {'ksize': {'id': 0, 'default': {'base': 2, 'mult': 1, 'limit': 10, 'add': 1}, 'type': 'int'},
                        'scale': {'id': 1, 'default': {'base': 1, 'mult': 0, 'limit': 10, 'add': 1}, 'type': 'int'},
                        'delta': {'id': 2, 'default': {'base': 1, 'mult': 0, 'limit': 255, 'add': 0}, 'type': 'int'},
    }
    params_sobel = {'ksize': {'id': 0, 'default': {'base': 2, 'mult': 1, 'limit': 10, 'add': 1}, 'type': 'int'},
                    'scale': {'id': 1, 'default': {'base': 1, 'mult': 0, 'limit': 10, 'add': 1}, 'type': 'int'},
                    'delta': {'id': 2, 'default': {'base': 1, 'mult': 0, 'limit': 255, 'add': 0}, 'type': 'int'},
                    'dx': {'id': 3, 'default': {'base': 1, 'mult': 1, 'limit': 5, 'add': 0}, 'type': 'int'},
                    'dy': {'id': 4, 'default': {'base': 1, 'mult': 0, 'limit': 5, 'add': 0}, 'type': 'int'}
    }
    params_scharr = {'scale': {'id': 0, 'default': {'base': 1, 'mult': 0, 'limit': 10, 'add': 1}, 'type': 'int'},
                     'delta': {'id': 1, 'default': {'base': 1, 'mult': 0, 'limit': 255, 'add': 0}, 'type': 'int'},
                     'dx': {'id': 2, 'default': {'base': 1, 'mult': 1, 'limit': 1, 'add': 0}, 'type': 'int'},
                     'dy': {'id': 3, 'default': {'base': 1, 'mult': 0, 'limit': 1, 'add': 0}, 'type': 'int'},
    }
    params_canny = {'low_thresh': {'id': 0, 'default': {'base': 1, 'mult': 20, 'limit': 50, 'add': 0}, 'type': 'int'},
                    'ratio': {'id': 1, 'default': {'base': 1, 'mult': 4, 'limit': 10, 'add': 0}, 'type': 'float'},
    }
    params_dog = {'ksize': {'id': 0, 'default': {'base': 2, 'mult': 1, 'limit': 10, 'add': 1}, 'type': 'int'},
                  'exc_std': {'id': 1, 'default': {'base': 0.1, 'mult': 20, 'limit': 100, 'add': 1}, 'type': 'float'},
                  'inh_std': {'id': 2, 'default': {'base': 0.1, 'mult': 28, 'limit': 100, 'add': 1}, 'type': 'float'},
                  'ratio': {'id': 3, 'default': {'base': 0.05, 'mult': 50, 'limit': 200, 'add': 0.0}, 'type': 'float'},
    }
    params_log = {'gauss_ksize': {'id': 0, 'default': {'base': 2, 'mult': 1, 'limit': 10, 'add': 1}, 'type': 'int'},
                  'std': {'id': 1, 'default': {'base': 0.1, 'mult': 20, 'limit': 100, 'add': 0.1}, 'type': 'float'},
                  'lap_ksize': {'id': 2, 'default': {'base': 2, 'mult': 1, 'limit': 5, 'add': 1}, 'type': 'int'},
                  'scale': {'id': 3, 'default': {'base': 1, 'mult': 0, 'limit': 10, 'add': 1}, 'type': 'int'},
                  'delta': {'id': 4, 'default': {'base': 1, 'mult': 0, 'limit': 255, 'add': 0}, 'type': 'int'}
    }
    filtering_params = {'gabor': params_gabor, 'laplacian': params_laplacian, 'sobel': params_sobel,
                        'scharr': params_scharr, 'canny': params_canny,
                        'DoG': params_dog, 'LoG': params_log}
    return filtering_params

