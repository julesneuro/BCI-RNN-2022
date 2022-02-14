import numpy as np

def generate_int(loc, scale):
    
    """
        Args:
        loc = center of dist
        scale = sd
        -- Generic function to randomly generate integers in a gaussian manner.
    """
    n = np.random.normal(loc, scale)
    n = np.round(n)
    return n

def generate_trialstats(vis_loc, aud_loc, vis_rel, aud_rel, spread, n_features):
    
    """
        Args:
        vis_loc = dist center
        aud_loc = aud dist center
        vis_rel = sd of vis dist, fixed
        aud_rel = sd of aud dist, variable by cond
        spread = spread when implanted into data
        n_features = length of input array
    """
    
    import scipy.stats as stats
    
    lower, upper = 1, (n_features - (spread + 2))
    l_v, u_v = (lower - vis_loc) / vis_rel, (upper - vis_loc) / vis_rel
    l_a, u_a = (lower - aud_loc) / aud_rel, (upper - aud_loc) / aud_rel
    
    d_vis = stats.truncnorm(l_v, u_v, loc = vis_loc, scale = vis_rel)
    d_aud = stats.truncnorm(l_a, u_a, loc = aud_loc, scale = aud_rel)
    
    vis_samp = np.round(d_vis.rvs(1))
    aud_samp = np.round(d_aud.rvs(1))
    
    return vis_samp, aud_samp

def startfin(x_loc, spread):
    
    """
        Args:
        x_loc -- given sampled location on a trial
        spread -- spread
        Returns start, fin
    """
    
    start = x_loc - ((spread - 1) / 2)
    fin = start + spread
    return start, fin

def generate_data(n_stim, len_azimuth, n_features, reliability_list, spread, n_positions = 5):
    
    """
        Args:
        train_size = size of training set
        len_azimuth = len of binary input representing spatial location
        n_positions = amount of distinct positions stimuli can appear at
        reliability_list = list of levels of reliability, indicated by SD
    """

    import itertools
    
    # Determine where the center of each discrete stim position is
    bin_center = len_azimuth / n_positions
    loc_list = [i*bin_center+bin_center for i in range(n_positions)]
    
    # Determine all possible audvis stim location combinations are 
    vis_loc, aud_loc = loc_list, loc_list
    loc_combinations = []
    for i in itertools.product(vis_loc, aud_loc, reliability_list):
        loc_combinations.append(i)
        
    # Initialize Empty Lists 
    X_l = []
    y_l_classes = []
    y_l_real = []
    
    n_per_combination = n_stim / (n_positions * n_positions * len(reliability_list))
    
    for combo in loc_combinations: # For all combinations, should be 100
        
        vis_loc, aud_loc, aud_rel = combo[0], combo[1], combo[2]
        vis_rel = reliability_list[0] # Most reliable condition should always be first!
        
        for i in range(int(n_per_combination)):
            
            # Initialize Holder for stim
            stim_array = np.zeros((2, n_features))
            label_array = np.zeros([0, 3]) * np.nan
            label_list = [vis_loc, aud_loc, aud_rel]
            label_arr = np.asarray(label_list)
            
            # Get generated centers of stimuli!
            vis_loc, aud_loc = generate_trialstats(vis_loc, aud_loc, 
                                                   vis_rel, aud_rel,
                                                   spread, n_features)
            real_list = [vis_loc, aud_loc]
            
            vis_start, vis_fin = startfin(vis_loc, spread)
            aud_start, aud_fin = startfin(aud_loc, spread)
            print(real_list)
            
            for i in range(int(vis_start), int(vis_fin)):
                stim_array[0][i] = 1
                
            for i in range(int(aud_start), int(aud_fin)):
                stim_array[1][i] = 1

            X_l.append(stim_array)
            y_l_classes.append(label_arr)
            y_l_real.append(real_list)
            
    X = np.asarray(X_l)
    y_labels = np.asarray(y_l_classes)
    y_real = np.asarray(y_l_real)
    
    return X, y_labels, y_real