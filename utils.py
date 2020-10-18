import numpy as np

LEFT     = 1
RIGHT    = 2
ACC      = 3
BRAKE    = 4
STRAIGHT = 0

action_to_id_dict = {LEFT     : np.array([-1.0, 0.0, 0.0]),
                     RIGHT    : np.array([+1.0, 0.0, 0.0]),
                     ACC      : np.array([0.0, +1.0, 0.0]),
                     BRAKE    : np.array([0.0, 0.0, +0.2]),
                     STRAIGHT : np.array([0.0, 0.0, 0.0 ])}

CUTOFF = 84 # for pixels                     
def rgb2gray(rgb):
    """ 
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32') 

def action_to_id(y_samples):
    """
    this method turns samples of actions into an id.
    y_samples should be of size (NUM_SAMPLES, 3)
    """

    y_ids = np.zeros((y_samples.shape[0]))

    for key, var in action_to_id_dict.items():
      curr_idxs = np.all(y_samples == var, axis=1)
      y_ids[curr_idxs] = key

    return y_ids
