import numpy as np
import skimage
from skimage.morphology import binary_dilation, star
from skimage import io
from skimage.morphology import skeletonize
import torch


def make_skeleton(image, sigma = 0.125, tr = 0.1):

    '''
    Create skeleton from binary image
    '''

    image = skimage.filters.gaussian(image, sigma)
    image[image > tr] = 1
    image[image != 1] = 0
    image = np.array(image, dtype=bool)
    skeleton = skeletonize(image)

    return skeleton



def find_neighs(patch, point):

      '''
      Find neighboring points in a given patch
      '''

      x, y = point
      neighs = []
      if patch[x,y] != 1:
          return []

      for i in [-1,0,1]:
        for j in [-1,0,1]:
            if i==0 and j==0:
              pass
            else:
              if x+i >= 0 and x+i < patch.shape[0] and y+j >= 0 and y+j < patch.shape[1]:
                  if patch[x+i, y+j] == 1:
                    neighs.append([x+i,y+j])   

      return neighs

def find_start_points(patch):

    '''
    Find start points in a given patch
    (those that lie on the border)
    '''

    start_points = []
    for x in range(patch.shape[0]):
        if patch[x,0] == 1:
          start_points.append([x, 0])
        if patch[x,patch.shape[1]-1] == 1:
          start_points.append([x, patch.shape[0]-1])

    for y in range(patch.shape[1]):
        if patch[0,y] == 1:
          start_points.append([0, y])
        if patch[patch.shape[0]-1,y] == 1:
          start_points.append([patch.shape[0]-1, y])

    return start_points



def check_in_paths(paths, point):

    '''
    Сheck if this point already was in the paths
    '''

    if len(paths) > 0:
        for path in paths:
            if point in path:
                return True
    return False

def find_path(start_point, patch, paths):

    '''
    Find longest path from the given start_point
    and crossroads points (with more than 1 neighbor)
    '''

    if len(find_neighs(patch, start_point)) == 0:
        return [start_point] , []

    prev_point = start_point
    point = find_neighs(patch, prev_point)[0]
    path, cross_points = [start_point, point], []
    while True:
        neighs = find_neighs(patch, point)
        for n in neighs:
            if check_in_paths(paths, n) == True:
                idx = neighs.index(n)
                neighs.pop(idx)
        for n in neighs:
            if n in path:
                idx = neighs.index(n)
                neighs.pop(idx)
            
        if prev_point in neighs:        
            idx = neighs.index(prev_point)
            neighs.pop(idx)
        prev_point = point
        if len(neighs) == 0:
            break

        idx = np.random.randint(len(neighs))
        point = neighs[-1]
        path.append(point)
        if len(neighs) > 1:
            for n in neighs[:-1]:
              cross_points.append(n)

    return path, cross_points


def all_nonzero(patch):

    '''
    Find all nonzero points in patch
    '''

    nonzero = []
    for i in range(patch.shape[0]):
        for j in range(patch.shape[1]):
          if patch[i,j] == 1:
              nonzero.append([i,j])
    return nonzero

def check_patch(patch, paths):

    '''
    Check if all roads in the patch have been found
    '''

    arr = np.zeros_like(patch)
    for path in paths:
        for p in path:
            arr[p[0], p[1]] = 1
    if (arr == patch).mean() == 1:
        return True
    else:
        return False


def find_additional_start_points(patch, paths):

    '''
    Find additional start points in a given patch
    (those that lie not on the border)
    '''

    start_points = []
    all_points = all_nonzero(patch)
    for point in all_points:
        skip = check_in_paths(paths, point)
        if skip == False:
          
              neighs = find_neighs(patch, point)
              if len(neighs) > 1:
                  for n in neighs:
                      if check_in_paths(paths, n) == True:
                          idx = neighs.index(n)
                          neighs.pop(idx)

              if len(neighs) == 1: 
                  start_points.append(point)
                  
    return start_points


def find_paths(patch):

    '''
    Find all paths in the patch
    '''

    start_points = find_start_points(patch)
    paths, cross_points = [], []
    if len(start_points) == 0:
        return []
    for start_point in start_points:
        skip = check_in_paths(paths, start_point)
        if skip == False:
            path, cross_points_ = find_path(start_point, patch, paths)
            paths.append(path)

            if len(cross_points_) > 0:
                [cross_points.append(p) for p in cross_points_]

    while len(cross_points) > 0:
        for cross_point in cross_points:
          skip = check_in_paths(paths, cross_point)
          if skip == False:
              path, cross_points_ = find_path(cross_point, patch, paths)
              paths.append(path)

              if len(cross_points_) > 0:
                  [cross_points.append(p) for p in cross_points_]
                  idx = cross_points.index(cross_point)
                  cross_points.pop(idx)

          if skip == True:
              idx = cross_points.index(cross_point)
              cross_points.pop(idx)

    if check_patch(patch, paths) == False:
        start_points = find_additional_start_points(patch, paths)
        for start_point in start_points:
            skip = check_in_paths(paths, start_point)
            if skip == False:
                path, cross_points_ = find_path(start_point, patch, paths)
                paths.append(path)

    return paths


def assing_label(prop_patch, paths):

    '''
    Assing label for the patch
    '''

    for path in paths:
        count = 0
        for p in path:
            if prop_patch[p[0], p[1]] != 1:
              count += 1
              if count > 4:
                  return 0
    return 1

def generate_sublabels(labels):

    '''
    Generate sublabels from given labels
    '''

    size = [int(labels.shape[0]/2), int(labels.shape[1]/2)]
    labels_ = torch.zeros((size))
    for i in range(size[0]):
        for j in range(size[1]):
            labels_[i, j] = int(labels[2*i:2*(i+1), 2*j:2*(j+1)].mean())

    return labels_

def generate_labels(mask_gt, mask_p, sigma=0.5):

    '''
    Generate labels for discriminator input
    '''
    
    gt_dil = binary_dilation(mask_gt,selem=star(1))
    skel_gt = make_skeleton(mask_gt, sigma=sigma)
    T_0 = gt_dil * mask_p

    labels0 = torch.zeros((8,8))
    for i in range(8):
        for j in range(8):
            prop_patch = T_0[i*32:(i+1)*32, j*32:(j+1)*32]
            patch = np.array(skel_gt[i*32:(i+1)*32, j*32:(j+1)*32], dtype=np.uint8)
            paths = find_paths(patch)
            labels0[i,j] = assing_label(prop_patch, paths)

    labels = [labels0]
    for i in range(3):
        labels.append(generate_sublabels(labels[-1]))

    return labels
