# Generated with SMOP  0.41
from libsmop import *
# create_mask.m

    # MASK = create_mask(IMAGE)
# 
# Using imply function built in MATLAB to create mask
# for localized Eulerian video processing
# 
# Copyright (c) 2011-2012 Massachusetts Institute of Technology, 
# Quanta Research Cambridge, Inc.
    
    # Authors: Hao-yu Wu, Michael Rubinstein, Eugene Shih, 
# License: Please refer to the LICENCE file
# Date: June 2012
    
    
@function
def create_mask(image=None,*args,**kwargs):
    varargin = create_mask.varargin
    nargin = create_mask.nargin

    
    imshow(image)
    h=copy(impoly)
# create_mask.m:17
    position=wait(h)
# create_mask.m:18
    BW=createMask(h)
# create_mask.m:19
    mask=zeros(size(BW))
# create_mask.m:20
    mask[BW]=1
# create_mask.m:21
    
    g=fspecial('gaussian',15,5)
# create_mask.m:24
    mask=imfilter(mask,g)
# create_mask.m:25
    mask=repmat(mask,concat([1,1,3]))
# create_mask.m:27
    save(concat([vidName,'.mat']),'mask')
    return mask
    
if __name__ == '__main__':
    pass
    