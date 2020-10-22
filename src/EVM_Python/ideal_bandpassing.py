# Generated with SMOP  0.41
from libsmop import *
# ideal_bandpassing.m

    # FILTERED = ideal_bandpassing(INPUT,DIM,WL,WH,SAMPLINGRATE)
# 
# Apply ideal band pass filter on INPUT along dimension DIM.
# 
# WL: lower cutoff frequency of ideal band pass filter
# WH: higher cutoff frequency of ideal band pass filter
# SAMPLINGRATE: sampling rate of INPUT
# 
# Copyright (c) 2011-2012 Massachusetts Institute of Technology, 
# Quanta Research Cambridge, Inc.
    
    # Authors: Hao-yu Wu, Michael Rubinstein, Eugene Shih, 
# License: Please refer to the LICENCE file
# Date: June 2012
    
    
@function
def ideal_bandpassing(input_=None,dim=None,wl=None,wh=None,samplingRate=None,*args,**kwargs):
    varargin = ideal_bandpassing.varargin
    nargin = ideal_bandpassing.nargin

    if (dim > size(size(input_),2)):
        error('Exceed maximum dimension')
    
    input_shifted=shiftdim(input_,dim - 1)
# ideal_bandpassing.m:22
    Dimensions=size(input_shifted)
# ideal_bandpassing.m:23
    n=Dimensions(1)
# ideal_bandpassing.m:25
    dn=size(Dimensions,2)
# ideal_bandpassing.m:26
    Freq=arange(1,n)
# ideal_bandpassing.m:29
    Freq=dot((Freq - 1) / n,samplingRate)
# ideal_bandpassing.m:30
    mask=Freq > logical_and(wl,Freq) < wh
# ideal_bandpassing.m:31
    Dimensions[1]=1
# ideal_bandpassing.m:33
    mask=ravel(mask)
# ideal_bandpassing.m:34
    mask=repmat(mask,Dimensions)
# ideal_bandpassing.m:35
    F=fft(input_shifted,[],1)
# ideal_bandpassing.m:38
    F[logical_not(mask)]=0
# ideal_bandpassing.m:40
    filtered=real(ifft(F,[],1))
# ideal_bandpassing.m:42
    filtered=shiftdim(filtered,dn - (dim - 1))
# ideal_bandpassing.m:44
    return filtered
    
if __name__ == '__main__':
    pass
    