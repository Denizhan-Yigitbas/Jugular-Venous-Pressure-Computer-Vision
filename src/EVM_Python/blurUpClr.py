# Generated with SMOP  0.41
from libsmop import *
# blurUpClr.m

    # 3-color version of upBlur
    
    
@function
def blurUpClr(im=None,nlevs=None,filt=None,*args,**kwargs):
    varargin = blurUpClr.varargin
    nargin = blurUpClr.nargin

    #------------------------------------------------------------
## OPTIONAL ARGS:
    
    if (exist('nlevs') != 1):
        nlevs=1
# blurUpClr.m:9
    
    if (exist('filt') != 1):
        filt='binom5'
# blurUpClr.m:13
    
    #------------------------------------------------------------
    
    tmp=upBlur(im(arange(),arange(),1),nlevs,filt)
# blurUpClr.m:18
    out=zeros(size(tmp,1),size(tmp,2),size(im,3))
# blurUpClr.m:19
    out[arange(),arange(),1]=tmp
# blurUpClr.m:20
    for clr in arange(2,size(im,3)).reshape(-1):
        out[arange(),arange(),clr]=upBlur(im(arange(),arange(),clr),nlevs,filt)
# blurUpClr.m:22
    