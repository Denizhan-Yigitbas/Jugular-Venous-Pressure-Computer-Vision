# Generated with SMOP  0.41
from libsmop import *
# blurDnClr.m

    # 3-color version of blurDn.
    

def blurDnClr(im, nlevs, filt):


    if (exist('nlevs') != 1):
        nlevs=1
# blurDnClr.m:9
    
    if (exist('filt') != 1):
        filt='binom5'
# blurDnClr.m:13
    
    #------------------------------------------------------------
    
    tmp=blurDn(im(arange(),arange(),1),nlevs,filt)

    out=zeros(size(tmp,1),size(tmp,2),size(im,3))
# blurDnClr.m:19
    out[arange(),arange(),1]=tmp
# blurDnClr.m:20
    for clr in arange(2,size(im,3)).reshape(-1):
        out[arange(),arange(),clr]=blurDn(im(arange(),arange(),clr),nlevs,filt)
# blurDnClr.m:22
    