# Generated with SMOP  0.41
from libsmop import *
# amplify_spatial_lpyr_temporal_ideal.m

    # amplify_spatial_lpyr_temporal_ideal(vidFile, outDir, alpha, lambda_c,
#                                     wl, wh, samplingRate, chromAttenuation)
# 
# Spatial Filtering: Laplacian pyramid
# Temporal Filtering: Ideal bandpass
# 
# Copyright (c) 2011-2012 Massachusetts Institute of Technology, 
# Quanta Research Cambridge, Inc.
    
    # Authors: Hao-yu Wu, Michael Rubinstein, Eugene Shih, 
# License: Please refer to the LICENCE file
# Date: June 2012
    
    
@function
def amplify_spatial_lpyr_temporal_ideal(vidFile=None,outDir=None,alpha=None,lambda_c=None,fl=None,fh=None,samplingRate=None,chromAttenuation=None,*args,**kwargs):
    varargin = amplify_spatial_lpyr_temporal_ideal.varargin
    nargin = amplify_spatial_lpyr_temporal_ideal.nargin

    
    __,vidName=fileparts(vidFile,nargout=2)
# amplify_spatial_lpyr_temporal_ideal.m:19
    outName=fullfile(outDir,concat([vidName,'-ideal-from-',num2str(fl),'-to-',num2str(fh),'-alpha-',num2str(alpha),'-lambda_c-',num2str(lambda_c),'-chromAtn-',num2str(chromAttenuation),'.avi']))
# amplify_spatial_lpyr_temporal_ideal.m:21
    
    vid=VideoReader(vidFile)
# amplify_spatial_lpyr_temporal_ideal.m:27
    
    vidHeight=vid.Height
# amplify_spatial_lpyr_temporal_ideal.m:29
    vidWidth=vid.Width
# amplify_spatial_lpyr_temporal_ideal.m:30
    nChannels=3
# amplify_spatial_lpyr_temporal_ideal.m:31
    fr=vid.FrameRate
# amplify_spatial_lpyr_temporal_ideal.m:32
    len_=vid.NumberOfFrames
# amplify_spatial_lpyr_temporal_ideal.m:33
    temp=struct('cdata',zeros(vidHeight,vidWidth,nChannels,'uint8'),'colormap',[])
# amplify_spatial_lpyr_temporal_ideal.m:34
    startIndex=1
# amplify_spatial_lpyr_temporal_ideal.m:36
    endIndex=len_ - 10
# amplify_spatial_lpyr_temporal_ideal.m:37
    vidOut=VideoWriter(outName)
# amplify_spatial_lpyr_temporal_ideal.m:39
    vidOut.FrameRate = copy(fr)
# amplify_spatial_lpyr_temporal_ideal.m:40
    open_(vidOut)
    # compute Laplacian pyramid for each frame
    pyr_stack,pind=build_Lpyr_stack(vidFile,startIndex,endIndex,nargout=2)
# amplify_spatial_lpyr_temporal_ideal.m:46
    
    # save(['pyrStack_' vidName '.mat'],'pyr_stack','pind','-v7.3');
    
    filtered_stack=ideal_bandpassing(pyr_stack,3,fl,fh,samplingRate)
# amplify_spatial_lpyr_temporal_ideal.m:52
    
    ind=size(pyr_stack(arange(),1,1),1)
# amplify_spatial_lpyr_temporal_ideal.m:56
    nLevels=size(pind,1)
# amplify_spatial_lpyr_temporal_ideal.m:57
    delta=lambda_c / 8 / (1 + alpha)
# amplify_spatial_lpyr_temporal_ideal.m:59
    
    # paper. (for better visualization)
    exaggeration_factor=2
# amplify_spatial_lpyr_temporal_ideal.m:63
    
    # freqency band of Laplacian pyramid
    
    lambda_=(vidHeight ** 2 + vidWidth ** 2) ** 0.5 / 3
# amplify_spatial_lpyr_temporal_ideal.m:68
    
    for l in arange(nLevels,1,- 1).reshape(-1):
        indices=arange(ind - prod(pind(l,arange())) + 1,ind)
# amplify_spatial_lpyr_temporal_ideal.m:71
        currAlpha=lambda_ / delta / 8 - 1
# amplify_spatial_lpyr_temporal_ideal.m:73
        currAlpha=dot(currAlpha,exaggeration_factor)
# amplify_spatial_lpyr_temporal_ideal.m:74
        if (l == nLevels or l == 1):
            filtered_stack[indices,arange(),arange()]=0
# amplify_spatial_lpyr_temporal_ideal.m:77
        else:
            if (currAlpha > alpha):
                filtered_stack[indices,arange(),arange()]=dot(alpha,filtered_stack(indices,arange(),arange()))
# amplify_spatial_lpyr_temporal_ideal.m:79
            else:
                filtered_stack[indices,arange(),arange()]=dot(currAlpha,filtered_stack(indices,arange(),arange()))
# amplify_spatial_lpyr_temporal_ideal.m:81
        ind=ind - prod(pind(l,arange()))
# amplify_spatial_lpyr_temporal_ideal.m:84
        # representative lambda will reduce by factor of 2
        lambda_=lambda_ / 2
# amplify_spatial_lpyr_temporal_ideal.m:87
    
    
    ## Render on the input video
    
    # output video
    k=0
# amplify_spatial_lpyr_temporal_ideal.m:93
    for i in arange(startIndex + 1,endIndex).reshape(-1):
        i
        k=k + 1
# amplify_spatial_lpyr_temporal_ideal.m:96
        temp.cdata = copy(read(vid,i))
# amplify_spatial_lpyr_temporal_ideal.m:97
        rgbframe,__=frame2im(temp,nargout=2)
# amplify_spatial_lpyr_temporal_ideal.m:98
        rgbframe=im2double(rgbframe)
# amplify_spatial_lpyr_temporal_ideal.m:99
        frame=rgb2ntsc(rgbframe)
# amplify_spatial_lpyr_temporal_ideal.m:100
        filtered=zeros(vidHeight,vidWidth,3)
# amplify_spatial_lpyr_temporal_ideal.m:102
        filtered[arange(),arange(),1]=reconLpyr(filtered_stack(arange(),1,k),pind)
# amplify_spatial_lpyr_temporal_ideal.m:104
        filtered[arange(),arange(),2]=dot(reconLpyr(filtered_stack(arange(),2,k),pind),chromAttenuation)
# amplify_spatial_lpyr_temporal_ideal.m:105
        filtered[arange(),arange(),3]=dot(reconLpyr(filtered_stack(arange(),3,k),pind),chromAttenuation)
# amplify_spatial_lpyr_temporal_ideal.m:106
        filtered=filtered + frame
# amplify_spatial_lpyr_temporal_ideal.m:108
        frame=ntsc2rgb(filtered)
# amplify_spatial_lpyr_temporal_ideal.m:110
        frame[frame > 1]=1
# amplify_spatial_lpyr_temporal_ideal.m:112
        frame[frame < 0]=0
# amplify_spatial_lpyr_temporal_ideal.m:113
        writeVideo(vidOut,im2uint8(frame))
    
    close_(vidOut)
    return
    
if __name__ == '__main__':
    pass
    