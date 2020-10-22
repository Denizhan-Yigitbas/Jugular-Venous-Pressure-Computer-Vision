# Generated with SMOP  0.41
from libsmop import *
# amplify_spatial_lpyr_temporal_butter.m

    # amplify_spatial_lpyr_temporal_butter(vidFile, outDir, alpha, lambda_c, 
#                                      fl, fh, samplingRate, chromAttenuation)
# 
# Spatial Filtering: Laplacian pyramid
# Temporal Filtering: substraction of two butterworth lowpass filters
#                     with cutoff frequencies fh and fl
# 
# Copyright (c) 2011-2012 Massachusetts Institute of Technology, 
# Quanta Research Cambridge, Inc.
    
    # Authors: Hao-yu Wu, Michael Rubinstein, Eugene Shih, 
# License: Please refer to the LICENCE file
# Date: June 2012
    
    
@function
def amplify_spatial_lpyr_temporal_butter(vidFile=None,outDir=None,alpha=None,lambda_c=None,fl=None,fh=None,samplingRate=None,chromAttenuation=None,*args,**kwargs):
    varargin = amplify_spatial_lpyr_temporal_butter.varargin
    nargin = amplify_spatial_lpyr_temporal_butter.nargin

    
    low_a,low_b=butter(1,fl / samplingRate,'low',nargout=2)
# amplify_spatial_lpyr_temporal_butter.m:19
    high_a,high_b=butter(1,fh / samplingRate,'low',nargout=2)
# amplify_spatial_lpyr_temporal_butter.m:20
    __,vidName=fileparts(vidFile,nargout=2)
# amplify_spatial_lpyr_temporal_butter.m:22
    outName=fullfile(outDir,concat([vidName,'-butter-from-',num2str(fl),'-to-',num2str(fh),'-alpha-',num2str(alpha),'-lambda_c-',num2str(lambda_c),'-chromAtn-',num2str(chromAttenuation),'.avi']))
# amplify_spatial_lpyr_temporal_butter.m:24
    
    vid=VideoReader(vidFile)
# amplify_spatial_lpyr_temporal_butter.m:29
    
    vidHeight=vid.Height
# amplify_spatial_lpyr_temporal_butter.m:31
    vidWidth=vid.Width
# amplify_spatial_lpyr_temporal_butter.m:32
    nChannels=3
# amplify_spatial_lpyr_temporal_butter.m:33
    fr=vid.FrameRate
# amplify_spatial_lpyr_temporal_butter.m:34
    len_=vid.NumberOfFrames
# amplify_spatial_lpyr_temporal_butter.m:35
    temp=struct('cdata',zeros(vidHeight,vidWidth,nChannels,'uint8'),'colormap',[])
# amplify_spatial_lpyr_temporal_butter.m:36
    startIndex=1
# amplify_spatial_lpyr_temporal_butter.m:40
    endIndex=len_ - 10
# amplify_spatial_lpyr_temporal_butter.m:41
    vidOut=VideoWriter(outName)
# amplify_spatial_lpyr_temporal_butter.m:43
    vidOut.FrameRate = copy(fr)
# amplify_spatial_lpyr_temporal_butter.m:44
    open_(vidOut)
    # firstFrame
    temp.cdata = copy(read(vid,startIndex))
# amplify_spatial_lpyr_temporal_butter.m:49
    rgbframe,__=frame2im(temp,nargout=2)
# amplify_spatial_lpyr_temporal_butter.m:50
    rgbframe=im2double(rgbframe)
# amplify_spatial_lpyr_temporal_butter.m:51
    frame=rgb2ntsc(rgbframe)
# amplify_spatial_lpyr_temporal_butter.m:52
    pyr,pind=buildLpyr(frame(arange(),arange(),1),'auto',nargout=2)
# amplify_spatial_lpyr_temporal_butter.m:54
    pyr=repmat(pyr,concat([1,3]))
# amplify_spatial_lpyr_temporal_butter.m:55
    pyr(arange(),2),__=buildLpyr(frame(arange(),arange(),2),'auto',nargout=2)
# amplify_spatial_lpyr_temporal_butter.m:56
    pyr(arange(),3),__=buildLpyr(frame(arange(),arange(),3),'auto',nargout=2)
# amplify_spatial_lpyr_temporal_butter.m:57
    lowpass1=copy(pyr)
# amplify_spatial_lpyr_temporal_butter.m:58
    lowpass2=copy(pyr)
# amplify_spatial_lpyr_temporal_butter.m:59
    pyr_prev=copy(pyr)
# amplify_spatial_lpyr_temporal_butter.m:60
    output=copy(rgbframe)
# amplify_spatial_lpyr_temporal_butter.m:62
    writeVideo(vidOut,im2uint8(output))
    nLevels=size(pind,1)
# amplify_spatial_lpyr_temporal_butter.m:65
    for i in arange(startIndex + 1,endIndex).reshape(-1):
        progmeter(i - startIndex,endIndex - startIndex + 1)
        temp.cdata = copy(read(vid,i))
# amplify_spatial_lpyr_temporal_butter.m:71
        rgbframe,__=frame2im(temp,nargout=2)
# amplify_spatial_lpyr_temporal_butter.m:72
        rgbframe=im2double(rgbframe)
# amplify_spatial_lpyr_temporal_butter.m:74
        frame=rgb2ntsc(rgbframe)
# amplify_spatial_lpyr_temporal_butter.m:75
        pyr(arange(),1),__=buildLpyr(frame(arange(),arange(),1),'auto',nargout=2)
# amplify_spatial_lpyr_temporal_butter.m:77
        pyr(arange(),2),__=buildLpyr(frame(arange(),arange(),2),'auto',nargout=2)
# amplify_spatial_lpyr_temporal_butter.m:78
        pyr(arange(),3),__=buildLpyr(frame(arange(),arange(),3),'auto',nargout=2)
# amplify_spatial_lpyr_temporal_butter.m:79
        lowpass1=(multiply(- high_b(2),lowpass1) + multiply(high_a(1),pyr) + multiply(high_a(2),pyr_prev)) / high_b(1)
# amplify_spatial_lpyr_temporal_butter.m:82
        lowpass2=(multiply(- low_b(2),lowpass2) + multiply(low_a(1),pyr) + multiply(low_a(2),pyr_prev)) / low_b(1)
# amplify_spatial_lpyr_temporal_butter.m:84
        filtered=(lowpass1 - lowpass2)
# amplify_spatial_lpyr_temporal_butter.m:87
        pyr_prev=copy(pyr)
# amplify_spatial_lpyr_temporal_butter.m:89
        ind=size(pyr,1)
# amplify_spatial_lpyr_temporal_butter.m:92
        delta=lambda_c / 8 / (1 + alpha)
# amplify_spatial_lpyr_temporal_butter.m:94
        # paper. (for better visualization)
        exaggeration_factor=2
# amplify_spatial_lpyr_temporal_butter.m:98
        # freqency band of Laplacian pyramid
        lambda_=(vidHeight ** 2 + vidWidth ** 2) ** 0.5 / 3
# amplify_spatial_lpyr_temporal_butter.m:103
        for l in arange(nLevels,1,- 1).reshape(-1):
            indices=arange(ind - prod(pind(l,arange())) + 1,ind)
# amplify_spatial_lpyr_temporal_butter.m:106
            currAlpha=lambda_ / delta / 8 - 1
# amplify_spatial_lpyr_temporal_butter.m:109
            currAlpha=dot(currAlpha,exaggeration_factor)
# amplify_spatial_lpyr_temporal_butter.m:110
            if (l == nLevels or l == 1):
                filtered[indices,arange()]=0
# amplify_spatial_lpyr_temporal_butter.m:113
            else:
                if (currAlpha > alpha):
                    filtered[indices,arange()]=dot(alpha,filtered(indices,arange()))
# amplify_spatial_lpyr_temporal_butter.m:115
                else:
                    filtered[indices,arange()]=dot(currAlpha,filtered(indices,arange()))
# amplify_spatial_lpyr_temporal_butter.m:117
            ind=ind - prod(pind(l,arange()))
# amplify_spatial_lpyr_temporal_butter.m:120
            # representative lambda will reduce by factor of 2
            lambda_=lambda_ / 2
# amplify_spatial_lpyr_temporal_butter.m:123
        ## Render on the input video
        output=zeros(size(frame))
# amplify_spatial_lpyr_temporal_butter.m:128
        output[arange(),arange(),1]=reconLpyr(filtered(arange(),1),pind)
# amplify_spatial_lpyr_temporal_butter.m:130
        output[arange(),arange(),2]=reconLpyr(filtered(arange(),2),pind)
# amplify_spatial_lpyr_temporal_butter.m:131
        output[arange(),arange(),3]=reconLpyr(filtered(arange(),3),pind)
# amplify_spatial_lpyr_temporal_butter.m:132
        output[arange(),arange(),2]=dot(output(arange(),arange(),2),chromAttenuation)
# amplify_spatial_lpyr_temporal_butter.m:134
        output[arange(),arange(),3]=dot(output(arange(),arange(),3),chromAttenuation)
# amplify_spatial_lpyr_temporal_butter.m:135
        output=frame + output
# amplify_spatial_lpyr_temporal_butter.m:137
        output=ntsc2rgb(output)
# amplify_spatial_lpyr_temporal_butter.m:139
        #             filtered = rgbframe + filtered.*mask;
        output[output > 1]=1
# amplify_spatial_lpyr_temporal_butter.m:142
        output[output < 0]=0
# amplify_spatial_lpyr_temporal_butter.m:143
        writeVideo(vidOut,im2uint8(output))
    
    close_(vidOut)
    return
    
if __name__ == '__main__':
    pass
    