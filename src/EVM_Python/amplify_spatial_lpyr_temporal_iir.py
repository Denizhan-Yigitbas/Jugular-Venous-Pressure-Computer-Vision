# Generated with SMOP  0.41
from libsmop import *
# amplify_spatial_lpyr_temporal_iir.m

    # amplify_spatial_lpyr_temporal_iir(vidFile, resultsDir, ...
#                                   alpha, lambda_c, r1, r2, chromAttenuation)
# 
# Spatial Filtering: Laplacian pyramid
# Temporal Filtering: substraction of two IIR lowpass filters
# 
# y1[n] = r1*x[n] + (1-r1)*y1[n-1]
# y2[n] = r2*x[n] + (1-r2)*y2[n-1]
# (r1 > r2)
    
    # y[n] = y1[n] - y2[n]
    
    # Copyright (c) 2011-2012 Massachusetts Institute of Technology, 
# Quanta Research Cambridge, Inc.
    
    # Authors: Hao-yu Wu, Michael Rubinstein, Eugene Shih, 
# License: Please refer to the LICENCE file
# Date: June 2012
    
    
@function
def amplify_spatial_lpyr_temporal_iir(vidFile=None,resultsDir=None,alpha=None,lambda_c=None,r1=None,r2=None,chromAttenuation=None,*args,**kwargs):
    varargin = amplify_spatial_lpyr_temporal_iir.varargin
    nargin = amplify_spatial_lpyr_temporal_iir.nargin

    
    __,vidName=fileparts(vidFile,nargout=2)
# amplify_spatial_lpyr_temporal_iir.m:23
    outName=fullfile(resultsDir,concat([vidName,'-iir-r1-',num2str(r1),'-r2-',num2str(r2),'-alpha-',num2str(alpha),'-lambda_c-',num2str(lambda_c),'-chromAtn-',num2str(chromAttenuation),'.avi']))
# amplify_spatial_lpyr_temporal_iir.m:24
    
    vid=VideoReader(vidFile)
# amplify_spatial_lpyr_temporal_iir.m:31
    
    vidHeight=vid.Height
# amplify_spatial_lpyr_temporal_iir.m:33
    vidWidth=vid.Width
# amplify_spatial_lpyr_temporal_iir.m:34
    nChannels=3
# amplify_spatial_lpyr_temporal_iir.m:35
    fr=vid.FrameRate
# amplify_spatial_lpyr_temporal_iir.m:36
    len_=vid.NumberOfFrames
# amplify_spatial_lpyr_temporal_iir.m:37
    temp=struct('cdata',zeros(vidHeight,vidWidth,nChannels,'uint8'),'colormap',[])
# amplify_spatial_lpyr_temporal_iir.m:38
    startIndex=1
# amplify_spatial_lpyr_temporal_iir.m:43
    endIndex=len_ - 10
# amplify_spatial_lpyr_temporal_iir.m:44
    vidOut=VideoWriter(outName)
# amplify_spatial_lpyr_temporal_iir.m:46
    vidOut.FrameRate = copy(fr)
# amplify_spatial_lpyr_temporal_iir.m:47
    open_(vidOut)
    # firstFrame
    temp.cdata = copy(read(vid,startIndex))
# amplify_spatial_lpyr_temporal_iir.m:52
    rgbframe,__=frame2im(temp,nargout=2)
# amplify_spatial_lpyr_temporal_iir.m:53
    rgbframe=im2double(rgbframe)
# amplify_spatial_lpyr_temporal_iir.m:54
    frame=rgb2ntsc(rgbframe)
# amplify_spatial_lpyr_temporal_iir.m:55
    pyr,pind=buildLpyr(frame(arange(),arange(),1),'auto',nargout=2)
# amplify_spatial_lpyr_temporal_iir.m:57
    pyr=repmat(pyr,concat([1,3]))
# amplify_spatial_lpyr_temporal_iir.m:58
    pyr(arange(),2),__=buildLpyr(frame(arange(),arange(),2),'auto',nargout=2)
# amplify_spatial_lpyr_temporal_iir.m:59
    pyr(arange(),3),__=buildLpyr(frame(arange(),arange(),3),'auto',nargout=2)
# amplify_spatial_lpyr_temporal_iir.m:60
    lowpass1=copy(pyr)
# amplify_spatial_lpyr_temporal_iir.m:62
    lowpass2=copy(pyr)
# amplify_spatial_lpyr_temporal_iir.m:63
    output=copy(rgbframe)
# amplify_spatial_lpyr_temporal_iir.m:65
    writeVideo(vidOut,im2uint8(output))
    nLevels=size(pind,1)
# amplify_spatial_lpyr_temporal_iir.m:68
    for i in arange(startIndex + 1,endIndex).reshape(-1):
        progmeter(i - startIndex,endIndex - startIndex + 1)
        temp.cdata = copy(read(vid,i))
# amplify_spatial_lpyr_temporal_iir.m:74
        rgbframe,__=frame2im(temp,nargout=2)
# amplify_spatial_lpyr_temporal_iir.m:75
        rgbframe=im2double(rgbframe)
# amplify_spatial_lpyr_temporal_iir.m:77
        frame=rgb2ntsc(rgbframe)
# amplify_spatial_lpyr_temporal_iir.m:78
        pyr(arange(),1),__=buildLpyr(frame(arange(),arange(),1),'auto',nargout=2)
# amplify_spatial_lpyr_temporal_iir.m:80
        pyr(arange(),2),__=buildLpyr(frame(arange(),arange(),2),'auto',nargout=2)
# amplify_spatial_lpyr_temporal_iir.m:81
        pyr(arange(),3),__=buildLpyr(frame(arange(),arange(),3),'auto',nargout=2)
# amplify_spatial_lpyr_temporal_iir.m:82
        lowpass1=dot((1 - r1),lowpass1) + dot(r1,pyr)
# amplify_spatial_lpyr_temporal_iir.m:85
        lowpass2=dot((1 - r2),lowpass2) + dot(r2,pyr)
# amplify_spatial_lpyr_temporal_iir.m:86
        filtered=(lowpass1 - lowpass2)
# amplify_spatial_lpyr_temporal_iir.m:88
        ind=size(pyr,1)
# amplify_spatial_lpyr_temporal_iir.m:92
        delta=lambda_c / 8 / (1 + alpha)
# amplify_spatial_lpyr_temporal_iir.m:94
        # paper. (for better visualization)
        exaggeration_factor=2
# amplify_spatial_lpyr_temporal_iir.m:98
        # freqency band of Laplacian pyramid
        lambda_=(vidHeight ** 2 + vidWidth ** 2) ** 0.5 / 3
# amplify_spatial_lpyr_temporal_iir.m:103
        for l in arange(nLevels,1,- 1).reshape(-1):
            indices=arange(ind - prod(pind(l,arange())) + 1,ind)
# amplify_spatial_lpyr_temporal_iir.m:106
            currAlpha=lambda_ / delta / 8 - 1
# amplify_spatial_lpyr_temporal_iir.m:108
            currAlpha=dot(currAlpha,exaggeration_factor)
# amplify_spatial_lpyr_temporal_iir.m:109
            if (l == nLevels or l == 1):
                filtered[indices,arange()]=0
# amplify_spatial_lpyr_temporal_iir.m:112
            else:
                if (currAlpha > alpha):
                    filtered[indices,arange()]=dot(alpha,filtered(indices,arange()))
# amplify_spatial_lpyr_temporal_iir.m:114
                else:
                    filtered[indices,arange()]=dot(currAlpha,filtered(indices,arange()))
# amplify_spatial_lpyr_temporal_iir.m:116
            ind=ind - prod(pind(l,arange()))
# amplify_spatial_lpyr_temporal_iir.m:119
            # representative lambda will reduce by factor of 2
            lambda_=lambda_ / 2
# amplify_spatial_lpyr_temporal_iir.m:122
        ## Render on the input video
        output=zeros(size(frame))
# amplify_spatial_lpyr_temporal_iir.m:127
        output[arange(),arange(),1]=reconLpyr(filtered(arange(),1),pind)
# amplify_spatial_lpyr_temporal_iir.m:129
        output[arange(),arange(),2]=reconLpyr(filtered(arange(),2),pind)
# amplify_spatial_lpyr_temporal_iir.m:130
        output[arange(),arange(),3]=reconLpyr(filtered(arange(),3),pind)
# amplify_spatial_lpyr_temporal_iir.m:131
        output[arange(),arange(),2]=dot(output(arange(),arange(),2),chromAttenuation)
# amplify_spatial_lpyr_temporal_iir.m:133
        output[arange(),arange(),3]=dot(output(arange(),arange(),3),chromAttenuation)
# amplify_spatial_lpyr_temporal_iir.m:134
        output=frame + output
# amplify_spatial_lpyr_temporal_iir.m:136
        output=ntsc2rgb(output)
# amplify_spatial_lpyr_temporal_iir.m:138
        #             filtered = rgbframe + filtered.*mask;
        output[output > 1]=1
# amplify_spatial_lpyr_temporal_iir.m:141
        output[output < 0]=0
# amplify_spatial_lpyr_temporal_iir.m:142
        writeVideo(vidOut,im2uint8(output))
    
    close_(vidOut)
    return
    
if __name__ == '__main__':
    pass
    