# Generated with SMOP  0.41
from libsmop import *
# build_Lpyr_stack.m

    # [LPYR_STACK, pind] = build_Lpyr_stack(VID_FILE, START_INDEX, END_INDEX)
# 
# Apply Laplacian pyramid decomposition on vidFile from startIndex to
# endIndex
# 
# LPYR_STACK: stack of Laplacian pyramid of each frame 
# the second dimension is the color channel
# the third dimension is the time
    
    # pind: see buildLpyr function in matlabPyrTools library
# 
# Copyright (c) 2011-2012 Massachusetts Institute of Technology, 
# Quanta Research Cambridge, Inc.
    
    # Authors: Hao-yu Wu, Michael Rubinstein, Eugene Shih, 
# License: Please refer to the LICENCE file
# Date: June 2012
    
    
@function
def build_Lpyr_stack(vidFile=None,startIndex=None,endIndex=None,*args,**kwargs):
    varargin = build_Lpyr_stack.varargin
    nargin = build_Lpyr_stack.nargin

    # Read video
    vid=VideoReader(vidFile)
# build_Lpyr_stack.m:22
    
    vidHeight=vid.Height
# build_Lpyr_stack.m:24
    vidWidth=vid.Width
# build_Lpyr_stack.m:25
    nChannels=3
# build_Lpyr_stack.m:26
    temp=struct('cdata',zeros(vidHeight,vidWidth,nChannels,'uint8'),'colormap',[])
# build_Lpyr_stack.m:27
    
    temp.cdata = copy(read(vid,startIndex))
# build_Lpyr_stack.m:31
    rgbframe,__=frame2im(temp,nargout=2)
# build_Lpyr_stack.m:32
    rgbframe=im2double(rgbframe)
# build_Lpyr_stack.m:33
    frame=rgb2ntsc(rgbframe)
# build_Lpyr_stack.m:34
    pyr,pind=buildLpyr(frame(arange(),arange(),1),'auto',nargout=2)
# build_Lpyr_stack.m:36
    
    Lpyr_stack=zeros(size(pyr,1),3,endIndex - startIndex + 1)
# build_Lpyr_stack.m:39
    Lpyr_stack[arange(),1,1]=pyr
# build_Lpyr_stack.m:40
    Lpyr_stack(arange(),2,1),__=buildLpyr(frame(arange(),arange(),2),'auto',nargout=2)
# build_Lpyr_stack.m:42
    Lpyr_stack(arange(),3,1),__=buildLpyr(frame(arange(),arange(),3),'auto',nargout=2)
# build_Lpyr_stack.m:43
    k=1
# build_Lpyr_stack.m:45
    for i in arange(startIndex + 1,endIndex).reshape(-1):
        k=k + 1
# build_Lpyr_stack.m:47
        temp.cdata = copy(read(vid,i))
# build_Lpyr_stack.m:48
        rgbframe,__=frame2im(temp,nargout=2)
# build_Lpyr_stack.m:49
        rgbframe=im2double(rgbframe)
# build_Lpyr_stack.m:51
        frame=rgb2ntsc(rgbframe)
# build_Lpyr_stack.m:52
        Lpyr_stack(arange(),1,k),__=buildLpyr(frame(arange(),arange(),1),'auto',nargout=2)
# build_Lpyr_stack.m:54
        Lpyr_stack(arange(),2,k),__=buildLpyr(frame(arange(),arange(),2),'auto',nargout=2)
# build_Lpyr_stack.m:55
        Lpyr_stack(arange(),3,k),__=buildLpyr(frame(arange(),arange(),3),'auto',nargout=2)
# build_Lpyr_stack.m:56
    
    return Lpyr_stack,pind
    
if __name__ == '__main__':
    pass
    