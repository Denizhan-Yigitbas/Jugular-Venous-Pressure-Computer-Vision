# Generated with SMOP  0.41
from libsmop import *
# progmeter.m

    
@function
def progmeter(i=None,n=None,w=None,*args,**kwargs):
    varargin = progmeter.varargin
    nargin = progmeter.nargin

    if nargin < 3:
        w=1
# progmeter.m:4
    
    if i == 0:
        fwrite(1,sprintf('00%%'))
        return
    else:
        if ischar(i):
            fwrite(1,sprintf('%s\n',i))
            fwrite(1,sprintf('00%%'))
            return
    
    if mod(i,dot(w,n) / 100) <= mod(i - 1,dot(w,n) / 100):
        fwrite(1,sprintf('\b\b\b'))
        fwrite(1,sprintf('%02d%%',round(dot(100,i) / n)))
    
    if i == n:
        fprintf(1,'\n')
    