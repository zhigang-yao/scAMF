function [val] = clip(val)
if val>4
    val = 4;
elseif val<-4
    val = -4;
end
end

