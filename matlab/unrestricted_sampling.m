function [ X ] = unrestricted_sampling( N, range )
%UNRESTRICTED_SAMPLING Summary of this function goes here
%   It generates samplings unrestrictedly by given sampler number and range

  dim = size(range,1);
  X = rand(dim, N);
  for i=1:1:dim
    X(i,:)=range(i,1)+X(i,:)*(range(i,2)-range(i,1));
  end

end

