function [ post ] = logpost( sample, nu_t, constraint_func_name )
%LOGPOST Summary of this function goes here
%   Detailed explanation goes here
  
  term = feval(constraint_func_name, sample);
  post = sum( log( normpdf( - term / nu_t ) ) );
end

