function L = Likelihood_subfunction( dXX , varargin )
    L = 0;
    for kk=1:size(dXX,2);
        L = L+Likelihood_subfunction1D(dXX(:,kk),varargin{:});
    end
end