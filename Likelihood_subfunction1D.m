function L = Likelihood_subfunction1D( dXX , D, sig2, dT, R )
% dXX is square dst of diff of trajectory with mean subtracted and
% k_th element multiplied by (-1)^k 

    if D<0 || sig2 < 0;
        L = -Inf;
    else
        N = numel(dXX);

        alpha = 2*D*dT*(1-2*R) + 2*sig2;
        beta = 2*R*D*dT - sig2;

        eigvec = alpha + 2*beta*cos(pi*(1:N)./(N+1));
        eigvec = reshape(eigvec, size(dXX));

        L = -.5*sum( log(eigvec) + (2/(N+1)).*dXX./eigvec );
    end;
end