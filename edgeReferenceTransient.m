function [dODReferenced,dODMeasured] = edgeReferenceTransient(Ion,Ioff,Eedge)

% Ensure that the pump on and pump off data are the same size
if any(size(Ion) ~= size(Ioff))
    error('Ion and Ioff do not have the same size');
% Ensure that the range given is the same size as the number of pixels/energy    
elseif size(Eedge) ~= size(Ion,3)
    error('Eedge does not have the same size as Ion along the energy dimension')
else
    
    % Define raw dOD
    if size(Ion,1) == 1  %if there is only one average
        dODMeasured = squeeze(-real(log10(Ion./Ioff)));
    else
        dODMeasured = squeeze(mean(-real(log10(Ion./Ioff))));
    end
    dODMeasured(isnan(dODMeasured)) = 0; dODMeasured(isinf(dODMeasured)) = 0; % Handling edge cases
    
    % Use all shots with pump off to construct an OD calibration set for the noise
    allShotsPumpOff = reshape(permute(Ioff,[2 1 3]),size(Ioff,1)*size(Ioff,2),[]);
    dODCalib = -real(log10(allShotsPumpOff(1:2:(end-1),:)./allShotsPumpOff(2:2:end,:)));

    % define edge zone
    dODEdgeCalib= dODCalib(:,Eedge);
    
    %check if there is more calibration points than edge pixels
    [p,m] = size(dODEdgeCalib);
    if p <= m
        msg = sprintf(['There are less calibration measurements (' num2str(p) ') than edge-pixels (' num2str(m)...
            '). Reduce size of edge-pixel region or use more calibration points. \nThe covariance matrix does not have full rank (RCond = ' num2str(rcond(cov(dODEdgeCalib))) '), results will be inaccurate.']);
        warning(msg);
        warning('off','MATLAB:nearlySingularMatrix') %turn off redundant default warning
    end
            
    %compute B matrix using Equ. (3)
    B= (dODEdgeCalib'*dODEdgeCalib) \ (dODEdgeCalib'*dODCalib);

    %Apply B using Equ. (2)
    dODReferenced = dODMeasured - dODMeasured(:,Eedge)*B;
    warning('on','MATLAB:nearlySingularMatrix') %turn default warning back on
end
end