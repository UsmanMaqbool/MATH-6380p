function features = ScatImg(x,Wop)
addpath 'scatnet-0.2';
addpath_scatnet;
% compute scattering with 5 scales, 6 orientations
% and an oversampling factor of 2^2
filt_opt.J = 5;
filt_opt.L = 6;
scat_opt.oversampling = 2;
[Wop, filters] = wavelet_factory_2d(size(x), filt_opt, scat_opt);
Sx = scat(x, Wop);
% display scattering coefficients
%image_scat(Sx)

features = sum(sum(format_scat(scat(x,Wop)),2),3);

%Corresponding python code
%import matlab
%eng = matlab.engine.start_matlab()
%def scatImg(x): # Input 2D numpy array (can convert picture to Gray scale)
%    sig = matlab.double(x.tolist())
%    ret = eng.ScatImg( sig ) 
%    return np.ndarray.flatten(np.array(ret))
