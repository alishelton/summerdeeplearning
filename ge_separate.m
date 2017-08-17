%{
    This file opens and separates GE scans into their high and low
    energy components
    
    Authors:
        Ali Shelton
        Serghei
%}

% File name and opening

fname = 'C:\Users\alishelton\Documents\ge_scans\SA_BTTP111407LE.img';
fid = fopen(fname,'r','b');

% Read in header and date
c1 = fread(fid, 252, 'uint8');
t1 = fread(fid, 1, 'uint8');
t2 = fread(fid, 1, 'uint8');
t3 = fread(fid, 1, 'uint8');
t4 = fread(fid, 1, 'uint8');
time_t2 = t1 + t2*256 + t3 * 256 * 256 + t4 *256 * 256 * 256;
date_str = datestr(time_t2/60/60/24 + 719529);

% Read in rows and columns
tp = fread(fid, 2, 'uint8');
f1 = fread(fid,1,'uint8');
f2 = fread(fid,1,'uint8');
nrows = f1 + f2*256;
f3 = fread(fid,1,'uint8');
f4 = fread(fid,1,'uint8');
ncols = f3 +f4*256;

% Move down to the image and read it
c2 = fread(fid, 250, 'uint8');
Ny = ncols*nrows;
Bv = fread(fid,Ny*2,'uint8');
Bv2 = Bv(1:2:end);
Gv2 = Bv(2:2:end);

% Combine odd and even bytes to form pixel data and shape image
B = Bv2 + Gv2 * 256;
C = rot90(reshape(B,[ncols, nrows]));
G = (flip(C,1));
figure;imagesc(G);colormap(gray);


% testing purposes only

fname2 = 'C:\Users\alishelton\Documents\ge_scans\RA108LHE17052017.img';
fid2 = fopen(fname2,'r','b');

% Read in header and date
c12 = fread(fid2, 252, 'uint8');
t12 = fread(fid2, 1, 'uint8');
t22 = fread(fid2, 1, 'uint8');
t32 = fread(fid2, 1, 'uint8');
t42 = fread(fid2, 1, 'uint8');
% Read in rows and columns
tp2 = fread(fid2, 2, 'uint8');
f12 = fread(fid2,1,'uint8');
f22 = fread(fid2,1,'uint8');
f32 = fread(fid2,1,'uint8');
f42 = fread(fid2,1,'uint8');
% Move down to the image and read it
c22 = fread(fid2, 250, 'uint8');
Bv22 = fread(fid2,Ny*2,'uint8');
Bv222 = Bv22(1:2:end);
Gv22 = Bv22(2:2:end);

% Combine odd and even bytes to form pixel data and shape image
B2 = Bv222 + Gv22 * 256;






