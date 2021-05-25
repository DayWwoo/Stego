clear all;clc;
addpath(genpath(pwd));
tic;
% matlabpool local 12
n=100;%��n��ͼ����д
payload=0.1;
QF=75;

%image_name = 'Lena.jpg';

for i=1:n
    COVER=[num2str(i), '.JPEG'];
    STEGO=[num2str(i),'_stego.JPEG'];
    S_STRUCT = UERD(COVER,payload);
    temp = load(strcat('default_gray_jpeg_obj_', num2str(QF), '.mat'));
    default_gray_jpeg_obj = temp.default_gray_jpeg_obj;
    C_STRUCT = default_gray_jpeg_obj;
    C_STRUCT.coef_arrays{1} = S_STRUCT.coef_arrays{1};
    jpeg_write(C_STRUCT,STEGO);
    fprintf(['�� ',num2str(i),' ��ͼ��-------- ok','\n']);
end
% matlabpool close
toc;
% -------------------------------------------------------------------------