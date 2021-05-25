clear;clc;
addpath(genpath(pwd));
%pathh='D:\BOSSbase\BOSSbase_1.01\JPEG\';
%cd(pathh);
%addpath(genpath(pathh));
%--------------------------------------------------------------------------
tic;
n=10000;%��n��ͼ����д
payload=0.4;
QF=75;

% image_name = 'Lena.jpg';%��ȡ���ļ���������jpg��ʽ��ͼ��
% matlabpool local 4
for i=1:10000
    for payload=0.1:0.1:0.4
        % COVER=image_name;
        % STEGO=['stego_',image_name];
        COVER=[num2str(i), '.JPEG'];
        STEGO=[num2str(i), '_',num2str(payload),'_stego.JPEG'];
        S_STRUCT = J_UNIWARD(COVER,payload);
        temp = load(strcat('default_gray_jpeg_obj_', num2str(QF), '.mat'));
        default_gray_jpeg_obj = temp.default_gray_jpeg_obj;
        C_STRUCT = default_gray_jpeg_obj;
        C_STRUCT.coef_arrays{1} = S_STRUCT.coef_arrays{1};
        jpeg_write(C_STRUCT,STEGO);
        fprintf(['�� ',num2str(i), '_',num2str(payload),' ��ͼ��-------- ok','\n']);
    end
end
% matlabpool close;
toc;
%--------------------------------------------------------------------------