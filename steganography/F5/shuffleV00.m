%ͼ�����ҳ���

%���ߣ�ltx1215
%���ڣ�2010��8��7��
%���õ��ǻ��������㷨
clear;
[filename, pathname] = uigetfile('*.jpg', '��ԭʼͼ��')
filename= [pathname filename];
J=imread(filename);
info=imfinfo(filename);
[m,n,p]=size(J);

%������������
x(1)=0.5;
for i=1:m*n-1
    x(i+1)=3.7*x(i)*(1-x(i));
end
[y,num]=sort(x);%�������Ļ������н�������
%���ԭͼΪ�Ҷ�ͼ
if info.BitDepth==8
    
    Scambled=uint8(zeros(m,n));%����һ����ԭͼ��С��ͬ��0����
    for i=1:m*n
        Scambled(i)=J(num(i));
    end
    
    
    %���ԭͼΪ��ֵͼ��
elseif info.BitDepth==1
    S=uint8(zeros(m,n));
    for i=1:m
        for j=1:n
            if J(i,j)==1
                S(i,j)=255;
            end
        end
    end
    Scambled=uint8(zeros(m,n));%����һ����ԭͼ��С��ͬ��0����
    for i=1:m*n
        Scambled(i)=S(num(i));
    end
end

%���Ϊ���ͼ
if p==3
    Red=uint8(zeros(m,n));
    Green=uint8(zeros(m,n));
    Blue=uint8(zeros(m,n));
    RedNew=J(:,:,1);
    GreenNew=J(:,:,2);
    BlueNew=J(:,:,3);
    
    Scambled=uint8(zeros(m,n,p));%����һ����ԭͼ��С��ͬ��0����
    for i=1:m*n
        Red(i)=RedNew(num(i));
        Green(i)=GreenNew(num(i));
        Blue(i)=BlueNew(num(i));
    end
    Scambled(:,:,1)=Red;
    Scambled(:,:,2)=Green;
    Scambled(:,:,3)=Blue;
end
imwrite(Scambled,'Scambled.jpg','quality',100);
imshow(Scambled);


%ͼ�����ҳ���
%���ߣ�ltx1215
%10��8��10��
[filename, pathname] = uigetfile('*.jpg', '��ԭʼͼ��')
filename= [pathname filename];
J=imread(filename);
info=imfinfo(filename);
[m,n,p]=size(J);

%������������
x(1)=0.5;
for i=1:m*n-1
    x(i+1)=3.7*x(i)*(1-x(i));
end
[y,num]=sort(x);%�������Ļ������н�������
%���ԭͼΪ�Ҷ�ͼ
if info.BitDepth==8
    IScamble=uint8(zeros(m,n));%����һ����ԭͼ��С��ͬ��0����
    for i=1:m*n
        IScamble(num(i))=J(i);
     end
    %���ԭͼΪ��ֵͼ��
elseif info.BitDepth==1
    S=uint8(zeros(m,n));
    for i=1:m        
        for j=1:n            
            if J(i,j)==1                
                S(i,j)=255;                
            end            
        end        
    end    
    IScamble=uint8(zeros(m,n));%����һ����ԭͼ��С��ͬ��0����    
    for i=1:m*n        
        IScamble(num(i))=S(i);        
    end    
end

%���Ϊ���ͼ
if p==3    
    Red=uint8(zeros(m,n));    
    Green=uint8(zeros(m,n));    
    Blue=uint8(zeros(m,n));    
    RedNew=J(:,:,1);    
    GreenNew=J(:,:,2);    
    BlueNew=J(:,:,3);   
    IScamble=uint8(zeros(m,n,p));%����һ����ԭͼ��С��ͬ��0����    
    for i=1:m*n        
        Red(num(i))=RedNew(i);        
        Green(num(i))=GreenNew(i);        
        Blue(num(i))=BlueNew(i);        
    end    
    IScamble(:,:,1)=Red;    
    IScamble(:,:,2)=Green;    
    IScamble(:,:,3)=Blue;    
end
imwrite(IScamble,'IScambled.jpg','quality',100);
imshow(IScamble);
