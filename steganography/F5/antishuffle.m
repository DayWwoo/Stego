%ͼ�����ҳ���
function y = antishuffle(x, key)
[m,n]=size(x);
%������������
s=zeros(1,m*n);
s(1)=key;
for i=2:m*n
    s(i)=3.7*s(i-1)*(1-s(i-1));
end
%�������Ļ������н�������
[~,num]=sort(s);
%����һ����ԭͼ��С��ͬ��0����
y=zeros(m,n);
for i=1:m*n
    y(num(i))=x(i);
end
