clear;
close all;
clc;

%图像标题会错位，不知什么原因

%% 1 read image rgb2gray

rgb = imread('gecko.bmp');

Ir2g = rgb2gray(rgb);
% Ir2g = medfilt2(rgb2gray(rgb));%转化灰度图后进行中值滤波，效果比未滤波有明显提升
% I = imhmin(Img, 1);
imwrite(Ir2g, 'gray.png');

se1 = strel('disk', 15); %构建多尺度
%高帽，低帽变换，增强亮处和暗处
Itop = imtophat(Ir2g, se1);
% figure
% imshow(Itop), title('Itop')
Ibot = imbothat(Ir2g, se1);
% figure
% imshow(Ibot), title('Ibot')
I = imsubtract(imadd(Itop, Ir2g),Ibot);
imwrite(I, 'gtb.png');
imshow(I)
% imshow(Img)
text(732,501,'Image courtesy of Corel',...
    'FontSize',7,'HorizontalAlignment','right')

%% 2 use the gradient magnitude as the segmentation function

hy = fspecial('sobel'); %索贝尔算子，计算纵向梯度
% hy = fspecial('prewitt');
%'gaussian', 'sobel', 'prewitt', 'laplacian', 'log', 'average', 'unsharp', 'disk', 'motion'

hx = hy';
Iy = imfilter(double(I), hy, 'replicate');
Ix = imfilter(double(I), hx, 'replicate');
gradmag = sqrt(Ix.^2 + Iy.^2); %gradient magnitude
% imwrite(gradmag, 'gradmag.png');
figure
imshow(gradmag, [])%, title('Gradinet magnitude (gradmag)')

%% oversegmentation 不进行其它处理会过分割

oversegL = watershed(gradmag);
oversegLrgb = label2rgb(oversegL);
% figure, imshow(oversegLrgb), title('watershed transform overseg')
imwrite(oversegLrgb,'overseg.png');

%% 3 foreground 标注前景目标

se = strel('disk', 5); 


%构建结构元素，即模板。例子中的参数（'disk',20）不行，关闭操作图像左半边就没了


% %使用se1模板，多尺度 figure3
% %直接调用开启函数的方法
% Io = imopen(I, se1);
% figure
% imshow(Io), title('Opening(Io)')
% 
% %使用se1模板，多尺度
% % 直接调用关闭函数的方法
% Ioc = imclose(Io, se1);
% figure
% imshow(Ioc), title('Opening-closing (Ioc)')




% 基于重建的开启。使用imerode和imreconstruct
%figure4
% Ie = imerode(Itop, se);%腐蚀，重建
% Iobr = imreconstruct(Ie, Itop);
Ie = imerode(I, se);%腐蚀，重建
Iobr = imreconstruct(Ie, I);
figure
imshow(Iobr), title('Opening-by-reconstruction (Iobr)')



%基于重建的关闭
Iobrd = imdilate(Iobr, se);%膨胀
Iobrcbr = imreconstruct(imcomplement(Iobrd), imcomplement(Iobr));%重建 
Iobrcbr = imcomplement(Iobrcbr);%重建结果取补
figure
imshow(Iobrcbr), title('Opening-closing by reconstruction (Iobrcbr)')
imwrite(Iobrcbr, 'o-c.png');

%修改此处识别大的（2），小的（1）
se3 = strel('disk', 2); 
% se3 = strel('disk', 1); 
%计算Iobrcbr的区域极大值从而得到好的前景标记
fgm = imregionalmax(Iobrcbr);%二值图，白色为前景
fgm = imclose(fgm, se3);
% fgm = imdilate(fgm, se4);
figure
imshow(fgm), title('Regional maxima of opening-closing by reconstruction (fgm)')
imwrite(fgm, 'fgm.png');

%将前景标记图叠加到原始图像
I2 = I;
I2(fgm) = 255;%将fgm中的前景区域（像素值为1）标记到原图上（置白色）  
figure
imshow(I2), title('Regional maxima superimposed on original image (I2)')


%注意到一些大部分重合或被阴影遮挡的物体没有被标记出来。这意味着这些物体最终可能不会被正确的分割出来。  
%并且，有些物体中前景标记正确的到达了物体的边缘。这意味着你应该清除掉标记斑块的边缘，向内收缩一点。  
%可以通过先闭操作，再腐蚀做到这点。  
se2 = strel(ones(3,3));
fgm2 = imclose(fgm, se2);
fgm3 = imerode(fgm2, se2);

%这个操作会导致遗留下一些离群的孤立点，这些是需要被移除的。  
%你可以通过bwareaopen做到这点，函数将移除那些包含像素点个数少于指定值的区域
fgm4 = bwareaopen(fgm3, 1);
I3 = I;
I3(fgm4) = 255;
figure
imshow(I3)
title('Modified regional maxima superimposed on original image (fgm4)')


%% 4 计算背景标注

%背景标记
sebw = strel('disk', 1); 
Iebw = imerode(I, sebw);%腐蚀，重建
Iobrbw = imreconstruct(Iebw, I);
Iobrdbw = imdilate(Iobrbw, sebw);%膨胀
Iobrcbrbw = imreconstruct(imcomplement(Iobrdbw), imcomplement(Iobrbw));%重建 
Iobrcbrbw = imcomplement(Iobrcbrbw);%重建结果取补


%原算法的前提假设是：图像中相对亮的是物体，相对暗的是背景
%二值化
% bw = imbinarize(Iobrcbr);

% bw = imregionalmin(Iobrcbr);%计算图像中大量局部最小区域的位置
% bw = imextendedmin(Iobrcbr,15); %计算图像中比周围点更深的点的集合（通过某个高度阈值）

se = strel('disk', 2); 



bw1 = im2bw(Iobrcbrbw, graythresh(Iobrcbrbw)); %这种二值化方法也可以，大津法otsu
bw2 = medfilt2(bw1);
bw = imopen(bw2, se);
% bw3 = imerode(bw2, se);
% bw3 = imdilate(bw3, se);

% bw4 = imreconstruct(bw3,bw2); 
bw = imfill(bw,'holes'); %填洞

figure
imshow(bw), title('Thresholded opening-closing by reconstruction (bw)')
imwrite(bw, 'bw.png');

%通过对bw的距离变换进行分水岭变换，寻找分水岭脊线（DL==0）
D = bwdist(bw);%距离变换
DL = watershed(D);
bgm = DL == 0;%区域分界线 
figure
imshow(bgm), title('Watershed ridge lines (bgm)')
imwrite(bgm, 'edgebgm.png'); 

%% 5 计算修改后的分割函数的分水岭变换

gradmag2 = imimposemin(gradmag, bgm | fgm4);

Lrow = watershed(gradmag2); %调用分水岭函数

%% kmeans分类

areastr = regionprops(Lrow, 'Area');
areacell = struct2cell(areastr);
areamat = cell2mat(areacell);
[m n] = size(areamat);
[idx, C] = kmeans(areamat(1, 2:n)',2);
[a b] = sort(idx, 'descend');
[m1 n1] = size(Lrow);
zero1 = zeros(size(Lrow));
zero2 = zero1;
numbig = sum(a == size(C, 1));
for i = 1:numbig
    c(i) = b(i) + 1;
    zero1(Lrow == c(i)) = 2;
end
zero2(Lrow == 1) = 1;
L = zero1 + zero2;


%% 6 可视化结果
I4 = I;
I4(imdilate(L == 0, ones(3, 3)) | bgm | fgm4) = 255;
figure
imshow(I4)
title('Markers and object boundaries superimposed on original image (I4)')

Lrgb = label2rgb(L, 'jet', 'w', 'shuffle');
figure
imshow(Lrgb)
title('Colored watershed label matrix (Lrgb)')
imwrite(Lrgb, 'Lrgb_big.png');

figure
imshow(I)
hold on
himage = imshow(Lrgb);
himage.AlphaData = 0.3;
title('Lrgb superimposed transparently on original image')