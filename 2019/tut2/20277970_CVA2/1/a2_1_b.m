A = imread("batman.png");
s = 1.55;
theta = 315;
alpha = s*cosd(theta);
beta = s*sind(theta);
m = size(A,1);
n = size(A,2);
x0 = ((m+1)/2);
y0 = (n+1)/2;
H = [alpha beta (((1-alpha)*x0)-(beta*y0));
    -beta alpha (beta*x0 + (1-alpha)*y0);
    0 0 1];
B = applyhomography(A,H);
imshow(B)