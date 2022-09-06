%% Reset workspace
clc;clear;close all
%% Mô hình của các agent trong graph
f = cell(1,5);
f{1} = [2 1 1 10 1];
f{2} = [2 1 1 3 1];
f{3} = [2 2 1 10 1];
f{4} = [2 1 1 10 3];
f{5} = [2 2 1 10 3];
A = cell(1,5);
B = cell(1,5);
C = cell(1,5);
D = cell(1,5);
for i = 1:5
    fi = f{i};
    A{i} = [0 1 0; 0 0 fi(3);0 -fi(4) -fi(1)];
    B{i} = [0;0;fi(2)];
    C{i} = [1 0 0];
    D{i} = [0;0;fi(5)];
end
%% Mô hình của các leader dynamic
S = [0 1;0 0];
R = [1 0];
%% Các mô hình rời rạc hóa của hệ thống với thời gian trích mẫu khác nhau
PHI = [0.1 0.2];
Az = cell(5,2);
Bz = cell(5,2);
Dz = cell(5,2);
for i = 1:5
    sys1 = ss(A{i},B{i},C{i},0);
    sys2 = ss(A{i},D{i},C{i},0);
    for j = 1:2
        sys1_z = c2d(sys1,PHI(j));
        Az{i,j} = sys1_z.A;
        Bz{i,j} = sys1_z.B;
        sys2_z = c2d(sys2,PHI(j));
        Dz{i,j} = sys2_z.B;
    end
end

%% Rời rạc hóa cho mô hình leader dynamic
Sz = cell(1,2);
for i = 1:2
    Sz{i} = expm(S*PHI(i));
end

%% Tìm tham số của phương trình điều khiển, ma trận kề của graph
A_aject=[0 0 1 1 0;
         0 0 1 0 1;
         1 1 0 0 0;
         1 0 0 0 1;
         0 1 0 1 0];
g_pin = [0 0 1 0 0];
PI = cell(1,5);
for i = 1:5
    PI{i} = [1 0;0 1;0 0];
end
GAMMA = cell(5,2);
for i = 1:2
    GAMMA{1,i} = [0 10];
    GAMMA{2,i} = [0 3];
    GAMMA{3,i} = [0 5];
    GAMMA{4,i} = [0 10];
    GAMMA{5,i} = [0 5];
end
%% Giải hệ phương trình LMIs cho các hệ số K
K_LMI = cell(5,2);
for i = 1:5
    alpha1 = 0.95;
    alpha2 = 0.97;
    to1 = 5;
    muy = 1.1;
    A1 = Az{i,1};
    B1 = Bz{i,1};
    A2 = Az{i,2};
    B2 = Bz{i,2};
    F = [1 0 0];
    D1 = Dz{i,1};
    D2 = Dz{i,2};
    C = [1 0 0];
    N = 20;
    setlmis([]);
    [P1,nP1,sxP1] = lmivar(1,[3 1]);
    [G1,nG1,sxG1] = lmivar(2,[3 3]);
    [L1,nL1,sxL1] = lmivar(1,[1 0]);
    [U1,nU1,sxU1] = lmivar(1,[1 0]);
    [P2,nP2,sxP2] = lmivar(1,[3 1]);
    [G2,nG2,sxG2] = lmivar(2,[3 3]);
    [L2,nL2,sxL2] = lmivar(1,[1 0]);
    [U2,nU2,sxU2] = lmivar(1,[1 0]);
    S1 = newlmi;
    lmiterm([S1 1 1 G1],-1,1,'s');
    lmiterm([S1 1 1 P1],1/alpha1,1);
    lmiterm([S1 2 2 0],-to1^2);
    lmiterm([S1 3 1 G1],A1,1);
    lmiterm([S1 3 1 L1],B1,F);
    lmiterm([S1 3 2 0],D1);
    lmiterm([S1 3 3 P1],-1,1);
    lmiterm([S1 4 1 G1],C,1);
    lmiterm([S1 4 4 0],-1);
    lmiterm([S1 5 1 G1],C,1);
    lmiterm([S1 5 1 U1],1,F);
    lmiterm([S1 5 3 -L1],N',B1');
    lmiterm([S1 5 5 U1],-1,N,'s');
    S2 = newlmi;
    lmiterm([S2 1 1 G2],-1,1,'s');
    lmiterm([S2 1 1 P2],1/alpha2,1);
    lmiterm([S2 2 2 0],-to1^2);
    lmiterm([S2 3 1 G2],A2,1);
    lmiterm([S2 3 1 L2],B2,F);
    lmiterm([S2 3 2 0],D2);
    lmiterm([S2 3 3 P2],-1,1);
    lmiterm([S2 4 1 G2],C,1);
    lmiterm([S2 4 4 0],-1);
    lmiterm([S2 5 1 G2],C,1);
    lmiterm([S2 5 1 U2],1,F);
    lmiterm([S2 5 3 -L2],N',B2');
    lmiterm([S2 5 5 U2],-1,N,'s');
    S3 = newlmi;
    lmiterm([S3 1 1 P1],1,1);
    lmiterm([-S3 1 1 P2],muy,1);
    S4 = newlmi;
    lmiterm([S4 1 1 P2],1,1);
    lmiterm([-S4 1 1 P1],muy,1);
    LMIs = getlmis;
    [~,xopt] = feasp(LMIs);
    L1_LMI = dec2mat(LMIs,xopt,L1);
    U1_LMI = dec2mat(LMIs,xopt,U1);
    L2_LMI = dec2mat(LMIs,xopt,L2);
    U2_LMI = dec2mat(LMIs,xopt,U2);
    K_LMI{i,1} = L1_LMI*U1_LMI^-1;
    K_LMI{i,2} = L2_LMI*U2_LMI^-1;
end
%% Tìm ma trận Laplacian của graph
D_aject = diag(sum(A_aject,1));
Laplacian = D_aject-A_aject;
G_aject = diag(g_pin);
eig_i = eig(Laplacian+G_aject);
%% Giải phương trình LMIs cho bộ điều khiển H
H_LMI = cell(1,2);
Ss1 = Sz{1};
Ss2 = Sz{2};
H_ = eye(2);
setlmis([]);
[Q1,nQ1,sxQ1] = lmivar(1,[2 1]);
[Q2,nQ2,sxQ2] = lmivar(1,[2 1]);
for i = 1:5
    lmiterm([2*i-1 1 1 Q1],-alpha1,1);
    lmiterm([2*i-1 2 1 Q1],1,Ss1);
    lmiterm([2*i-1 2 1 0],-eig_i(i)*H_);
    lmiterm([2*i-1 2 2 Q1],-1,1);
    lmiterm([2*i 1 1 Q2],-alpha2,1);
    lmiterm([2*i 2 1 Q2],1,Ss2);
    lmiterm([2*i 2 1 0],-eig_i(i)*H_);
    lmiterm([2*i 2 2 Q2],-1,1);
end
lmiterm([11 1 1 Q1],1,1);
lmiterm([-11 1 1 Q2],muy,1);
lmiterm([12 1 1 Q2],1,1);
lmiterm([-12 1 1 Q1],muy,1);
LMIs = getlmis;
[~,xopt] = feasp(LMIs);
Q1_LMI = dec2mat(LMIs,xopt,Q1);
Q2_LMI = dec2mat(LMIs,xopt,Q2);
H_LMI{1} = pinv(Q1_LMI)*H_;
H_LMI{2} = pinv(Q2_LMI)*H_;
%% Mô phỏng hệ thống khi không có nhiễu đầu vào
k_max = 400;
t = zeros(1,k_max+1);
x = cell(5,k_max+1);
u = cell(5,k_max+1);
xi0 = cell(1,k_max+1);
xi = cell(5,k_max+1);
y0 = cell(1,k_max+1);
w = cell(5,k_max+1);
y = cell(5,k_max+1);
xi0{1} = [1;2];
t(1) = 0;
for i = 1:5
    x{i,1} = [1;2;3];
    xi{i,1} = [1;2];
end

for i = 1:k_max+1
    if mod(i-1,10) < 5
        ro = 1;
    else 
        ro = 2;
    end
    y0{i} = [1 0]*xi0{i};
    SIGMA = cell(5,1);
    for j = 1:5
        y{j,i} = C*x{j,i};
        SIGMA{j} = 0;
        for k = 1:5
            SIGMA{j} = SIGMA{j}+A_aject(j,k)*(xi{k,i}-xi{j,i});
        end
        SIGMA{j} = SIGMA{j} + g_pin(j)*(xi0{i}-xi{j,i});
        u{j,i} = K_LMI{j,ro}*(y{j,i}-C*PI{j}*xi{j,i})+GAMMA{j,ro}*xi{j,i};
        w{j,i} = j*sin(0.1*t(i))*exp(-0.1*t(i));
%         w{j,i} = j*sin(t(i));
    end


    if i == k_max+1
        break
    end

    if mod(i-1,10) < 5
        t(i+1) = t(i) + 0.1;
    else
        t(i+1) = t(i) + 0.2;
    end
    xi0{i+1} = Sz{ro}*xi0{i};
    for j = 1:5
        xi{j,i+1} = Sz{ro}*xi{j,i} + H_LMI{ro}*SIGMA{j};
        x{j,i+1} = Az{j,ro}*x{j,i} + Bz{j,ro}*u{j,i} + Dz{j,ro}*w{j,i};
    end
end

y = cell2mat(y);
y0 = cell2mat(y0);
plot(t,y-y0);