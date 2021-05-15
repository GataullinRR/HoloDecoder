% код с мягким решением для числа ошибок от 0 до 100%

clear
% размер объекта
N=input('размер объекта ');
% расстояние между объектом и голограммой
l=N 0.5;
% длина волны
w=1;

% преобразование входного кода в единичный - построение объекта
A=input('кодируемое значение ');

% обнуление объекта и голограммы
Err=0;
for i=1:310;
O(i)=0;
G(i)=0;
F(i)=0;
FInv(i)=0;
L(i)=0;
R(i)=0;
S(i)=0;
T(i)=0;
RInv(i)=0;
SInv(i)=0;
TInv(i)=0;
S1(i)=0;
S2(i)=0;
SInv1(i)=0;
SInv2(i)=0;
S0(i)=0;
SInv0(i)=0;
Out(i)=0;
OutMax(i)=0;
OutG=0;
OutMaxG=0;
OutGD(i)=0;
OutGG1=0;
OutGG2=0;
OutGG3=0;
OutGG4=0;
StatEr=0;

% построение объекта O(i) в единичном коде
if i==A;
x=i;
O(i)=1;
end;
end;

Err=input('количество пакетных ошибок ');
ErrorShift=input('смещение пакета ');
Errors=input('количество случайных ошибок ');

% построение голограммы G
for i=1:N;
j=i-x;
L(i)=hypot(l,j);
G(i)=O(x)*cos((L(i)/w-fix(L(i)/w))/w*2*pi);
if G(i)<=0;
F(i)=0;
else;
F(i)=1;
end;
end;

%вывод голограммы
r (F)
%return

%пакетные ошибки
% наиболее сложный случай - длина пакета 50% длины кодовой комбинации и
% 50% длины каждой половины (занимает вторую и третью четверть): пакет
% Errors=N/2 и расположен с N/4 до N*3/4

%figure;
r(F);

Shift=ErrorShift;
for i=(Shift 1):N;
if Err>0;
F(i)=1-F(i);
Err=Err-1;
end;
end;

%figure;
r(F);

%случайные ошибки
% массив случайных чисел
[r]=randperm(N,N);

for i=1:Errors;
F(r(i))=1-F(r(i));
end;

%декодирование полного прямого и инверсного кода
for i=1:N;
FInv(i)=1-F(i);
for j=1:N;
k=j-i;
L(j)=hypot(l,k);
Rcos(j)=cos((L(j)/w-fix(L(j)/w))/w*2*pi);
R(j)=F(i)*Rcos(j);
S0(j)=S0(j) R(j);
RInv(j)=FInv(i)*Rcos(j);
SInv0(j)=SInv0(j) RInv(j);
end;
end;

figure;
bar (S0);

figure;
bar (SInv0);

%декодирование первой половины прямого и инверсного кода
for i=1:(N/2);
FInv(i)=1-F(i);
for j=1:N;
k=j-i;
L(j)=hypot(l,k);
Rcos(j)=cos((L(j)/w-fix(L(j)/w))/w*2*pi);
R(j)=F(i)*Rcos(j);
S1(j)=S1(j) R(j);
RInv(j)=FInv(i)*Rcos(j);
SInv1(j)=SInv1(j) RInv(j);
end;
end;

figure;
bar (S1);

figure;
bar (SInv1);

%декодирование второй половины прямого и инверсного кода
for i=(N/2 1):N;
FInv(i)=1-F(i);
for j=1:N;
k=j-i;
L(j)=hypot(l,k);
Rcos(j)=cos((L(j)/w-fix(L(j)/w))/w*2*pi);
R(j)=F(i)*Rcos(j);
S2(j)=S2(j) R(j);
RInv(j)=FInv(i)*Rcos(j);
SInv2(j)=SInv2(j) RInv(j);
end;
end;

figure;
bar (S2);

figure;
bar (SInv2);

% поиск положительных максимумов
%поиск первого максимума
Out10P=0;
OutMax10P=0;
Out10PI=0;
OutMax10PI=0;

Out11P=0;
OutMax11P=0;
Out11PI=0;
OutMax11PI=0;
Out12P=0;
OutMax12P=0;
Out12PI=0;
OutMax12PI=0;

for i=1:N;
if S0(i)>OutMax10P;
OutMax10P=S0(i);
%определение выходного значения
Out10P=i;
end;
if SInv0(i)>OutMax10PI;
OutMax10PI=SInv0(i);
%определение выходного значения
Out10PI=i;
end;
end;

for i=1:N;
if S1(i)>OutMax11P;
OutMax11P=S1(i);
%определение выходного значения
Out11P=i;
end;
if SInv1(i)>OutMax11PI;
OutMax11PI=SInv1(i);
%определение выходного значения
Out11PI=i;
end;
end;

Out(161)=Out10P;
OutMax(161)=OutMax10P;
Out(162)=Out10PI;
OutMax(162)=OutMax10PI;

Out(113)=Out11P;
OutMax(113)=OutMax11P;
Out(114)=Out11PI;
OutMax(114)=OutMax11PI;

for i=1:N;
if S2(i)>OutMax12P;
OutMax12P=S2(i);
%определение выходного значения
Out12P=i;
end;
if SInv2(i)>OutMax12PI;
OutMax12PI=SInv2(i);
%определение выходного значения
Out12PI=i;
end;
end;
Out(115)=Out12P;
OutMax(115)=OutMax12P;
Out(116)=Out12PI;
OutMax(116)=OutMax12P;

%поиск второго максимума
Out20P=0;
OutMax20P=0;
Out20PI=0;
OutMax20PI=0;

Out21P=0;
OutMax21P=0;
Out21PI=0;
OutMax21PI=0;
Out22P=0;
OutMax22P=0;
Out22PI=0;
OutMax22PI=0;

for i=1:N;
if S0(i)~=OutMax10P;
if S0(i)>OutMax20P;
OutMax20P=S0(i);
%определение выходного значения
Out20P=i;
end;
end;
if SInv0(i)~=OutMax10PI;
if SInv0(i)>OutMax20PI;
OutMax20PI=SInv0(i);
%определение выходного значения
Out20PI=i;
end;
end;
end;

for i=1:N;
if S1(i)~=OutMax11P;
if S1(i)>OutMax21P;
OutMax21P=S1(i);
%определение выходного значения
Out21P=i;
end;
end;
if SInv1(i)~=OutMax11PI;
if SInv1(i)>OutMax21PI;
OutMax21PI=SInv1(i);
%определение выходного значения
Out21PI=i;
end;
end;
end;
for i=1:N;
if S2(i)~=OutMax12P;
if S2(i)>OutMax22P;
OutMax22P=S2(i);
%определение выходного значения
Out22P=i;
end;
end;
if SInv2(i)~=OutMax12PI;
if SInv2(i)>OutMax22PI;
OutMax22PI=SInv2(i);
%определение выходного значения
Out22PI=i;
end;
end;
end;

%сравнение первых двух максимумов на наличие полуторакратного превышения

if OutMax10P>(1.5*OutMax20P);
OutMax(163)=OutMax10P;
Out(163)=Out10P;
Out(249)=Out(163);
else
OutMax(163)=0;
end;

if OutMax10P>(3*OutMax20P);
OutMax(267)=OutMax10P;
Out(267)=Out10P;
Out(268)=Out(267);
Out(269)=Out(267);
end;

if OutMax10PI>(1.5*OutMax20PI);
OutMax(164)=OutMax10PI;
Out(164)=Out10PI;
Out(250)=Out(164);
else
OutMax(164)=0;
end;

if OutMax10PI>(3*OutMax20PI);
OutMax(270)=OutMax10PI;
Out(270)=Out10PI;
Out(271)=Out(270);
Out(272)=Out(270);
end;

if OutMax11P>(1.5*OutMax21P);
OutMax(105)=OutMax11P;
Out(105)=Out11P;
Out(251)=Out(105);
else
OutMax(105)=0;
end;

if OutMax11P>(3*OutMax21P);
OutMax(273)=OutMax11P;
Out(273)=Out11P;
Out(274)=Out(273);
Out(275)=Out(273);
end;

if OutMax11PI>(1.5*OutMax21PI);
OutMax(106)=OutMax11PI;
Out(106)=Out11PI;
Out(252)=Out(106);
else
OutMax(106)=0;
end;

if OutMax11PI>(3*OutMax21PI);
OutMax(276)=OutMax11PI;
Out(276)=Out11PI;
Out(277)=Out(276);
Out(278)=Out(276);
end;


if OutMax12P>(1.5*OutMax22P);
OutMax(107)=OutMax12P;
Out(107)=Out12P;
Out(253)=Out(107);
else
OutMax(107)=0;
end;

if OutMax12P>(3*OutMax22P);
OutMax(279)=OutMax12P;
Out(279)=Out12P;
Out(280)=Out(279);
Out(281)=Out(279);
end;

if OutMax12PI>(1.5*OutMax22PI);
OutMax(108)=OutMax12PI;
Out(108)=Out12PI;
Out(254)=Out(108);
else
OutMax(108)=0;
end;

if OutMax12PI>(3*OutMax22PI);
OutMax(282)=OutMax12PI;
Out(282)=Out12PI;
Out(283)=Out(282);
Out(284)=Out(282);
end;

%поиск третьего максимума
Out30P=0;
OutMax30P=0;
Out30PI=0;
OutMax30PI=0;

Out31P=0;
OutMax31P=0;
Out31PI=0;
OutMax31PI=0;
Out32P=0;
OutMax32P=0;
Out32PI=0;
OutMax32PI=0;

for i=1:N;
if S0(i)~=OutMax10P&&S0(i)~=OutMax20P;
if S0(i)>OutMax30P;
OutMax30P=S0(i);
%определение выходного значения
Out30P=i;
end;
end;
if SInv0(i)~=OutMax10PI&&SInv0(i)~=OutMax20PI;
if SInv0(i)>OutMax30PI;
OutMax30PI=SInv0(i);
%определение выходного значения
Out30PI=i;
end;
end;
end;


for i=1:N;
if S1(i)~=OutMax11P&&S1(i)~=OutMax21P;
if S1(i)>OutMax31P;
OutMax31P=S1(i);
%определение выходного значения
Out31P=i;
end;
end;
if SInv1(i)~=OutMax11PI&&SInv1(i)~=OutMax21PI;
if SInv1(i)>OutMax31PI;
OutMax31PI=SInv1(i);
%определение выходного значения
Out31PI=i;
end;
end;
end;
for i=1:N;
if S2(i)~=OutMax12P&&S2(i)~=OutMax22P;
if S2(i)>OutMax32P;
OutMax32P=S2(i);
%определение выходного значения
Out32P=i;
end;
end;
if SInv2(i)~=OutMax12PI&&SInv2(i)~=OutMax22PI;
if SInv2(i)>OutMax32PI;
OutMax32PI=SInv2(i);
%определение выходного значения
Out32PI=i;
end;
end;
end;

%поиск четвертого максимума
Out40P=0;
OutMax40P=0;
Out40PI=0;
OutMax40PI=0;

Out41P=0;
OutMax41P=0;
Out41PI=0;
OutMax41PI=0;
Out42P=0;
OutMax42P=0;
Out42PI=0;
OutMax42PI=0;

for i=1:N;
if S0(i)~=OutMax10P&&S0(i)~=OutMax20P&&S0(i)~=OutMax30P;
if S0(i)>OutMax40P;
OutMax40P=S0(i);
%определение выходного значения
Out40P=i;
end;
end;
if SInv0(i)~=OutMax10PI&&SInv0(i)~=OutMax20PI&&SInv0(i)~=OutMax30PI;
if SInv0(i)>OutMax40PI;
OutMax40PI=SInv0(i);
%определение выходного значения
Out40PI=i;
end;
end;
end;

for i=1:N;
if S1(i)~=OutMax11P&&S1(i)~=OutMax21P&&S1(i)~=OutMax31P;
if S1(i)>OutMax41P;
OutMax41P=S1(i);
%определение выходного значения
Out41P=i;
end;
end;
if SInv1(i)~=OutMax11PI&&SInv1(i)~=OutMax21PI&&SInv1(i)~=OutMax31PI;
if SInv1(i)>OutMax41PI;
OutMax41PI=SInv1(i);
%определение выходного значения
Out41PI=i;
end;
end;
end;
for i=1:N;
if S2(i)~=OutMax12P&&S2(i)~=OutMax22P&&S2(i)~=OutMax32P;
if S2(i)>OutMax42P;
OutMax42P=S2(i);
%определение выходного значения
Out42P=i;
end;
end;
if SInv2(i)~=OutMax12PI&&SInv2(i)~=OutMax22PI&&SInv2(i)~=OutMax32PI;
if SInv2(i)>OutMax42PI;
OutMax42PI=SInv2(i);
%определение выходного значения
Out42PI=i;
end;
end;
end;

% поиск отрицательных максимумов
%поиск первого максимума
Out10N=0;
OutMax10N=0;
Out10NI=0;
OutMax10NI=0;

Out11N=0;
OutMax11N=0;
Out11NI=0;
OutMax11NI=0;
Out12N=0;
OutMax12N=0;
Out12NI=0;
OutMax12NI=0;

for i=1:N;
if S0(i)<OutMax10N;
OutMax10N=S0(i);
%определение выходного значения
Out10N=i;
end;
if SInv0(i)<OutMax10NI;
OutMax10NI=SInv0(i);
%определение выходного значения
Out10NI=i;
end;
end;

for i=1:N;
if S1(i)<OutMax11N;
OutMax11N=S1(i);
%определение выходного значения
Out11N=i;
end;
if SInv1(i)<OutMax11NI;
OutMax11NI=SInv1(i);
%определение выходного значения
Out11NI=i;
end;
end;

Out(165)=Out10N;
OutMax(165)=OutMax10N;
Out(166)=Out10NI;
OutMax(166)=OutMax10N;

Out(117)=Out11N;
OutMax(117)=OutMax11N;
Out(118)=Out11NI;
OutMax(118)=OutMax11N;

for i=1:N;
if S2(i)<OutMax12N;
OutMax12N=S2(i);
%определение выходного значения
Out12N=i;
end;
if SInv2(i)<OutMax12NI;
OutMax12NI=SInv2(i);
%определение выходного значения
Out12NI=i;
end;
end;
Out(119)=Out12N;
OutMax(119)=OutMax12N;
Out(120)=Out12NI;
OutMax(120)=OutMax12NI;

%поиск второго максимума
Out20N=0;
OutMax20N=0;
Out20NI=0;
OutMax20NI=0;

Out21N=0;
OutMax21N=0;
Out21NI=0;
OutMax21NI=0;
Out22N=0;
OutMax22N=0;
Out22NI=0;
OutMax22NI=0;

for i=1:N;
if S0(i)~=OutMax10N;
if S0(i)<OutMax20N;
OutMax20N=S0(i);
%определение выходного значения
Out20N=i;
end;
end;
if SInv0(i)~=OutMax10NI;
if SInv0(i)<OutMax20NI;
OutMax20NI=SInv0(i);
%определение выходного значения
Out20NI=i;
end;
end;
end;

for i=1:N;
if S1(i)~=OutMax11N;
if S1(i)<OutMax21N;
OutMax21N=S1(i);
%определение выходного значения
Out21N=i;
end;
end;
if SInv1(i)~=OutMax11NI;
if SInv1(i)<OutMax21NI;
OutMax21NI=SInv1(i);
%определение выходного значения
Out21NI=i;
end;
end;
end;
for i=1:N;
if S2(i)~=OutMax12N;
if S2(i)<OutMax22N;
OutMax22N=S2(i);
%определение выходного значения
Out22N=i;
end;
end;
if SInv2(i)~=OutMax12NI;
if SInv2(i)<OutMax22NI;
OutMax22NI=SInv2(i);
%определение выходного значения
Out22NI=i;
end;
end;
end;

%сравнение первых двух максимумов на наличие полуторакратного превышения
if OutMax10N<(1.5*OutMax20N);
OutMax(167)=OutMax10N;
Out(167)=Out10N;
Out(255)=Out(167);
else
OutMax(167)=0;
end;

if OutMax10N<(3*OutMax20N);
OutMax(285)=OutMax10N;
Out(285)=Out10N;
Out(286)=Out(285);
Out(287)=Out(285);
end;

if OutMax10NI<(1.5*OutMax20NI);
OutMax(168)=OutMax10NI;
Out(168)=Out10NI;
Out(256)=Out(168);
else
OutMax(168)=0;
end;

if OutMax10NI<(3*OutMax20NI);
OutMax(288)=OutMax10NI;
Out(288)=Out10NI;
Out(289)=Out(288);
Out(290)=Out(288);
end;


if OutMax11N<(1.5*OutMax21N);
OutMax(109)=OutMax11N;
Out(109)=Out11N;
Out(257)=Out(109);
else
OutMax(109)=0;
end;

if OutMax11N<(3*OutMax21N);
OutMax(291)=OutMax11N;
Out(291)=Out11N;
Out(292)=Out(291);
Out(293)=Out(291);
end;

if OutMax11NI<(1.5*OutMax21NI);
OutMax(110)=OutMax11NI;
Out(110)=Out11NI;
Out(258)=Out(110);
else
OutMax(110)=0;
end;

if OutMax11NI<(3*OutMax21NI);
OutMax(294)=OutMax11NI;
Out(294)=Out11NI;
Out(295)=Out(294);
Out(296)=Out(294);
end;


if OutMax12N<(1.5*OutMax22N);
OutMax(111)=OutMax12N;
Out(111)=Out12N;
Out(259)=Out(111);
else
OutMax(111)=0;
end;

if OutMax12N<(3*OutMax22N);
OutMax(297)=OutMax12N;
Out(297)=Out12N;
Out(298)=Out(297);
Out(299)=Out(297);
end;

if OutMax12NI<(1.5*OutMax22NI);
OutMax(112)=OutMax12NI;
Out(112)=Out12NI;
Out(260)=Out(112);
else
OutMax(112)=0;
end;

if OutMax12NI<(3*OutMax22NI);
OutMax(300)=OutMax12NI;
Out(300)=Out12NI;
Out(301)=Out(300);
Out(302)=Out(300);
end;

%поиск третьего максимума
Out30N=0;
OutMax30N=0;
Out30NI=0;
OutMax30NI=0;

Out31N=0;
OutMax31N=0;
Out31NI=0;
OutMax31NI=0;
Out32N=0;
OutMax32N=0;
Out32NI=0;
OutMax32NI=0;

for i=1:N;
if S0(i)~=OutMax10N&&S0(i)~=OutMax20N;
if S0(i)<OutMax30N;
OutMax30N=S0(i);
%определение выходного значения
Out30N=i;
end;
end;
if SInv0(i)~=OutMax10NI&&SInv0(i)~=OutMax20NI;
if SInv0(i)<OutMax30NI;
OutMax30NI=SInv0(i);
%определение выходного значения
Out31N0=i;
end;
end;
end;

for i=1:N;
if S1(i)~=OutMax11N&&S1(i)~=OutMax21N;
if S1(i)<OutMax31N;
OutMax31N=S1(i);
%определение выходного значения
Out31N=i;
end;
end;
if SInv1(i)~=OutMax11NI&&SInv1(i)~=OutMax21NI;
if SInv1(i)<OutMax31NI;
OutMax31NI=SInv1(i);
%определение выходного значения
Out31NI=i;
end;
end;
end;
for i=1:N;
if S2(i)~=OutMax12N&&S2(i)~=OutMax22N;
if S2(i)<OutMax32N;
OutMax32N=S2(i);
%определение выходного значения
Out32N=i;
end;
end;
if SInv2(i)~=OutMax12NI&&SInv2(i)~=OutMax22NI;
if SInv2(i)<OutMax32NI;
OutMax32NI=SInv2(i);
%определение выходного значения
Out32NI=i;
end;
end;
end;

%поиск четвертого максимума
Out40N=0;
OutMax40N=0;
Out40NI=0;
OutMax40NI=0;

Out41N=0;
OutMax41N=0;
Out41NI=0;
OutMax41NI=0;
Out42N=0;
OutMax42N=0;
Out42NI=0;
OutMax42NI=0;

for i=1:N;
if S0(i)~=OutMax10N&&S0(i)~=OutMax20N&&S0(i)~=OutMax30N;
if S0(i)<OutMax40N;
OutMax40N=S0(i);
%определение выходного значения
Out40N=i;
end;
end;
if SInv0(i)~=OutMax10NI&&SInv0(i)~=OutMax20NI&&SInv0(i)~=OutMax30NI;
if SInv0(i)<OutMax40NI;
OutMax40NI=SInv0(i);
%определение выходного значения
Out40NI=i;
end;
end;
end;

for i=1:N;
if S1(i)~=OutMax11N&&S1(i)~=OutMax21N&&S1(i)~=OutMax31N;
if S1(i)<OutMax41N;
OutMax41N=S1(i);
%определение выходного значения
Out41N=i;
end;
end;
if SInv1(i)~=OutMax11NI&&SInv1(i)~=OutMax21NI&&SInv1(i)~=OutMax31NI;
if SInv1(i)<OutMax41NI;
OutMax41NI=SInv1(i);
%определение выходного значения
Out41NI=i;
end;
end;
end;
for i=1:N;
if S2(i)~=OutMax12N&&S2(i)~=OutMax22N&&S2(i)~=OutMax32N;
if S2(i)<OutMax42N;
OutMax42N=S2(i);
%определение выходного значения
Out42N=i;
end;
end;
if SInv2(i)~=OutMax12NI&&SInv2(i)~=OutMax22NI&&SInv2(i)~=OutMax32NI;
if SInv2(i)<OutMax42NI;
OutMax42NI=SInv2(i);
%определение выходного значения
Out42NI=i;
end;
end;
end;

%Out=Out1

% построение глобального максимума на основе выбора пар максимумов,
% расположенных симметрично на расстоянии 2,4 и 6 точек по осям голограмм.
% Кандидат на результат декодирования - в центре между ними

%анализ двух полных голограмм
%положительный прямой код
if abs(Out10P-Out20P)==2;
Out(169)=(Out10P Out20P)/2;
OutMax(169)=(OutMax10P OutMax20P)/2;
end;

%анализ превышения среднего значения двух первых максимумов над третьим
if OutMax(169)>(1.4*OutMax30P);
Out(241)=Out(169);
OutMax(241)=OutMax(169);
Out(245)=Out(241);
end;

if OutMax(169)>(2*OutMax30P);
Out(261)=Out(169);
OutMax(261)=OutMax(169);
Out(262)=Out(261);
Out(263)=Out(261);
OutMax(262)=OutMax(169);
OutMax(263)=OutMax(169);
end;
%

if abs(Out10P-Out30P)==2;
Out(170)=(Out10P Out30P)/2;
OutMax(170)=(OutMax10P OutMax30P)/2;
end;

if abs(Out10P-Out40P)==2;
Out(171)=(Out10P Out40P)/2;
OutMax(171)=(OutMax10P OutMax40P)/2;
end;

if abs(Out20P-Out30P)==2;
Out(172)=(Out20P Out30P)/2;
OutMax(172)=(OutMax20P OutMax30P)/2;
end;

if abs(Out20P-Out40P)==2;
Out(173)=(Out20P Out40P)/2;
OutMax(173)=(OutMax20P OutMax40P)/2;
end;

if abs(Out30P-Out40P)==2;
Out(174)=(Out30P Out40P)/2;
OutMax(174)=(OutMax30P OutMax40P)/2;
end;

if abs(Out10P-Out20P)==4;
Out(175)=(Out10P Out20P)/2;
OutMax(175)=(OutMax10P OutMax20P)/2;
end;

if abs(Out10P-Out30P)==4;
Out(176)=(Out10P Out30P)/2;
OutMax(176)=(OutMax10P OutMax30P)/2;
end;

if abs(Out10P-Out40P)==4;
Out(177)=(Out10P Out40P)/2;
OutMax(177)=(OutMax10P OutMax40P)/2;
end;

if abs(Out20P-Out30P)==4;
Out(178)=(Out20P Out30P)/2;
OutMax(178)=(OutMax20P OutMax30P)/2;
end;

if abs(Out20P-Out40P)==4;
Out(179)=(Out20P Out40P)/2;
OutMax(179)=(OutMax20P OutMax40P)/2;
end;

if abs(Out30P-Out40P)==4;
Out(180)=(Out30P Out40P)/2;
OutMax(180)=(OutMax30P OutMax40P)/2;
end;

if abs(Out10P-Out20P)==6;
Out(181)=(Out10P Out20P)/2;
OutMax(181)=(OutMax10P OutMax20P)/2;
end;

if abs(Out10P-Out30P)==6;
Out(182)=(Out10P Out30P)/2;
OutMax(182)=(OutMax10P OutMax20P)/2;
end;

if abs(Out10P-Out40P)==6;
Out(183)=(Out10P Out40P)/2;
OutMax(183)=(OutMax10P OutMax40P)/2;
end;

if abs(Out20P-Out30P)==6;
Out(184)=(Out20P Out30P)/2;
OutMax(184)=(OutMax20P OutMax30P)/2;
end;

if abs(Out20P-Out40P)==6;
Out(185)=(Out20P Out40P)/2;
OutMax(185)=(OutMax20P OutMax40P)/2;
end;

if abs(Out30P-Out40P)==6;
Out(186)=(Out30P Out40P)/2;
OutMax(186)=(OutMax30P OutMax40P)/2;
end;

%положительный инверсный код
if abs(Out10PI-Out20PI)==2;
Out(187)=(Out10PI Out20PI)/2;
OutMax(187)=(OutMax10PI OutMax20PI)/2;
end;

%анализ превышения среднего значения двух первых максимумов над третьим
if OutMax(187)>(1.4*OutMax30PI);
Out(242)=Out(187);
OutMax(242)=OutMax(187);
Out(246)=Out(242);
end;

if OutMax(187)>(2*OutMax30PI);
Out(264)=Out(187);
OutMax(264)=OutMax(187);
Out(265)=Out(264);
Out(266)=Out(264);
OutMax(265)=OutMax(187);
OutMax(266)=OutMax(187);
end;
%
if abs(Out10PI-Out30PI)==2;
Out(188)=(Out10PI Out30PI)/2;
OutMax(188)=(OutMax10P OutMax30P)/2;
end;

if abs(Out10PI-Out40PI)==2;
Out(189)=(Out10PI Out40PI)/2;
OutMax(189)=(OutMax10PI OutMax40PI)/2;
end;

if abs(Out20PI-Out30PI)==2;
Out(190)=(Out20PI Out30PI)/2;
OutMax(190)=(OutMax20PI OutMax30PI)/2;
end;

if abs(Out20PI-Out40PI)==2;
Out(191)=(Out20PI Out40PI)/2;
OutMax(191)=(OutMax20P OutMax40P)/2;
end;

if abs(Out30PI-Out40PI)==2;
Out(192)=(Out30PI Out40PI)/2;
OutMax(192)=(OutMax30P OutMax40P)/2;
end;

if abs(Out10PI-Out20PI)==4;
Out(193)=(Out10PI Out20PI)/2;
OutMax(193)=(OutMax10PI OutMax20PI)/2;
end;

if abs(Out10PI-Out30PI)==4;
Out(194)=(Out10PI Out30PI)/2;
OutMax(194)=(OutMax10PI OutMax30PI)/2;
end;

if abs(Out10PI-Out40PI)==4;
Out(195)=(Out10PI Out40PI)/2;
OutMax(195)=(OutMax10PI OutMax40PI)/2;
end;

if abs(Out20PI-Out30PI)==4;
Out(196)=(Out20PI Out30PI)/2;
OutMax(196)=(OutMax20PI OutMax30PI)/2;
end;

if abs(Out20PI-Out40PI)==4;
Out(197)=(Out20PI Out40PI)/2;
OutMax(197)=(OutMax20PI OutMax40PI)/2;
end;

if abs(Out30PI-Out40PI)==4;
Out(198)=(Out30PI Out40PI)/2;
OutMax(198)=(OutMax30PI OutMax40PI)/2;
end;

if abs(Out10PI-Out20PI)==6;
Out(199)=(Out10PI Out20PI)/2;
OutMax(199)=(OutMax10PI OutMax20PI)/2;
end;

if abs(Out10PI-Out30PI)==6;
Out(200)=(Out10PI Out30PI)/2;
OutMax(200)=(OutMax10PI OutMax30PI)/2;
end;

if abs(Out10PI-Out40PI)==6;
Out(201)=(Out10PI Out40PI)/2;
OutMax(201)=(OutMax10PI OutMax40PI)/2;
end;

if abs(Out20PI-Out30PI)==6;
Out(202)=(Out20PI Out30PI)/2;
OutMax(202)=(OutMax20PI OutMax30PI)/2;
end;

if abs(Out20PI-Out40PI)==6;
Out(203)=(Out20PI Out40PI)/2;
OutMax(203)=(OutMax20PI OutMax40PI)/2;
end;

if abs(Out30PI-Out40PI)==6;
Out(204)=(Out30PI Out40PI)/2;
OutMax(204)=(OutMax30PI OutMax40PI)/2;
end;

%отрицательный прямой код
if abs(Out10N-Out20N)==2;
Out(205)=(Out10N Out20N)/2;
OutMax(205)=(OutMax10N OutMax20N)/2;
end;

%анализ превышения среднего значения двух первых максимумов над третьим
if OutMax(205)<(1.4*OutMax30N);
Out(243)=Out(205);
OutMax(243)=OutMax(205);
Out(247)=Out(243);
end;
if OutMax(205)<(2*OutMax30N);
Out(303)=Out(205);
OutMax(303)=OutMax(205);
Out(304)=Out(303);
Out(305)=Out(303);
OutMax(304)=OutMax(205);
OutMax(305)=OutMax(205);
end;
%

if abs(Out10N-Out30N)==2;
Out(206)=(Out10N Out30N)/2;
OutMax(206)=(OutMax10N OutMax30N)/2;
end;

if abs(Out10N-Out40N)==2;
Out(207)=(Out10N Out40N)/2;
OutMax(207)=(OutMax10N OutMax40N)/2;
end;

if abs(Out20N-Out30N)==2;
Out(208)=(Out20N Out30N)/2;
OutMax(208)=(OutMax20N OutMax30N)/2;
end;

if abs(Out20N-Out40N)==2;
Out(209)=(Out20N Out40N)/2;
OutMax(209)=(OutMax20N OutMax40N)/2;
end;

if abs(Out30N-Out40N)==2;
Out(210)=(Out30N Out40N)/2;
OutMax(210)=(OutMax30N OutMax40N)/2;
end;

if abs(Out10N-Out20N)==4;
Out(211)=(Out10N Out20N)/2;
OutMax(211)=(OutMax10N OutMax20N)/2;
end;

if abs(Out10N-Out30N)==4;
Out(212)=(Out10N Out30N)/2;
OutMax(212)=(OutMax10N OutMax30N)/2;
end;

if abs(Out10N-Out40N)==4;
Out(213)=(Out10N Out40N)/2;
OutMax(213)=(OutMax10N OutMax40N)/2;
end;

if abs(Out20N-Out30N)==4;
Out(214)=(Out20N Out30N)/2;
OutMax(214)=(OutMax20N OutMax30N)/2;
end;

if abs(Out20N-Out40N)==4;
Out(215)=(Out20N Out40N)/2;
OutMax(215)=(OutMax20N OutMax40N)/2;
end;

if abs(Out30N-Out40N)==4;
Out(216)=(Out30N Out40N)/2;
OutMax(216)=(OutMax30N OutMax40N)/2;
end;

if abs(Out10N-Out20N)==6;
Out(217)=(Out10N Out20N)/2;
OutMax(217)=(OutMax10N OutMax20N)/2;
end;

if abs(Out10N-Out30N)==6;
Out(218)=(Out10N Out30N)/2;
OutMax(218)=(OutMax10N OutMax30N)/2;
end;

if abs(Out10N-Out40N)==6;
Out(219)=(Out10N Out40N)/2;
OutMax(219)=(OutMax10N OutMax40N)/2;
end;

if abs(Out20N-Out30N)==6;
Out(220)=(Out20N Out30N)/2;
OutMax(220)=(OutMax20N OutMax30N)/2;
end;

if abs(Out20N-Out40N)==6;
Out(221)=(Out20N Out40N)/2;
OutMax(221)=(OutMax20N OutMax40N)/2;
end;

if abs(Out30N-Out40N)==6;
Out(222)=(Out30N Out40N)/2;
OutMax(222)=(OutMax30N OutMax40N)/2;
end;

%отрицательный инверсный код
if abs(Out10NI-Out20NI)==2;
Out(223)=(Out10NI Out20NI)/2;
OutMax(223)=(OutMax10NI OutMax20NI)/2;
end;

%анализ превышения среднего значения двух первых максимумов над третьим
if OutMax(223)<(1.4*OutMax30NI);
Out(244)=Out(223);
OutMax(244)=OutMax(223);
Out(248)=Out(244);
end;
if OutMax(223)<(2*OutMax30NI);
Out(306)=Out(223);
OutMax(306)=OutMax(223);
Out(307)=Out(306);
Out(308)=Out(306);
OutMax(307)=OutMax(223);
OutMax(308)=OutMax(223);
end;
%

if abs(Out10NI-Out30NI)==2;
Out(224)=(Out10NI Out30NI)/2;
OutMax(224)=(OutMax10NI OutMax30NI)/2;
end;

if abs(Out10NI-Out40NI)==2;
Out(225)=(Out10NI Out40NI)/2;
OutMax(225)=(OutMax10NI OutMax40NI)/2;
end;

if abs(Out20NI-Out30NI)==2;
Out(226)=(Out20NI Out30NI)/2;
OutMax(226)=(OutMax20NI OutMax30NI)/2;
end;

if abs(Out20NI-Out40NI)==2;
Out(227)=(Out20NI Out40NI)/2;
OutMax(227)=(OutMax20NI OutMax40NI)/2;
end;

if abs(Out30NI-Out40NI)==2;
Out(228)=(Out30NI Out40NI)/2;
OutMax(228)=(OutMax30NI OutMax40NI)/2;
end;

if abs(Out10NI-Out20NI)==4;
Out(229)=(Out10NI Out20NI)/2;
OutMax(229)=(OutMax10NI OutMax20NI)/2;
end;

if abs(Out10NI-Out30NI)==4;
Out(230)=(Out10NI Out30NI)/2;
OutMax(230)=(OutMax10NI OutMax30NI)/2;
end;

if abs(Out10NI-Out40NI)==4;
Out(231)=(Out10NI Out40NI)/2;
OutMax(231)=(OutMax10NI OutMax40NI)/2;
end;

if abs(Out20NI-Out30NI)==4;
Out(232)=(Out20NI Out30NI)/2;
OutMax(232)=(OutMax20NI OutMax30NI)/2;
end;

if abs(Out20NI-Out40NI)==4;
Out(233)=(Out20NI Out40NI)/2;
OutMax(233)=(OutMax20NI OutMax40NI)/2;
end;

if abs(Out30NI-Out40NI)==4;
Out(234)=(Out30NI Out40NI)/2;
OutMax(234)=(OutMax30NI OutMax40NI)/2;
end;

if abs(Out10NI-Out20NI)==6;
Out(235)=(Out10NI Out20NI)/2;
OutMax(235)=(OutMax10NI OutMax20NI)/2;
end;

if abs(Out10NI-Out30NI)==6;
Out(236)=(Out10NI Out30NI)/2;
OutMax(236)=(OutMax10NI OutMax30NI)/2;
end;

if abs(Out10NI-Out40NI)==6;
Out(237)=(Out10NI Out40NI)/2;
OutMax(237)=(OutMax10NI OutMax40NI)/2;
end;

if abs(Out20NI-Out30NI)==6;
Out(238)=(Out20NI Out30NI)/2;
OutMax(238)=(OutMax20NI OutMax30NI)/2;
end;

if abs(Out20NI-Out40NI)==6;
Out(239)=(Out20NI Out40NI)/2;
OutMax(239)=(OutMax20NI OutMax40NI)/2;
end;

if abs(Out30NI-Out40NI)==6;
Out(240)=(Out30NI Out40NI)/2;
OutMax(240)=(OutMax30NI OutMax40NI)/2;
end;

%анализ по половинам четырех голограмм

%положительный прямой код первой половины

if abs(Out11P-Out21P)==2;
Out(1)=(Out11P Out21P)/2;
OutMax(1)=(OutMax11P OutMax21P)/2;
end;

if abs(Out11P-Out31P)==2;
Out(121)=(Out11P Out31P)/2;
OutMax(121)=(OutMax11P OutMax31P)/2;
end;

if abs(Out11P-Out41P)==2;
Out(122)=(Out11P Out41P)/2;
OutMax(122)=(OutMax11P OutMax41P)/2;
end;

if abs(Out21P-Out31P)==2;
Out(123)=(Out21P Out31P)/2;
OutMax(123)=(OutMax21P OutMax31P)/2;
end;

if abs(Out21P-Out41P)==2;
Out(124)=(Out21P Out41P)/2;
OutMax(124)=(OutMax21P OutMax41P)/2;
end;

if abs(Out31P-Out41P)==2;
Out(125)=(Out31P Out41P)/2;
OutMax(125)=(OutMax31P OutMax41P)/2;
end;

if abs(Out11P-Out21P)==4;
Out(2)=(Out11P Out21P)/2;
OutMax(2)=(OutMax11P OutMax21P)/2;
end;

if abs(Out11P-Out31P)==4;
Out(3)=(Out11P Out31P)/2;
OutMax(3)=(OutMax11P OutMax31P)/2;
end;

if abs(Out11P-Out41P)==4;
Out(4)=(Out11P Out41P)/2;
OutMax(4)=(OutMax11P OutMax41P)/2;
end;

if abs(Out21P-Out31P)==4;
Out(5)=(Out21P Out31P)/2;
OutMax(5)=(OutMax21P OutMax31P)/2;
end;

if abs(Out21P-Out41P)==4;
Out(6)=(Out21P Out41P)/2;
OutMax(6)=(OutMax21P OutMax41P)/2;
end;

if abs(Out31P-Out41P)==4;
Out(7)=(Out31P Out41P)/2;
OutMax(7)=(OutMax31P OutMax41P)/2;
end;

if abs(Out11P-Out21P)==6;
Out(8)=(Out11P Out21P)/2;
OutMax(8)=(OutMax11P OutMax21P)/2;
end;

if abs(Out11P-Out31P)==6;
Out(9)=(Out11P Out31P)/2;
OutMax(9)=(OutMax11P OutMax21P)/2;
end;

if abs(Out11P-Out41P)==6;
Out(10)=(Out11P Out41P)/2;
OutMax(10)=(OutMax11P OutMax41P)/2;
end;

if abs(Out21P-Out31P)==6;
Out(11)=(Out21P Out31P)/2;
OutMax(11)=(OutMax21P OutMax31P)/2;
end;

if abs(Out21P-Out41P)==6;
Out(12)=(Out21P Out41P)/2;
OutMax(12)=(OutMax21P OutMax41P)/2;
end;

if abs(Out31P-Out41P)==6;
Out(13)=(Out31P Out41P)/2;
OutMax(13)=(OutMax31P OutMax41P)/2;
end;

%положительный инверсный код первой половины
if abs(Out11PI-Out21PI)==2;
Out(14)=(Out11PI Out21PI)/2;
OutMax(14)=(OutMax11PI OutMax21PI)/2;
end;

if abs(Out11PI-Out31PI)==2;
Out(126)=(Out11PI Out31PI)/2;
OutMax(126)=(OutMax11P OutMax31P)/2;
end;

if abs(Out11PI-Out41PI)==2;
Out(127)=(Out11PI Out41PI)/2;
OutMax(127)=(OutMax11PI OutMax41PI)/2;
end;

if abs(Out21PI-Out31PI)==2;
Out(128)=(Out21PI Out31PI)/2;
OutMax(128)=(OutMax21PI OutMax31PI)/2;
end;

if abs(Out21PI-Out41PI)==2;
Out(129)=(Out21PI Out41PI)/2;
OutMax(129)=(OutMax21P OutMax41P)/2;
end;

if abs(Out31PI-Out41PI)==2;
Out(130)=(Out31PI Out41PI)/2;
OutMax(130)=(OutMax31P OutMax41P)/2;
end;

if abs(Out11PI-Out21PI)==4;
Out(15)=(Out11PI Out21PI)/2;
OutMax(15)=(OutMax11PI OutMax21PI)/2;
end;

if abs(Out11PI-Out31PI)==4;
Out(16)=(Out11PI Out31PI)/2;
OutMax(16)=(OutMax11PI OutMax31PI)/2;
end;

if abs(Out11PI-Out41PI)==4;
Out(17)=(Out11PI Out41PI)/2;
OutMax(17)=(OutMax11PI OutMax41PI)/2;
end;

if abs(Out21PI-Out31PI)==4;
Out(18)=(Out21PI Out31PI)/2;
OutMax(18)=(OutMax21PI OutMax31PI)/2;
end;

if abs(Out21PI-Out41PI)==4;
Out(19)=(Out21PI Out41PI)/2;
OutMax(19)=(OutMax21PI OutMax41PI)/2;
end;

if abs(Out31PI-Out41PI)==4;
Out(20)=(Out31PI Out41PI)/2;
OutMax(20)=(OutMax31PI OutMax41PI)/2;
end;

if abs(Out11PI-Out21PI)==6;
Out(21)=(Out11PI Out21PI)/2;
OutMax(21)=(OutMax11PI OutMax21PI)/2;
end;

if abs(Out11PI-Out31PI)==6;
Out(22)=(Out11PI Out31PI)/2;
OutMax(22)=(OutMax11PI OutMax31PI)/2;
end;

if abs(Out11PI-Out41PI)==6;
Out(23)=(Out11PI Out41PI)/2;
OutMax(23)=(OutMax11PI OutMax41PI)/2;
end;

if abs(Out21PI-Out31PI)==6;
Out(24)=(Out21PI Out31PI)/2;
OutMax(24)=(OutMax21PI OutMax31PI)/2;
end;

if abs(Out21PI-Out41PI)==6;
Out(25)=(Out21PI Out41PI)/2;
OutMax(25)=(OutMax21PI OutMax41PI)/2;
end;

if abs(Out31PI-Out41PI)==6;
Out(26)=(Out31PI Out41PI)/2;
OutMax(26)=(OutMax31PI OutMax41PI)/2;
end;

%отрицательный прямой код первой половины
if abs(Out11N-Out21N)==2;
Out(27)=(Out11N Out21N)/2;
OutMax(27)=(OutMax11N OutMax21N)/2;
end;

if abs(Out11N-Out31N)==2;
Out(131)=(Out11N Out31N)/2;
OutMax(131)=(OutMax11N OutMax31N)/2;
end;

if abs(Out11N-Out41N)==2;
Out(132)=(Out11N Out41N)/2;
OutMax(132)=(OutMax11N OutMax41N)/2;
end;

if abs(Out21N-Out31N)==2;
Out(133)=(Out21N Out31N)/2;
OutMax(133)=(OutMax21N OutMax31N)/2;
end;

if abs(Out21N-Out41N)==2;
Out(134)=(Out21N Out41N)/2;
OutMax(134)=(OutMax21N OutMax41N)/2;
end;

if abs(Out31N-Out41N)==2;
Out(135)=(Out31N Out41N)/2;
OutMax(135)=(OutMax31N OutMax41N)/2;
end;

if abs(Out11N-Out21N)==4;
Out(28)=(Out11N Out21N)/2;
OutMax(28)=(OutMax11N OutMax21N)/2;
end;

if abs(Out11N-Out31N)==4;
Out(29)=(Out11N Out31N)/2;
OutMax(29)=(OutMax11N OutMax31N)/2;
end;

if abs(Out11N-Out41N)==4;
Out(30)=(Out11N Out41N)/2;
OutMax(30)=(OutMax11N OutMax41N)/2;
end;

if abs(Out21N-Out31N)==4;
Out(31)=(Out21N Out31N)/2;
OutMax(31)=(OutMax21N OutMax31N)/2;
end;

if abs(Out21N-Out41N)==4;
Out(32)=(Out21N Out41N)/2;
OutMax(32)=(OutMax21N OutMax41N)/2;
end;

if abs(Out31N-Out41N)==4;
Out(33)=(Out31N Out41N)/2;
OutMax(33)=(OutMax31N OutMax41N)/2;
end;

if abs(Out11N-Out21N)==6;
Out(34)=(Out11N Out21N)/2;
OutMax(34)=(OutMax11N OutMax21N)/2;
end;

if abs(Out11N-Out31N)==6;
Out(35)=(Out11N Out31N)/2;
OutMax(35)=(OutMax11N OutMax31N)/2;
end;

if abs(Out11N-Out41N)==6;
Out(36)=(Out11N Out41N)/2;
OutMax(36)=(OutMax11N OutMax41N)/2;
end;

if abs(Out21N-Out31N)==6;
Out(37)=(Out21N Out31N)/2;
OutMax(37)=(OutMax21N OutMax31N)/2;
end;

if abs(Out21N-Out41N)==6;
Out(38)=(Out21N Out41N)/2;
OutMax(38)=(OutMax21N OutMax41N)/2;
end;

if abs(Out31N-Out41N)==6;
Out(39)=(Out31N Out41N)/2;
OutMax(39)=(OutMax31N OutMax41N)/2;
end;

%отрицательный инверсный код первой половины
if abs(Out11NI-Out21NI)==2;
Out(40)=(Out11NI Out21NI)/2;
OutMax(40)=(OutMax11NI OutMax21NI)/2;
end;

if abs(Out11NI-Out31NI)==2;
Out(136)=(Out11NI Out31NI)/2;
OutMax(136)=(OutMax11NI OutMax31NI)/2;
end;

if abs(Out11NI-Out41NI)==2;
Out(137)=(Out11NI Out41NI)/2;
OutMax(137)=(OutMax11NI OutMax41NI)/2;
end;

if abs(Out21NI-Out31NI)==2;
Out(138)=(Out21NI Out31NI)/2;
OutMax(138)=(OutMax21NI OutMax31NI)/2;
end;

if abs(Out21NI-Out41NI)==2;
Out(139)=(Out21NI Out41NI)/2;
OutMax(139)=(OutMax21NI OutMax41NI)/2;
end;

if abs(Out31NI-Out41NI)==2;
Out(140)=(Out31NI Out41NI)/2;
OutMax(140)=(OutMax31NI OutMax41NI)/2;
end;

if abs(Out11NI-Out21NI)==4;
Out(41)=(Out11NI Out21NI)/2;
OutMax(41)=(OutMax11NI OutMax21NI)/2;
end;

if abs(Out11NI-Out31NI)==4;
Out(42)=(Out11NI Out31NI)/2;
OutMax(42)=(OutMax11NI OutMax31NI)/2;
end;

if abs(Out11NI-Out41NI)==4;
Out(43)=(Out11NI Out41NI)/2;
OutMax(43)=(OutMax11NI OutMax41NI)/2;
end;

if abs(Out21NI-Out31NI)==4;
Out(44)=(Out21NI Out31NI)/2;
OutMax(44)=(OutMax21NI OutMax31NI)/2;
end;

if abs(Out21NI-Out41NI)==4;
Out(45)=(Out21NI Out41NI)/2;
OutMax(45)=(OutMax21NI OutMax41NI)/2;
end;

if abs(Out31NI-Out41NI)==4;
Out(46)=(Out31NI Out41NI)/2;
OutMax(46)=(OutMax31NI OutMax41NI)/2;
end;

if abs(Out11NI-Out21NI)==6;
Out(47)=(Out11NI Out21NI)/2;
OutMax(47)=(OutMax11NI OutMax21NI)/2;
end;

if abs(Out11NI-Out31NI)==6;
Out(48)=(Out11NI Out31NI)/2;
OutMax(48)=(OutMax11NI OutMax31NI)/2;
end;

if abs(Out11NI-Out41NI)==6;
Out(49)=(Out11NI Out41NI)/2;
OutMax(49)=(OutMax11NI OutMax41NI)/2;
end;

if abs(Out21NI-Out31NI)==6;
Out(50)=(Out21NI Out31NI)/2;
OutMax(50)=(OutMax21NI OutMax31NI)/2;
end;

if abs(Out21NI-Out41NI)==6;
Out(51)=(Out21NI Out41NI)/2;
OutMax(51)=(OutMax21NI OutMax41NI)/2;
end;

if abs(Out31NI-Out41NI)==6;
Out(52)=(Out31NI Out41NI)/2;
OutMax(52)=(OutMax31NI OutMax41NI)/2;
end;

%положительный прямой код второй половины

if abs(Out12P-Out22P)==2;
Out(53)=(Out12P Out22P)/2;
OutMax(53)=(OutMax12P OutMax22P)/2;
end;

if abs(Out12P-Out32P)==2;
Out(141)=(Out12P Out32P)/2;
OutMax(141)=(OutMax12P OutMax32P)/2;
end;

if abs(Out12P-Out42P)==2;
Out(142)=(Out12P Out42P)/2;
OutMax(142)=(OutMax12P OutMax42P)/2;
end;

if abs(Out22P-Out32P)==2;
Out(143)=(Out22P Out32P)/2;
OutMax(143)=(OutMax22P OutMax32P)/2;
end;

if abs(Out22P-Out42P)==2;
Out(144)=(Out22P Out42P)/2;
OutMax(144)=(OutMax22P OutMax42P)/2;
end;

if abs(Out32P-Out42P)==2;
Out(145)=(Out32P Out42P)/2;
OutMax(145)=(OutMax32P OutMax42P)/2;
end;

if abs(Out12P-Out22P)==4;
Out(54)=(Out12P Out22P)/2;
OutMax(54)=(OutMax12P OutMax22P)/2;
end;

if abs(Out12P-Out32P)==4;
Out(55)=(Out12P Out32P)/2;
OutMax(55)=(OutMax12P OutMax32P)/2;
end;

if abs(Out12P-Out42P)==4;
Out(56)=(Out12P Out42P)/2;
OutMax(56)=(OutMax12P OutMax42P)/2;
end;

if abs(Out22P-Out32P)==4;
Out(57)=(Out22P Out32P)/2;
OutMax(57)=(OutMax22P OutMax32P)/2;
end;

if abs(Out22P-Out42P)==4;
Out(58)=(Out22P Out42P)/2;
OutMax(58)=(OutMax22P OutMax42P)/2;
end;

if abs(Out32P-Out42P)==4;
Out(59)=(Out32P Out42P)/2;
OutMax(59)=(OutMax32P OutMax42P)/2;
end;

if abs(Out12P-Out22P)==6;
Out(60)=(Out12P Out22P)/2;
OutMax(60)=(OutMax12P OutMax22P)/2;
end;

if abs(Out12P-Out32P)==6;
Out(61)=(Out12P Out32P)/2;
OutMax(61)=(OutMax12P OutMax22P)/2;
end;

if abs(Out12P-Out42P)==6;
Out(62)=(Out12P Out42P)/2;
OutMax(62)=(OutMax12P OutMax42P)/2;
end;

if abs(Out22P-Out32P)==6;
Out(63)=(Out22P Out32P)/2;
OutMax(63)=(OutMax22P OutMax32P)/2;
end;

if abs(Out22P-Out42P)==6;
Out(64)=(Out22P Out42P)/2;
OutMax(64)=(OutMax22P OutMax42P)/2;
end;

if abs(Out32P-Out42P)==6;
Out(65)=(Out32P Out42P)/2;
OutMax(65)=(OutMax32P OutMax42P)/2;
end;

%положительный инверсный код второй половины
if abs(Out12PI-Out22PI)==2;
Out(66)=(Out12PI Out22PI)/2;
OutMax(66)=(OutMax12PI OutMax22PI)/2;
end;

if abs(Out12PI-Out32PI)==2;
Out(146)=(Out12PI Out32PI)/2;
OutMax(146)=(OutMax12PI OutMax32PI)/2;
end;

if abs(Out12PI-Out42PI)==2;
Out(147)=(Out12PI Out42PI)/2;
OutMax(147)=(OutMax12PI OutMax42PI)/2;
end;

if abs(Out22PI-Out32PI)==2;
Out(148)=(Out22PI Out32PI)/2;
OutMax(148)=(OutMax22PI OutMax32PI)/2;
end;

if abs(Out22PI-Out42PI)==2;
Out(149)=(Out22PI Out42PI)/2;
OutMax(149)=(OutMax22PI OutMax42PI)/2;
end;

if abs(Out32PI-Out42PI)==2;
Out(150)=(Out32PI Out42PI)/2;
OutMax(150)=(OutMax32PI OutMax42PI)/2;
end;

if abs(Out12PI-Out22PI)==4;
Out(67)=(Out12PI Out22PI)/2;
OutMax(67)=(OutMax12PI OutMax22PI)/2;
end;

if abs(Out12PI-Out32PI)==4;
Out(68)=(Out12PI Out32PI)/2;
OutMax(68)=(OutMax12PI OutMax32PI)/2;
end;

if abs(Out12PI-Out42PI)==4;
Out(69)=(Out12PI Out42PI)/2;
OutMax(69)=(OutMax12PI OutMax42PI)/2;
end;

if abs(Out22PI-Out32PI)==4;
Out(70)=(Out22PI Out32PI)/2;
OutMax(70)=(OutMax22PI OutMax32PI)/2;
end;

if abs(Out22PI-Out42PI)==4;
Out(71)=(Out22PI Out42PI)/2;
OutMax(71)=(OutMax22PI OutMax42PI)/2;
end;

if abs(Out32PI-Out42PI)==4;
Out(72)=(Out32PI Out42PI)/2;
OutMax(72)=(OutMax32PI OutMax42PI)/2;
end;

if abs(Out12PI-Out22PI)==6;
Out(73)=(Out12PI Out22PI)/2;
OutMax(73)=(OutMax12PI OutMax22PI)/2;
end;

if abs(Out12PI-Out32PI)==6;
Out(74)=(Out12PI Out32PI)/2;
OutMax(74)=(OutMax12PI OutMax32PI)/2;
end;

if abs(Out12PI-Out42PI)==6;
Out(75)=(Out12PI Out42PI)/2;
OutMax(75)=(OutMax12PI OutMax42PI)/2;
end;

if abs(Out22PI-Out32PI)==6;
Out(76)=(Out22PI Out32PI)/2;
OutMax(76)=(OutMax22PI OutMax32PI)/2;
end;

if abs(Out22PI-Out42PI)==6;
Out(77)=(Out22PI Out42PI)/2;
OutMax(77)=(OutMax22PI OutMax42PI)/2;
end;

if abs(Out32PI-Out42PI)==6;
Out(78)=(Out32PI Out42PI)/2;
OutMax(78)=(OutMax32PI OutMax42PI)/2;
end;

%отрицательный прямой код второй половины
if abs(Out12N-Out22N)==2;
Out(79)=(Out12N Out22N)/2;
OutMax(79)=(OutMax12N OutMax22N)/2;
end;

if abs(Out12N-Out32N)==2;
Out(151)=(Out12N Out32N)/2;
OutMax(151)=(OutMax12N OutMax32N)/2;
end;

if abs(Out12N-Out42N)==2;
Out(152)=(Out12N Out42N)/2;
OutMax(152)=(OutMax12N OutMax42N)/2;
end;

if abs(Out22N-Out32N)==2;
Out(153)=(Out22N Out32N)/2;
OutMax(153)=(OutMax22N OutMax32N)/2;
end;

if abs(Out22N-Out42N)==2;
Out(154)=(Out22N Out42N)/2;
OutMax(154)=(OutMax22N OutMax42N)/2;
end;

if abs(Out32N-Out42N)==2;
Out(155)=(Out32N Out42N)/2;
OutMax(155)=(OutMax32N OutMax42N)/2;
end;


if abs(Out12N-Out22N)==4;
Out(80)=(Out12N Out22N)/2;
OutMax(80)=(OutMax12N OutMax22N)/2;
end;

if abs(Out12N-Out32N)==4;
Out(81)=(Out12N Out32N)/2;
OutMax(81)=(OutMax12N OutMax32N)/2;
end;

if abs(Out12N-Out42N)==4;
Out(82)=(Out12N Out42N)/2;
OutMax(82)=(OutMax12N OutMax42N)/2;
end;

if abs(Out22N-Out32N)==4;
Out(83)=(Out22N Out32N)/2;
OutMax(83)=(OutMax22N OutMax32N)/2;
end;

if abs(Out22N-Out42N)==4;
Out(84)=(Out22N Out42N)/2;
OutMax(84)=(OutMax22N OutMax42N)/2;
end;

if abs(Out32N-Out42N)==4;
Out(85)=(Out32N Out42N)/2;
OutMax(85)=(OutMax32N OutMax42N)/2;
end;

if abs(Out12N-Out22N)==6;
Out(86)=(Out12N Out22N)/2;
OutMax(86)=(OutMax12N OutMax22N)/2;
end;

if abs(Out12N-Out32N)==6;
Out(87)=(Out12N Out32N)/2;
OutMax(87)=(OutMax12N OutMax32N)/2;
end;

if abs(Out12N-Out42N)==6;
Out(88)=(Out12N Out42N)/2;
OutMax(88)=(OutMax12N OutMax42N)/2;
end;

if abs(Out22N-Out32N)==6;
Out(89)=(Out22N Out32N)/2;
OutMax(89)=(OutMax22N OutMax32N)/2;
end;

if abs(Out22N-Out42N)==6;
Out(90)=(Out22N Out42N)/2;
OutMax(90)=(OutMax22N OutMax42N)/2;
end;

if abs(Out32N-Out42N)==6;
Out(91)=(Out32N Out42N)/2;
OutMax(91)=(OutMax32N OutMax42N)/2;
end;

%отрицательный инверсный код второй половины
if abs(Out12NI-Out22NI)==2;
Out(92)=(Out12NI Out22NI)/2;
OutMax(92)=(OutMax12NI OutMax22NI)/2;
end;

if abs(Out12NI-Out32NI)==2;
Out(156)=(Out12NI Out32NI)/2;
OutMax(156)=(OutMax12NI OutMax32NI)/2;
end;

if abs(Out12NI-Out42NI)==2;
Out(157)=(Out12NI Out42NI)/2;
OutMax(157)=(OutMax12NI OutMax42NI)/2;
end;

if abs(Out22NI-Out32NI)==2;
Out(158)=(Out22NI Out32NI)/2;
OutMax(158)=(OutMax22NI OutMax32NI)/2;
end;

if abs(Out22NI-Out42NI)==2;
Out(159)=(Out22NI Out42NI)/2;
OutMax(159)=(OutMax22NI OutMax42NI)/2;
end;

if abs(Out32NI-Out42NI)==2;
Out(160)=(Out32NI Out42NI)/2;
OutMax(160)=(OutMax32NI OutMax42NI)/2;
end;

if abs(Out12NI-Out22NI)==4;
Out(93)=(Out12NI Out22NI)/2;
OutMax(93)=(OutMax12NI OutMax22NI)/2;
end;

if abs(Out12NI-Out32NI)==4;
Out(94)=(Out12NI Out32NI)/2;
OutMax(94)=(OutMax12NI OutMax32NI)/2;
end;

if abs(Out12NI-Out42NI)==4;
Out(95)=(Out12NI Out42NI)/2;
OutMax(95)=(OutMax12NI OutMax42NI)/2;
end;

if abs(Out22NI-Out32NI)==4;
Out(96)=(Out22NI Out32NI)/2;
OutMax(96)=(OutMax22NI OutMax32NI)/2;
end;

if abs(Out22NI-Out42NI)==4;
Out(97)=(Out22NI Out42NI)/2;
OutMax(97)=(OutMax22NI OutMax42NI)/2;
end;

if abs(Out32NI-Out42NI)==4;
Out(98)=(Out32NI Out42NI)/2;
OutMax(98)=(OutMax32NI OutMax42NI)/2;
end;

if abs(Out12NI-Out22NI)==6;
Out(99)=(Out12NI Out22NI)/2;
OutMax(99)=(OutMax12NI OutMax22NI)/2;
end;

if abs(Out12NI-Out32NI)==6;
Out(100)=(Out12NI Out32NI)/2;
OutMax(100)=(OutMax12NI OutMax32NI)/2;
end;

if abs(Out12NI-Out42NI)==6;
Out(101)=(Out12NI Out42NI)/2;
OutMax(101)=(OutMax12NI OutMax42NI)/2;
end;

if abs(Out22NI-Out32NI)==6;
Out(102)=(Out22NI Out32NI)/2;
OutMax(102)=(OutMax22NI OutMax32NI)/2;
end;

if abs(Out22NI-Out42NI)==6;
Out(103)=(Out22NI Out42NI)/2;
OutMax(103)=(OutMax22NI OutMax42NI)/2;
end;

if abs(Out32NI-Out42NI)==6;
Out(104)=(Out32NI Out42NI)/2;
OutMax(104)=(OutMax32NI OutMax42NI)/2;
end;

% поиск значения массива Out, которое встречается чаще
OutGG1=0;
OutGG2=0;
for i=1:302;
if Out(i)~=0;
OutGD(Out(i))=OutGD(Out(i)) 1;
end;
end;
%OutGD
% значения массива OutGD определяют, сколько раз встречается каждое
% значение из массива максимумов Out

OutG=0;
for i=1:N;
if OutGD(i)>OutG;
OutG=OutGD(i);
OutGG1=i;
end;
end;
% OutGG1 - выходное значение, встречающееся чаще

OutGG1