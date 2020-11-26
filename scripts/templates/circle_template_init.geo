
rad = $rad;
h = $h;
eta = 3*$h;
theta = 2*Pi/3;

Point(0) = { 0, 0, 0, h};
Point(1) = { rad, 0, 0, h};
Point(2) = { 0, rad, 0, h};
Point(3) = { -rad, 0, 0, h};
Point(4) = { 0, -rad, 0, h};

Point(10) = { eta, 0, 0, h};
Point(11) = { eta*Cos(theta), eta*Sin(theta), 0, h};
Point(12) = { eta*Cos(2*theta), eta*Sin(2*theta), 0, h};

Circle(1) = {1, 0, 2};
Circle(2) = {2, 0, 3};
Circle(3) = {3, 0, 4};
Circle(4) = {4, 0, 1};

Line(100) = {0, 10};
Line(200) = {0, 11};
Line(300) = {0, 12};

Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Line{100} In Surface {1};
Line{200} In Surface {1};
Line{300} In Surface {1};

Physical Surface(1) = {1};
Physical Line ("init") = {100, 200, 300};
