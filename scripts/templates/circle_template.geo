
rad = $rad;
h = $meshsize;

Point(0) = { 0, 0, 0, h};
Point(1) = { rad, 0, 0, h};
Point(2) = { 0, rad, 0, h};
Point(3) = { -rad, 0, 0, h};
Point(4) = { 0, -rad, 0, h};

Circle(1) = {1, 0, 2};
Circle(2) = {2, 0, 3};
Circle(3) = {3, 0, 4};
Circle(4) = {4, 0, 1};

Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Physical Surface(1) = {1};
