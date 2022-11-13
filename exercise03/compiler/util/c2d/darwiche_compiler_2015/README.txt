The original repository you find here:

https://github.com/nesl/automated-reasoning/tree/master/c2D_code
or
http://reasoning.cs.ucla.edu/minic2d/

This Version is adapted for compilation with cmake.

----------------------------------------------------------------------------------

compilation instructions:
cd src
mkdir -p bin
cd bin
cmake -DCMAKE_BUILD_TYPE=Release ..
make
ls

--> miniC2D is the created binary

In order for working with miniC2D, you need also the binary hgr2htree,
which is provided by the authors of the miniC2D compiler.
(you'll find the binary in the binary directory)

Doesn't work on Windows! The Authors of miniC2D provided only "Unix"
compatible precompiled binaries and libraries for compilation.

----------------------------------------------------------------------------------

type "./miniC2D --help" for usage information.

Example usage:
./miniC2D -c graphical_model.cnf

If the program doesn't run you probably need 32-bit libraries for 
hgr2htree

sudo apt-get install libc6-i386
sudo apt-get install lib32stdc++6

see here for detailed information:
https://github.com/nesl/automated-reasoning/tree/master/c2D_code


