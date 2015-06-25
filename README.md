Features
======
1. Forked from https://www.github.com/BVLC/caffe master branch in 2015/6/20.
2. Build on CPU_ONLY mode without using CUDA.
3. Export **libcaffe.lib** and **libcaffe.dll** files for further development.

Requirements
======
1. Windows X64.
2. Visual Studio 2012.
3. Pre-compiled 3rdparty dependencies can download from [here](http://pan.baidu.com/s/1sjQIb7J).
 
Build
======
1. Download pre-compiled dependencies from [here](http://pan.baidu.com/s/1sjQIb7J). 
2. Put **3rdparty/include** and **3rdparty/lib** folders into **caffe-windows-lib/3rdparty**.
3. Put **3rdparty/bin** folder into **caffe-windows-lib/build**.
4. Open vsproject/vsproject.sln. 
5. Compile libcaffe.
6. Compile other caffe tools (optional).
7. Compile your own code using libcaffe.dll (remember to add **CPU_ONLY** macro in preprocessor).
   
MNIST example
======
1. Download the mnist lmdb database from [here](http://pan.baidu.com/s/1dDliZDJ).
2. Put **mnist/mnist_test_lmdb** and **mnist/mnist_test_lmdb** folders into **caffe-windows-lib/examples/mnist**.
3. Run examples\mnist\train_lenet.bat.
