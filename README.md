### 环境准备

- OpenCV C++版:

参考网上教程, 非常多!

- LibTorch:

```sh
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
```

### 编译

```sh
mkdir build && cd build
cmake ..
make
```

### 执行

```
# in build/
./example-app
```
