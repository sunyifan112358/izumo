# Izumo
A CUDA wrapper for Go language

## Overview
The goal of Izumo is to allow programmers to run CUDA code with Go directly. Traditionally, to invoke a CUDA kernel from Go needs the developer to write specialized wrappers in C and to connect Go with the wrappers with CGO. In general, a wrapper is required for one algorithm or one kernel. However, manipulating the memory and pointers with CGO is not an easy task, and writing excessive wrappers leads to redundant code and can be error prone. Therefore, we decided to do the heavy-lifting for you.

Izumo provides a runtime API in Go that allows you to invoke CUDA kernels without writing any C code. In order to smooth the learning curve, we try to follow the fashion of both Go programming language and the official CUDA runtime API library. Therefore, if you are an experienced CUDA programmer, you should be able to start writing your own program after reading a few examples. 
