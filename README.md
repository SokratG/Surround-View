## The 3D surround view on Jetson Tegra TX2 platform - this is my research project for master thesis.
The essence of the project is to build a good quality stitching of images from 4 cameras and project the image onto a mesh model with acceptable performance. The project work on ~20 fps with constraints from the connection adaptor 30 fps due the synchronize frames mode.

### Cite(related paper):

```
@incollection{gasparyan2022real,
  title={Real-Time 3D Surround View System for Vehicle Based on Panoramic Stitching Image},
  author={Gasparyan, Sokrat and Vasilianov, Georgii},
  booktitle={Networked Control Systems for Connected and Automated Vehicles: Volume 1},
  pages={85--93},
  year={2022},
  publisher={Springer}
}
```

### The 3D bowl view:
<img src="gitresource/svdemo3d.gif">

### The top view:
<img src="gitresource/demotopview.gif" width="65%" height="auto">

### The hardware setup:
| Device type |  Device name |
|-------------|--------------|
| Camera | e-CAM30A CUMI0330 MOD |
| Connection adaptor board | e-CAM130 TRICUTX2 ADAPTOR |
| Connecting wires | High-speed 2-lane MIPI CSI-2 |
| Embedded platform | Jetson Tegra TX2 |
| Tripod | - |

<img src="gitresource/camerasetup.jpg" width="58%" height="auto">

### The software setup:
* OS - Linux Ubuntu LTS v16.04
* C++ 14, Python 3.6, GLSL, C CUDA
* CMake >= v3.16
* CUDA Toolkit  v9.0
* V4L2 driver
* OpenGL ES v3.2
* OpenMP

### The data processing flowchart 
<img src="gitresource/svsoftwareflow.jpg" width="125%" height="auto">

### The 3d party library:
* [OpenCV v4.1.2](https://github.com/opencv/opencv)
* [Mesa 3D (EGL, GLES)](https://docs.mesa3d.org/download.html)
* [GLM](https://github.com/g-truc/glm)
* [GLFW](https://www.glfw.org)
* [ASSIMP v4.0.1](https://www.assimp.org/index.php/downloads)
* [meshgrid](https://github.com/xiaohongchen1991/meshgen)
* [Common CUDA blender](https://github.com/Avandrea/OpenCV-BlenderGPU)

### [3D Car model](https://www.cgtrader.com/free-3d-models/car/sport/low-poly-dodge-challenger-srt-hellcat-2015)
