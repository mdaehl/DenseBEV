from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if __name__ == "__main__":
    setup(
        name="iou3d_nms",
        version="0.1",
        ext_modules=[
            CUDAExtension(
                name="iou3d_nms",
                sources=[
                    "src/iou3d_nms.cpp",
                    "src/iou3d_nms_kernel.cu",
                    "src/iou3d_nms_api.cpp",
                ],
            ),
        ],
        cmdclass={"build_ext": BuildExtension},
    )
