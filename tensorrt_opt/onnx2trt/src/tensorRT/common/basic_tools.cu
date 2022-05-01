#include "basic_tools.hpp"
#include "cuda_tools.hpp"
#include <unistd.h>
#include <stdio.h>

using namespace std;
using namespace cv;
bool BaiscTools::exists(const string& path){

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

vector<string> BaiscTools::load_labels(const char* file){
    vector<string> lines;

    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open()){
        printf("open %d failed.\n", file);
        return lines;
    }
    
    string line;
    while(getline(in, line)){
        lines.push_back(line);
    }
    in.close();
    return lines;
}


void BaiscTools::AffineMatrix::invertAffineTransform(float imat[6], float omat[6]){
        float i00 = imat[0];  float i01 = imat[1];  float i02 = imat[2];
        float i10 = imat[3];  float i11 = imat[4];  float i12 = imat[5];

        float D = i00 * i11 - i01 * i10;
        D = D != 0 ? 1.0 / D : 0;
        
        printf("DDD:%f",D);

        float A11 = i11 * D;
        float A22 = i00 * D;
        float A12 = -i01 * D;
        float A21 = -i10 * D;
        float b1 = -A11 * i02 - A12 * i12;
        float b2 = -A21 * i02 - A22 * i12;
        omat[0] = A11;  omat[1] = A12;  omat[2] = b1;
        omat[3] = A21;  omat[4] = A22;  omat[5] = b2;
};

void  BaiscTools::AffineMatrix::compute(const MySize& from, const MySize& to){
    float scale_x = to.width / (float)from.width;
    float scale_y = to.height / (float)from.height;
    float scale = min(scale_x, scale_y); 
    i2d[0] = scale;  
    i2d[1] = 0;  
    i2d[2] = -scale * from.width  * 0.5  + to.width * 0.5 + scale * 0.5 - 0.5;
    i2d[3] = 0;  
    i2d[4] = scale;  
    i2d[5] = -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;
    invertAffineTransform(i2d, d2i);
};

__device__ void BaiscTools::affine_project(float* matrix, int x, int y, float* proj_x, float* proj_y){
    
    *proj_x = matrix[0] * x + matrix[1] * y + matrix[2];
    *proj_y = matrix[3] * x + matrix[4] * y + matrix[5];
}

__global__  void BaiscTools::warp_affine_bilinear_kernel(
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    uint8_t* dst, int dst_line_size, int dst_width, int dst_height, 
	uint8_t fill_value, AffineMatrix matrix
){
    int dx = blockDim.x * blockIdx.x + threadIdx.x; 
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= dst_width || dy >= dst_height)  return;
    float c0 = fill_value, c1 = fill_value, c2 = fill_value;
    float src_x = 0; float src_y = 0;
    affine_project(matrix.d2i, dx, dy, &src_x, &src_y);

    if(src_x < -1 || src_x >= src_width || src_y < -1 || src_y >= src_height){
    }else{

        // p1  p2
        //   p
        // p3  p4
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_values[] = {fill_value, fill_value, fill_value};
        
        //双线性差值，和python版一致
        float ly    = src_y - y_low;
        float lx    = src_x - x_low;
        float hy    = 1 - ly;
        float hx    = 1 - lx;
        float w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        uint8_t* v1 = const_values;
        uint8_t* v2 = const_values;
        uint8_t* v3 = const_values;
        uint8_t* v4 = const_values;
        if(y_low >= 0){
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3;

            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3;
        }
        
        if(y_high < src_height){
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;

            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }
        
        // 该点的像素值
        c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
        c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
        c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
    }

    uint8_t* pdst = dst + dy * dst_line_size + dx * 3;
    pdst[0] = c0; pdst[1] = c1; pdst[2] = c2;

    //BGR -> RGB
    // pdst[2] = c0; pdst[1] = c1; pdst[2] = c0;

    // (p - mean) / std 
    // pdst[0] = (c0 - mean) / std;  pdst[1] = (c1 - mean) / std; pdst[2] = (c2 - mean) / std;

}

void BaiscTools::warp_affine_bilinear(
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    uint8_t* dst, int dst_line_size, int dst_width, int dst_height, 
	uint8_t fill_value
){
    dim3 block_size(32, 32); 
    dim3 grid_size((dst_width + 31) / 32, (dst_height + 31) / 32);
    AffineMatrix affine;
    // affine 
    affine.compute(MySize(src_width, src_height), MySize(dst_width, dst_height));
    warp_affine_bilinear_kernel<<<grid_size, block_size, 0, nullptr>>>(
        src, src_line_size, src_width, src_height,
        dst, dst_line_size, dst_width, dst_height,
        fill_value, affine
    );
}

Mat BaiscTools::warpaffine_to_center_align(const Mat& image, const Size& size){  

    Mat output_image(size, CV_8UC3);
    uint8_t* psrc_device = nullptr;
    uint8_t* pdst_device = nullptr;
    size_t src_size = image.cols * image.rows * 3;
    size_t dst_size = size.width * size.height * 3;

    checkCudaRuntime(cudaMalloc(&psrc_device, src_size)); 
    checkCudaRuntime(cudaMalloc(&pdst_device, dst_size));
    checkCudaRuntime(cudaMemcpy(psrc_device, image.data, src_size, cudaMemcpyHostToDevice));
    
    // 在cuda上执行warpaffine
    warp_affine_bilinear(
        psrc_device, image.cols * 3, image.cols, image.rows,
        pdst_device, size.width * 3, size.width, size.height,
        114
    );
    checkCudaRuntime(cudaPeekAtLastError());
    checkCudaRuntime(cudaMemcpy(output_image.data, pdst_device, dst_size, cudaMemcpyDeviceToHost));
    checkCudaRuntime(cudaFree(psrc_device));
    checkCudaRuntime(cudaFree(pdst_device));
    return output_image;
}

