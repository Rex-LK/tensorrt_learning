#include <common/basic_tools.hpp>
#include <common/cuda-tools.hpp>
#include <unistd.h>
#include <stdio.h>
#include <sys/time.h>
#include <algorithm>
using namespace std;
using namespace cv;
bool BaiscTools::exists(const string& path){

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

bool BaiscTools::build_model(string model,int maxbatch){
    if(exists(model + ".trtmodel")){
        printf("%s.trtmodel has exists.\n",model.c_str());
        return true;
    }

    TRT::compile(
        TRT::Mode::FP32,
        maxbatch,
        model + ".onnx",
        model+ ".trtmodel",
        1 << 28
    );
    INFO("Done.");
    return true;
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


BaiscTools::affined_img_matrix BaiscTools::warpaffine_cpu(Mat &ori_img,int dst_height,int dst_width){
    int ori_hegiht = ori_img.rows;
    int ori_weight = ori_img.cols;
    Mat dst_image(dst_height, dst_width, CV_8UC3);
    float scale_x = dst_width / (float)ori_weight;
    float scale_y = dst_height / (float)ori_hegiht;
    float scale = min(scale_x, scale_y);
    float i2d[6], d2i[6];
    i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * ori_weight + dst_width + scale - 1) * 0.5;
    i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * ori_hegiht + dst_height + scale - 1) * 0.5;
    Mat m2x3_i2d(2, 3, CV_32F, i2d);
    Mat m2x3_d2i(2, 3, CV_32F, d2i);
    invertAffineTransform(m2x3_i2d, m2x3_d2i);
    warpAffine(ori_img, dst_image, m2x3_i2d, dst_image.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar::all(114));
    
    // (image -mean) / std  && BGR->RGB
    // int image_area = input_image.cols * input_image.rows;
    // unsigned char* pimage = input_image.data;
    // float* phost_b = input_data_host + image_area * 0;
    // float* phost_g = input_data_host + image_area * 1;
    // float* phost_r = input_data_host + image_area * 2;
    // for(int i = 0; i < image_area; ++i, pimage += 3){
    //     *phost_r++ = pimage[0] / 255.0f;
    //     *phost_g++ = pimage[1] / 255.0f;
    //     *phost_b++ = pimage[2] / 255.0f;
    // }
    // ///////////////////////////////////////////////////
    // checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));

    affined_img_matrix job;
    m2x3_i2d.copyTo(job.m2x3_i2d);
    m2x3_d2i.copyTo(job.m2x3_d2i);
    dst_image.copyTo(job.dst_image);
    return job;
}


//gpu warpafine

void BaiscTools::AffineMatrix::invertAffineTransform(float imat[6], float omat[6]){
        float i00 = imat[0];  float i01 = imat[1];  float i02 = imat[2];
        float i10 = imat[3];  float i11 = imat[4];  float i12 = imat[5];

        float D = i00 * i11 - i01 * i10;
        D = D != 0 ? 1.0 / D : 0;
        

        float A11 = i11 * D;
        float A22 = i00 * D;
        float A12 = -i01 * D;
        float A21 = -i10 * D;
        float b1 = -A11 * i02 - A12 * i12;
        float b2 = -A21 * i02 - A22 * i12;
        omat[0] = A11;  omat[1] = A12;  omat[2] = b1;
        omat[3] = A21;  omat[4] = A22;  omat[5] = b2;
}
void   BaiscTools::AffineMatrix::compute(const MySize& from, const MySize& to){
    float scale_x = to.width / (float)from.width;
    float scale_y = to.height / (float)from.height;

    float scale = min(scale_x, scale_y); 
    
    //正变换M矩阵

    // M = [ scale,    0,     -scale * from.width * 0.5 + to.width * 0.5
    //     0,     scale,    -scale * from.height * 0.5 + to.height * 0.5
    //     0, 0, 1]
    i2d[0] = scale;  
    i2d[1] = 0;  
    i2d[2] = -scale * from.width  * 0.5  + to.width * 0.5 + scale * 0.5 - 0.5;
    i2d[3] = 0;  
    i2d[4] = scale;  
    i2d[5] = -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;
    

    //计算M矩阵的逆变换
    invertAffineTransform(i2d, d2i);
}

__device__ void  BaiscTools::affine_project(float* matrix, int x, int y, float* proj_x, float* proj_y){
    
    *proj_x = matrix[0] * x + matrix[1] * y + matrix[2];
    *proj_y = matrix[3] * x + matrix[4] * y + matrix[5];
}

__global__  void  BaiscTools::warp_affine_bilinear_kernel(
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    uint8_t* dst, int dst_line_size, int dst_width, int dst_height, 
	uint8_t fill_value, AffineMatrix matrix
){
    // 线程ID的全局索引
    int dx = blockDim.x * blockIdx.x + threadIdx.x; 
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    // 线程ID 超过图像大小时 return
    if (dx >= dst_width || dy >= dst_height)  return;
    // 目标图像为640*640*3 用fill_value填充
    float c0 = fill_value, c1 = fill_value, c2 = fill_value;
    float src_x = 0; float src_y = 0;
    //将目标图上一点映射回原图大小
    affine_project(matrix.d2i, dx, dy, &src_x, &src_y);
    if(src_x < -1 || src_x >= src_width || src_y < -1 || src_y >= src_height){
        // out of range
        // src_x < -1，high_x < 0，超出范围
        // src_x >= -1，high_x >= 0，存在取值
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

void  BaiscTools::warp_affine_bilinear(
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

cv::Mat  BaiscTools::warpaffine_gpu(const cv::Mat& ori_image, const int dst_height,const int dst_width){
    int ori_height = ori_image.rows;
    int ori_weight = ori_image.cols;

    cv::Mat output_image(dst_height, dst_width, CV_8UC3);
    uint8_t* psrc_device = nullptr;
    uint8_t* pdst_device = nullptr;
    
    size_t src_size = ori_weight * ori_height * 3;
    size_t dst_size = dst_width * dst_height * 3;

    checkRuntime(cudaMalloc(&psrc_device, src_size)); 
    checkRuntime(cudaMalloc(&pdst_device, dst_size));
    checkRuntime(cudaMemcpy(psrc_device, ori_image.data, src_size, cudaMemcpyHostToDevice));
    
    // 在cuda上执行warpaffine
    warp_affine_bilinear(
        psrc_device, ori_weight * 3, ori_weight, ori_height,
        pdst_device, dst_width * 3, dst_width, dst_height,
        114
    );
    checkRuntime(cudaPeekAtLastError());
    //如果需要返回GPU上的指针则只需要在这里返回 pdst_device
    checkRuntime(cudaMemcpy(output_image.data, pdst_device, dst_size, cudaMemcpyDeviceToHost));
    checkRuntime(cudaFree(psrc_device));
    checkRuntime(cudaFree(pdst_device));
    return output_image;
};
//gpu warpafine



static vector<int> _classes_colors = {
    0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 
    128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 
    64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 12
};


tuple<cv::Mat, cv::Mat> BaiscTools::unet_post_process(float* output, int output_width, int output_height, int num_class, int ibatch){

    cv::Mat output_prob(output_height, output_width, CV_32F);
    cv::Mat output_index(output_height, output_width, CV_8U);

    float* pnet   = output + ibatch * output_width * output_height * num_class;
    float* prob   = output_prob.ptr<float>(0);
    uint8_t* pidx = output_index.ptr<uint8_t>(0);

    for(int k = 0; k < output_prob.cols * output_prob.rows; ++k, pnet+=num_class, ++prob, ++pidx){
        int ic = max_element(pnet, pnet + num_class) - pnet;
        *prob  = pnet[ic];
        *pidx  = ic;
    }
    return make_tuple(output_prob, output_index);
};

void BaiscTools::render(cv::Mat& image, const cv::Mat& prob, const cv::Mat& iclass){

    auto pimage = image.ptr<cv::Vec3b>(0);
    auto pprob  = prob.ptr<float>(0);
    auto pclass = iclass.ptr<uint8_t>(0);

    for(int i = 0; i < image.cols*image.rows; ++i, ++pimage, ++pprob, ++pclass){

        int iclass        = *pclass;
        float probability = *pprob;
        auto& pixel       = *pimage;
        float foreground  = min(0.6f + probability * 0.2f, 0.8f);
        float background  = 1 - foreground;
        for(int c = 0; c < 3; ++c){
            auto value = pixel[c] * background + foreground * _classes_colors[iclass * 3 + 2-c];
            pixel[c] = min((int)value, 255);
        }
    }
}

long BaiscTools::get_current_time(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    long timestamp = tv.tv_sec * 1000 + tv.tv_usec / 1000;
    return timestamp;
}

bool BaiscTools::string_to_bool(string value)
{
    if (value == "True" || value == "true" || value == "1")
    {
        return true;
    }
    else if (value == "False" || value == "false" || value == "0" || value == "")
    {
        return false;
    }
    else
    {
        exit(-1);
    }
}

vector<string> BaiscTools::parse_string_v1(string &param_string)
{
    vector<string> result;
    istringstream param_string_stream(param_string);
    string temp;

    while (param_string_stream)
    {
        if (!getline(param_string_stream, temp, ','))
            break;
        temp.erase(remove(temp.begin(), temp.end(), ' '), temp.end());
        result.push_back(temp);
    }
    return result;
}



vector<vector<string>> BaiscTools::parse_string_v2(string &param_string)
{
    vector<string> param_list;
    vector<string> param;
    vector<vector<string>> result;
    istringstream param_string_stream(param_string);
    string temp;
    while (param_string_stream)
    {
        if (!getline(param_string_stream, temp, ';'))
            break;
        param_list.push_back(temp);
    }

    for (int i = 0; i < param_list.size(); i++)
    {
        istringstream param_list_stream(param_list[i]);
        while (param_list_stream)
        {
            if (!getline(param_list_stream, temp, ','))
                break;
            temp.erase(remove(temp.begin(), temp.end(), ' '), temp.end());
            param.push_back(temp);
        }
        result.push_back(param);
        param.clear();
    }

    param_list.clear();
    param.clear();
    vector<string>().swap(param_list);
    vector<string>().swap(param);
    return result;
}


float BaiscTools::box_overlap(vector<float>region1, vector<float>region2)
{
    if (region1[0] > region2[2] || region1[1] > region2[3] || region1[2] < region2[0] || region1[3] < region2[1])
        return 0;
    float width = min(region1[2], region2[2]) - max(region1[0], region2[0]);
    float height = min(region1[3], region2[3]) - max(region1[1], region2[1]);
    float intersection_area = width * height;
    float area1 = (region1[3] - region1[1]) * (region1[2] - region1[0]);
    float area2 = (region2[3] - region2[1]) * (region2[2] - region2[0]);
    float iou = intersection_area * 1.0 / (area1 + area2 - intersection_area);
    return iou;
}