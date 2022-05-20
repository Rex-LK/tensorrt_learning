void unet_inference(){
    TRTLogger logger;
    auto engine_data = load_file("unet.trtmodel");
    auto runtime   = make_nvshared(nvinfer1::createInferRuntime(logger));
    auto engine = make_nvshared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }

    if(engine->getNbBindings() != 2){
        printf("你的onnx导出有问题，必须是1个输入和1个输出，你这明显有：%d个输出.\n", engine->getNbBindings() - 1);
        return;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    auto execution_context = make_nvshared(engine->createExecutionContext());

    int input_batch = 1;
    int input_channel = 3;
    int input_height = 512;
    int input_width = 512;
    int input_numel = input_batch * input_channel * input_height * input_width;
    float* input_data_host = nullptr;
    float* input_data_device = nullptr;
    cudaMallocHost(&input_data_host, input_numel * sizeof(float));
    cudaMalloc(&input_data_device, input_numel * sizeof(float));

    ///////////////////////////////////////////////////
    // letter box
    auto image = cv::imread("street.jpg");
    float scale_x = input_width / (float)image.cols;
    float scale_y = input_height / (float)image.rows;
    float scale = std::min(scale_x, scale_y);
    float i2d[6], d2i[6];
    i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * image.cols + input_width + scale  - 1) * 0.5;
    i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * image.rows + input_height + scale - 1) * 0.5;

    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);

    cv::Mat input_image(input_height, input_width, CV_8UC3);
    cv::warpAffine(image, input_image, m2x3_i2d, input_image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));
    cv::imwrite("input-image.jpg", input_image);

    int image_area = input_image.cols * input_image.rows;
    unsigned char* pimage = input_image.data;
    float* phost_b = input_data_host + image_area * 0;
    float* phost_g = input_data_host + image_area * 1;
    float* phost_r = input_data_host + image_area * 2;
    for(int i = 0; i < image_area; ++i, pimage += 3){
        *phost_r++ = pimage[0] / 255.0f;
        *phost_g++ = pimage[1] / 255.0f;
        *phost_b++ = pimage[2] / 255.0f;
    }
    ///////////////////////////////////////////////////
    (cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));

    // 3x3输入，对应3x3输出
    auto output_dims   = engine->getBindingDimensions(1);
    int output_height  = output_dims.d[1];
    int output_width   = output_dims.d[2];
    int num_classes    = output_dims.d[3];
    int output_numel = input_batch * output_height * output_width * num_classes;
    float* output_data_host = nullptr;
    float* output_data_device = nullptr;
    (cudaMallocHost(&output_data_host, sizeof(float) * output_numel));
    (cudaMalloc(&output_data_device, sizeof(float) * output_numel));

    // 明确当前推理时，使用的数据输入大小
    auto input_dims = engine->getBindingDimensions(0);
    input_dims.d[0] = input_batch;

    execution_context->setBindingDimensions(0, input_dims);
    float* bindings[] = {input_data_device, output_data_device};
    bool success      = execution_context->enqueueV2((void**)bindings, stream, nullptr);
    (cudaMemcpyAsync(output_data_host, output_data_device, sizeof(float) * output_numel, cudaMemcpyDeviceToHost, stream));
    (cudaStreamSynchronize(stream));

    cv::Mat prob, iclass;
    tie(prob, iclass) = post_process(output_data_host, output_width, output_height, num_classes, 0);
    cv::warpAffine(prob, prob, m2x3_d2i, image.size(), cv::INTER_LINEAR);
    cv::warpAffine(iclass, iclass, m2x3_d2i, image.size(), cv::INTER_NEAREST);
    render(image, prob, iclass);

    printf("Done, Save to image-draw.jpg\n");
    cv::imwrite("image-draw.jpg", image);

    (cudaStreamDestroy(stream));
    (cudaFreeHost(input_data_host));
    (cudaFreeHost(output_data_host));
    (cudaFree(input_data_device));
    (cudaFree(output_data_device));
}