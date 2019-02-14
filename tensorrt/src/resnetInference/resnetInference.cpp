#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include <vector>
#include "NvInfer.h"
#include "NvUffParser.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc.hpp>
#include "NvUtils.h"
#include <cstdio>
#include <ctime>
#include <sys/stat.h>
#include <sys/types.h>

using namespace nvuffparser;
using namespace nvinfer1;
#include "common.h"

static Logger gLogger;

#define MAX_WORKSPACE (1 << 30)

#define RETURN_AND_LOG(ret, severity, message)                                              \
    do {                                                                                    \
        std::string error_message = "resnet_inference: " + std::string(message);            \
        gLogger.log(ILogger::Severity::k ## severity, error_message.c_str());               \
        return (ret);                                                                       \
    } while(0)

inline int64_t volume(const Dims& d)
{
	int64_t v = 1;
	for (int64_t i = 0; i < d.nbDims; i++)
		v *= d.d[i];
	return v;
}


inline unsigned int elementSize(DataType t)
{
	switch (t)
	{
	case DataType::kINT32:
		// Fallthrough, same as kFLOAT
	case DataType::kFLOAT: return 4;
	case DataType::kHALF: return 2;
	case DataType::kINT8: return 1;
	}
	assert(0);
	return 0;
}


char outputDirectory[256];
static const int INPUT_C = 3;
static const int INPUT_H = 224;
static const int INPUT_W = 224;
char trtFilename[32] = "resnet.trt";
char uffFilename[32] = "resnet.uff"; // UFF file generated using TensorRT's Python API
char scoresFilename[32] = "resnet_scores.csv";
char timesFilename[32] = "resnet_times.csv";


std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"../../images/", "../../models/", "../../labels/"};
    return locateFile(input,dirs);
}


void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}


std::vector<std::pair<int64_t, DataType>>
calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize)
{
    std::vector<std::pair<int64_t, DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i)
    {
        Dims dims = engine.getBindingDimensions(i);
        DataType dtype = engine.getBindingDataType(i);

        int64_t eltCount = volume(dims) * batchSize;
        sizes.push_back(std::make_pair(eltCount, dtype));
    }

    return sizes;
}

void* loadRGBImageForCuda(const std::string& filename, int64_t eltCount, DataType dtype)
{
	/* in that specific case, eltCount == INPUT_C * INPUT_H * INPUT_W */
	assert(eltCount == INPUT_C * INPUT_H * INPUT_W);
	assert(elementSize(dtype) == sizeof(float));

	size_t memSize = eltCount * elementSize(dtype);
	float* inputs = new float[eltCount];

	// Read OpenCV Mat image
	cv::Mat image;
	auto imagefile = locateFile(filename);
	image = cv::imread(imagefile.c_str());
	// Downsample with nearest neighbour
	cv::resize(image, image, cv::Size(INPUT_H, INPUT_W), 0, 0, cv::INTER_NEAREST);
	// Convert from HWC to CHW format
	auto size = image.size();
	cv::Size newsize(size.width, size.height * 3);
	cv::Mat destination(newsize, CV_8U);
	for (int i = 0; i < image.channels(); ++i)
	{
		cv::extractChannel(
			image,
			cv::Mat(
				size.height,
				size.width,
				CV_8U,
				&(destination.at<uint8_t>(size.height*size.width*i))),
			2 - i); // BGR 2 RGB
	}
	// Convert to uint8_t data array
	int size_ = destination.total() * destination.elemSize();
	uint8_t* data = new uint8_t[size_];
	std::memcpy(data, destination.data, size_ * sizeof(uint8_t));

	/* initialize the inputs buffer */
	for (int i = 0; i < eltCount; i++)
		inputs[i] = float(data[i]) / 255.0;

	void* deviceMem = safeCudaMalloc(memSize);
	CHECK(cudaMemcpy(deviceMem, inputs, memSize, cudaMemcpyHostToDevice));

	delete[] inputs;
	return deviceMem;
}

void saveScores(int64_t eltCount, DataType dtype, void* buffer, int label, const std::string& filename)
{
	assert(elementSize(dtype) == sizeof(float));
	size_t memSize = eltCount * elementSize(dtype);
	float* outputs = new float[eltCount];
	CHECK(cudaMemcpy(outputs, buffer, memSize, cudaMemcpyDeviceToHost));

	// Save scores to csv
	ofstream scoresFile;
	char scoresFilepath[256];
	sprintf(scoresFilepath, "%s/%s", outputDirectory, scoresFilename);
	scoresFile.open(scoresFilepath, ofstream::out | ofstream::app);
	if (scoresFile.is_open())
	{
		// Write filename and label to row
		scoresFile << filename.c_str() << "," << label;
		// Write scores to row
		for (unsigned int i = 0; i < eltCount; i++)
		{
			scoresFile << "," << outputs[i];
		}
		scoresFile << "\n";
		scoresFile.close();
	}
}


void printOutput(int64_t eltCount, DataType dtype, void* buffer, int label)
{
    assert(elementSize(dtype) == sizeof(float));
    std::cout << "--- OUTPUT ---" << std::endl;

    size_t memSize = eltCount * elementSize(dtype);
    float* outputs = new float[eltCount];
    CHECK(cudaMemcpy(outputs, buffer, memSize, cudaMemcpyDeviceToHost));

    int maxIdx = 0;
    for (int i = 0; i < eltCount; ++i)
        if (outputs[i] > outputs[maxIdx])
            maxIdx = i;

    for (int64_t eltIdx = 0; eltIdx < eltCount; ++eltIdx)
    {
        std::cout << eltIdx << " => " << outputs[eltIdx] << "\t : ";
        if (eltIdx == maxIdx)
            std::cout << "***";
        if (eltIdx == label)
            std::cout << "~~~";
        std::cout << "\n";
    }

    std::cout << std::endl;
    delete[] outputs;
}


ICudaEngine* loadModelAndCreateEngine(const char* uffFile, int maxBatchSize,
                                      IUffParser* parser)
{
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();

#if 1
    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT))
        RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
#else
    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kHALF))
        RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
    builder->setFp16Mode(true);
#endif

    /* we create the engine */
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(MAX_WORKSPACE);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (!engine)
        RETURN_AND_LOG(nullptr, ERROR, "Unable to create engine");

    /* we can clean the network and the parser */
    network->destroy();
    builder->destroy();

    return engine;
}

ICudaEngine* loadEngine(const char* filename)
{
	std::stringstream trtModelStream;
	trtModelStream.seekg(0, trtModelStream.beg);
	std::ifstream cache(filename);
	trtModelStream << cache.rdbuf();
	cache.close();
	IRuntime* runtime = createInferRuntime(gLogger);
	trtModelStream.seekg(0, std::ios::end);
	const int modelSize = trtModelStream.tellg();
	trtModelStream.seekg(0, std::ios::beg);
	void* modelMem = malloc(modelSize);
	trtModelStream.read((char*)modelMem, modelSize);
	nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(modelMem, modelSize, NULL);
	free(modelMem);
	return engine;
}

void saveEngine(ICudaEngine& engine)
{
	IHostMemory* trtModelStream;
	trtModelStream = engine.serialize();
	std::ofstream p(trtFilename);
	p.write((const char*)trtModelStream->data(), trtModelStream->size());
}

void execute(ICudaEngine& engine)
{
    IExecutionContext* context = engine.createExecutionContext();
    int batchSize = 1;

    int nbBindings = engine.getNbBindings();
    assert(nbBindings == 2);

    std::vector<void*> buffers(nbBindings);
    auto buffersSizes = calculateBindingBufferSizes(engine, nbBindings, batchSize);

    int bindingIdxInput = 0;
    for (int i = 0; i < nbBindings; ++i)
    {
        if (engine.bindingIsInput(i))
            bindingIdxInput = i;
        else
        {
            auto bufferSizesOutput = buffersSizes[i];
            buffers[i] = safeCudaMalloc(bufferSizesOutput.first *
                                        elementSize(bufferSizesOutput.second));
        }
    }

    auto bufferSizesInput = buffersSizes[bindingIdxInput];
    
    // Read labels and filenames
    string line;
	auto filename = locateFile("labels.csv");
	ifstream labelsFile(filename.c_str());
    std::vector<std::string> filenames;
    std::vector<int> labels;
    if (labelsFile.is_open())
    {
        getline(labelsFile, line); // Skip header
        while (getline(labelsFile, line))
        {
            int tokenCount = 0;
            size_t pos = 0;
            std::string token;
            while ((pos = line.find(",")) != std::string::npos)
            {
                token = line.substr(0, pos);
                if (tokenCount == 0) {
                    filenames.push_back(token);
                }
                else if (tokenCount == 1)
                    labels.push_back(std::stoi(token));
                line.erase(0, pos + 1);
                tokenCount = tokenCount + 1;
            }
        }
        labelsFile.close();
    }
	printf("Read all %i labels.\n", filenames.size());

	// Create scores file
	ofstream scoresFile;
	char scoresFilepath[256];
	sprintf(scoresFilepath, "%s/%s", outputDirectory, scoresFilename);
	printf("Saving scores to %s...\n", scoresFilepath);
	scoresFile.open(scoresFilepath, ofstream::out);
	if (scoresFile.is_open())
	{
		// Write header
		scoresFile << "Filename,Label,0,1,2,3,4,5,6,7,8\n";
		scoresFile.close();
	}
    
    float ms;
    std::vector<float> inferenceTimes;
    std::vector<float> preprocessingTimes;
    for (unsigned int i = 0; i < filenames.size(); i++)
    {
        auto t_start = std::chrono::high_resolution_clock::now();
        buffers[bindingIdxInput] = loadRGBImageForCuda(filenames[i],
                                                       bufferSizesInput.first,
                                                       bufferSizesInput.second);
        auto t_end = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
        preprocessingTimes.push_back(ms);
        
        t_start = std::chrono::high_resolution_clock::now();
        context->execute(batchSize, &buffers[0]);
        t_end = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
        inferenceTimes.push_back(ms);

        for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
        {
            if (engine.bindingIsInput(bindingIdx))
                continue;

            auto bufferSizesOutput = buffersSizes[bindingIdx];
            printOutput(bufferSizesOutput.first, bufferSizesOutput.second,
                        buffers[bindingIdx], labels[i]);
			saveScores(bufferSizesOutput.first, bufferSizesOutput.second,
				buffers[bindingIdx], labels[i], filenames[i]);
        }
        CHECK(cudaFree(buffers[bindingIdxInput]));
    }
    
    // Save inference times to csv
	ofstream timesFile;
	char timesFilepath[256];
	sprintf(timesFilepath, "%s/%s", outputDirectory, timesFilename);
	printf("Saving times to %s...\n", timesFilepath);
	timesFile.open(timesFilepath, ofstream::out);
    if (timesFile.is_open())
    {
        // Write header
		timesFile << "Filename,Preprocessing time (ms),Inference time (ms)\n";
        // Write rows
        for (unsigned int i = 0; i < inferenceTimes.size(); i++)
        {
			timesFile << filenames[i] << "," << preprocessingTimes[i] << ","
                    << inferenceTimes[i] << "\n";
        }
		timesFile.close();
    }
    printf("Finished inferencing.\n");

    for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
        if (!engine.bindingIsInput(bindingIdx))
            CHECK(cudaFree(buffers[bindingIdx]));
    context->destroy();
}


int main(int argc, char** argv)
{
	// Get timestamp
	time_t rawtime;
	tm* timeinfo;
	char timestamp[80];
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	strftime(timestamp, 80, "%Y%m%d-%H%M%S", timeinfo);

	// Create output directory
	sprintf(outputDirectory, "../../outputs/%s/", timestamp);
	printf("Saving to %s...\n", outputDirectory);
	mkdir(outputDirectory, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    int maxBatchSize = 1;
    auto parser = createUffParser();

    /* Register tensorflow input */
    parser->registerInput("input_1", Dims3(INPUT_C, INPUT_H, INPUT_W), UffInputOrder::kNCHW);
    parser->registerOutput("fc9/Sigmoid");

	// If TRT file exists, load first
	ICudaEngine* engine;
	ifstream f(trtFilename);
	if (f.good())
	{
		printf("TRT file exists, constructing engine...");
		f.close();
		engine = loadEngine(trtFilename);
		printf("Loaded the TRT model and created the engine.");
	}
	else
	{
		// Else load from UFF and save TRT
		printf("No TRT model file, constructing engine from UFF.\n");
		auto uffFile = locateFile(uffFilename);
		engine = loadModelAndCreateEngine(uffFile.c_str(), maxBatchSize, parser);
		saveEngine(*engine);
		printf("Loaded the UFF model, created and saved the engine.\n");
	}
	if (!engine)
		RETURN_AND_LOG(EXIT_FAILURE, ERROR, "Model load failed.");

    /* we need to keep the memory created by the parser */
    parser->destroy();

    execute(*engine);
    engine->destroy();
    shutdownProtobufLibrary();
    return EXIT_SUCCESS;
}
