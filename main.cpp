#include <mvnc.h>
extern "C" {
#include <./movidius/fp16.h>    // TODO: Make yourr own float2fp16 , fp162float to avoid double licensing
}
#include <opencv2/opencv.hpp>
#include <time.h>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <streambuf>
#include <sstream>
#include <memory>

const unsigned int net_data_width = 224;
const unsigned int net_data_height = 224;
const unsigned int net_data_channels = 3;
const cv::Scalar   net_mean(0.40787054*255.0, 0.45752458*255.0, 0.48109378*255.0);

void prepareTensor(std::unique_ptr<unsigned char[]>& input, std::string& imageName,unsigned int* inputLength)
{
  // load an image using OpenCV
  cv::Mat imagefp32 = cv::imread(imageName, -1);
  if (imagefp32.empty())
    throw std::string("Error reading image: ") + imageName;

  // Convert to expected format
  cv::Mat samplefp32;
  if (imagefp32.channels() == 4 && net_data_channels == 3)
    cv::cvtColor(imagefp32, samplefp32, cv::COLOR_BGRA2BGR);
  else if (imagefp32.channels() == 1 && net_data_channels == 3)
    cv::cvtColor(imagefp32, samplefp32, cv::COLOR_GRAY2BGR);
  else
    samplefp32 = imagefp32;
  
  // Resize input image to expected geometry
  cv::Size input_geometry(net_data_width, net_data_height);

  cv::Mat samplefp32_resized;
  if (samplefp32.size() != input_geometry)
    cv::resize(samplefp32, samplefp32_resized, input_geometry);
  else
    samplefp32_resized = samplefp32;

  // Convert to float32
  cv::Mat samplefp32_float;
  samplefp32_resized.convertTo(samplefp32_float, CV_32FC3);

  // Mean subtract
  cv::Mat sample_fp32_normalized;
  cv::Mat mean = cv::Mat(input_geometry, CV_32FC3, net_mean);
  cv::subtract(samplefp32_float, mean, sample_fp32_normalized);

  // Separate channels (caffe format: NCHW)
  std::vector<cv::Mat> input_channels(net_data_channels);
  cv::split(sample_fp32_normalized, input_channels);

  // convert image data into float16
  input.reset(new unsigned char[sizeof(short)*net_data_width*net_data_height*net_data_channels]);
  floattofp16(input.get(), reinterpret_cast<float*>(sample_fp32_normalized.data),
        net_data_width*net_data_height*net_data_channels);
 
  *inputLength = sizeof(short)*net_data_width*net_data_height*net_data_channels;
}


void printPredictions(void* outputTensor,unsigned int outputLength)
{
	unsigned int net_output_width = outputLength/sizeof(short);

	std::vector<float> predictions(net_output_width);
  fp16tofloat(&predictions[0],reinterpret_cast<unsigned char*>(outputTensor),net_output_width);
	int top1_index= -1;
	float top1_result = -1.0;

	// find top1 results	
	for(int i = 0; i<net_output_width;++i) {
		if(predictions[i] > top1_result) {
			top1_result = predictions[i];
			top1_index = i;
		}
	}

	// Print top-1 result (class name , prob)
	std::ifstream synset_words("./synset_words.txt");
  std::string top1_class;
	for (int i=0; i<=top1_index; ++i) {
		std::getline(synset_words,top1_class);	
	}
	std::cout << "top-1: " << top1_result << " (" << top1_class << ")" << std::endl;
}


void loadGraphFromFile(std::unique_ptr<char[]>& graphFile, const std::string& graphFileName, unsigned int* graphSize)
{
  std::ifstream ifs;
  ifs.open(graphFileName, std::ifstream::binary);

  if (ifs.good() == false) {
    throw std::string("Error: Unable to open graph file: ") + graphFileName;
  }

  // Get size of file
  ifs.seekg(0, ifs.end);
  *graphSize = ifs.tellg();
  ifs.seekg(0, ifs.beg);


  graphFile.reset(new char[*graphSize]);

  ifs.read(graphFile.get(),*graphSize);

  // TODO: check if whole file was read
  
  ifs.close();
}


int main(int argc, char** argv) {

  void * graphHandle = nullptr;
  const std::string graphFileName("myGoogleNetGraph");
  int exit_code = 0;
  mvncStatus ret = MVNC_OK;
  try {
    
    if(argc != 2 ) {
      throw std::string("ERROR: Wrong syntax. Valid syntax:\n \
               test-ncs <name of image to process> \n \
                ");
    }
    std::string imageFileName(argv[1]);

    std::vector<std::string> ncs_names;
    char tmpncsname[200]; // How to determine max size automatically
    int index = 0;  // Index of device to query for
    while(ret == MVNC_OK) {
      ret = mvncGetDeviceName(index++,tmpncsname,200); // hardcoded max name size 
      if (ret == MVNC_OK) {
        ncs_names.push_back(tmpncsname); 
        std::cout << "Found NCS: " << tmpncsname << std::endl;
      }
    }

    // If not devices present the exit
    if (ncs_names.size() == 0) {
      throw std::string("Error: No Intel Movidius identified in a system!\n");
    }

    // Using first device
    // TODO: run workload on many devices
    void* dev_handle = 0;
    ret = mvncOpenDevice(ncs_names[0].c_str(), &dev_handle);
    if(ret != MVNC_OK) {
      throw std::string("Error: Could not open NCS device: ") + ncs_names[0];
    }

    // Allocate graph
    unsigned int graphSize = 0;
    std::unique_ptr<char[]> graphFile;
    loadGraphFromFile(graphFile, graphFileName, &graphSize);

    ret = mvncAllocateGraph(dev_handle,&graphHandle,static_cast<void*>(graphFile.get()),graphSize);
    if (ret != MVNC_OK) {
      throw std::string("Error: Graph allocation on NCS failed!");
    }
    
    // Loading tensor, tensor is of a HalfFloat data type 
    std::unique_ptr<unsigned char[]> tensor;
    unsigned int inputLength;
    prepareTensor(tensor, imageFileName, &inputLength);
    ret = mvncLoadTensor(graphHandle, tensor.get(), inputLength,
                         nullptr/* user param*/);  // TODO: What are user params??? 
    if (ret != MVNC_OK) {
      throw std::string("Error: Loading Tensor failed!");
    }

    void* outputTensor;
    unsigned int outputLength;
		void* userParam;
    // This function normally blocks till results are available
    ret = mvncGetResult(graphHandle,&outputTensor, &outputLength,&userParam);
    
    if (ret != MVNC_OK) {
      throw std::string("Error: Getting results from NCS failed!");
    }
    printPredictions(outputTensor, outputLength);
  }
  catch (std::string err) {
    std::cout << err << std::endl;
    exit_code = -1;
  }

  // Cleaning up
  ret = mvncDeallocateGraph(graphHandle);
  if (ret != MVNC_OK) {
    std::cerr << "Error: Deallocation of Graph failed!" << std::endl;
  }

	// TODO: close Device

  return exit_code;
}
