#include <mvnc.h>
#include <./movidius/fp16.h>    // TODO: Make yourr own float2fp16 , fp162float to avoid double licensing
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

void prepareTensor(void* input, std::string& imageName)
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
  // TODO: convert image data into float16
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
      std::cerr << "Error: Graph allocation on NCS failed!" << std::endl;
      exit(-1);
    }
    
    // Loading tensor, tensor is of a HalfFloat data type 
    std::unique_ptr<char[]> tensor;
    prepareTensor(static_cast<void*>(tensor.get()), imageFileName);
//    ret = mvncLoadTensor(graphHandle, input, inputLength, /* user param???*/) 
//    if (ret != MVNC_OK) {
//      std::cerr << "Error: Loading Tensor failed!" << std::endl;
//    }

  }
  catch (std::string err) {
    exit_code = -1;
  }

  // Cleaning up
  ret = mvncDeallocateGraph(graphHandle);
  if (ret != MVNC_OK) {
    std::cerr << "Error: Deallocation of Graph failed!" << std::endl;
  }
  return exit_code;
}
