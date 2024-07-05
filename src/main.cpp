#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/fast_random.h>
#include <libutils/timer.h>

#include <cassert>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

template<typename T>
std::string to_string(T value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line) {
    if (CL_SUCCESS == err)
        return;

    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

int main() {
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // find device and platform
    cl_uint platformCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformCount));
    std::vector<cl_platform_id> platforms(platformCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformCount, platforms.data(), nullptr));

    bool gpuDeviceSelected = false;
    bool anyDeviceSelected = false;
    cl_platform_id selectedPlatform;
    cl_device_id selectedDevice;
    for (int platformIndex = 0; platformIndex < platformCount; ++platformIndex) {
        cl_platform_id platform = platforms[platformIndex];
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));
        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            cl_device_id device = devices[deviceIndex];
            cl_device_type deviceType;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr));

            if (deviceType & CL_DEVICE_TYPE_GPU) {
                gpuDeviceSelected = true;
                anyDeviceSelected = true;
                selectedPlatform = platform;
                selectedDevice = device;
                break;
            }
            if (deviceType & CL_DEVICE_TYPE_CPU) {
                anyDeviceSelected = true;
                selectedPlatform = platform;
                selectedDevice = device;
            }
        }
        if (gpuDeviceSelected)
            break;
    }
    if (!anyDeviceSelected)
        throw std::runtime_error("No device found");

    // create context
    cl_int errcode;
    cl_context_properties contextProperties[]{CL_CONTEXT_PLATFORM, cl_context_properties(selectedPlatform), 0};
    cl_device_id devices[]{selectedDevice};
    cl_context context = clCreateContext(contextProperties, 1, devices, nullptr, nullptr, &errcode);
    OCL_SAFE_CALL(errcode);

    // create command queue
    cl_command_queue commandQueue = clCreateCommandQueue(context, selectedDevice, 0, &errcode);
    OCL_SAFE_CALL(errcode);

    // generate data
    constexpr unsigned int n = 100 * 1000 * 1000;
    bool *as, *bs, *cs;
    as = (bool *) malloc(n);
    bs = (bool *) malloc(n);
    cs = (bool *) malloc(n);
    srand(time(0));
    for (int i = 0; i < n; i++) {
        as[i] = rand() % 2 == 0;
        bs[i] = rand() % 2 == 0;
    }

    // create buffers
    size_t bufferSize = n;
    cl_mem a_gpu = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bufferSize, as, &errcode);
    OCL_SAFE_CALL(errcode);
    cl_mem b_gpu = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bufferSize, bs, &errcode);
    OCL_SAFE_CALL(errcode);
    cl_mem c_gpu = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, &errcode);
    OCL_SAFE_CALL(errcode);

    // load kernel text
    std::string kernelSources;
    {
        std::ifstream file("/src/cl/kernel.cl");
        kernelSources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernelSources.size() == 0) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
    }

    // create program with kernerl
    const char *kernelSourcesChars = kernelSources.c_str();
    size_t length[]{kernelSources.size()};
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSourcesChars, length, &errcode);

    // build program
    cl_int buildStatus = clBuildProgram(program, 1, devices, nullptr, nullptr, nullptr);

    // check building info
    size_t log_size = 0;
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, selectedDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));
    std::vector<char> log(log_size, 0);
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, selectedDevice, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr));
    if (log_size > 1) {
        std::cout << "Log:" << std::endl;
        std::cout << log.data() << std::endl;
    }
    OCL_SAFE_CALL(buildStatus);

    // create kernel
    cl_kernel kernel = clCreateKernel(program, "xor", &errcode);
    OCL_SAFE_CALL(errcode);

    {
        unsigned int i = 0;
        clSetKernelArg(kernel, i++, sizeof(cl_mem), &a_gpu);
        clSetKernelArg(kernel, i++, sizeof(cl_mem), &b_gpu);
        clSetKernelArg(kernel, i++, sizeof(cl_mem), &c_gpu);
        clSetKernelArg(kernel, i++, sizeof(unsigned int), &n);
    }

    // run calculating
    {
        size_t workGroupSize = 128;
        size_t globalWorkSize = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        cl_event event;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(commandQueue, kernel, 1, nullptr, &globalWorkSize, &workGroupSize, 0,
                                                 nullptr, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
        }
    }

    // get results from VRAM
    {
        cl_event event;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueReadBuffer(commandQueue, c_gpu, CL_TRUE, 0, bufferSize, cs, 0, nullptr, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
        }
    }

    // check results
    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != (as[i] ^ bs[i])) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    // free resureses
    OCL_SAFE_CALL(clReleaseProgram(program));
    OCL_SAFE_CALL(clReleaseKernel(kernel));
    OCL_SAFE_CALL(clReleaseMemObject(a_gpu));
    OCL_SAFE_CALL(clReleaseMemObject(b_gpu));
    OCL_SAFE_CALL(clReleaseMemObject(c_gpu));
    OCL_SAFE_CALL(clReleaseCommandQueue(commandQueue));
    OCL_SAFE_CALL(clReleaseContext(context));

    return 0;
}