#ifndef LOGGER_H
#define LOGGER_H

#include <NvInfer.h>
#include <NvInferRuntime.h>


inline const char* severity_string(nvinfer1::ILogger::Severity t)
{
	switch (t)
	{
	case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
	case nvinfer1::ILogger::Severity::kERROR:   return "error";
	case nvinfer1::ILogger::Severity::kWARNING: return "warning";
	case nvinfer1::ILogger::Severity::kINFO:    return "info";
	case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
	default: return "unknow";
	}
}


class TRTLogger : public nvinfer1::ILogger
{
public:
	virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override
	{
		if (severity <= Severity::kINFO)
		{
			if (severity == Severity::kWARNING)
				printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
			else if (severity <= Severity::kERROR)
				printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
			else
				printf("%s: %s\n", severity_string(severity), msg);
		}
	}
} logger;

#endif // LOGGER_H