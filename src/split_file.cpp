
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <filesystem>
#include <cstring>
#include <cmath>
#include <omp.h>

namespace nb = nanobind;

namespace fs = std::filesystem;

// 极致性能的XYZ文件分割
void split_xyz(
    const std::string input_file,
    const std::string output_dir,
    const std::string output_prefix)
{
    // 创建输出目录
    fs::create_directories(output_dir);

    // 一次性读取整个文件到内存
    std::ifstream infile(input_file, std::ios::binary | std::ios::ate);
    if (!infile)
    {
        throw std::runtime_error("Cannot open input file: " + input_file);
    }

    size_t file_size = infile.tellg();
    infile.seekg(0, std::ios::beg);

    std::vector<char> buffer(file_size);
    infile.read(buffer.data(), file_size);
    infile.close();

    // 快速扫描找到所有帧的位置
    // reserve不会限制大小，只是预分配内存，超过会自动扩容
    std::vector<size_t> frame_starts;
    std::vector<size_t> frame_sizes;
    frame_starts.reserve(10000);
    frame_sizes.reserve(10000);

    const char *data = buffer.data();
    const char *end = data + file_size;
    size_t offset = 0;

    while (offset < file_size)
    {
        const char *line_start = data + offset;
        const char *line_end = static_cast<const char *>(
            memchr(line_start, '\n', end - line_start));

        if (!line_end)
            break;

        // 快速解析原子数
        int n_atoms = 0;
        const char *p = line_start;

        // 跳过空格
        while (p < line_end && (*p == ' ' || *p == '\t'))
            ++p;

        // 解析数字
        while (p < line_end && *p >= '0' && *p <= '9')
        {
            n_atoms = n_atoms * 10 + (*p - '0');
            ++p;
        }

        if (n_atoms == 0)
            break;

        frame_starts.push_back(offset);

        // 跳过 n_atoms + 1 行
        const char *current = line_end + 1;
        int lines_to_skip = n_atoms + 1;

        for (int i = 0; i < lines_to_skip && current < end; ++i)
        {
            const char *next_line = static_cast<const char *>(
                memchr(current, '\n', end - current));
            if (!next_line)
            {
                current = end;
                break;
            }
            current = next_line + 1;
        }

        frame_sizes.push_back(current - line_start);
        offset = current - data;
    }

    const int total_frames = frame_starts.size();
    if (total_frames == 0)
    {
        throw std::runtime_error("No valid frames found in file");
    }

    // 计算需要的位数（如：10000帧需要5位，100000帧需要6位）
    const int num_digits = static_cast<int>(std::log10(total_frames - 1)) + 1;

// 并行写入所有帧
#pragma omp parallel
    {
        // 每个线程使用独立的字符串流，避免锁竞争
        std::ostringstream oss;
        oss << output_dir << "/" << output_prefix << ".";
        std::string base_path = oss.str();

#pragma omp for schedule(static)
        for (int frame = 0; frame < total_frames; ++frame)
        {
            // 构建文件名（线程局部）
            std::ostringstream filename;
            filename << base_path
                     << std::setw(num_digits) << std::setfill('0')
                     << frame << ".xyz";

            // 直接写入，使用较大的缓冲区
            std::ofstream outfile(filename.str(), std::ios::binary);
            if (outfile)
            {
                // 设置大缓冲区以提高写入性能
                constexpr size_t buf_size = 1024 * 1024; // 1MB buffer
                char write_buffer[buf_size];
                outfile.rdbuf()->pubsetbuf(write_buffer, buf_size);

                const char *frame_data = buffer.data() + frame_starts[frame];
                outfile.write(frame_data, frame_sizes[frame]);
            }
        }
    }
}

// Python绑定
NB_MODULE(_split, m)
{
    m.def("split_xyz", &split_xyz,
          nb::arg("input_file"),
          nb::arg("output_dir"),
          nb::arg("output_prefix"),
          "Ultra-fast parallel splitting of multi-frame XYZ file (maximum I/O performance)");
}