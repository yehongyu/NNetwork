#ifndef _MNISTLOADER_CPP_
#define _MNISTLOADER_CPP_

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>

namespace mnist
{

    uint32_t read_header(const std::unique_ptr<char[]>& buffer, size_t pos)
    {
        uint32_t* header = reinterpret_cast<uint32_t*>(buffer.get());
        uint32_t value = *(header + pos);
        return (value << 24) | ((value << 8) & 0x00FF0000) | ((value >> 8) & 0X0000FF00) | (value >> 24);
    }

    std::vector<std::vector<int> > readMNistFile(const std::string& path)
    {
        std::vector<std::vector<int> > res;
        std::ifstream file;
        file.open(path, std::ios::in | std::ios::binary | std::ios::ate);
        if (!file) {
            std::cout << "Error opening file" << std::endl;
            return res;
        }
        std::streampos size = file.tellg();
        std::unique_ptr<char[]> buffer(new char[size]);

        // read entire file to buffer
        file.seekg(0, std::ios::beg);
        file.read(buffer.get(), size);
        file.close();

        uint32_t magic = read_header(buffer, 0);
        uint32_t count = read_header(buffer, 1);

        if (magic == 0x803) {
            uint32_t rows = read_header(buffer, 2);
            uint32_t cols = read_header(buffer, 3);
            //std::cout << "images count:" << count << std::endl;
            uint8_t* image_buffer = reinterpret_cast<uint8_t*>(buffer.get() + 16);
            size_t pixel_size = rows * cols;
            for (size_t i=0; i<count; ++i) {
                std::vector<int> cur_res;
                for (size_t j = 0; j<pixel_size; ++j) {
                    cur_res.push_back(static_cast<int>(*image_buffer++));
                    //std::cout << (int)cur_res[j] << "\n";
                }
                res.push_back(cur_res);
            }
        } else if (magic == 0x801) {
            //std::cout << "label count:" << count << std::endl;
            uint8_t* label_buffer = reinterpret_cast<uint8_t*>(buffer.get() + 8);
            for (size_t i=0; i<count; ++i) {
                std::vector<int> cur_res;
                cur_res.push_back(static_cast<int>(*label_buffer++));
                //std::cout << (int)cur_res[0] << "\n";
                res.push_back(cur_res);
            }
        }
        return res;
    }

}

#endif

