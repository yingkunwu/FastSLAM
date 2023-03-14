#include "../include/world.h"
#include <iostream>
#include <assert.h>
#include <opencv2/opencv.hpp>

World::World(int size_x, int size_y) : x__(size_x), y__(size_y) {}

void World::read_map(std::string filename)
{
    // Read the image file 
    cv::Mat img = cv::imread("../scene1.png");

    std::vector<std::vector<int>> grid = std::vector<std::vector<int>>(img.rows, std::vector<int>(img.cols, 0));

    for(int i = 0; i < img.rows; i++) {
        for(int j = 0; j < img.cols; j++) {
            cv::Vec3b rgb = img.at<cv::Vec3b>(i, j);
            if (((int) rgb[0] == 255) && ((int) rgb[1] == 255) && ((int) rgb[2] == 255)) {
                grid[i][j] = 1;
            } else {
                grid[i][j] = 0;
            }
        }
    }
}

void World::set_landmarks(double x, double y)
{
    landmarks__.push_back(x);
    landmarks__.push_back(y);
    // check invariant condition
    assert(landmarks__.size() % 2 == 0);
    std::cout << "[Landmark] x:" << x << " y:" << y <<std::endl;
}

std::vector<double> World::get_landmarks()
{
    return landmarks__;
}

int World::get_x() { return x__; }

int World::get_y() { return y__; }

