#pragma once

#include <vector>

class World
{
public:
    World(int size_x, int size_y);
    void set_landmarks(double x, double y);
    std::vector<double> get_landmarks();

    int get_x();
    int get_y();
    
private:
    std::vector<double> landmarks__;
    // world size
    double x__;
    double y__;
};
