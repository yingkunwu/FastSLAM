#include "../include/world.h"
#include <iostream>
#include <assert.h>

World::World(int size_x, int size_y) : x__(size_x), y__(size_y) {}

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

