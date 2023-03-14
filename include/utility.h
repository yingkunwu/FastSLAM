#pragma once

#include <vector>
#include "../cvplot/include/cvplot/cvplot.h"
#include <opencv2/opencv.hpp>

class World;
struct particle;
class Robot;

namespace utility
{
    // Get random number from normal distribution given its mean and variance
    double get_gaussian_random_number(double mean, double var);
    // get random number between [0,1.0]
    double get_random_number(double min, double max);
    // Get normal distribution probability of x given mean and variance
    double get_gaussian_probability(double mean, double var, double x);
    // evaluate each belief
    double evaluation(Robot *r, std::vector<particle> *belief, World *w);
    // visualize the robot and particles
    void visualization(Robot *robot, int step, std::vector<particle> *belief, World *w);
} // namespace utility
