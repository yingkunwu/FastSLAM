#include "../include/utility.h"
#include "../include/world.h"
#include "../include/MCL.h"

#include <iostream>
#include <math.h>
#include <random>

double utility::get_gaussian_random_number(double mean, double var)
{
    // Random Generators
    std::random_device rd;
    std::mt19937 gen(rd());
    // get rangom number from normal distribution
    std::normal_distribution<double> dist(mean, var);
    return dist(gen);
}

double utility::get_random_number(double min, double max)
{
    // Random Generators
    std::random_device rd;
    std::mt19937 gen(rd());
    // get random number from a uniform distribution
    std::uniform_real_distribution<double> dist(min, max);
    return dist(gen);
}

double utility::get_gaussian_probability(double mean, double var, double x)
{
    //std::cout << mean << " " << var << " " << x << std::endl;
    // Probability of x given normal ditribution with mean and variance
    double p = exp(-(pow((mean - x), 2)) / (pow(var, 2)) / 2.0) / sqrt(2.0 * M_PI * (pow(var, 2)));
    //std::cout << p << std::endl;
    return p;
}

double utility::evaluation(Robot *r, std::vector<particle> *p, World *world)
{
    //Calculate the mean error of the system
    double sum = 0.0;
    for (int i = 0; i < p->size(); i++)
    {
        //the second part is because of world's cyclicity
        double dx = fmod(((*p)[i].r.get_x() - r->get_x() + (world->get_x() / 2.0)), world->get_x()) - (world->get_x() / 2.0);
        double dy = fmod(((*p)[i].r.get_y() - r->get_y() + (world->get_y() / 2.0)), world->get_y()) - (world->get_y() / 2.0);
        double err = sqrt(pow(dx, 2) + pow(dy, 2));
        sum += err;
    }
    return sum / p->size();
}

void utility::visualization(Robot *robot, int step, std::vector<particle> *belief, World *w)
{
    float x_min = 0.0;
    float y_min = 0.0;
    float x_max = 100.0;
    float y_max = 100.0;

    //Draw cvplot scatter plot for the robot, landmarks, particles and resampled particles on a graph
    int n = belief->size();
    std::vector<std::pair<float, float>> data;

    //Graph Format
    auto name = "MCL";
    cvplot::setWindowTitle(name, "step" + std::to_string(step));
    cvplot::moveWindow(name, 0, 0);
    cvplot::resizeWindow(name, 800, 600);
    auto &figure = cvplot::figure(name);
    figure.origin(true, true);
    figure.setAxes(x_min, y_min, x_max, y_max);

    //Draw robot position in blue
    data.clear();
    data.push_back({robot->get_x(), robot->get_y()});
    figure.series("Robot").set(data).type(cvplot::Dots).color(cvplot::Black);

    //Draw particles in green
    data.clear();
    for (auto i = 0; i < n; i++)
    {
        data.push_back({(*belief)[i].r.get_x(), (*belief)[i].r.get_y()});
    }
    figure.series("Particles").set(data).type(cvplot::Dots).color(cvplot::Green);

    //Draw landmarks in red
    data.clear();
    std::vector<double> lm = w->get_landmarks();
    for (int i = 1; i < lm.size(); i += 2)
    {
        data.push_back({lm[i - 1], lm[i]});
    }
    figure.series("Landmarks").set(data).type(cvplot::Dots).color(cvplot::Red);

    //Show the plot
    cvplot::figure(name).show();

}