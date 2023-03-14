#include "../include/robot.h"
#include "../include/utility.h"

double Robot::forward_noise__;
double Robot::turn_noise__;
double Robot::sense_noise__;

Robot::Robot(int x, int y, std::vector<double> landmarks) : landmarks(landmarks)
{
    // Randomly and uniformly position the robot inside the world 
    x__ = utility::get_random_number(0.0, 1.0) * x;
    y__ = utility::get_random_number(0.0, 1.0) * y;
    orientation__ = utility::get_random_number(0.0, 1.0) * 2 * M_PI;
}

void Robot::set_states(double new_x, double new_y, double new_orient)
{
    // replace robot states with new values
    x__ = new_x;
    y__ = new_y;
    orientation__ = new_orient;
}

void Robot::set_noise(double new_f_noise, double new_t_noise, double new_s_noise)
{
    // set noises
    forward_noise__ = new_f_noise;
    turn_noise__ = new_t_noise;
    sense_noise__ = new_s_noise;
}

std::vector<double> Robot::sense(bool noise)
{
    std::vector<double> measurements;
    // iterate through landmarks
    for (int i = 1; i < landmarks.size(); i += 2)
    {
        // get Euclidean distance to each landmark and add noise to simulate range finder data
        double m = sqrt(pow((landmarks[i - 1] - x__), 2) + pow((landmarks[i] - y__), 2));
        noise ? m += utility::get_gaussian_random_number(0.0, sense_noise__) : m += 0.0;
        measurements.push_back(m);
    }

    return measurements;
}

void Robot::move(double turn, double forward)
{
    // set rotation, add gaussian noise with mean of rotation bias and turn_noise as variance
    // here we assume trn bias is zero
    orientation__ = fmod((orientation__ + turn + utility::get_gaussian_random_number(0.0, turn_noise__)), (2 * M_PI));

    double dist = forward + utility::get_gaussian_random_number(0, forward_noise__);
    x__ = x__ + (dist * cos(orientation__));
    y__ = y__ + (dist * sin(orientation__));
}

std::string Robot::get_pose()
{
    std::string pose = std::string("[") + std::string("X=") + std::to_string(x__) + std::string(", Y=") + std::to_string(y__) + std::string(" Theta=") + std::to_string(orientation__) + std::string("]");
    return pose;
}

double Robot::get_sensor_noise() { return sense_noise__; }

double Robot::get_x() { return x__; }

double Robot::get_y() { return y__; }