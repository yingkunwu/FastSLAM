#include "../include/robot.h"
#include "../include/MCL.h"
#include "../include/utility.h"
#include <iostream>

int main()
{
    // create a world
    World w(100, 100);
    // create landmarks positions
    double landmarks[16] = {20.0, 20.0, 20.0, 80.0, 20.0, 50.0, 50.0, 20.0, 50.0, 80.0, 80.0, 80.0, 80.0, 20.0, 80.0, 50.0};
    // set world landmarks
    for (int i = 1; i < sizeof(landmarks) / sizeof(landmarks[0]); i += 2)
    {
        w.set_landmarks(landmarks[i - 1], landmarks[i]);
    }

    // create a robot
    Robot R(w);
    // set robot noise
    R.set_noise(0.2, 0.1, 3.0);
    // set robot position inside the world
    R.set_states(40.0, 40.0, M_PI / 2.0);
    std::cout << R.get_pose() << std::endl;
    //std::cout << R.get_sensor_readings() << std::endl;
    // create control command
    command u(10.0, -M_PI / 2.0);
    // move the robot
    R.move(u.turn, u.forward);
    std::cout << R.get_pose() << std::endl;
    //std::cout << R.get_sensor_readings() << std::endl;

    // first belief; create particle set
    int NUMBER_OF_PARTICLES = 1000;
    std::vector<particle> p;
    for (int i = 0; i < NUMBER_OF_PARTICLES; ++i)
    {
        p.emplace_back(Robot(w), 0.0);
    }

    // set number of iterations for mcl
    int NUMBER_OF_ITERATIONS = 100;

    // create control command
    command u1(0.5, 0.1);

    // resample
    for (int i = 0; i < NUMBER_OF_ITERATIONS; ++i)
    {
        // move the robot
        R.move(u1.turn, u1.forward);
        // get sensor measurements
        std::vector<double> z = R.sense();
        std::cout << R.get_pose() << std::endl;
        // mcl
        std::vector<particle> belief = MCL(p, u1, z).resample();
        std::cout << "[Error]" << utility::evaluation(&R, &belief, &w) << std::endl;
        //set new blief
        utility::visualization(&R, i, &p, &belief, &w);
        p = belief;
    }

    return 0;
}