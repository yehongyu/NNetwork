#ifndef _NNETWORK_H_
#define _NNETWORK_H_

#include <memory>
#include <vector>
#include <Eigen/Dense>

namespace nnmachine
{

    class NNetwork
    {
      public:
        NNetwork(std::vector<int> const& layers, double learningRate);

        void train(Eigen::MatrixXd const& constInputs, Eigen::MatrixXd const& targets);
        Eigen::MatrixXd predict(Eigen::MatrixXd const& constInput);

      private:
        std::vector<int> layers;
        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::MatrixXd> biass;
        double lr;
        size_t layerNum;

    };

    Eigen::MatrixXd Sigmoid(Eigen::MatrixXd const&);
    Eigen::MatrixXd DSigmoid(Eigen::MatrixXd const&);

}

#endif
