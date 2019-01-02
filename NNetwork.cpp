#include "NNetwork.h"
#include <memory>
#include <iostream>

using namespace Eigen;
using namespace std;

namespace nnmachine
{

MatrixXd Sigmoid(MatrixXd const& x)
{
    // matrix.array(): 将matrix转成array，再点对点操作
    return 1.0 / (1.0 + (-x).array().exp());
}

MatrixXd DSigmoid(MatrixXd const& x)
{
    // cwiseProduct: matrix 与 matrix 对应位置相乘
    return x.cwiseProduct((1.0 - x.array()).matrix());
}

NNetwork::NNetwork(vector<int> const& layersIn, double learningRate):layers(layersIn),lr(learningRate)
{
    layerNum = layers.size();
    for (int i=1; i<layerNum; ++i) {
        MatrixXd curWeight = MatrixXd::Random(layers[i], layers[i-1]);
        MatrixXd curBias = MatrixXd::Random(layers[i], 1);
        weights.push_back(curWeight);
        biass.push_back(curBias);
    }
    std::cout << "init layer num: " << layerNum << std::endl;
    copy(layers.begin(), layers.end(), ostream_iterator<int> (cout, "->"));
    std::cout << std::endl;
    std::cout << "init lr: " << lr << std::endl;
}

void NNetwork::train(MatrixXd const& constInputs, MatrixXd const& targets)
{
    // forward: calc output of each layer
    MatrixXd inputs = constInputs;
    vector<MatrixXd> layerOutputs;
    layerOutputs.push_back(inputs);
    MatrixXd & nextLayerIn = inputs;
    for (int i=0; i<layerNum-1; ++i) {
        MatrixXd curIn = (weights[i] * nextLayerIn) + biass[i];
        MatrixXd curOut = Sigmoid(curIn);
        layerOutputs.push_back(curOut);
        nextLayerIn = curOut;
    }

    MatrixXd & finalOut = nextLayerIn;
    MatrixXd outputErrors = targets - finalOut;

    // backward: update weight and bias
    MatrixXd & preLayerErrIn = outputErrors;
    for (int i=layerNum-2; i>=0; --i) {
        MatrixXd & curOut = layerOutputs[i];
        MatrixXd & postOut = layerOutputs[i+1];
        MatrixXd curErr = preLayerErrIn.cwiseProduct(DSigmoid(postOut));
        MatrixXd postError = weights[i].transpose() * preLayerErrIn;
        preLayerErrIn = postError;
        weights[i] += (lr * (curErr * curOut.transpose()).array()).matrix();
        biass[i] += (lr * curErr.array()).matrix();
    }

    layerOutputs.clear();
}

MatrixXd NNetwork::predict(MatrixXd const& constInputs)
{
    MatrixXd inputs = constInputs;

    MatrixXd & nextLayerIn = inputs;
    for (int i=0; i<layerNum-1; ++i) {
        MatrixXd curIn = (weights[i] * nextLayerIn) + biass[i];
        MatrixXd curOut = Sigmoid(curIn);
        nextLayerIn = curOut;
    }

    return nextLayerIn;
}

}
