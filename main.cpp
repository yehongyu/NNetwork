#include <iostream>
#include <fstream>
#include <ctime>
#include <map>
#include "NNetwork.h"
#include "MNistLoader.cpp"

using namespace std;
using namespace Eigen;
using namespace nnmachine;
using namespace mnist;

char * getNowTime()
{
    time_t now = time(0);
    char * dt = ctime(&now);
    return dt;
}

MatrixXd normalizeInputs(vector<int> &img, double ratio)
{
    MatrixXi inputs = VectorXi::Map(&img[0], img.size());
    MatrixXd dinputs = inputs.cast <double> ();
    dinputs = dinputs * ratio;
    return (dinputs.array() + 0.01).matrix();
}

VectorXd genTarget(int outputNodeNum, int label)
{
    VectorXd target = VectorXd::Zero(outputNodeNum);
    target = (target.array() + 0.01).matrix();
    target[label] = 0.99;
    return target;
}

double nnTest(NNetwork & nnet, vector<vector<int> > & images, vector<vector<int> > &labels)
{

    int count = images.size();
    double ratio = 0.99 / 255.0;
    int error = 0;
    for (int i=0; i<count; ++i) {
        int correctLabel = labels[i][0];
        MatrixXd dinputs = normalizeInputs(images[i], ratio);
        MatrixXd target = nnet.predict(dinputs);
        int label; int j;
        double maxCoeff = target.maxCoeff(&label, &j);
        if (label != correctLabel) {
            //cout << "correct:" << correctLabel << ", predict:" << label << endl;
            error += 1;
        }
    }
    int right = count - error;
    double pref = right / (double) count;
    //cout << "total=" << count << ", right=" << right << ",error=" << error << ", pref=" << pref << endl;
    return pref;
}

void nnTrain(NNetwork & nnet, int outputNodeNum, int epochs)
{
    string trainImagePath = "train-images-idx3-ubyte";
    string trainLabelPath = "train-labels-idx1-ubyte";
    vector<vector<int> > images = readMNistFile(trainImagePath);
    vector<vector<int> > labels = readMNistFile(trainLabelPath);

    string testImagePath = "t10k-images-idx3-ubyte";
    string testLabelPath = "t10k-labels-idx1-ubyte";
    vector<vector<int> > testImages = readMNistFile(testImagePath);
    vector<vector<int> > testLabels = readMNistFile(testLabelPath);

    cout << "epochs:" << epochs << endl;
    int count = images.size();
    double ratio = 0.99 / 255.0;
    for (int t=0; t<epochs; ++t) {
        for (int i=0; i<count; ++i) {
            int label = labels[i][0];
            MatrixXd dinputs = normalizeInputs(images[i], ratio);
            VectorXd target = genTarget(outputNodeNum, label);
            nnet.train(dinputs, target);
            if ((i % 10000) == 0) {
                cout << "Iter-"<< t <<", i-" << i << ", time:" << getNowTime();
            }
        }
        double pref = nnTest(nnet, testImages, testLabels);
        cout << "Iter-" << t << ", pref="<< pref << ", time:" << getNowTime();
    }
    testImages.clear();
    testLabels.clear();
    images.clear();
    labels.clear();
}

map<string, string> readConfigFile(string const& path)
{
    ifstream configFile;
    configFile.open(path.c_str());
    string strLine;
    string filepath;
    map<string, string> configMap;
    if(configFile.is_open())
    {
        while (!configFile.eof())
        {
            getline(configFile, strLine);
            size_t pos = strLine.find('=');
            string key = strLine.substr(0, pos);
            string value = strLine.substr(pos + 1);
            configMap[key] = value;
        }
    } else {
        cout << "Cannot open config file:" << path << endl;
    }
    return configMap;
}

vector<int> split(const string &s, char delim) {
    vector<int> elems;
    stringstream ss(s);
    string number;
    while(std::getline(ss, number, delim)) {
        elems.push_back(std::stoi(number));
    }
    return elems;
}

int main(int argc, char const *argv[])
{

    // read config file
    cout << "Config reading..." << endl;
    string path = "nn.conf";
    map<string, string> configMap = readConfigFile(path);
    vector<int> layers = split(configMap["layers"], ',');
    double lr = std::stod(configMap["lr"]);
    int epochs = std::stoi(configMap["epochs"]);

    // train
    cout << "NNetwork train..." << endl;
    NNetwork nnet(layers, lr);
    int layerNum = layers.size();
    int outputNodeNum = layers[layerNum - 1];
    nnTrain(nnet, outputNodeNum, epochs);


    return 0;
}



