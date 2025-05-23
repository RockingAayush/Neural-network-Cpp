/*Activation-feedforward-backpropagation*/

#define e 2.718281828459045235360

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <random>

using namespace std;

// Utility functions
// Display array
void display(vector<float> arr){
    for (int i = 0; i < arr.size(); i++){
        cout << arr[i] << " "; 
    }
    cout << endl;
}

// Generate random normal weight matrix
vector<vector <float>> generateRandomNormalWeights(int rows,int columns){
    vector<vector <float>> matrix(rows,vector<float>(columns));
    
    random_device rd; // obtain a random seed
    mt19937 gen(42); // Mersenne Twister RNG (Replace 42 by rd() for random number)
    normal_distribution<float> dist(0.0f, 1.0f); // mean 0, stddev 1

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            matrix[i][j] = dist(gen);
        }
    }

    return matrix;
}

// Generate random normal bias array
vector <float> generateRandomNormalBiases(int columns){
    vector <float> biases(columns,0.0f);
    
    random_device rd; // obtain a random seed
    mt19937 gen(42); // Mersenne Twister RNG (Replace 42 by rd() for random number)
    normal_distribution<float> dist(0.0f, 1.0f); // mean 0, stddev 1

    
    for (int j = 0; j < columns; ++j) {
        biases[j] = dist(gen);
    }

    return biases;
}

// Matrix multiplication (A x B)
vector<float> matMultiply(vector<float> matA, vector<vector<float>> matB){
    int matA_rows = 1;
    int matB_rows = matB.size();
    int matA_columns = matA.size();
    int matB_columns = matB[0].size();

    if (matA_columns != matB_rows){
        throw invalid_argument("Dimensions for matrix multiplication doesn't match.");
    }
    
    vector<float> result(matB_columns,0.0f);

    // Multiplication logic
    
    for (int j = 0; j < matB_columns; j++) {    // matB_columns or result columns
        for (int k = 0; k < matA_columns; k++) {    // shared dimension
            result[j] += matA[k] * matB[k][j];
        }
    }
    
    return result;
}

// Adding two vectors
vector <float> addVectors(vector <float> vecA, vector <float> vecB){
    if (vecA.size() != vecB.size()){
        throw invalid_argument("Dimensions for vector addition doesn't match.");
    }
    
    vector <float> result(vecA.size());

    for (int i = 0; i < vecA.size(); i++){
        result[i] = vecA[i] + vecB[i];
    }
    return result;
}


// Relu activation function
vector<float> ReLU(vector<float> arr){
    for (int i = 0; i < arr.size(); i++){
    if (arr[i]<0){
        arr[i] = 0;
        }
    }
    return arr;
}

// Sigmoid activation function
vector<float> sigmoid(vector<float> arr){
    for (int i = 0; i < arr.size(); i++){
        arr[i] = 1/(1+pow(e,arr[i]));
    }
    return arr;
}

int main(){

    // Initialization of weights and biases
    vector<vector<float>> weights_first_layer = generateRandomNormalWeights(4,4);
    vector<float> biases_first_layer = generateRandomNormalBiases(4);  

    vector<vector<float>> weights_second_layer = generateRandomNormalWeights(4,4);
    vector<float> biases_second_layer = generateRandomNormalBiases(4);
    
    // Feedforward
    // input layer
    vector<float> inputs = {1,2,3,4};
     
    // hidden layer
    vector<float> output_first_layer = sigmoid(addVectors(matMultiply(inputs,weights_first_layer),biases_first_layer));
    display(output_first_layer);
    
    // output layer
    vector<float> output_second_layer = sigmoid(addVectors(matMultiply(output_first_layer,weights_second_layer),biases_second_layer));
    display(output_second_layer);

    
    return 0;
}