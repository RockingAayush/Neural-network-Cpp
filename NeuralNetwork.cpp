/*Activation-feedforward-backpropagation*/

#define e 2.718281828459045235360

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
using namespace std;

// Display array
void display(vector<float> arr){
    for (int i = 0; i < arr.size(); i++){
        cout << arr[i] << " "; 
    }
    cout << endl;
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

    vector<float> inputs = {1,2,3,4};  // 1x4
    vector<vector<float>> weights_first_layer = {
        {-1,0,1,1},
        {2,2,2,-2},
        {3,-3,0,3},
        {4,-4,4,-4}
    };    // 4x5
    vector<vector<float>> weights_second_layer = {
        {-1,0,1,1},
        {2,2,2,-2},
        {3,-3,0,3},
        {4,-4,4,-4}
    };
    
    vector<float> output_first_layer = sigmoid(matMultiply(inputs,weights_first_layer));
    display(output_first_layer);
    vector<float> output_second_layer = ReLU(matMultiply(output_first_layer,weights_second_layer));
    display(output_second_layer);

    return 0;
}