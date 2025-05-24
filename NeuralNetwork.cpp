#define e 2.718281828459045235360
#define LEARNING_RATE 0.01
#define EPOCHS 100

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <random>

using namespace std;

// Utility functions
// Display array
void displayArr(vector<float> arr){
    for (int i = 0; i < arr.size(); i++){
        cout << arr[i] << " "; 
    }
    cout << endl;
}

// Display matrix
void displayMatrix(const vector<vector<float>>& matrix) {
    for (const auto& row : matrix) {
        for (float val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
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

// Subtracting two vectors
vector <float> subVectors(vector <float> vecA, vector <float> vecB){
    if (vecA.size() != vecB.size()){
        throw invalid_argument("Dimensions for vector addition doesn't match.");
    }
    
    vector <float> result(vecA.size());

    for (int i = 0; i < vecA.size(); i++){
        result[i] = vecA[i] - vecB[i];
    }
    return result;
}

// Multiplying two vectors
vector <float> mulVectors(vector <float> vecA, vector <float> vecB){
    if (vecA.size() != vecB.size()){
        throw invalid_argument("Dimensions for vector addition doesn't match.");
    }
    
    vector <float> result(vecA.size());

    for (int i = 0; i < vecA.size(); i++){
        result[i] = vecA[i] * vecB[i];
    }
    return result;
}

// Sum of all elements of a vector
float sumElements(vector <float> arr){
    float sum = 0;
    for (int i = 0; i < arr.size(); i++){
        sum += arr[i];
    }
    return sum;
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
        arr[i] = 1/(1+pow(e,-arr[i]));
    }
    return arr;
}

int main(){

    vector<float> inputs  = {0.3f, 0.3f, 0.3f, 0.3f};
    vector<float> targets = {0.3f, 0.3f, 0.3f, 0.3f};

    // Initialization of weights and biases
    vector<vector<float>> weights_first_layer = generateRandomNormalWeights(4,4);
    vector<float> biases_first_layer = generateRandomNormalBiases(4);  

    vector<vector<float>> weights_second_layer = generateRandomNormalWeights(4,4);
    vector<float> biases_second_layer = generateRandomNormalBiases(4);
    
    // Training loop
    for (int i = 1; i <= EPOCHS; i++){
        // Feedforward
        // Hidden layer
        vector<float> output_first_layer = sigmoid(addVectors(matMultiply(inputs,weights_first_layer),biases_first_layer));
        
        // Output layer
        vector<float> output_second_layer = sigmoid(addVectors(matMultiply(output_first_layer,weights_second_layer),biases_second_layer));
    
        // Calculating output loss
        vector<float> error = subVectors(output_second_layer,targets);
        float loss = sumElements(mulVectors(mulVectors(error,error),vector<float>(error.size(),0.5f)));  // Display loss
        cout << "EPOCH: " << i << " Loss: " << loss << endl;
        
        vector<float> delta_output_layer = mulVectors(mulVectors(output_second_layer,error),subVectors(vector<float>(output_second_layer.size(), 1.0f),output_second_layer));  // (output)delj = Oj(1-Oj)(Oj-tj) for sigmoid
        
        // Calculating delta for hidden layer
        vector<float> delta_hidden_layer = mulVectors(output_first_layer,subVectors(vector<float>(output_first_layer.size(), 1.0f),output_first_layer));  // (hidden)delj = Oj(1-Oj)

        for (int j = 0; j < delta_hidden_layer.size(); j++){
            float sum = 0;
            for (int k = 0; k < delta_output_layer.size(); k++){
                sum += weights_second_layer[k][j] * delta_output_layer[k]; 
            }
            delta_hidden_layer[j] *= sum;
        }

        
        // Weight updation hidden layer
        for (int i = 0; i < weights_first_layer.size(); i++){
            for (int j = 0; j < weights_first_layer[0].size(); j++){
                weights_first_layer[i][j] -= LEARNING_RATE * delta_hidden_layer[j] * inputs[i]; 
            }
            biases_first_layer[i] -= LEARNING_RATE * delta_hidden_layer[i];
        }

        // Weight updation output layer
        for (int i = 0; i < weights_second_layer.size(); i++){
            for (int j = 0; j < weights_second_layer[0].size(); j++){
                weights_second_layer[i][j] -= LEARNING_RATE * delta_output_layer[j] * output_first_layer[i]; 
            }
            biases_second_layer[i] -= LEARNING_RATE * delta_output_layer[i];
        }
    }
    
    
    vector<float> output_first_layer_final = sigmoid(addVectors(matMultiply(inputs,weights_first_layer),biases_first_layer));   
    vector<float> output_second_layer_final = sigmoid(addVectors(matMultiply(output_first_layer_final,weights_second_layer),biases_second_layer));
    vector<float> error = subVectors(output_second_layer_final,targets);  //Oj - tj

    float loss = sumElements(mulVectors(mulVectors(error,error),vector<float>(error.size(),0.5f)));  // Display loss
    cout << "Final Loss: " << loss;
    
    // To display updated weights and biases
    /*
    cout << "Updated weights first layer: " << endl;
    displayMatrix(weights_first_layer);
    cout << "Updated biases first layer: " << endl;
    displayArr(biases_first_layer);
    cout << endl;
    cout << "Updated weights second layer: " << endl;
    displayMatrix(weights_second_layer);
    cout << "Updated biases second layer: " << endl;
    displayArr(biases_second_layer);
    */


    return 0;
}
