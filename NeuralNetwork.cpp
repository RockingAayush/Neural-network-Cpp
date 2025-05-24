#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>

#define LEARNING_RATE 0.1f
#define EPOCHS 5000

using namespace std;

// Utility functions
vector<vector<float>> transpose(const vector<vector<float>>& mat) {
    int rows = mat.size();
    int cols = mat[0].size();
    vector<vector<float>> result(cols, vector<float>(rows));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result[j][i] = mat[i][j];
    return result;
}

vector<vector<float>> randomMatrix(int rows, int cols) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> dist(0.0f, 1.0f);
    vector<vector<float>> mat(rows, vector<float>(cols));
    for (auto& row : mat)
        for (auto& val : row)
            val = dist(gen);
    return mat;
}

vector<float> randomVector(int size) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> dist(0.0f, 1.0f);
    vector<float> vec(size);
    for (auto& val : vec)
        val = dist(gen);
    return vec;
}

vector<float> matMul(const vector<float>& a, const vector<vector<float>>& b) {
    vector<float> result(b[0].size(), 0.0f);
    for (int j = 0; j < b[0].size(); ++j)
        for (int i = 0; i < a.size(); ++i)
            result[j] += a[i] * b[i][j];
    return result;
}

vector<float> add(const vector<float>& a, const vector<float>& b) {
    vector<float> result(a.size());
    for (int i = 0; i < a.size(); ++i)
        result[i] = a[i] + b[i];
    return result;
}

vector<float> subtract(const vector<float>& a, const vector<float>& b) {
    vector<float> result(a.size());
    for (int i = 0; i < a.size(); ++i)
        result[i] = a[i] - b[i];
    return result;
}

vector<float> multiply(const vector<float>& a, const vector<float>& b) {
    vector<float> result(a.size());
    for (int i = 0; i < a.size(); ++i)
        result[i] = a[i] * b[i];
    return result;
}

vector<float> scalarMul(const vector<float>& a, float s) {
    vector<float> result(a.size());
    for (int i = 0; i < a.size(); ++i)
        result[i] = a[i] * s;
    return result;
}

float mse(const vector<float>& a, const vector<float>& b) {
    float sum = 0;
    for (int i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum / a.size();
}

// Activation functions
vector<float> tanhActivation(const vector<float>& x) {
    vector<float> y(x.size());
    for (int i = 0; i < x.size(); ++i)
        y[i] = tanhf(x[i]);
    return y;
}

vector<float> tanhDerivative(const vector<float>& y) {
    vector<float> dydx(y.size());
    for (int i = 0; i < y.size(); ++i)
        dydx[i] = 1 - y[i] * y[i];
    return dydx;
}

vector<float> sigmoid(const vector<float>& x) {
    vector<float> y(x.size());
    for (int i = 0; i < x.size(); ++i)
        y[i] = 1.0f / (1.0f + expf(-x[i]));
    return y;
}

vector<float> sigmoidDerivative(const vector<float>& y) {
    vector<float> dydx(y.size());
    for (int i = 0; i < y.size(); ++i)
        dydx[i] = y[i] * (1 - y[i]);
    return dydx;
}

// Main loop
int main() {
    // Dataset
    vector<vector<float>> inputs;
    vector<vector<float>> targets;
    for (int i = 0; i < 100; ++i) {
        float x = static_cast<float>((2 * M_PI * i) / 100);
        float y = sinf(x);
        inputs.push_back({static_cast<float>(x / (2 * M_PI))});  // Normalize x to [0, 1]
        targets.push_back({(y + 1.0f) / 2.0f});                   // Normalize y to [0, 1]
    }

    // Neural network architecture
    int input_size = 1;
    int hidden_size = 32;
    int output_size = 1;

    auto w1 = randomMatrix(input_size, hidden_size);
    auto b1 = randomVector(hidden_size);

    auto w2 = randomMatrix(hidden_size, output_size);
    auto b2 = randomVector(output_size);

    // Training loop
    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        float total_loss = 0;
        for (int i = 0; i < inputs.size(); ++i) {
            // Forward
            auto z1 = add(matMul(inputs[i], w1), b1);
            auto a1 = tanhActivation(z1);
            auto z2 = add(matMul(a1, w2), b2);
            auto a2 = sigmoid(z2);

            // Loss
            total_loss += mse(a2, targets[i]);

            // Backpropagation
            auto d_a2 = multiply(subtract(a2, targets[i]), sigmoidDerivative(a2));
            auto d_a1_raw = matMul(d_a2, transpose(w2));
            auto d_a1 = multiply(d_a1_raw, tanhDerivative(a1));

            // Update weights and biases (output layer)
            for (int j = 0; j < hidden_size; ++j)
                for (int k = 0; k < output_size; ++k)
                    w2[j][k] -= LEARNING_RATE * d_a2[k] * a1[j];

            for (int k = 0; k < output_size; ++k)
                b2[k] -= LEARNING_RATE * d_a2[k];

            // Update weights and biases (hidden layer)
            for (int j = 0; j < input_size; ++j)
                for (int k = 0; k < hidden_size; ++k)
                    w1[j][k] -= LEARNING_RATE * d_a1[k] * inputs[i][j];

            for (int k = 0; k < hidden_size; ++k)
                b1[k] -= LEARNING_RATE * d_a1[k];
        }

        if (epoch % 50 == 0 || epoch == 1)
            cout << "Epoch " << epoch << " Loss: " << total_loss / inputs.size() << endl;
    }

    // Saving predictions
    ofstream outfile("predictions.csv");
    outfile << "x,sin_x,predicted\n";

    for (int i = 0; i <= 100; i++) {
        float x = (2 * M_PI * i) / 100;
        float x_norm = x / (2 * M_PI);

        auto a1 = tanhActivation(add(matMul({x_norm}, w1), b1));
        auto a2 = sigmoid(add(matMul(a1, w2), b2));

        float y_pred = 2.0f * a2[0] - 1.0f;  // Denormalize
        float y_actual = sin(x);

        outfile << x << "," << y_actual << "," << y_pred << "\n";
    }

    outfile.close();
    cout << "\nSaved predictions to predictions.csv\n";
    return 0;
}
