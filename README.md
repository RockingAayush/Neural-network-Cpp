# C++ Neural Network Implementation for Educational Purposes  

This compact C++ implementation demonstrates core neural network mechanics through clean, didactic code—ideal for understanding fundamental machine learning concepts without framework abstractions.  

## Pedagogical Value  

The project emphasizes **algorithmic transparency** by implementing:  

1. **Forward Propagation**  
   ```cpp  
   // Matrix multiplication and activation application  
   vector output_first_layer = sigmoid(  
       addVectors(matMultiply(inputs, weights_first_layer), biases_first_layer)  
   );  
   ```
 Shows raw matrix operations for layer computations, clarifying how inputs transform through weights/biases.  

2. **Backpropagation Mechanics**  
   ```cpp  
   // Output layer delta calculation  
   vector delta_output_layer = mulVectors(  
       mulVectors(output_second_layer, error),  
       subVectors(vector(output_second_layer.size(), 1.0f), output_second_layer)  
   );  
   ```
 Explicitly computes gradients using chain rule derivatives, revealing how networks adjust parameters.  

3. **Gradient Descent Implementation**  
   ```cpp  
   // Weight update logic  
   weights_first_layer[i][j] -= LEARNING_RATE * delta_hidden_layer[j] * inputs[i];  
   biases_first_layer[i] -= LEARNING_RATE * delta_hidden_layer[i];  
   ```
 Demonstrates parameter optimization without black-box optimizers.  

## Code Organization  

### Core Components  
**Mathematical Operations Library**  
```cpp  
vector matMultiply(vector matA, vector> matB) {  
    // Implements matrix multiplication  
    for(int j=0; j weights_first_layer = generateRandomNormalWeights(INPUT_SIZE, HIDDEN_SIZE);  
   ```
 Adjust `INPUT_SIZE`/`HIDDEN_SIZE` to explore underfitting/overfitting.  

2. **Activation Function Comparison**  
   Replace `sigmoid` with `ReLU` in forward passes:  
   ```cpp  
   vector output_first_layer = ReLU(...);  
   ```
 Observe training dynamics changes from nonlinearity differences.  

This implementation serves as a **conceptual scaffold**—learners can extend it with momentum terms, batch processing, or additional layers while retaining visibility into core mechanics.
